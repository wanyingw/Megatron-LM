# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Memory footprint and MFU/HFU accounting for the NS distribution strategies.

Companion to ``bench_ns_lsh.py``: same model, same LPT ownership, same timing
methodology, but additionally reports

  1. which matrices land on which GPU, per mode (ownership + LSH home assignment),
  2. measured peak CUDA memory per mode (max over ranks, so the worst GPU),
  3. issued / useful FLOPs per GPU, and MFU / HFU against a *measured* bf16 GEMM
     peak on the same hardware.

Definitions used here, stated explicitly because "MFU" for an optimizer-side
kernel is not the usual training-step definition:

  useful   the irreducible work: orthogonalizing each matrix the group owns
           exactly once, divided evenly across the group_size GPUs that are
           occupied for the duration. Mode-independent. This is exactly the
           ``useful`` of ``bench_ns_strategies.flop_model``.
  issued   the FLOPs a GPU actually executes. Mode-dependent.
  MFU      useful / (elapsed * peak_flops)
  HFU      issued / (elapsed * peak_flops)

  A replicated weight (shard_count == 1, the MoE router) is orthogonalized
  identically on every rank in every mode with no collective, so issued ==
  useful for it. That is ``flop_model``'s convention and is kept here.

peak_flops is MEASURED on this hardware (largest achieved bf16 GEMM rate over a
size sweep), not taken from a datasheet, so MFU/HFU are fractions of what these
GPUs actually deliver on dense bf16 GEMMs.

Launch exactly like bench_ns_lsh.py (world == group size)::

    torchrun --nproc-per-node 4 bench_ns_mem_mfu.py --group egtp --num-ns-steps 16
"""
import argparse
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ns_strategies import (  # noqa: E402
    build_model_matrices,
    flop_model,
    ns_cost,
    ns_step_flops,
    owned_matrices,
    time_strategy,
)
from bench_ns_lsh import lpt_homes, run_ns_batched, time_profile_lsh  # noqa: E402

from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)
from megatron.core.optimizer.layer_sharded_a2a import (  # noqa: E402
    layer_sharded_all_to_all_bwd,
    layer_sharded_all_to_all_fwd,
)

MIB = 1024.0 * 1024.0
LSH_MODES = ["layer_sharded", "layer_sharded_batched"]


# --------------------------------------------------------------------------------------
# Peak-FLOPs calibration (the MFU/HFU denominator)
# --------------------------------------------------------------------------------------


def measure_gemm_peak(iters: int = 20) -> Tuple[float, str]:
    """Return (best achieved bf16 TFLOP/s, description) over a square-GEMM sweep.

    NS at fp32_matmul_precision="medium" casts to bf16 and issues ``torch.addmm``,
    so both ``matmul`` and ``addmm`` are swept and the best rate is taken.
    """
    best, best_desc = 0.0, ""
    for n in (4096, 8192, 12288, 16384):
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        for name, fn in (("matmul", lambda: a @ b), ("addmm", lambda: torch.addmm(c, a, b))):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters
            tflops = 2.0 * n**3 / 1e12 / (ms / 1e3)
            if tflops > best:
                best, best_desc = tflops, f"{name} {n}x{n}x{n}"
        del a, b, c
        torch.cuda.empty_cache()
    return best, best_desc


# --------------------------------------------------------------------------------------
# Memory measurement
# --------------------------------------------------------------------------------------


def measure_mem(once) -> Tuple[float, float]:
    """Return (base bytes live before the call, peak bytes during it).

    A warmup call runs first so cuBLAS workspaces and any lazily-created buffers
    are already resident and do not inflate the delta.
    """
    once()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    once()
    torch.cuda.synchronize()
    return float(base), float(torch.cuda.max_memory_allocated())


def dist_max(value: float, group) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX, group=group)
    return float(tensor.item())


def make_dupdist_once(shard, group, mode, steps, coeff, shard_count, use_syrk):
    partition_dim = None if shard_count == 1 else 0

    def once():
        newton_schulz_tp(
            shard,
            steps=steps,
            coefficient_type=coeff,
            tp_group=group,
            partition_dim=partition_dim,
            tp_mode=mode,
            use_syrk=use_syrk,
        )

    return once


def make_lsh_once(profile, group, steps, coeff, ns_batch, use_syrk):
    """Mirror of ``time_profile_lsh``'s inner step, exposed so memory can be measured."""
    world, rank = group.size(), group.rank()
    sharded = [((r, c), sc) for (r, c), sc in profile if sc > 1]
    repl = [((r, c), sc) for (r, c), sc in profile if sc == 1]
    shards = [torch.randn(r, c, device="cuda", dtype=torch.float32) for (r, c), _ in sharded]
    repl_mats = [torch.randn(r, c, device="cuda", dtype=torch.float32) for (r, c), _ in repl]
    homes = lpt_homes([ns_cost(m) for m in sharded], world)

    def once():
        if shards:
            complete, my_idx = layer_sharded_all_to_all_fwd(
                shards, homes, rank, world, group, gtp_dim=0
            )
            outs = run_ns_batched(complete, steps, coeff, ns_batch, use_syrk=use_syrk)
            layer_sharded_all_to_all_bwd(
                outs, my_idx, shards, homes, rank, world, group, gtp_dim=0
            )
        if repl_mats:
            run_ns_batched(repl_mats, steps, coeff, ns_batch, use_syrk=use_syrk)

    return once, shards, repl_mats, homes, sharded


# --------------------------------------------------------------------------------------
# FLOP accounting
# --------------------------------------------------------------------------------------


def profile_flops(sig, mode: str, steps: int, group_size: int) -> Tuple[float, float, float]:
    """Return (issued mean per GPU, issued max over GPUs, useful per GPU) for a profile.

    dup/dist are symmetric across ranks, so mean == max. layer_sharded is not: a
    home rank pays the full NS of every matrix homed to it and nothing for the rest.
    """
    expanded = [m for m, n in sig for _ in range(n)]
    useful = sum(flop_model(m, "duplicated", steps, group_size)[1] for m in expanded)

    if mode in ("duplicated", "distributed"):
        issued = sum(flop_model(m, mode, steps, group_size)[0] for m in expanded)
        return issued, issued, useful

    sharded = [m for m in expanded if m[1] > 1]
    repl = [m for m in expanded if m[1] == 1]
    homes = lpt_homes([ns_cost(m) for m in sharded], group_size)
    per_rank = [0.0] * group_size
    for idx, ((rows, cols), shard_count) in enumerate(sharded):
        full_rows, full_cols = rows * shard_count, cols
        fm, fn = min(full_rows, full_cols), max(full_rows, full_cols)
        per_rank[homes[idx]] += ns_step_flops(fm, fn) * steps
    # Replicated weights run on every rank in every mode.
    repl_each = sum(flop_model(m, "duplicated", steps, group_size)[0] for m in repl)
    per_rank = [x + repl_each for x in per_rank]
    return sum(per_rank) / group_size, max(per_rank), useful


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["gtp", "egtp"], required=True)
    parser.add_argument("--modelled-world-size", type=int, default=12288)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gtp", type=int, default=64)
    parser.add_argument("--ep", type=int, default=64)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--egtp", type=int, default=4)
    parser.add_argument("--shard-latent-proj", action="store_true")
    parser.add_argument("--num-ns-steps", type=int, default=16)
    parser.add_argument("--coefficient-type", type=str, default="polar_express")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ns-batch-size", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=["duplicated", "distributed"])
    parser.add_argument(
        "--fp32-matmul-prec", type=str, default="medium", choices=["medium", "high", "highest"]
    )
    parser.add_argument("--use-syrk", action="store_true")
    config = parser.parse_args()

    torch.set_float32_matmul_precision(config.fp32_matmul_prec)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    dense, expert = build_model_matrices(config)
    if config.group == "gtp":
        matrices, group_size = dense, config.gtp
        dp_size = config.modelled_world_size // (config.tp * config.gtp)
    else:
        matrices, group_size = expert, config.egtp
        dp_size = config.modelled_world_size // (config.etp * config.egtp * config.ep)
    assert world == group_size, f"launch with {group_size} ranks; got {world}"

    profiles: Dict[Tuple, List[int]] = {}
    for dp_rank in range(dp_size):
        sig = tuple(sorted(Counter(owned_matrices(matrices, dp_size, dp_rank)).items()))
        profiles.setdefault(sig, []).append(dp_rank)
    ordered = sorted(profiles.items(), key=lambda kv: -len(kv[1]))

    log(
        f"group={config.group} group_size={group_size} dp_size={dp_size} "
        f"profiles={len(profiles)} ns_steps={config.num_ns_steps} "
        f"coeff={config.coefficient_type} prec={torch.get_float32_matmul_precision()} "
        f"nsb={config.ns_batch_size} use_syrk={config.use_syrk}"
    )
    log(f"device={torch.cuda.get_device_name()} "
        f"total_hbm={torch.cuda.get_device_properties(0).total_memory / MIB:.0f} MiB")

    # ---------------------------------------------------------------- peak calibration
    peak_tflops, peak_desc = measure_gemm_peak()
    peak_tflops = dist_max(peak_tflops, group)
    log(f"\nMEASURED bf16 GEMM PEAK: {peak_tflops:.1f} TFLOP/s per GPU (best: {peak_desc})")
    log("  MFU/HFU below are fractions of this measured rate, not of a datasheet number.")

    # ---------------------------------------------------------------- ownership dump
    log("\n" + "=" * 100)
    log("OWNERSHIP: which matrices a dp_rank owns, and where they live per mode")
    log("=" * 100)
    log(f"{'ranks':>6}  {'local shard (per GPU)':<26}{'full matrix':<16}{'n':>4}"
        f"{'MiB/shard':>11}{'MiB/full':>10}")
    for sig, dp_ranks in ordered:
        first = True
        for (shape, shard_count), n in sorted(sig, key=lambda kv: -ns_cost(kv[0])):
            rows, cols = shape
            full = f"{rows * shard_count}x{cols}"
            shard_mib = rows * cols * 4 / MIB
            full_mib = rows * shard_count * cols * 4 / MIB
            tag = "(replicated)" if shard_count == 1 else ""
            log(f"{len(dp_ranks) if first else '':>6}  {f'{rows}x{cols}':<26}{full:<16}{n:>4}"
                f"{shard_mib:>11.1f}{full_mib:>10.1f}  {tag}")
            first = False
        log("")

    # ---------------------------------------------------------------- lsh home mapping
    log("=" * 100)
    log("LAYER_SHARDED HOME ASSIGNMENT (LPT over the group; GPUs not listed hold no")
    log("complete matrix and idle through the NS, but still pay both all_to_alls)")
    log("=" * 100)
    for sig, dp_ranks in ordered:
        expanded = [m for m, n in sig for _ in range(n)]
        sharded = [m for m in expanded if m[1] > 1]
        repl = [m for m in expanded if m[1] == 1]
        homes = lpt_homes([ns_cost(m) for m in sharded], group_size)
        by_home: Dict[int, List] = {}
        for idx, m in enumerate(sharded):
            by_home.setdefault(homes[idx], []).append(m)
        owns = " + ".join(f"{n}x[{e[0][0]}x{e[0][1]}]" for e, n in sig)
        log(f"\nprofile ({len(dp_ranks)} dp_ranks): {owns}")
        log(f"  {len(sharded)} sharded matrices over {group_size} GPUs -> "
            f"{len(by_home)} GPUs hold a complete matrix, {group_size - len(by_home)} idle")
        for home in sorted(by_home):
            names = Counter(f"{m[0][0] * m[1]}x{m[0][1]}" for m in by_home[home])
            mib = sum(m[0][0] * m[1] * m[0][1] * 4 / MIB for m in by_home[home])
            log(f"    gpu[{home:>3}] : "
                f"{' + '.join(f'{c}x[{s}]' for s, c in names.items()):<40} {mib:>9.1f} MiB complete")
        if repl:
            names = Counter(f"{m[0][0]}x{m[0][1]}" for m in repl)
            log(f"    ALL GPUs  : {' + '.join(f'{c}x[{s}]' for s, c in names.items())} "
                f"(replicated, local NS on every rank in every mode)")

    # ---------------------------------------------------------------- per-shape dup/dist
    distinct = sorted({m for sig in profiles for m, _ in sig}, key=lambda e: -ns_cost(e))
    per_shape_ms: Dict[Tuple, Dict[str, float]] = {}
    per_shape_mem: Dict[Tuple, Dict[str, float]] = {}
    log("\n" + "=" * 100)
    log("PER-SHAPE, dup/dist: time and transient memory above the resident shard")
    log("=" * 100)
    log(f"{'local shard':>14}{'full':>16}{'mode':>13}{'ms':>10}"
        f"{'shard MiB':>11}{'transient MiB':>15}{'peak MiB':>11}")
    for matrix in distinct:
        (rows, cols), shard_count = matrix
        shard = torch.randn((rows, cols), device="cuda", dtype=torch.float32)
        per_shape_ms[matrix] = {}
        per_shape_mem[matrix] = {}
        for mode in config.modes:
            ms = time_strategy(
                shard, group, mode, config.num_ns_steps, config.coefficient_type,
                config.iters, config.warmup, shard_count, use_syrk=config.use_syrk,
            )
            once = make_dupdist_once(
                shard, group, mode, config.num_ns_steps, config.coefficient_type,
                shard_count, config.use_syrk,
            )
            base, peak = measure_mem(once)
            transient = dist_max(peak - base, group)
            per_shape_ms[matrix][mode] = ms
            per_shape_mem[matrix][mode] = transient
            log(f"{f'{rows}x{cols}':>14}{f'{rows * shard_count}x{cols}':>16}{mode:>13}{ms:>10.2f}"
                f"{rows * cols * 4 / MIB:>11.1f}{transient / MIB:>15.1f}"
                f"{(rows * cols * 4 + transient) / MIB:>11.1f}")
        del shard
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- per-profile
    log("\n" + "=" * 100)
    log("PER-PROFILE: time, memory, FLOPs, MFU/HFU")
    log("=" * 100)

    all_modes = list(config.modes) + LSH_MODES
    totals: Dict[str, List[float]] = {m: [] for m in all_modes}
    mem_totals: Dict[str, List[float]] = {m: [] for m in all_modes}
    rows_out = []

    for sig, dp_ranks in ordered:
        expanded = [m for m, n in sig for _ in range(n)]
        owns = " + ".join(f"{n}x[{e[0][0]}x{e[0][1]}]" for e, n in sig)
        resident = sum(e[0][0] * e[0][1] * 4 * n for e, n in sig)  # fp32 momentum shards
        entry = {"owns": owns, "ranks": len(dp_ranks), "resident": resident, "modes": {}}

        for mode in config.modes:
            ms = sum(per_shape_ms[m][mode] * n for m, n in sig)
            # dup/dist process one matrix at a time: peak = resident + worst single transient
            transient = max(per_shape_mem[m][mode] for m, _ in sig)
            iss_mean, iss_max, useful = profile_flops(sig, mode, config.num_ns_steps, group_size)
            totals[mode].append(ms)
            mem_totals[mode].append(resident + transient)
            entry["modes"][mode] = (ms, resident + transient, iss_mean, iss_max, useful)

        for mode, nsb in zip(LSH_MODES, (1, config.ns_batch_size)):
            ms = time_profile_lsh(
                expanded, group, config.num_ns_steps, config.coefficient_type,
                config.iters, config.warmup, ns_batch=nsb, use_syrk=config.use_syrk,
            )
            once, shards, repl_mats, _, _ = make_lsh_once(
                expanded, group, config.num_ns_steps, config.coefficient_type,
                nsb, config.use_syrk,
            )
            base, peak = measure_mem(once)
            peak = dist_max(peak, group)
            del once, shards, repl_mats
            torch.cuda.empty_cache()
            iss_mean, iss_max, useful = profile_flops(sig, mode, config.num_ns_steps, group_size)
            totals[mode].append(ms)
            mem_totals[mode].append(peak)
            entry["modes"][mode] = (ms, peak, iss_mean, iss_max, useful)

        rows_out.append(entry)

    hdr = (f"{'mode':>22}{'ms':>10}{'peak MiB':>11}{'issued TF':>11}{'useful TF':>11}"
           f"{'redund':>8}{'HFU%':>8}{'MFU%':>8}")
    for entry in rows_out:
        log(f"\n{entry['ranks']} dp_ranks | {entry['owns']}")
        log(f"  resident fp32 momentum shards: {entry['resident'] / MIB:.1f} MiB (same in every mode)")
        log(hdr)
        for mode in all_modes:
            ms, peak, iss_mean, iss_max, useful = entry["modes"][mode]
            sec = ms / 1e3
            hfu = iss_mean / 1e12 / sec / peak_tflops * 100
            mfu = useful / 1e12 / sec / peak_tflops * 100
            note = ""
            if mode in LSH_MODES and iss_max != iss_mean:
                note = f"  (busiest GPU issues {iss_max / 1e12:.1f} TF)"
            log(f"{mode:>22}{ms:>10.2f}{peak / MIB:>11.1f}{iss_mean / 1e12:>11.2f}"
                f"{useful / 1e12:>11.2f}{iss_mean / useful:>7.2f}x{hfu:>8.1f}{mfu:>8.1f}{note}")

    # ---------------------------------------------------------------- step summary
    log("\n" + "=" * 100)
    log("STEP COST = slowest profile;  STEP MEMORY = worst profile's peak")
    log("=" * 100)
    base_ms = max(totals[config.modes[0]])
    log(f"{'mode':>22}{'step ms':>10}{'speedup':>9}{'worst peak MiB':>16}{'HFU%':>8}{'MFU%':>8}")
    for mode in all_modes:
        worst_i = max(range(len(totals[mode])), key=lambda i: totals[mode][i])
        ms = totals[mode][worst_i]
        entry = rows_out[worst_i]
        _, _, iss_mean, _, useful = entry["modes"][mode]
        sec = ms / 1e3
        hfu = iss_mean / 1e12 / sec / peak_tflops * 100
        mfu = useful / 1e12 / sec / peak_tflops * 100
        log(f"{mode:>22}{ms:>10.2f}{base_ms / ms:>8.2f}x{max(mem_totals[mode]) / MIB:>16.1f}"
            f"{hfu:>8.1f}{mfu:>8.1f}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
