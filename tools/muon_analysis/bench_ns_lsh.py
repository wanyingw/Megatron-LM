# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""layer_sharded extension of bench_ns_strategies.py — same model, same framing.

Reuses the original harness's model description, LPT ownership, flop model and
per-shape timing for duplicated/distributed. Adds ``layer_sharded`` with the
composition it actually has:

  dup/dist  process the profile's matrices one at a time with the whole group
            -> per-shape timing x count sums correctly (their composition).
  lsh       processes ALL of a profile's matrices concurrently: whole matrices
            are LPT-placed onto homes across the group, one fused all_to_all in,
            batched Newton-Schulz per home, one all_to_all back. Per-shape-x-count
            would serialize the homes and overstate lsh by ~group_size, so lsh is
            timed PER PROFILE, end to end, with the group's real collective.

Replicated weights (shard_count == 1, the router) run a local NS on every rank in
every mode, identically; they are timed once and added to every profile total.

Launch exactly like bench_ns_strategies.py (world == group size):
  torchrun --nproc-per-node 4 bench_ns_lsh.py --group egtp --num-ns-steps 16
Requires: emerging_optimizers >= 0.3.0 (batched 3-D NS) and megatron-core on
PYTHONPATH for layer_sharded_a2a (pure torch).
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
    HAVE_USE_SYRK,
    build_model_matrices,
    flop_model,
    ns_cost,
    owned_matrices,
    time_strategy,
)

from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz  # noqa: E402
from megatron.core.optimizer.layer_sharded_a2a import (  # noqa: E402
    layer_sharded_all_to_all_bwd,
    layer_sharded_all_to_all_fwd,
)


def lpt_homes(full_costs: List[int], world: int) -> Dict[int, int]:
    loads = [0] * world
    homes = {}
    for idx in sorted(range(len(full_costs)), key=lambda i: -full_costs[i]):
        h = min(range(world), key=lambda r: loads[r])
        homes[idx] = h
        loads[h] += full_costs[idx]
    return homes


def run_ns_batched(
    mats: List[torch.Tensor], steps, coeff, batch: int, use_syrk: bool = False
) -> List[torch.Tensor]:
    """Group same-shape matrices, stack, run batched NS (EO >= 0.3), unstack in order."""
    by_shape: Dict[Tuple[int, int], List[int]] = {}
    for i, m in enumerate(mats):
        by_shape.setdefault(tuple(m.shape), []).append(i)
    out: List[torch.Tensor] = [None] * len(mats)  # type: ignore[list-item]
    for shape, idxs in by_shape.items():
        for lo in range(0, len(idxs), batch):
            chunk = idxs[lo : lo + batch]
            if len(chunk) == 1:
                out[chunk[0]] = newton_schulz(mats[chunk[0]], steps, coeff, use_syrk=use_syrk)
            else:
                stacked = torch.stack([mats[i] for i in chunk])
                res = newton_schulz(stacked, steps, coeff, use_syrk=use_syrk)
                for j, i in enumerate(chunk):
                    out[i] = res[j]
    return out


def time_profile_lsh(
    profile, group, steps, coeff, iters, warmup, ns_batch, use_syrk: bool = False
) -> float:
    """Median ms for one FULL profile step under layer sharding."""
    world, rank = group.size(), group.rank()
    sharded = [((r, c), sc) for (r, c), sc in profile if sc > 1]
    repl = [((r, c), sc) for (r, c), sc in profile if sc == 1]
    for (_, _), sc in sharded:
        assert sc == world, f"shard_count {sc} != world {world}"
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

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    timings = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.distributed.barrier(group=group)
        start.record()
        once()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


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
        "--fp32-matmul-prec", type=str, default="medium",
        choices=["medium", "high", "highest"])
    parser.add_argument(
        "--use-syrk", action="store_true",
        help="Route Newton-Schulz through the Triton batched-SYRK kernel instead of GEMMs "
        "for duplicated/distributed and both layer_sharded modes. Requires an "
        "emerging_optimizers install with batched-SYRK support (e.g. PR #276).")
    config = parser.parse_args()

    assert not config.use_syrk or HAVE_USE_SYRK, (
        "--use-syrk requires an emerging_optimizers install whose newton_schulz_tp accepts "
        "use_syrk (e.g. checkout PR #276); this install's signature does not."
    )

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
    log(f"group={config.group} group_size={group_size} dp_size={dp_size} "
        f"profiles={len(profiles)} ns_steps={config.num_ns_steps} "
        f"coeff={config.coefficient_type} prec={torch.get_float32_matmul_precision()} "
        f"nsb={config.ns_batch_size} use_syrk={config.use_syrk}")

    # dup/dist: original per-shape composition.
    distinct = sorted({m for sig in profiles for m, _ in sig}, key=lambda e: -ns_cost(e))
    per_shape: Dict[Tuple, Dict[str, float]] = {}
    for matrix in distinct:
        (rows, cols), sc = matrix
        shard = torch.randn((rows, cols), device="cuda", dtype=torch.float32)
        per_shape[matrix] = {}
        for mode in config.modes:
            per_shape[matrix][mode] = time_strategy(
                shard, group, mode, config.num_ns_steps, config.coefficient_type,
                config.iters, config.warmup, sc, use_syrk=config.use_syrk)
        del shard

    # lsh: per-profile end-to-end.
    lsh_modes = ["layer_sharded", "layer_sharded_batched"]
    log(f"\n{'ranks':>6}  {'owns':<44}" +
        "".join(f"{m:>13}" for m in config.modes) +
        "".join(f"{m:>22}" for m in lsh_modes))
    totals = {m: [] for m in config.modes + lsh_modes}
    for sig, dp_ranks in sorted(profiles.items(), key=lambda kv: -len(kv[1])):
        expanded = [m for m, n in sig for _ in range(n)]
        owns = " + ".join(f"{n}x[{e[0][0]}x{e[0][1]}]" for e, n in sig)
        line = f"{len(dp_ranks):>6}  {owns:<44}"
        for mode in config.modes:
            t = sum(per_shape[m][mode] * n for m, n in sig)
            totals[mode].append(t)
            line += f"{t:>12.2f}m"
        t_lsh = time_profile_lsh(expanded, group, config.num_ns_steps,
                                 config.coefficient_type, config.iters,
                                 config.warmup, ns_batch=1, use_syrk=config.use_syrk)
        totals["layer_sharded"].append(t_lsh)
        line += f"{t_lsh:>21.2f}m"
        t_lsh_b = time_profile_lsh(expanded, group, config.num_ns_steps,
                                   config.coefficient_type, config.iters,
                                   config.warmup, config.ns_batch_size, use_syrk=config.use_syrk)
        totals["layer_sharded_batched"].append(t_lsh_b)
        line += f"{t_lsh_b:>21.2f}m"
        log(line)

    log("\nSTEP COST (slowest profile):")
    base = max(totals[config.modes[0]])
    for m in config.modes + lsh_modes:
        worst = max(totals[m])
        log(f"  {m:>22}: {worst:9.2f} ms   speedup vs {config.modes[0]}: {base/worst:5.2f}x")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
