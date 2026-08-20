# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ownership, LSH home assignment and FLOP accounting — no GPUs, no collectives.

Everything here is a pure function of the model description and the LPT packer, so
it runs on one process and does not need the sharding degree's worth of ranks.
Pair it with measured per-profile times (``--times-json``) and a measured peak
(``--peak-tflops``) to get MFU/HFU without re-running the distributed benchmark.

Definitions match ``bench_ns_mem_mfu.py``:
  useful  = orthogonalize each matrix once, divided across group_size GPUs
  issued  = what one GPU actually executes (mode-dependent)
  MFU/HFU = useful/issued over (elapsed * peak_flops)
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ns_strategies import (  # noqa: E402
    build_model_matrices,
    flop_model,
    ns_cost,
    ns_step_flops,
    owned_matrices,
)
from bench_ns_lsh import lpt_homes  # noqa: E402
from bench_ns_mem_mfu import profile_flops  # noqa: E402

MIB = 1024.0 * 1024.0
LSH_MODES = ["layer_sharded", "layer_sharded_batched"]


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
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--times-json", type=str, default=None)
    config = parser.parse_args()

    dense, expert = build_model_matrices(config)
    if config.group == "gtp":
        matrices, group_size = dense, config.gtp
        dp_size = config.modelled_world_size // (config.tp * config.gtp)
    else:
        matrices, group_size = expert, config.egtp
        dp_size = config.modelled_world_size // (config.etp * config.egtp * config.ep)

    profiles: Dict[Tuple, List[int]] = {}
    for dp_rank in range(dp_size):
        sig = tuple(sorted(Counter(owned_matrices(matrices, dp_size, dp_rank)).items()))
        profiles.setdefault(sig, []).append(dp_rank)
    ordered = sorted(profiles.items(), key=lambda kv: -len(kv[1]))

    times = {}
    if config.times_json:
        with open(config.times_json) as handle:
            times = json.load(handle)

    print(f"group={config.group} group_size={group_size} dp_size={dp_size} "
          f"profiles={len(profiles)} ns_steps={config.num_ns_steps}")
    print(f"total matrices in model = {len(matrices)}, dp shards = {dp_size}, "
          f"mean matrices per shard = {len(matrices) / dp_size:.2f}")

    print("\n" + "=" * 104)
    print("OWNERSHIP + LAYER_SHARDED HOMES")
    print("=" * 104)
    for sig, dp_ranks in ordered:
        expanded = [m for m, n in sig for _ in range(n)]
        sharded = [m for m in expanded if m[1] > 1]
        repl = [m for m in expanded if m[1] == 1]
        homes = lpt_homes([ns_cost(m) for m in sharded], group_size)
        by_home: Dict[int, List] = {}
        for idx, m in enumerate(sharded):
            by_home.setdefault(homes[idx], []).append(m)
        owns = " + ".join(f"{n}x[{e[0][0]}x{e[0][1]}]" for e, n in sig)
        resident = sum(e[0][0] * e[0][1] * 4 * n for e, n in sig) / MIB
        print(f"\nprofile: {len(dp_ranks)} dp_ranks | {owns}")
        print(f"  resident fp32 momentum on every GPU: {resident:.1f} MiB")
        for (shape, shard_count), n in sorted(sig, key=lambda kv: -ns_cost(kv[0])):
            rows, cols = shape
            kind = "replicated" if shard_count == 1 else f"sharded {shard_count}-way"
            print(f"    {n:>3}x  local {rows}x{cols:<8} -> full "
                  f"{rows * shard_count}x{cols:<8} ({kind})")
        print(f"  LSH: {len(sharded)} sharded matrices -> {len(by_home)}/{group_size} GPUs "
              f"are a home, {group_size - len(by_home)} GPUs idle during NS")
        loads = sorted(
            (sum(ns_step_flops(min(m[0][0] * m[1], m[0][1]), max(m[0][0] * m[1], m[0][1]))
                 for m in mats) * config.num_ns_steps / 1e12, home, len(mats))
            for home, mats in by_home.items()
        )
        if loads:
            print(f"       home load  min {loads[0][0]:.2f} TF ({loads[0][2]} mats) / "
                  f"max {loads[-1][0]:.2f} TF ({loads[-1][2]} mats)  -> "
                  f"LPT imbalance {loads[-1][0] / loads[0][0]:.2f}x")
        if repl:
            names = Counter(f"{m[0][0]}x{m[0][1]}" for m in repl)
            print(f"       replicated on ALL GPUs: "
                  f"{' + '.join(f'{c}x[{s}]' for s, c in names.items())}")

    print("\n" + "=" * 104)
    print("FLOPS PER GPU  (issued = executed on one GPU; useful = irreducible share)")
    if config.peak_tflops:
        print(f"peak = {config.peak_tflops:.1f} TFLOP/s bf16 (measured)")
    print("=" * 104)
    header = f"{'mode':>22}{'issued TF':>11}{'useful TF':>11}{'redund':>9}"
    if times:
        header += f"{'ms':>10}{'HFU%':>8}{'MFU%':>8}"
    for sig, dp_ranks in ordered:
        owns = " + ".join(f"{n}x[{e[0][0]}x{e[0][1]}]" for e, n in sig)
        print(f"\n{len(dp_ranks)} dp_ranks | {owns}")
        print(header)
        for mode in ["duplicated", "distributed"] + LSH_MODES:
            iss_mean, iss_max, useful = profile_flops(
                sig, mode, config.num_ns_steps, group_size
            )
            line = (f"{mode:>22}{iss_mean / 1e12:>11.2f}{useful / 1e12:>11.2f}"
                    f"{iss_mean / useful:>8.2f}x")
            entry = times.get(owns, {})
            if entry.get(mode) and config.peak_tflops:
                ms = entry[mode]
                sec = ms / 1e3
                line += (f"{ms:>10.2f}"
                         f"{iss_mean / 1e12 / sec / config.peak_tflops * 100:>8.1f}"
                         f"{useful / 1e12 / sec / config.peak_tflops * 100:>8.1f}")
            print(line)


if __name__ == "__main__":
    main()
