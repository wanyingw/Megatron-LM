# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Profiling harness for the backward <-> Muon overlap study.

Everything here is a *measurement* aid for deciding whether, and where, overlapping the Muon
optimizer step with the backward pass can pay off. None of it runs unless one of the
``--muon-overlap-*`` flags is set, and the E1/E2 switches deliberately do NOT update Muon
parameters: a run with them on is a timing experiment, never a training run.

Three switches (see ``megatron/training/arguments.py``):

``--muon-overlap-probe``
    CUDA-event telemetry, logged every ``--muon-overlap-probe-interval`` iterations as
    world-wide min/median/max. Per DDP bucket group: when its gradient reduce-scatter
    completes on the GPU timeline, relative to the end of backward compute, the end of
    ``finalize_model_grads`` (all grads final), the global grad norm (the first collective the
    optimizer step waits on) and the end of ``optimizer.step()``. Numerics are unchanged.

``--muon-overlap-skip-step`` (E1)
    The Muon inner step is a no-op (its state is still allocated). The resulting iteration
    time is the floor any overlap design can reach; baseline minus E1 is the overlap ceiling.

``--muon-overlap-shadow {tail,bucket:K}`` (E2)
    On top of E1, a *shadow* copy of every ``LayerShardedMuon`` -- cloned params and momentum,
    random grads, the same NS homes and exchange plan -- runs on a side stream: launched when
    backward compute has been enqueued (``tail``) or when the K-th Muon-managed bucket group
    dispatches its reduce-scatter (``bucket:K``, 1-based in dispatch order), and joined where
    the real Muon step would run. E2 minus E1 is the cost of co-running the Muon work (the
    exposed remainder plus the slowdown it inflicts on backward); baseline minus E2 is the
    achievable gain of that launch point. ``--muon-overlap-shadow-comm`` picks whether the
    shadow's all_to_alls share the model's gtp_remat/tp communicators (what LayerShardedMuon
    does today, so NCCL serializes them behind the backward's GTP collectives) or get dedicated
    clones with the same ranks (truly concurrent, which exercises NCCL co-residency).
"""

import functools
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import torch

from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from megatron.core.optimizer.layer_wise_optimizer import _bucket_is_managed_by_layer_wise_optimizer
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_PROFILER: Optional["MuonOverlapProfiler"] = None

# The telemetry vector has a fixed width so one all_gather_into_tensor collects it.
_MAX_BUCKET_GROUPS = 64
_SCALAR_MARKS = ("bwd_end", "grads_final", "norm_done", "step_end", "shadow_launch", "shadow_done")

# Sharding / routing attributes LayerShardedMuon reads off a param (ParamShardSpec.from_param
# and the exchange plan), copied onto the shadow clones so they route exactly like the originals.
_SHADOW_PARAM_ATTRS = (
    "tensor_model_parallel",
    "partition_dim",
    "is_gtp_weight_remat",
    "pad_length",
    "group",
    "gtp_remat_size",
    "expert_tp",
    "is_expert_parallel",
    "allreduce",
    "shared",
    "use_muon",
    "is_embedding_or_output_parameter",
    "is_qkv",
)


def get_muon_overlap_profiler() -> Optional["MuonOverlapProfiler"]:
    """The active profiler, or None when no ``--muon-overlap-*`` flag is set."""
    return _PROFILER


def parse_shadow_launch(spec: str) -> Tuple[str, int]:
    """``'tail'`` -> ``('tail', 0)``; ``'bucket:K'`` (K >= 1) -> ``('bucket', K)``."""
    if spec == "tail":
        return "tail", 0
    match = re.fullmatch(r"bucket:(\d+)", spec)
    if match and int(match.group(1)) >= 1:
        return "bucket", int(match.group(1))
    raise ValueError(
        f"--muon-overlap-shadow must be 'tail' or 'bucket:K' with K >= 1, got {spec!r}"
    )


def iter_muon_inner_optimizers(optimizer):
    """Yield every TensorParallelMuon (incl. LayerShardedMuon) under a Megatron optimizer tree.

    Walks ``chained_optimizers`` (ChainedOptimizer, LayerWiseDistributedOptimizer) and reads
    ``.optimizer`` only on leaves, since ChainedOptimizer's ``.optimizer`` asserts a single child.
    """
    stack = [optimizer]
    while stack:
        opt = stack.pop()
        chained = getattr(opt, "chained_optimizers", None)
        if chained is not None:
            stack.extend(reversed(chained))  # keep declaration order
            continue
        inner = getattr(opt, "optimizer", None)
        if isinstance(inner, TensorParallelMuon):
            yield inner


def _is_muon_managed(bucket_group) -> bool:
    """Whether a DDP bucket group holds LayerWise/Muon-managed params (vs. Adam's)."""
    if not bucket_group.buckets:
        return False
    return _bucket_is_managed_by_layer_wise_optimizer(
        bucket_group.buckets[0], default_for_untagged=False
    )


class ShadowMuon:
    """Shadow copies of the LayerShardedMuon optimizers, stepped on one side stream.

    A clone keeps everything that determines the GPU work -- shapes, sharding attributes, NS
    homes, per-group process groups, NS batching, SYRK -- and nothing that would touch training
    state: params are copies, grads are fixed random tensors, results are never read.
    """

    def __init__(
        self, inner_optimizers: List[LayerShardedMuon], comm: str, max_ctas: Optional[int] = None
    ) -> None:
        assert torch.cuda.is_available(), "the shadow optimizer needs CUDA streams"
        assert comm in ("shared", "dedicated"), comm
        self.comm = comm
        self.max_ctas = max_ctas
        self.stream = torch.cuda.Stream()
        self.launch_event = torch.cuda.Event(enable_timing=True)
        self.done_event = torch.cuda.Event(enable_timing=True)
        self._dedicated: Dict[tuple, torch.distributed.ProcessGroup] = {}
        self.extra_bytes = 0
        self.shadows = [self._clone(inner) for inner in inner_optimizers]
        self._launched = False

    def _pg(self, group):
        """The group the shadow communicates over: the model's own, or a dedicated clone."""
        if group is None or self.comm == "shared" or torch.distributed.get_world_size(group) <= 1:
            return group
        ranks = tuple(torch.distributed.get_process_group_ranks(group))
        clone = self._dedicated.get(ranks)
        if clone is None:
            kwargs = {}
            if self.max_ctas is not None:
                options = torch.distributed.ProcessGroupNCCL.Options()
                options.config.max_ctas = self.max_ctas
                kwargs["pg_options"] = options
            # Only the member ranks call this (each rank clones its own groups), hence the local
            # synchronization; a world-wide new_group would need every rank to enumerate every
            # group of every domain in the same order.
            clone = torch.distributed.new_group(
                ranks=list(ranks), use_local_synchronization=True, **kwargs
            )
            self._dedicated[ranks] = clone
        return clone

    def _clone(self, inner: LayerShardedMuon) -> LayerShardedMuon:
        kwargs = dict(inner._init_kwargs)
        kwargs["gtp_remat_group"] = self._pg(kwargs["gtp_remat_group"])
        kwargs["tp_group"] = self._pg(kwargs["tp_group"])
        id_map: Dict[int, int] = {}
        groups = []
        for group in inner.param_groups:
            clones = []
            for p in group["params"]:
                # A Parameter like the fp32 main params it mirrors (same dtype, shape, attrs).
                q = torch.nn.Parameter(p.detach().clone())
                for attr in _SHADOW_PARAM_ATTRS:
                    if hasattr(p, attr):
                        setattr(q, attr, getattr(p, attr))
                q.grad = torch.randn_like(q)
                self.extra_bytes += 3 * q.numel() * q.element_size()  # param + grad + momentum
                id_map[id(p)] = id(q)
                clones.append(q)
            groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": clones})
        shadow = LayerShardedMuon(groups, **kwargs)
        shadow.set_group_process_groups(
            {
                index: (self._pg(gtp), self._pg(tp))
                for index, (gtp, tp) in inner._group_process_groups.items()
            }
        )
        shadow.set_param_ns_homes(
            {id_map[pid]: home for pid, home in inner._param_ns_homes.items() if pid in id_map}
        )
        return shadow

    def launch(self, wait_handle=None, wait_event=None) -> None:
        """Enqueue every shadow step on the side stream (once per iteration; host never blocks).

        ``wait_handle`` (a grad-sync collective handle) or ``wait_event`` orders the side stream
        behind the point the real design would have to wait for; with neither, behind the
        caller's current stream.
        """
        if self._launched:
            return
        self._launched = True
        outer = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            if wait_handle is not None:
                wait_handle.wait()
            elif wait_event is not None:
                self.stream.wait_event(wait_event)
            else:
                self.stream.wait_stream(outer)
            self.launch_event.record(self.stream)
            torch.cuda.nvtx.range_push("muon_overlap/shadow")
            try:
                for shadow in self.shadows:
                    shadow.step()
            finally:
                torch.cuda.nvtx.range_pop()
            self.done_event.record(self.stream)

    def reset(self) -> None:
        """Allow the next iteration's launch."""
        self._launched = False

    def warmup(self) -> None:
        """One synchronous shadow step at setup.

        Creates the NCCL communicators (a lazy init inside a backward hook could deadlock
        against in-flight collectives), allocates the momentum state and compiles any Triton
        kernels before the first measured iteration.
        """
        self.launch()
        torch.cuda.current_stream().wait_stream(self.stream)
        torch.cuda.synchronize()
        self.reset()


class MuonOverlapProfiler:
    """Owns the probe telemetry, the E1 switch and the E2 shadow for one training run."""

    def __init__(self, args, model_chunks, optimizer, config) -> None:
        self.probe = bool(args.muon_overlap_probe)
        self.skip_step = bool(args.muon_overlap_skip_step)
        self.shadow_launch = (
            parse_shadow_launch(args.muon_overlap_shadow) if args.muon_overlap_shadow else None
        )
        self.probe_interval = args.muon_overlap_probe_interval or args.log_interval
        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

        inner_optimizers = list(iter_muon_inner_optimizers(optimizer))
        if (self.skip_step or self.shadow_launch is not None) and not inner_optimizers:
            raise ValueError(
                "--muon-overlap-skip-step/--muon-overlap-shadow: no Muon optimizer found under "
                f"{type(optimizer).__name__}"
            )
        for inner in inner_optimizers:
            inner.profiling_skip_step = self.skip_step

        self.shadow: Optional[ShadowMuon] = None
        if self.shadow_launch is not None:
            layer_sharded = [o for o in inner_optimizers if isinstance(o, LayerShardedMuon)]
            if not layer_sharded:
                raise ValueError(
                    "--muon-overlap-shadow clones LayerShardedMuon, but no LayerShardedMuon was "
                    "built; use --muon-tp-mode layer_sharded (or --muon-expert-tp-mode)."
                )
            self.shadow = ShadowMuon(
                layer_sharded, args.muon_overlap_shadow_comm, args.muon_overlap_shadow_max_ctas
            )
            for inner in inner_optimizers:
                inner.profiling_join_streams = [self.shadow.stream]
            log_single_rank(
                logger,
                logging.INFO,
                f"muon-overlap shadow: {len(layer_sharded)} LayerShardedMuon clone(s), "
                f"+{self.shadow.extra_bytes / 2**30:.2f} GB on this rank (params+grads+momentum), "
                f"launch={args.muon_overlap_shadow}, comm={args.muon_overlap_shadow_comm}, "
                f"max_ctas={args.muon_overlap_shadow_max_ctas}",
            )

        # DDP bucket groups in registration order (dense then expert, per model chunk). The
        # index is rank-uniform within a pipeline stage: every DP/TP/GTP peer holds the same
        # buffers in the same order.
        self._bucket_groups: List[Tuple[object, bool]] = []
        for chunk in model_chunks:
            groups = list(getattr(chunk, "bucket_groups", [])) + list(
                getattr(chunk, "expert_parallel_bucket_groups", [])
            )
            for bucket_group in groups:
                index = len(self._bucket_groups)
                self._bucket_groups.append((bucket_group, _is_muon_managed(bucket_group)))
                bucket_group.grad_sync_started_callbacks.append(
                    functools.partial(self._on_grad_sync_started, index)
                )
        if len(self._bucket_groups) > _MAX_BUCKET_GROUPS:
            raise ValueError(
                f"muon-overlap probe supports up to {_MAX_BUCKET_GROUPS} bucket groups, "
                f"found {len(self._bucket_groups)}"
            )
        self.num_muon_bucket_groups = sum(1 for _, muon in self._bucket_groups if muon)
        if self.shadow_launch is not None and self.shadow_launch[0] == "bucket":
            if self.shadow_launch[1] > self.num_muon_bucket_groups:
                raise ValueError(
                    f"--muon-overlap-shadow bucket:{self.shadow_launch[1]} but only "
                    f"{self.num_muon_bucket_groups} Muon-managed bucket groups exist on this "
                    "rank; every rank of a domain must launch, so the index must exist everywhere."
                )

        # Per-iteration state. _rs is appended from backward hooks (autograd thread) and read
        # after optimizer.step() on the main thread; nothing reads it concurrently.
        self._probe_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._events: Dict[str, torch.cuda.Event] = {}
        self._rs: List[Tuple[int, float, torch.cuda.Event]] = []
        self._cpu_iter_start = 0.0
        self._muon_dispatched = 0

        # Marks around the gradient finalization: 'bwd_end' when backward compute is enqueued,
        # 'grads_final' once every grad is reduced and post-processed.
        inner_finalize = config.finalize_model_grads_func

        @functools.wraps(inner_finalize)
        def finalize_with_marks(*fargs, **fkwargs):
            self._on_finalize_begin()
            try:
                return inner_finalize(*fargs, **fkwargs)
            finally:
                self._on_finalize_end()

        config.finalize_model_grads_func = finalize_with_marks

        callbacks = getattr(optimizer, "post_grad_norm_callbacks", None)
        if callbacks is not None:
            callbacks.append(self._on_grad_norm)
        elif self.probe:
            log_single_rank(
                logger,
                logging.WARNING,
                f"muon-overlap probe: {type(optimizer).__name__} has no post_grad_norm_callbacks;"
                " the 'norm_done' mark will be missing.",
            )

        if self.shadow is not None:
            self.shadow.warmup()
        log_single_rank(
            logger,
            logging.INFO,
            f"muon-overlap profiling: probe={self.probe} (every {self.probe_interval} iters), "
            f"skip_step={self.skip_step}, shadow={args.muon_overlap_shadow}, "
            f"{len(self._bucket_groups)} bucket groups ({self.num_muon_bucket_groups} Muon-managed)"
            f", {len(inner_optimizers)} Muon optimizer(s).",
        )

    # ------------------------------------------------------------------ marks
    def _record(self, name: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream())
        self._events[name] = event

    def mark_iter_start(self) -> None:
        """Start of a training iteration (after the grad buffers are zeroed)."""
        self._events = {}
        self._rs = []
        self._muon_dispatched = 0
        self._cpu_iter_start = time.perf_counter()
        if self.shadow is not None:
            self.shadow.reset()
        self._record("iter_start")

    def _on_grad_sync_started(self, index: int, bucket_group) -> None:
        """Bucket group callback (backward hook): RS dispatched, handle set. Enqueue-only."""
        if "iter_start" not in self._events:
            return
        handle = bucket_group.grad_reduce_handle
        if self.probe:
            event = torch.cuda.Event(enable_timing=True)
            if handle is not None and self._probe_stream is not None:
                # The probe stream waits for the collective (a stream dependency, not a host
                # block), so the event fires when the reduce-scatter has completed on the GPU.
                with torch.cuda.stream(self._probe_stream):
                    handle.wait()
                    event.record(self._probe_stream)
            else:
                # Synchronous grad sync: the collective already ran on the current stream.
                event.record(torch.cuda.current_stream())
            self._rs.append((index, (time.perf_counter() - self._cpu_iter_start) * 1e3, event))
        if self._bucket_groups[index][1]:
            self._muon_dispatched += 1
            if (
                self.shadow is not None
                and self.shadow_launch[0] == "bucket"
                and self._muon_dispatched == self.shadow_launch[1]
            ):
                self.shadow.launch(wait_handle=handle)

    def _on_finalize_begin(self) -> None:
        if "iter_start" not in self._events:
            return
        self._record("bwd_end")
        if self.shadow is not None and self.shadow_launch[0] == "tail":
            self.shadow.launch(wait_event=self._events["bwd_end"])

    def _on_finalize_end(self) -> None:
        if "iter_start" in self._events:
            self._record("grads_final")

    def _on_grad_norm(self, grad_norm) -> None:
        if "iter_start" in self._events:
            self._record("norm_done")

    def mark_step_end(self) -> None:
        """Right after ``optimizer.step()`` returned."""
        if "iter_start" in self._events:
            self._record("step_end")

    # ---------------------------------------------------------------- logging
    def _local_vector(self) -> torch.Tensor:
        """This rank's telemetry, ms since iter_start on the GPU timeline, NaN where absent."""
        start = self._events["iter_start"]
        values = [float("nan")] * (len(_SCALAR_MARKS) + 1 + 3 * _MAX_BUCKET_GROUPS)
        for i, name in enumerate(_SCALAR_MARKS):
            if name in self._events:
                values[i] = start.elapsed_time(self._events[name])
        if self.shadow is not None and self.shadow._launched:
            values[_SCALAR_MARKS.index("shadow_launch")] = start.elapsed_time(
                self.shadow.launch_event
            )
            values[_SCALAR_MARKS.index("shadow_done")] = start.elapsed_time(self.shadow.done_event)
        values[len(_SCALAR_MARKS)] = float(len(self._bucket_groups))
        base = len(_SCALAR_MARKS) + 1
        for index, cpu_ms, event in self._rs:
            values[base + index] = start.elapsed_time(event)
            values[base + _MAX_BUCKET_GROUPS + index] = cpu_ms
            values[base + 2 * _MAX_BUCKET_GROUPS + index] = float(self._bucket_groups[index][1])
        return torch.tensor(values, dtype=torch.float32, device="cuda")

    def maybe_log(self, iteration: int) -> None:
        """Gather every rank's telemetry and log the summary from rank 0 (probe only).

        Called on every rank at the same point of the training loop: the all_gather is a
        collective. Runs ``torch.cuda.synchronize()`` first so every event has completed.
        """
        if not self.probe or iteration % self.probe_interval != 0:
            return
        if "iter_start" not in self._events or "step_end" not in self._events:
            return
        torch.cuda.synchronize()
        local = self._local_vector()
        gathered = torch.empty(self.world_size * local.numel(), dtype=local.dtype, device="cuda")
        torch.distributed.all_gather_into_tensor(gathered, local)
        if self.rank == 0:
            logger.info(self.format_report(iteration, gathered.view(self.world_size, -1).cpu()))

    @staticmethod
    def _stats(column: torch.Tensor) -> str:
        valid = column[~torch.isnan(column)]
        if valid.numel() == 0:
            return "      -"
        return f"{valid.min():7.1f}/{valid.median():7.1f}/{valid.max():7.1f}"

    def format_report(self, iteration: int, table: torch.Tensor) -> str:
        """Render the gathered ``[world, vector]`` telemetry as a rank-0 log block."""
        scalars = {name: table[:, i] for i, name in enumerate(_SCALAR_MARKS)}
        base = len(_SCALAR_MARKS) + 1
        rs_done = table[:, base : base + _MAX_BUCKET_GROUPS]
        rs_cpu = table[:, base + _MAX_BUCKET_GROUPS : base + 2 * _MAX_BUCKET_GROUPS]
        rs_muon = table[:, base + 2 * _MAX_BUCKET_GROUPS : base + 3 * _MAX_BUCKET_GROUPS]

        lines = [
            f"[muon-overlap-probe] iteration {iteration}: ms since iter_start on the GPU "
            f"timeline, min/median/max over {table.shape[0]} ranks"
        ]
        lines.append(f"  bwd_end      {self._stats(scalars['bwd_end'])}")
        lines.append(
            f"  grads_final  {self._stats(scalars['grads_final'])}   "
            f"tail(grads_final-bwd_end) {self._stats(scalars['grads_final'] - scalars['bwd_end'])}"
        )
        lines.append(
            f"  norm_done    {self._stats(scalars['norm_done'])}   "
            f"norm_wait(norm_done-grads_final) "
            f"{self._stats(scalars['norm_done'] - scalars['grads_final'])}"
        )
        lines.append(
            f"  step_end     {self._stats(scalars['step_end'])}   "
            f"opt(step_end-norm_done) {self._stats(scalars['step_end'] - scalars['norm_done'])}"
        )
        if not torch.isnan(scalars["shadow_launch"]).all():
            beyond = torch.clamp(scalars["shadow_done"] - scalars["norm_done"], min=0.0)
            lines.append(
                f"  shadow       launch {self._stats(scalars['shadow_launch'])}   "
                f"done {self._stats(scalars['shadow_done'])}   "
                f"exposed_after_norm {self._stats(beyond)}"
            )
        lines.append(
            "  bucket  muon  rs_done                     "
            "slack(bwd_end-rs_done)      cpu_dispatch(rank0)"
        )
        for index in range(_MAX_BUCKET_GROUPS):
            column = rs_done[:, index]
            if torch.isnan(column).all():
                continue
            muon_flags = rs_muon[:, index]
            muon = "Y" if (muon_flags[~torch.isnan(muon_flags)] > 0.5).any() else "n"
            cpu = rs_cpu[0, index]
            cpu_text = "      -" if torch.isnan(cpu) else f"{cpu:7.1f}"
            lines.append(
                f"  {index:<6d}  {muon:<4s}  {self._stats(column)}   "
                f"{self._stats(scalars['bwd_end'] - column)}   {cpu_text}"
            )
        return "\n".join(lines)


def setup_muon_overlap_profiling(
    args, model_chunks, optimizer, config
) -> Optional[MuonOverlapProfiler]:
    """Install the profiler once the optimizer exists and ``config.finalize_model_grads_func``
    is set. No-op (returns None) unless a ``--muon-overlap-*`` flag is on."""
    global _PROFILER
    if not (
        getattr(args, "muon_overlap_probe", False)
        or getattr(args, "muon_overlap_skip_step", False)
        or getattr(args, "muon_overlap_shadow", None)
    ):
        return None
    _PROFILER = MuonOverlapProfiler(args, model_chunks, optimizer, config)
    return _PROFILER
