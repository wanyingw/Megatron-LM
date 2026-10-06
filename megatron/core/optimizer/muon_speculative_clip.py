# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Speculative gradient clipping for layer-sharded Muon.

Megatron clips gradients by the GLOBAL norm, which exists only after every gradient does: the
Muon step waits for the last reduce-scatter, ``finalize_model_grads``, the norm kernels and a
world-wide all-reduce before it launches its first kernel. But clipping touches the gradients
only when it binds (coefficient < 1), and after warmup it essentially never binds. So:

1. ``launch`` (before the norm): every eligible :class:`LayerShardedMuon` runs its pipeline
   speculatively on side streams, on the final but unclipped gradients, and stages the
   updates (:meth:`LayerShardedMuon.speculate`). The main stream goes on to compute the norm.
2. ``verify`` (after the norm): the clip decision is replayed with the exact arithmetic of
   :func:`clip_grad_by_total_norm_fp32`. No bind: the staged pass is committed inside the
   regular step, bit-identical to a non-speculative step because an unbound clip leaves the
   gradients untouched. Bind, non-finite norm or a ``grad_norm_skip_threshold`` skip: the
   pass is dropped and the regular step runs on the clipped gradients, also bit-identical.
3. Hysteresis: speculation is attempted only after ``warmup_steps`` consecutive non-binding
   iterations; any bind resets the count, so the always-binding warmup never pays for doomed
   passes. The decision uses the all-reduced norm, so every rank takes the same branch and
   the collective schedule stays rank-uniform.

What it buys on the critical path: the norm kernels, the norm all-reduce (including the wait
for the slowest rank) and the clip no longer precede the Muon pipeline; they run under it.
It does not move the pipeline into the backward pass.
"""

import logging
import math
from typing import List, Optional

import torch

from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from megatron.core.optimizer.optimizer import _get_param_grad_norm_group
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


def clip_binds(clip_grad: float, grad_norm) -> bool:
    """Whether :func:`clip_grad_by_total_norm_fp32` would scale the gradients.

    Mirrors its arithmetic exactly, type for type: with a tensor norm the coefficient is
    ``clip_grad / (norm + 1e-6)`` clamped to 1.0 and the multi-tensor scale is applied
    unconditionally, so the gradients change unless the clamped coefficient is exactly 1.0;
    with a float norm the scale is applied iff the coefficient is < 1.0. Mirroring includes
    the NaN corner: the tensor path binds (NaN survives the clamp, ``!= 1.0``), the float path
    does not (``nan < 1.0`` is False), which is why :meth:`SpeculativeMuonClip.verify`
    rejects non-finite norms before consulting this function.
    """
    if clip_grad <= 0.0:
        return False
    if isinstance(grad_norm, torch.Tensor):
        coefficient = (clip_grad / (grad_norm + 1.0e-6)).clamp_max_(1.0)
        return coefficient.item() != 1.0
    return clip_grad / (grad_norm + 1.0e-6) < 1.0


class SpeculativeMuonClip:
    """Launch / verify controller for one or more LayerShardedMuon optimizers.

    Register :meth:`launch` as a ``pre_grad_norm_callbacks`` entry and :meth:`verify` as a
    ``post_grad_norm_callbacks`` entry of the optimizer whose ``step()`` computes the norm
    (:meth:`wire` does this for a LayerWiseDistributedOptimizer).
    """

    def __init__(
        self,
        optimizers: List[LayerShardedMuon],
        clip_grad: float,
        grad_norm_skip_threshold: float = float("inf"),
        warmup_steps: int = 10,
        log_every: int = 100,
    ) -> None:
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        self.optimizers = list(optimizers)
        self.clip_grad = float(clip_grad)
        self.grad_norm_skip_threshold = float(grad_norm_skip_threshold)
        self.warmup_steps = int(warmup_steps)
        self.log_every = int(log_every)
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._consecutive_ok = 0
        self._launched = False
        self.stats = {"steps": 0, "speculated": 0, "hits": 0, "misses": 0, "binds": 0}

    @classmethod
    def wire(cls, layer_wise, optimizers, config) -> Optional["SpeculativeMuonClip"]:
        """Build a controller for the eligible LayerShardedMuon children of ``layer_wise`` and
        register its hooks on it. Returns None (and logs why) when nothing is eligible."""
        eligible = []
        for opt in optimizers:
            inner = getattr(opt, "optimizer", opt)
            if not isinstance(inner, LayerShardedMuon):
                continue
            if getattr(inner, "weight_decay_method", "l2") == "l2" and any(
                g.get("weight_decay", 0.0) != 0.0 for g in inner.param_groups
            ):
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "muon_speculative_clip: skipping a LayerShardedMuon with l2 weight decay "
                    "(it modifies the gradient inside the step; use decoupled weight decay).",
                )
                continue
            if any(
                _get_param_grad_norm_group(p) is not None
                for g in inner.param_groups
                for p in g["params"]
            ):
                # Such params are clipped by their GROUP norm, which the global verdict does
                # not cover; a group-only bind would wrongly commit. Keep them exact.
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "muon_speculative_clip: skipping a LayerShardedMuon whose params belong to "
                    "a separate grad-norm group (verified against the global norm only).",
                )
                continue
            eligible.append(inner)
        if not eligible:
            log_single_rank(
                logger, logging.WARNING, "muon_speculative_clip: no eligible LayerShardedMuon."
            )
            return None
        controller = cls(
            eligible,
            clip_grad=config.clip_grad,
            grad_norm_skip_threshold=getattr(config, "grad_norm_skip_threshold", float("inf")),
            warmup_steps=config.muon_speculative_clip_warmup_steps,
        )
        layer_wise.pre_grad_norm_callbacks.append(controller.launch)
        layer_wise.post_grad_norm_callbacks.append(controller.verify)
        log_single_rank(
            logger,
            logging.INFO,
            f"muon_speculative_clip: wired {len(eligible)} LayerShardedMuon optimizer(s), "
            f"clip_grad={controller.clip_grad}, warmup_steps={controller.warmup_steps}.",
        )
        return controller

    @property
    def armed(self) -> bool:
        """Whether the next step speculates (hysteresis satisfied)."""
        return self._consecutive_ok >= self.warmup_steps

    def launch(self) -> None:
        """pre-grad-norm hook: start the speculative passes when armed. Enqueue-only."""
        self._launched = False
        if not self.armed or self.stream is None:
            return
        launched = [opt.speculate(self.stream) for opt in self.optimizers]
        self._launched = any(launched)
        if self._launched:
            self.stats["speculated"] += 1

    def verify(self, grad_norm) -> bool:
        """post-grad-norm hook: commit or drop the staged passes, update the hysteresis.

        Returns True on a hit (staged passes will be committed by the optimizers' step()).
        """
        self.stats["steps"] += 1
        if isinstance(grad_norm, torch.Tensor):
            norm = float(grad_norm.item())
        else:
            norm = float(grad_norm) if grad_norm is not None else float("nan")
        finite = math.isfinite(norm)
        binds = (not finite) or clip_binds(self.clip_grad, grad_norm)
        skip = math.isfinite(self.grad_norm_skip_threshold) and bool(
            norm > self.grad_norm_skip_threshold
        )
        hit = self._launched and finite and not binds and not skip
        for opt in self.optimizers:
            if hit:
                opt.set_speculation_verdict(True)
            else:
                opt.discard_speculation()
        if self._launched:
            self.stats["hits" if hit else "misses"] += 1
        self._launched = False
        if binds or skip:
            self.stats["binds"] += 1
            self._consecutive_ok = 0
        else:
            self._consecutive_ok += 1
        if self.log_every and self.stats["steps"] % self.log_every == 0:
            log_single_rank(logger, logging.INFO, self.summary())
        return hit

    def summary(self) -> str:
        """One-line counters for the log."""
        s = self.stats
        return (
            f"muon_speculative_clip: steps={s['steps']} speculated={s['speculated']} "
            f"hits={s['hits']} misses={s['misses']} binds={s['binds']} armed={self.armed}"
        )
