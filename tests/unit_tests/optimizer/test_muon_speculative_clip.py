# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for speculative gradient clipping of layer-sharded Muon.

The contract under test: a speculative pass that is committed (clip did not bind) leaves
params AND momentum bit-identical to a plain ``step()``; a pass that is dropped (clip bound,
non-finite norm, skipped update) leaves the state untouched so the plain step on the clipped
gradients is also bit-identical; the hysteresis arms only after ``warmup_steps`` consecutive
non-binding iterations and any bind resets it; ``clip_binds`` mirrors
``clip_grad_by_total_norm_fp32`` for float and tensor norms.

Domains: a single-rank domain (local Newton-Schulz, any world size) and TP(2) x GTP_remat(2)
plus an expert GTP(2) group (4 ranks, exercises the exchanges and the per-group streams).

Launch: torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_muon_speculative_clip.py
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("emerging_optimizers", reason="requires emerging-optimizers")

from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from megatron.core.optimizer.muon_speculative_clip import SpeculativeMuonClip, clip_binds
from megatron.core.optimizer.optimizer import ChainedOptimizer
from tests.unit_tests.test_utilities import Utils

_MUON_KW = dict(
    lr=1e-2,
    momentum=0.95,
    weight_decay=0.0,
    coefficient_type="quintic",
    num_ns_steps=5,
    scale_mode="spectral",
    extra_scale_factor=1.0,
    fp32_matmul_prec="highest",
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")


@pytest.fixture(scope="module", autouse=True)
def _dist_init():
    Utils.initialize_model_parallel()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.manual_seed(42)
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(prev)
    global _TP_GROUP, _GTP_REMAT_GROUP, _EGTP_GROUP
    for group in (_TP_GROUP, _GTP_REMAT_GROUP, _EGTP_GROUP):
        if group is not None:
            dist.destroy_process_group(group)
    _TP_GROUP = _GTP_REMAT_GROUP = _EGTP_GROUP = None
    Utils.destroy_model_parallel()


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Process groups and param construction
# ---------------------------------------------------------------------------

_TP_GROUP = None
_GTP_REMAT_GROUP = None
_EGTP_GROUP = None


def _get_2d_groups():
    """TP(2) x GTP_remat(2) subgroups, created once. Rank convention: r = g * T + t."""
    global _TP_GROUP, _GTP_REMAT_GROUP
    if _TP_GROUP is None:
        _TP_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 1], [2, 3]])
        _GTP_REMAT_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 2], [1, 3]])
    return _TP_GROUP, _GTP_REMAT_GROUP


def _get_expert_group():
    """Expert GTP_remat(2) subgroups {0,1} and {2,3}: a different partition than the dense one."""
    global _EGTP_GROUP
    if _EGTP_GROUP is None:
        _EGTP_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 1], [2, 3]])
    return _EGTP_GROUP


def _require_four_ranks():
    if dist.get_world_size() != 4:
        pytest.skip("Requires exactly 4 ranks (launch with torchrun --nproc-per-node=4)")


def _column_parallel_gtp_shards(dev, seed, n_params=2, full_shape=(32, 16)):
    """This rank's (TP-dim-0, then GTP_remat-dim-0) shards of ``n_params`` full matrices; every
    rank seeds identically so the ranks hold shards of the same matrices. Returns the shard
    tensors (not Parameters) so a pair of identical optimizers can be built from them."""
    rank = dist.get_rank()
    t, g = rank % 2, rank // 2
    gen = torch.Generator(device="cpu").manual_seed(seed)
    fulls = [torch.randn(*full_shape, generator=gen) for _ in range(n_params)]
    rows = full_shape[0] // 4
    shards = []
    for w in fulls:
        start = t * (full_shape[0] // 2) + g * rows
        shards.append(w[start : start + rows].clone().to(dev))
    return shards


def _expert_gtp_shards(dev, seed, n_params=3, full_shape=(16, 8)):
    """Expert params sharded over the expert GTP(2) group only (rank % 2 is the egtp rank)."""
    rank = dist.get_rank()
    egtp_rank = rank % 2
    gen = torch.Generator(device="cpu").manual_seed(seed + 1000 * (rank // 2))  # per EP group
    rows = full_shape[0] // 2
    shards = []
    for _ in range(n_params):
        w = torch.randn(*full_shape, generator=gen)
        shards.append(w[egtp_rank * rows : (egtp_rank + 1) * rows].clone().to(dev))
    return shards


def _as_params(shards, tp_sharded, gtp_sharded):
    params = []
    for s in shards:
        p = torch.nn.Parameter(s.clone())
        if tp_sharded:
            p.tensor_model_parallel = True
            p.partition_dim = 0
        if gtp_sharded:
            p.is_gtp_weight_remat = True
        params.append(p)
    return params


class _Pair:
    """A reference LayerShardedMuon and an identical one driven speculatively."""

    def __init__(self, groups, nesterov, weight_decay, concurrent_groups=True):
        """``groups``: list of (shard_tensors, tp_sharded, gtp_sharded, (gtp_group, tp_group),
        homes_by_index) -- one param group each."""
        self.ref_groups, self.spec_groups = [], []
        ref_param_groups, spec_param_groups = [], []
        ref_homes, spec_homes = {}, {}
        ref_pgs, spec_pgs = {}, {}
        for gi, (shards, tp_sharded, gtp_sharded, pgs, homes) in enumerate(groups):
            ref = _as_params(shards, tp_sharded, gtp_sharded)
            spec = _as_params(shards, tp_sharded, gtp_sharded)
            self.ref_groups.append(ref)
            self.spec_groups.append(spec)
            ref_param_groups.append({"params": ref})
            spec_param_groups.append({"params": spec})
            ref_pgs[gi] = pgs
            spec_pgs[gi] = pgs
            for i, home in homes.items():
                ref_homes[id(ref[i])] = home
                spec_homes[id(spec[i])] = home
        kw = {**_MUON_KW, "nesterov": nesterov, "weight_decay": weight_decay}
        gtp0, tp0 = groups[0][3]
        self.ref = LayerShardedMuon(
            ref_param_groups,
            gtp_remat_group=gtp0,
            tp_group=tp0,
            concurrent_groups=concurrent_groups,
            **kw,
        )
        self.spec = LayerShardedMuon(
            spec_param_groups,
            gtp_remat_group=gtp0,
            tp_group=tp0,
            concurrent_groups=concurrent_groups,
            **kw,
        )
        self.ref.set_group_process_groups(ref_pgs)
        self.spec.set_group_process_groups(spec_pgs)
        self.ref.set_param_ns_homes(ref_homes)
        self.spec.set_param_ns_homes(spec_homes)

    def set_grads(self, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed + 7 * dist.get_rank())
        for ref_group, spec_group in zip(self.ref_groups, self.spec_groups):
            for pr, ps in zip(ref_group, spec_group):
                g = torch.randn(pr.shape, generator=gen).to(pr.device)
                pr.grad = g.clone()
                ps.grad = g.clone()

    def all_params(self):
        return [p for g in self.ref_groups for p in g], [p for g in self.spec_groups for p in g]

    def assert_identical(self):
        ref_params, spec_params = self.all_params()
        for pr, ps in zip(ref_params, spec_params):
            assert torch.equal(pr, ps), "params diverged"
            mr = self.ref.state[pr].get("momentum_buffer")
            ms = self.spec.state[ps].get("momentum_buffer")
            assert (mr is None) == (ms is None)
            if mr is not None:
                assert torch.equal(mr, ms), "momentum diverged"


def _single_domain(dev, nesterov, weight_decay):
    shards = [torch.randn(16, 8, device=dev), torch.randn(8, 24, device=dev)]
    return _Pair([(shards, False, False, (None, None), {})], nesterov, weight_decay)


def _two_domains(dev, nesterov, weight_decay, concurrent_groups=True):
    tp, gtp = _get_2d_groups()
    egtp = _get_expert_group()
    dense = _column_parallel_gtp_shards(dev, seed=100)
    expert = _expert_gtp_shards(dev, seed=200)
    return _Pair(
        [
            (dense, True, True, (gtp, tp), {0: (0, 1), 1: (1, 0)}),
            (expert, False, True, (egtp, None), {0: (0, 0), 1: (1, 0), 2: (0, 0)}),
        ],
        nesterov,
        weight_decay,
        concurrent_groups=concurrent_groups,
    )


def _norm(value, kind):
    if kind == "tensor":
        return torch.tensor([value], dtype=torch.float32, device=_device())
    return float(value)


# ---------------------------------------------------------------------------
# clip_binds mirrors clip_grad_by_total_norm_fp32
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["float", "tensor"])
def test_clip_binds_matches_clip_function(kind):
    dev = _device()
    for clip_grad, norm in [(1.0, 0.5), (1.0, 0.9999), (1.0, 1.0), (1.0, 1.5), (0.0, 7.0)]:
        p = torch.nn.Parameter(torch.ones(4, 4, device=dev))
        p.grad = torch.ones(4, 4, device=dev)
        before = p.grad.clone()
        if clip_grad > 0.0:
            clip_grad_by_total_norm_fp32([p], clip_grad, _norm(norm, kind))
        changed = not torch.equal(p.grad, before)
        assert clip_binds(clip_grad, _norm(norm, kind)) == changed, (clip_grad, norm, kind)
    # A NaN norm: the tensor path's clamp keeps NaN (!= 1.0 -> binds); the float path's
    # ``nan < 1.0`` is False, exactly like clip_grad_by_total_norm_fp32, which would not
    # scale either. The controller rejects non-finite norms before consulting clip_binds.
    assert clip_binds(1.0, _norm(float("nan"), kind)) == (kind == "tensor")


# ---------------------------------------------------------------------------
# Hit / miss parity
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("domain", ["single", "two"])
@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("kind", ["tensor", "float"])
def test_committed_pass_is_bitwise_identical_to_step(domain, nesterov, weight_decay, kind):
    dev = _device()
    if domain == "two":
        _require_four_ranks()
        pair = _two_domains(dev, nesterov, weight_decay)
    else:
        pair = _single_domain(dev, nesterov, weight_decay)
    controller = SpeculativeMuonClip([pair.spec], clip_grad=1.0, warmup_steps=1)
    hits = []
    for step in range(4):
        pair.set_grads(seed=10 + step)
        controller.launch()  # step 0: not armed yet (hysteresis), later steps speculate
        assert pair.spec.has_speculation() == (step >= 1)
        hit = controller.verify(_norm(0.25, kind))  # clip 1.0 does not bind
        hits.append(hit)
        pair.spec.step()
        pair.ref.step()
        torch.cuda.synchronize()
        pair.assert_identical()
        assert not pair.spec.has_speculation()
    assert hits == [False, True, True, True]
    assert controller.stats == {"steps": 4, "speculated": 3, "hits": 3, "misses": 0, "binds": 0}


@needs_cuda
@pytest.mark.parametrize("domain", ["single", "two"])
@pytest.mark.parametrize("kind", ["tensor", "float"])
def test_dropped_pass_recomputes_exactly_on_clipped_grads(domain, kind):
    dev = _device()
    if domain == "two":
        _require_four_ranks()
        pair = _two_domains(dev, nesterov=True, weight_decay=0.1)
    else:
        pair = _single_domain(dev, nesterov=True, weight_decay=0.1)
    clip_grad = 0.5
    controller = SpeculativeMuonClip([pair.spec], clip_grad=clip_grad, warmup_steps=1)
    # Step 0 arms (no bind); step 1 speculates and the clip binds -> drop + exact recompute.
    for step, norm in enumerate([0.1, 2.0, 0.1, 0.1]):
        pair.set_grads(seed=50 + step)
        controller.launch()
        hit = controller.verify(_norm(norm, kind))
        binds = clip_binds(clip_grad, _norm(norm, kind))
        assert hit == (step >= 1 and not binds and controller.stats["speculated"] > 0 and step != 2)
        if binds:
            assert not pair.spec.has_speculation(), "a bound clip must drop the pass at verify"
            ref_params, spec_params = pair.all_params()
            clip_grad_by_total_norm_fp32(ref_params, clip_grad, _norm(norm, kind))
            clip_grad_by_total_norm_fp32(spec_params, clip_grad, _norm(norm, kind))
        pair.spec.step()
        pair.ref.step()
        torch.cuda.synchronize()
        pair.assert_identical()
    # step 0: arm; step 1: speculate+miss (reset); step 2: not armed; step 3: speculate+hit.
    assert controller.stats["speculated"] == 2
    assert controller.stats["hits"] == 1 and controller.stats["misses"] == 1
    assert controller.stats["binds"] == 1


@needs_cuda
@pytest.mark.parametrize("reason", ["nan", "skip_threshold", "inf"])
def test_non_finite_or_skipped_norm_drops_the_pass(reason):
    dev = _device()
    pair = _single_domain(dev, nesterov=True, weight_decay=0.0)
    controller = SpeculativeMuonClip(
        [pair.spec], clip_grad=0.0, grad_norm_skip_threshold=1.0, warmup_steps=1
    )
    pair.set_grads(seed=1)
    controller.launch()
    controller.verify(0.5)  # arms
    pair.spec.step()
    pair.ref.step()
    pair.set_grads(seed=2)
    controller.launch()
    assert pair.spec.has_speculation()
    norm = {"nan": float("nan"), "inf": float("inf"), "skip_threshold": 5.0}[reason]
    assert controller.verify(norm) is False
    assert not pair.spec.has_speculation()
    assert not controller.armed, "a bind/skip resets the hysteresis"
    # The state is untouched: a plain step still equals the reference.
    pair.spec.step()
    pair.ref.step()
    torch.cuda.synchronize()
    pair.assert_identical()


@needs_cuda
def test_hysteresis_arms_after_warmup_and_resets_on_bind():
    dev = _device()
    pair = _single_domain(dev, nesterov=False, weight_decay=0.0)
    controller = SpeculativeMuonClip([pair.spec], clip_grad=1.0, warmup_steps=3)
    speculated = []
    for step, norm in enumerate([0.1, 0.1, 0.1, 0.1, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1]):
        pair.set_grads(seed=300 + step)
        controller.launch()
        speculated.append(pair.spec.has_speculation())
        controller.verify(norm)
        pair.spec.step()
        pair.ref.step()
    torch.cuda.synchronize()
    pair.assert_identical()
    # Armed after 3 non-binding steps (steps 3, 4, 5 speculate); the bind at step 5 resets,
    # so steps 6-8 do not speculate and step 9 does again.
    assert speculated == [False, False, False, True, True, True, False, False, False, True]


@needs_cuda
def test_unverified_pass_is_dropped_by_step():
    dev = _device()
    pair = _single_domain(dev, nesterov=True, weight_decay=0.0)
    controller = SpeculativeMuonClip([pair.spec], clip_grad=1.0, warmup_steps=1)
    pair.set_grads(seed=1)
    controller.launch()
    controller.verify(0.1)
    pair.spec.step()
    pair.ref.step()
    pair.set_grads(seed=2)
    controller.launch()
    assert pair.spec.has_speculation()
    # No verify (e.g. the step was aborted): step() must not commit an unverified pass.
    pair.spec.step()
    pair.ref.step()
    torch.cuda.synchronize()
    pair.assert_identical()
    assert not pair.spec.has_speculation()


@needs_cuda
def test_speculate_refuses_l2_weight_decay():
    dev = _device()
    p = torch.nn.Parameter(torch.randn(8, 4, device=dev))
    p.grad = torch.randn(8, 4, device=dev)
    opt = LayerShardedMuon(
        [p],
        gtp_remat_group=None,
        use_decoupled_weight_decay=False,
        **{**_MUON_KW, "weight_decay": 0.1},
    )
    opt.set_param_ns_homes({})
    assert opt.speculate(torch.cuda.Stream()) is False
    assert not opt.has_speculation()


# ---------------------------------------------------------------------------
# Wiring: hooks on a LayerWise-like optimizer, fired by the top-level ChainedOptimizer.step
# ---------------------------------------------------------------------------


def test_wire_registers_hooks_and_skips_ineligible_optimizers():
    dev = _device()
    p = torch.nn.Parameter(torch.randn(8, 4, device=dev))
    eligible = LayerShardedMuon([p], gtp_remat_group=None, **_MUON_KW)
    q = torch.nn.Parameter(torch.randn(8, 4, device=dev))
    l2 = LayerShardedMuon(
        [q],
        gtp_remat_group=None,
        use_decoupled_weight_decay=False,
        **{**_MUON_KW, "weight_decay": 0.1},
    )
    layer_wise = SimpleNamespace(pre_grad_norm_callbacks=[], post_grad_norm_callbacks=[])
    config = SimpleNamespace(
        clip_grad=1.0, grad_norm_skip_threshold=float("inf"), muon_speculative_clip_warmup_steps=2
    )
    wrapped = [
        SimpleNamespace(optimizer=eligible),
        SimpleNamespace(optimizer=l2),
        torch.optim.SGD([q], lr=0.1),
    ]
    controller = SpeculativeMuonClip.wire(layer_wise, wrapped, config)
    assert controller is not None
    assert controller.optimizers == [eligible]
    assert layer_wise.pre_grad_norm_callbacks == [controller.launch]
    assert layer_wise.post_grad_norm_callbacks == [controller.verify]
    assert controller.warmup_steps == 2
    assert SpeculativeMuonClip.wire(layer_wise, [SimpleNamespace(optimizer=l2)], config) is None


def test_chained_optimizer_fires_pre_and_post_hooks_of_children():
    config = SimpleNamespace(
        clip_grad=0.0,
        log_num_zeros_in_grad=False,
        grad_norm_skip_threshold=float("inf"),
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather_with_optimizer_step=False,
    )
    events = []

    class _Child:
        is_stub_optimizer = False

        def __init__(self):
            self.config = config
            self.pre_grad_norm_callbacks = [lambda: events.append("child-pre")]
            self.post_grad_norm_callbacks = [lambda n: events.append(("child-post", n))]

        def prepare_grads(self):
            return False

        def get_grad_norm(self):
            events.append("norm")
            return 2.0

        def get_parameters(self):
            return []

        def step_with_ready_grads(self):
            events.append("step")
            return True

    chained = ChainedOptimizer([_Child()])
    chained.pre_grad_norm_callbacks.append(lambda: events.append("top-pre"))
    assert chained.step() == (True, 2.0, None)
    assert events == ["top-pre", "child-pre", "norm", ("child-post", 2.0), "step"]
