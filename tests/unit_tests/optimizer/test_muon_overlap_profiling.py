# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the backward <-> Muon overlap profiling harness.

- ``parse_shadow_launch``: the ``--muon-overlap-shadow`` spec grammar.
- ``profiling_skip_step`` (E1): allocates state, applies no update, honours the join streams.
- ``ShadowMuon`` (E2): clones a LayerShardedMuon (homes remapped, sharding attrs copied) and
  steps it on a side stream without touching the real optimizer, over the model's own or
  dedicated communicators (4 ranks, TP 2 x GTP_remat 2).
- ``grad_sync_started_callbacks``: fire once per bucket group per grad sync, handle set.
- ``post_grad_norm_callbacks``: ``ChainedOptimizer.step`` reports the global norm.

Launch: torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_muon_overlap_profiling.py
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("emerging_optimizers", reason="requires emerging-optimizers")

from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon
from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.training.muon_overlap_profiling import (
    ShadowMuon,
    iter_muon_inner_optimizers,
    parse_shadow_launch,
)
from tests.unit_tests.test_utilities import Utils

_MUON_KW = dict(
    lr=1e-2,
    momentum=0.95,
    nesterov=True,
    weight_decay=0.0,
    coefficient_type="quintic",
    num_ns_steps=5,
    scale_mode="spectral",
    extra_scale_factor=1.0,
    fp32_matmul_prec="highest",
)


@pytest.fixture(scope="module", autouse=True)
def _dist_init():
    Utils.initialize_model_parallel()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.manual_seed(42)
    yield
    Utils.destroy_model_parallel()


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Spec grammar and optimizer discovery
# ---------------------------------------------------------------------------


def test_parse_shadow_launch():
    assert parse_shadow_launch("tail") == ("tail", 0)
    assert parse_shadow_launch("bucket:3") == ("bucket", 3)
    for bad in ("bucket:0", "bucket", "head", "bucket:-1", "bucket:x", ""):
        with pytest.raises(ValueError):
            parse_shadow_launch(bad)


def test_iter_muon_inner_optimizers_walks_chained_and_wrapped():
    p = torch.nn.Parameter(torch.randn(8, 4, device=_device()))
    muon = LayerShardedMuon([p], gtp_remat_group=None, **_MUON_KW)
    adam = torch.optim.Adam([torch.nn.Parameter(torch.randn(4, device=_device()))])
    # ChainedOptimizer -> [LayerWise-like chained -> [Float16 wrapper -> muon], DistOpt -> adam]
    tree = SimpleNamespace(
        chained_optimizers=[
            SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=muon)]),
            SimpleNamespace(optimizer=adam),
        ]
    )
    assert list(iter_muon_inner_optimizers(tree)) == [muon]


# ---------------------------------------------------------------------------
# E1: profiling_skip_step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ["layer_sharded", "tensor_parallel"])
def test_skip_step_allocates_state_without_updating(cls):
    dev = _device()
    p = torch.nn.Parameter(torch.randn(16, 8, device=dev))
    p.grad = torch.randn(16, 8, device=dev)
    if cls == "layer_sharded":
        opt = LayerShardedMuon([p], gtp_remat_group=None, **_MUON_KW)
        opt.set_param_ns_homes({})
    else:
        opt = TensorParallelMuon([p], **{**_MUON_KW, "split_qkv": False})
    before = p.detach().clone()

    opt.profiling_skip_step = True
    assert opt.step() is None
    assert torch.equal(p, before), "the E1 no-op must not update the param"
    state = opt.state[p]
    assert "momentum_buffer" in state, "state must be allocated like a real step"
    assert torch.count_nonzero(state["momentum_buffer"]) == 0

    opt.profiling_skip_step = False
    opt.step()
    assert not torch.equal(p, before), "the real step still updates once the switch is off"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")
def test_skip_step_joins_streams():
    dev = _device()
    p = torch.nn.Parameter(torch.randn(16, 8, device=dev))
    p.grad = torch.randn(16, 8, device=dev)
    opt = LayerShardedMuon([p], gtp_remat_group=None, **_MUON_KW)
    opt.set_param_ns_homes({})
    opt.profiling_skip_step = True
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        marker = torch.zeros(1, device=dev)
        for _ in range(50):  # some side-stream work the join has to wait for
            marker = marker + torch.randn(2048, 2048, device=dev).sum() * 0
        marker += 1.0
    opt.profiling_join_streams = [side]
    opt.step()
    # The current stream is ordered behind the side stream now; this read is safe only if so.
    probe = marker.clone()
    torch.cuda.synchronize()
    assert probe.item() == 1.0


# ---------------------------------------------------------------------------
# E2: ShadowMuon on a TP(2) x GTP_remat(2) grid
# ---------------------------------------------------------------------------

_TP_GROUP = None
_GTP_REMAT_GROUP = None


def _get_2d_groups():
    """TP(2) x GTP_remat(2) subgroups, created once. Rank convention: r = g * T + t."""
    global _TP_GROUP, _GTP_REMAT_GROUP
    if _TP_GROUP is None:
        _TP_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 1], [2, 3]])
        _GTP_REMAT_GROUP, _ = dist.new_subgroups_by_enumeration([[0, 2], [1, 3]])
    return _TP_GROUP, _GTP_REMAT_GROUP


def _column_parallel_gtp_params(dev, t, g, n_params=2, full_shape=(32, 16)):
    """Column-parallel (partition_dim 0) weights, GTP_remat-sharding the TP-local rows; every
    rank seeds identically so the ranks hold shards of the same full matrices."""
    torch.manual_seed(100)
    fulls = [torch.randn(*full_shape, device=dev) for _ in range(n_params)]
    grads = [torch.randn(*full_shape, device=dev) for _ in range(n_params)]
    params = []
    rows = full_shape[0] // 4
    for w, gr in zip(fulls, grads):
        start = t * (full_shape[0] // 2) + g * rows
        p = torch.nn.Parameter(w[start : start + rows].clone())
        p.grad = gr[start : start + rows].clone()
        p.tensor_model_parallel = True
        p.partition_dim = 0
        p.is_gtp_weight_remat = True
        params.append(p)
    return params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")
@pytest.mark.parametrize("comm", ["shared", "dedicated"])
def test_shadow_steps_without_touching_the_real_optimizer(comm):
    if dist.get_world_size() != 4:
        pytest.skip("Requires exactly 4 ranks (launch with torchrun --nproc-per-node=4)")
    tp_group, gtp_group = _get_2d_groups()
    rank = dist.get_rank()
    t, g = rank % 2, rank // 2
    dev = _device()
    params = _column_parallel_gtp_params(dev, t, g)
    opt = LayerShardedMuon(params, gtp_remat_group=gtp_group, tp_group=tp_group, **_MUON_KW)
    homes = {id(params[0]): (0, 1), id(params[1]): (1, 0)}
    opt.set_param_ns_homes(homes)
    before = [p.detach().clone() for p in params]
    opt.profiling_skip_step = True

    shadow = ShadowMuon([opt], comm=comm)
    assert len(shadow.shadows) == 1
    clone = shadow.shadows[0]
    clone_params = [q for group in clone.param_groups for q in group["params"]]
    assert all(torch.equal(q, p) for q, p in zip(clone_params, params))
    assert all(q is not p for q, p in zip(clone_params, params))
    assert clone._param_ns_homes == {id(q): homes[id(p)] for q, p in zip(clone_params, params)}
    assert clone_params[0].partition_dim == 0 and clone_params[0].is_gtp_weight_remat
    assert shadow.extra_bytes == sum(3 * q.numel() * q.element_size() for q in clone_params)
    if comm == "dedicated":
        assert len(shadow._dedicated) == 2
        for group, dedicated in ((gtp_group, clone.gtp_remat_group), (tp_group, clone.tp_group)):
            assert dedicated is not group
            assert dist.get_process_group_ranks(dedicated) == dist.get_process_group_ranks(group)
    else:
        assert clone.gtp_remat_group is gtp_group and clone.tp_group is tp_group

    try:
        shadow.warmup()
        opt.profiling_join_streams = [shadow.stream]
        for _ in range(2):
            shadow.launch()
            opt.step()  # E1 no-op + join behind the shadow stream
            shadow.reset()
        torch.cuda.synchronize()
        assert all(torch.equal(p, b) for p, b in zip(params, before)), "real params untouched"
        for q, b in zip(clone_params, before):
            assert torch.isfinite(q).all()
            assert not torch.equal(q, b), "the shadow must have stepped its clones"
        assert "momentum_buffer" in clone.state[clone_params[0]]
        assert shadow.launch_event.elapsed_time(shadow.done_event) >= 0.0
    finally:
        for dedicated in shadow._dedicated.values():
            dist.destroy_process_group(dedicated)


# ---------------------------------------------------------------------------
# Hooks: DDP bucket-group callbacks and ChainedOptimizer grad-norm callbacks
# ---------------------------------------------------------------------------


def test_grad_sync_started_callbacks_fire_once_per_bucket_group():
    from tests.unit_tests.distributed.test_param_and_grad_buffer import get_model_and_buffers

    model, _, bucket_groups = get_model_and_buffers(
        input_dim=100,
        output_dim=100,
        num_layers=4,
        bias=True,
        shared_embedding=False,
        bucket_size=None,
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        average_in_collective=False,
    )
    seen = []
    for bucket_group in bucket_groups:
        bucket_group.grad_sync_started_callbacks.append(
            lambda group: seen.append((group, group.grad_reduce_handle))
        )
    params = list(model.parameters())
    param_to_bucket_group = {p: bg for bg in bucket_groups for p in bg.params}

    # First batch: register_grad_ready only seeds the golden counts; finish_grad_sync dispatches.
    for p in params:
        param_to_bucket_group[p].register_grad_ready(p)
    for bucket_group in bucket_groups:
        bucket_group.finish_grad_sync()
    assert [group for group, _ in seen] == list(bucket_groups)
    assert all(handle is not None for _, handle in seen), "async dispatch sets the handle first"

    for bucket_group in bucket_groups:
        bucket_group.reset()
    seen.clear()
    # Steady state: the last register_grad_ready of a bucket group dispatches (mid-backward).
    for p in params:
        param_to_bucket_group[p].register_grad_ready(p)
    assert len(seen) == len(bucket_groups)
    for bucket_group in bucket_groups:
        bucket_group.finish_grad_sync()
    assert len(seen) == len(bucket_groups), "finish_grad_sync must not re-fire the callbacks"


def test_post_grad_norm_callbacks_receive_the_norm():
    config = SimpleNamespace(
        clip_grad=0.0,
        log_num_zeros_in_grad=False,
        grad_norm_skip_threshold=float("inf"),
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather_with_optimizer_step=False,
    )

    class _Stub:
        is_stub_optimizer = False

        def __init__(self):
            self.config = config

        def prepare_grads(self):
            return False

        def get_grad_norm(self):
            return 3.0

        def get_parameters(self):
            return []

        def step_with_ready_grads(self):
            return True

    chained = ChainedOptimizer([_Stub()])
    seen = []
    chained.post_grad_norm_callbacks.append(seen.append)
    assert chained.step() == (True, 3.0, None)
    assert seen == [3.0]
