# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Same-forward routing/capture checks; collectives are CPU boundary mocks.

Expected expert IDs come from the original token IDs, independently of the
layout under test. These tests do not establish HCCL or graph-replay support.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.layers.fused_moe import routed_experts_capturer as capture_module
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

import vllm_ascend.ascend_forward_context as ascend_context
import vllm_ascend.ops.fused_moe.prepare_finalize as prepare_module
import vllm_ascend.ops.fused_moe.routed_experts as experts_module
from vllm_ascend.ops import register_custom_ops
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    AlltoAllCommImpl,
    FusedMC2CommImpl,
    MC2CommImpl,
)


class _CpuRouter(BaseRouter):
    """Exercise the upstream callback ordering with deterministic CPU top-k."""

    routing_method_type = None

    def __init__(self, capture_fn):
        super().__init__(top_k=2, global_num_experts=128)
        self.set_capture_fn(capture_fn)

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        weights, ids = torch.topk(router_logits, 2, dim=-1)
        return weights, ids.to(torch.int32)


def _expected_ids(tokens):
    first = torch.as_tensor(tokens, dtype=torch.int32) * 2 % 128
    return torch.stack((first, first + 1), dim=-1)


def _logits(tokens):
    ids = _expected_ids(tokens).long()
    logits = torch.full((len(ids), 128), -1000.0)
    logits.scatter_(1, ids, torch.tensor([2.0, 1.0]).expand(len(ids), -1))
    return logits


def _capturer():
    capturer = capture_module.RoutedExpertsCapturer.__new__(capture_module.RoutedExpertsCapturer)
    # Sentinel checks detect overwrites of padding, other layers and old rows.
    capturer.device_buffer = torch.full((32, 2, 2), -1, dtype=torch.int32)
    capturer.dp_rank = 0
    capturer.tp_size = 2
    return capturer


@pytest.fixture
def capture_env(monkeypatch):
    group = object()
    config = SimpleNamespace(dp_rank=0, dp_size=2, pcp_size=1, is_sequence_parallel=False, num_experts=128)
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([5, 13]), local_sizes=None)
    ctx = SimpleNamespace(dp_metadata=metadata)
    extra = SimpleNamespace(num_tokens=5, padded_num_tokens=14, mc2_mask=torch.arange(14) < 5)
    monkeypatch.setattr(prepare_module, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(prepare_module, "_EXTRA_CTX", extra)
    monkeypatch.setattr(prepare_module, "get_tp_group", lambda: SimpleNamespace(device_group=group))
    monkeypatch.setattr(prepare_module, "get_dynamic_mx_quant_scale_alg", lambda: 0)
    # Padding device allocation is outside the layout logic being tested.
    monkeypatch.setattr(
        prepare_module, "_pad_tokens_with_cat", lambda x, n: torch.nn.functional.pad(x, (0, 0, 0, n - len(x)))
    )
    monkeypatch.setattr(experts_module, "get_ascend_config", lambda: SimpleNamespace(enable_force_eplb=False))
    return SimpleNamespace(config=config, ctx=ctx, extra=extra, group=group)


def _comm(comm_cls, env, monkeypatch, rank, tp):
    monkeypatch.setattr(prepare_module, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(prepare_module, "get_tensor_model_parallel_rank", lambda: rank)
    comm = comm_cls.__new__(comm_cls)
    comm.moe_config = env.config
    # Use each communication class's actual choice of prepare implementation.
    comm.prepare_finalize = comm._get_prepare_finalize()
    return comm


def _route(prepared, capturer, env, *, log2phy=None):
    layer = SimpleNamespace(
        router=_CpuRouter(partial(capturer.capture, 1)),
        log2phy=log2phy,
        n_shared_experts=0,
        moe_config=env.config,
        global_redundant_expert_num=0,
    )
    _, ids = experts_module.AscendRoutedExperts._select_experts(
        layer, prepared.hidden_states, prepared.router_logits, False, capture_layout=prepared.capture_layout
    )
    return ids


def _assert_record(capturer, tokens):
    expected = _expected_ids(tokens)
    torch.testing.assert_close(capturer.device_buffer[: len(expected), 1], expected)
    assert torch.all(capturer.device_buffer[len(expected) :, 1] == -1)
    assert torch.all(capturer.device_buffer[:, 0] == -1)


@pytest.mark.parametrize("comm_cls", [AlltoAllCommImpl, MC2CommImpl, FusedMC2CommImpl])
@pytest.mark.parametrize("sp", [False, True], ids=["no-sp", "sp"])
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 5, 7])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_tp_shards_record_exact_logical_ids(capture_env, monkeypatch, comm_cls, sp, tp, tokens, dp_size):
    env = capture_env
    env.config.dp_size = dp_size
    if dp_size == 1:
        env.ctx.dp_metadata = None
    else:
        env.ctx.dp_metadata.num_tokens_across_dp_cpu[0] = tokens
    env.extra.num_tokens = tokens
    env.config.is_sequence_parallel = sp
    # A larger DP peer forces MC2's per-shard padding gaps. DP=1 also
    # checks that TP collection does not depend on DP metadata being present.
    capacity = 13 if dp_size == 2 else max(tokens, 1)
    env.extra.padded_num_tokens = ((capacity + tp - 1) // tp) * tp
    env.extra.mc2_mask = torch.arange(env.extra.padded_num_tokens) < tokens
    original_logits = _logits(range(tokens))
    shard_rows = (tokens + tp - 1) // tp
    sp_logits = torch.nn.functional.pad(original_logits, (0, 0, 0, shard_rows * tp - tokens))
    prepared = []
    for rank in range(tp):
        comm = _comm(comm_cls, env, monkeypatch, rank, tp)
        logits = sp_logits[rank * shard_rows : (rank + 1) * shard_rows] if sp else original_logits
        prepared.append(comm.prepare(torch.ones(len(logits), 8), logits, sp, capture_enabled=True))

    peer_ids = [torch.topk(item.router_logits, 2, dim=-1).indices.to(torch.int32) for item in prepared]
    for rank, item in enumerate(prepared):
        gathered = Mock()

        def all_gather(outputs, local, group, rank=rank, gathered=gathered):
            assert group is env.group
            torch.testing.assert_close(local, peer_ids[rank])
            assert [len(out) for out in outputs] == [len(peer) for peer in peer_ids]
            for out, peer in zip(outputs, peer_ids):
                out.copy_(peer)
            gathered()

        monkeypatch.setattr(capture_module.dist, "get_world_size", lambda group: tp)
        monkeypatch.setattr(capture_module.dist, "get_rank", lambda group, rank=rank: rank)
        monkeypatch.setattr(capture_module.dist, "all_gather", all_gather)
        capturer = _capturer()
        # An explicit layout must bypass the platform's default restoration.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                capture_module.current_platform, "get_routed_experts_local_rows", Mock(side_effect=AssertionError)
            )
            _route(item, capturer, env)
        _assert_record(capturer, range(tokens))
        assert gathered.call_count == (1 if tp > 1 else 0)
        assert item.capture_layout.gather_group is (env.group if tp > 1 else None)


@pytest.mark.parametrize("dp_rank,tokens", [(0, 5), (1, 7)])
def test_allgather_dp_ownership(capture_env, monkeypatch, dp_rank, tokens):
    env = capture_env
    env.config.dp_rank = dp_rank
    env.ctx.dp_metadata.num_tokens_across_dp_cpu = torch.tensor([5, 7])
    env.extra.max_tokens_across_dp = 7
    local_tokens = range(10 * dp_rank, 10 * dp_rank + tokens)
    # Each DP occupies seven physical rows; DP0's last two are padding.
    blocks = [
        torch.nn.functional.pad(_logits(range(d * 10, d * 10 + n)), (0, 0, 0, 7 - n)) for d, n in enumerate((5, 7))
    ]
    env.config.dp_group = SimpleNamespace(
        all_gather=lambda x, dim: torch.cat(blocks) if x.shape[1] == 128 else torch.ones(14, 8)
    )
    monkeypatch.setattr(capture_module.dist, "all_gather", Mock(side_effect=AssertionError("redundant TP gather")))
    comm = _comm(AllGatherCommImpl, env, monkeypatch, 0, 2)
    prepared = comm.prepare(torch.ones(tokens, 8), _logits(local_tokens), capture_enabled=True)
    assert prepared.capture_layout.source_ranges == ((dp_rank * 7, tokens),)
    capturer = _capturer()
    _route(prepared, capturer, env)
    _assert_record(capturer, local_tokens)


@pytest.mark.parametrize("dp_rank", [0, 1])
def test_ep_sp_layout_matches_real_gather_unpadding(capture_env, monkeypatch, dp_rank):
    env = capture_env
    env.config.dp_rank = dp_rank
    env.config.is_sequence_parallel = True
    env.ctx.dp_metadata.local_sizes = [3, 3, 4, 4]
    env.ctx.dp_metadata.get_chunk_sizes_across_dp_rank = lambda: env.ctx.dp_metadata.local_sizes
    env.ctx.dp_metadata.num_tokens_across_dp_cpu = torch.tensor([5, 7])
    blocks = []
    for d, count in enumerate((5, 7)):
        valid = _logits(range(d * 10, d * 10 + count))
        shard_size = (count + 1) // 2
        valid = torch.nn.functional.pad(valid, (0, 0, 0, shard_size * 2 - count))
        blocks.extend(torch.nn.functional.pad(chunk, (0, 0, 0, 4 - len(chunk))) for chunk in valid.chunk(2))
    ep = SimpleNamespace(
        world_size=len(blocks),
        all_gather=lambda x, dim: torch.cat(blocks) if x.shape[1] == 128 else torch.ones(len(blocks) * 4, 8),
    )
    monkeypatch.setattr(register_custom_ops, "get_ep_group", lambda: ep)
    monkeypatch.setattr(register_custom_ops, "get_forward_context", lambda: env.ctx)
    monkeypatch.setattr(
        torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", register_custom_ops._maybe_all_gather_and_maybe_unpad_impl
    )
    comm = _comm(AllGatherCommImpl, env, monkeypatch, 0, 2)
    prepared = comm.prepare(torch.ones(3, 8), _logits(range(3)), capture_enabled=True)
    capturer = _capturer()
    _route(prepared, capturer, env)
    start = dp_rank * 10
    _assert_record(capturer, range(start, start + (5 if dp_rank == 0 else 7)))
    assert prepared.capture_layout.gather_group is None


def test_capture_disabled_needs_no_new_upstream_api(capture_env, monkeypatch):
    monkeypatch.delattr(capture_module, "RoutedExpertsCaptureLayout")
    comm = _comm(AlltoAllCommImpl, capture_env, monkeypatch, 0, 1)
    prepared = comm.prepare(torch.ones(5, 8), _logits(range(5)))
    assert prepared.capture_layout is None
    with pytest.raises(RuntimeError, match="requires vLLM with RoutedExpertsCaptureLayout"):
        comm.prepare(torch.ones(5, 8), _logits(range(5)), capture_enabled=True)


@pytest.mark.parametrize("v2", [False, True])
def test_single_dp_token_count_uses_v1_v2_context(monkeypatch, v2):
    ctx = SimpleNamespace(dp_metadata=None, num_tokens=5, additional_kwargs={"num_tokens": 5})
    monkeypatch.setattr(prepare_module, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(ascend_context, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(ascend_context, "_USE_V2_EXTRA_KWARGS", v2)
    assert prepare_module._capture_local_rows(0) == 5


@pytest.mark.parametrize("has_metadata", [False, True])
def test_allgather_single_dp_requires_no_capture_collective(capture_env, monkeypatch, has_metadata):
    env = capture_env
    env.config.dp_size = 1
    env.ctx.dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([5])) if has_metadata else None
    comm = _comm(AllGatherCommImpl, env, monkeypatch, 0, 2)
    prepared = comm.prepare(torch.ones(5, 8), _logits(range(5)), capture_enabled=True)
    monkeypatch.setattr(capture_module.dist, "all_gather", Mock(side_effect=AssertionError("redundant gather")))
    capturer = _capturer()
    _route(prepared, capturer, env)
    _assert_record(capturer, range(5))
    assert prepared.capture_layout.source_ranges == ((0, 5),)


def test_fixed_capacity_layout_does_not_depend_on_mask_values(capture_env, monkeypatch):
    # Graph metadata describes a fixed capacity. Changing mask bits must not
    # introduce CPU reads or change the captured gather/copy plan. This is a
    # metadata check, not an ACL graph replay test.
    env = capture_env
    env.ctx.dp_metadata.num_tokens_across_dp_cpu = torch.tensor([14, 14])
    comm = _comm(MC2CommImpl, env, monkeypatch, 0, 2)
    layouts = []
    for valid in (5, 7):
        env.extra.mc2_mask = torch.arange(14) < valid
        result = comm.prepare(torch.ones(7, 8), _logits(range(7)), True, capture_enabled=True)
        layouts.append(result.capture_layout)
    assert layouts[0] == layouts[1]
    assert layouts[0].source_ranges == ((0, 7), (7, 7))


def test_capture_disabled_preserves_legacy_router_signature(capture_env):
    def legacy_select(hidden_states, router_logits, input_ids):
        return torch.ones(5, 2), _expected_ids(range(5))

    layer = SimpleNamespace(
        router=SimpleNamespace(_select_experts=legacy_select),
        log2phy=None,
        n_shared_experts=0,
        moe_config=capture_env.config,
        global_redundant_expert_num=0,
    )
    _, ids = experts_module.AscendRoutedExperts._select_experts(layer, torch.ones(5, 8), _logits(range(5)), False)
    torch.testing.assert_close(ids, _expected_ids(range(5)))


@pytest.mark.parametrize("missing_layout", [False, True])
def test_forward_passes_layout_and_captures_before_log2phy(capture_env, monkeypatch, missing_layout):
    env = capture_env
    comm = _comm(AlltoAllCommImpl, env, monkeypatch, 0, 1)
    real_prepare = comm.prepare
    prepare_spy = Mock(wraps=real_prepare)
    comm.prepare = prepare_spy
    if missing_layout:
        prepare_spy.side_effect = lambda **kwargs: real_prepare(**{**kwargs, "capture_enabled": False})
    comm.finalize = lambda hidden_states, **kwargs: hidden_states
    monkeypatch.setattr(experts_module, "get_moe_comm_method", lambda *args: comm)
    monkeypatch.setattr(
        experts_module,
        "_EXTRA_CTX",
        SimpleNamespace(in_profile_run=False, moe_comm_type=SimpleNamespace(name="ALLTOALL")),
    )
    monkeypatch.setattr(experts_module, "get_forward_context", lambda: env.ctx)
    capturer = _capturer()
    apply = Mock(return_value=SimpleNamespace(routed_out=torch.ones(5, 8)))
    layer = SimpleNamespace(
        router=_CpuRouter(partial(capturer.capture, 1)),
        log2phy=torch.arange(127, -1, -1),
        n_shared_experts=0,
        moe_config=env.config,
        global_redundant_expert_num=0,
        quant_type=prepare_module.QuantType.NONE,
        quant_method=SimpleNamespace(apply=apply),
        enable_npugraph_ex_static_kernel=False,
        _use_v2_model_runner=False,
        dynamic_eplb=False,
        return_with_event=False,
        layer_name="model.layers.1.mlp.experts",
    )
    layer._select_experts = partial(experts_module.AscendRoutedExperts._select_experts, layer)
    env.ctx.cudagraph_runtime_mode = "NONE"
    monkeypatch.setattr(experts_module, "get_tensor_model_parallel_rank", lambda: 0)
    if missing_layout:
        with pytest.raises(RuntimeError, match="did not provide a capture layout"):
            experts_module.AscendRoutedExperts.forward_impl(
                layer, hidden_states=torch.ones(5, 8), router_logits=_logits(range(5))
            )
        apply.assert_not_called()
    else:
        experts_module.AscendRoutedExperts.forward_impl(
            layer, hidden_states=torch.ones(5, 8), router_logits=_logits(range(5))
        )
        assert prepare_spy.call_args.kwargs["capture_enabled"] is True
        _assert_record(capturer, range(5))
        torch.testing.assert_close(apply.call_args.kwargs["topk_ids"], 127 - _expected_ids(range(5)).long())


def test_uses_upstream_capture_implementation():
    assert capture_module.RoutedExpertsCapturer.capture.__module__ == capture_module.__name__
