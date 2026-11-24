# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe, mega_moe_context, moe_comm_method
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.mega_moe_context import (
    bind_channel_layers,
    collect_channel_layers,
    register_channel_layer,
)
from vllm_ascend.quantization.quant_type import QuantType


def _factory(
    group,
    num_experts,
    num_topk,
    num_max_tokens_per_rank,
    hidden,
    intermediate_hidden,
    max_recv_token_num,
    dispatch_quant_mode,
    dispatch_quant_out_dtype,
):
    return SimpleNamespace(
        **locals(),
        context=torch.ones(1),
        ep_world_size=2,
        rank_num_per_server=2,
        topo_type=0,
        ccl_buffer_size=4096,
        rank_id=0,
        group_name=str(group),
        _ctx_manager=object(),
    )


def _get_mega_moe_ccl_buffer_size(moe_expert_num, num_topk, intermediate_hidden, server_num=0):
    return 4096


@pytest.fixture
def pair(monkeypatch):
    scope = SimpleNamespace(device_group=object(), cpu_group=object(), world_size=2)
    monkeypatch.setattr(mega_moe_context, "_agree", lambda scope, value: [value, value])
    calls = []

    def make(experts, topk, quant):
        runner = torch.nn.Module()
        runner.routed_experts = torch.nn.Module()
        runner.moe_config = SimpleNamespace(
            num_experts=experts,
            num_local_experts=experts // 2,
            experts_per_token=topk,
            hidden_dim=64,
            intermediate_size_per_partition=128,
        )
        runner.quant_type = quant
        comm = Mock()
        comm.get_symm_buffer_for_mega_moe = Mock(wraps=_factory)
        comm.get_symm_buffer_for_mega_moe.__signature__ = mega_moe_context.signature(_factory)
        comm.get_symm_buffer_for_mega_moe.__module__ = __name__
        comm.token_dispatcher.global_bs = 0
        comm.token_dispatcher.max_num_tokens_per_rank = 128
        comm.mega_moe_buffer_params.side_effect = lambda mode, dtype: dict(
            num_experts=experts,
            num_topk=topk,
            num_max_tokens_per_rank=128,
            hidden=64,
            intermediate_hidden=128,
            max_recv_token_num=0,
            dispatch_quant_mode=mode,
            dispatch_quant_out_dtype=dtype,
        )
        calls.append(comm)
        register_channel_layer(runner, lambda kind, cfg: comm)
        return runner

    monkeypatch.setattr(moe_comm_method, "_MoECommMethods", {})
    anchor = Mock()
    monkeypatch.setattr(moe_comm_method, "get_moe_comm_method", lambda kind, cfg: anchor)

    def run(experts=128, topk=3, quant=QuantType.W4A8MXFP, allow_sharing=True):
        with collect_channel_layers() as layers:
            # Registration order can differ from first forward order.
            draft = make(experts, topk, quant)
            main = make(384, 6, QuantType.W8A8MXFP)
        bind_channel_layers(layers, main, scope, allow_sharing=allow_sharing)
        assert sum(comm.get_symm_buffer_for_mega_moe.call_count for comm in calls) == 1
        return main, draft, anchor._channel_window

    return run


def test_smaller_draft_shares_addresses_but_retains_its_experts(pair):
    main, draft, _ = pair()
    owner, view = main.routed_experts.channel_mega_moe_buffer, draft.routed_experts.channel_mega_moe_buffer
    assert view.context is owner.context
    assert (owner.num_experts, owner.num_topk) == (384, 6)
    assert (view.num_experts, view.num_topk) == (128, 3)
    assert not draft.channel_mega_moe_fallback


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experts": 384, "quant": QuantType.W4A4MXFP},
        {"topk": 7},
        {"experts": 512},
        {"quant": QuantType.NONE},
        {"allow_sharing": False},
    ],
)
def test_incompatible_draft_uses_ordinary_moe(pair, kwargs):
    main, draft, _ = pair(**kwargs)
    assert draft.routed_experts.channel_mega_moe_buffer is None
    assert draft.channel_mega_moe_fallback
    assert not main.channel_mega_moe_fallback


def test_one_byte_over_capacity_falls_back(pair, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_get_mega_moe_ccl_buffer_size", lambda moe_expert_num: 4097)
    _, draft, _ = pair()
    assert draft.channel_mega_moe_fallback


def test_one_rank_rejects_sharing_for_entire_group(pair, monkeypatch):
    monkeypatch.setattr(
        mega_moe_context, "_agree", lambda scope, val: [val, False] if isinstance(val, bool) else [val, val]
    )
    _, draft, _ = pair()
    assert draft.channel_mega_moe_fallback


def test_restore_keeps_captured_context_and_rebinds_all_views(pair, monkeypatch):
    main, draft, window = pair()
    context = window.owner.context
    monkeypatch.setattr(torch.distributed, "barrier", Mock())
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    group = object()
    window.restore(group)
    window.restore(group)
    assert window.factory.call_count == 2
    assert draft.routed_experts.channel_mega_moe_buffer.context is context
    assert draft.routed_experts.channel_mega_moe_buffer.group is group
    assert main.routed_experts.channel_mega_moe_buffer.group is group


def test_fallback_is_scoped_to_the_actual_layer(monkeypatch):
    ctx = SimpleNamespace(
        moe_comm_type=MoECommType.FUSED_MC2, moe_comm_method=object(), use_mega_moe=True, max_tokens_across_dp=4
    )
    original_method = ctx.moe_comm_method
    monkeypatch.setattr(fused_moe, "_EXTRA_CTX", ctx)
    monkeypatch.setattr(fused_moe, "get_unfused_mc2_capacity", lambda tp_size: 16)
    runner = object.__new__(AscendMoERunner)
    torch.nn.Module.__init__(runner)
    runner.channel_mega_moe_fallback = True
    runner.moe_config = SimpleNamespace(ep_size=8, tp_size=1, experts_per_token=3)
    with pytest.raises(ValueError), runner._unfused_channel_context():
        assert ctx.moe_comm_type is MoECommType.MC2
        assert not ctx.use_mega_moe
        ctx.moe_comm_method = None
        raise ValueError("failed forward")
    assert ctx.moe_comm_type is MoECommType.FUSED_MC2
    assert ctx.use_mega_moe and ctx.moe_comm_method is original_method


def test_ordinary_mc2_never_uses_the_larger_megamoe_capacity(monkeypatch):
    from vllm_ascend import ascend_forward_context as ctx

    monkeypatch.setattr(ctx, "get_mc2_tokens_capacity", lambda: 8192)
    cap = ctx.get_unfused_mc2_capacity(tp_size=2)
    assert cap is None
    assert ctx.select_unfused_a5_moe(1, cap, 8, 3) is MoECommType.ALLTOALL
    assert ctx.select_unfused_a5_moe(1, cap, 2, 3) is MoECommType.ALLGATHER


def test_a3_and_single_model_registration_is_inert_without_collection():
    get_method = Mock(side_effect=AssertionError("unexpected A5 registration"))
    register_channel_layer(None, get_method)
    get_method.assert_not_called()


def test_missing_owner_setting_does_not_match_an_optional_none(pair):
    _, _, window = pair()
    del window.owner.dispatch_quant_out_dtype
    assert not window.fits(dict(window.params, dispatch_quant_out_dtype=None))


def test_size_query_uses_draft_shape_and_owner_topology(pair, monkeypatch):
    _, draft, window = pair()
    window.owner.ep_world_size = 8
    window.owner.topo_type = 1
    window.owner.rank_num_per_server = 2
    observed = []

    def query(ep_world_size, moe_expert_num, num_topk, intermediate_hidden, server_num):
        observed.append((ep_world_size, moe_expert_num, num_topk, intermediate_hidden, server_num))
        return 4096

    monkeypatch.setattr(sys.modules[__name__], "_get_mega_moe_ccl_buffer_size", query)
    params = dict(window.params, num_experts=128, num_topk=3, intermediate_hidden=32)
    assert window.fits(params)
    assert observed == [(8, 128, 3, 32, 4)]
    assert draft.routed_experts.channel_mega_moe_buffer.context is window.owner.context


@pytest.mark.parametrize(("ep_size", "ranks"), [(0, 2), (8, 0), (8, -1), (8, 3)])
def test_invalid_topology_rejects_shared_window(pair, ep_size, ranks):
    _, _, window = pair()
    window.owner.ep_world_size = ep_size
    window.owner.topo_type = 1
    window.owner.rank_num_per_server = ranks
    assert not window.fits(window.params)


def test_legacy_size_query_without_server_count_is_local_only(pair, monkeypatch):
    _, _, window = pair()
    monkeypatch.setattr(sys.modules[__name__], "_get_mega_moe_ccl_buffer_size", lambda moe_expert_num, num_topk: 4096)
    assert window.fits(window.params)
    window.owner.topo_type = 1
    assert not window.fits(window.params)
