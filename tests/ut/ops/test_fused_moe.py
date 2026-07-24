# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendMoERunner,
    AscendUnquantizedFusedMoEMethod,
    make_eplb_placement_config,
    use_multistage_eplb_load,
)
from vllm_ascend.quantization.quant_type import QuantType


def _build_weight_layer():
    return SimpleNamespace(
        w13_weight=nn.Parameter(torch.randn(2, 3, 4)),
        w2_weight=nn.Parameter(torch.randn(2, 4, 3)),
    )


def _build_apply_layer():
    return SimpleNamespace(
        w13_weight=nn.Parameter(torch.randn(4, 3, 8)),
        w2_weight=nn.Parameter(torch.randn(4, 8, 3)),
        w13_bias=None,
        w2_bias=None,
        zero_expert_num=0,
        zero_expert_type=None,
        n_shared_experts=0,
        swiglu_limit=0.0,
    )


def _build_unquantized_method(*, dynamic_eplb: bool = False):
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    method.dynamic_eplb = dynamic_eplb
    method.tid2eid = None
    method.moe = SimpleNamespace(has_bias=False)
    method._maybe_pad_weight = MagicMock(side_effect=lambda weight: weight)
    return method


@pytest.mark.parametrize(
    ("dynamic_eplb", "policy_type", "collection_interval", "expected"),
    [
        (True, 2, 600, False),
        (True, 3, 600, True),
        (True, 3, 1, False),
        (False, 3, 600, False),
    ],
)
def test_use_multistage_eplb_load(dynamic_eplb, policy_type, collection_interval, expected):
    assert use_multistage_eplb_load(dynamic_eplb, policy_type, collection_interval) is expected


def test_make_eplb_placement_config_does_not_copy_source():
    source = SimpleNamespace(expert_map_path=None, dynamic_eplb=True, num_redundant_experts=0)

    placement_config = make_eplb_placement_config(source, num_redundant_experts=8)

    assert placement_config.expert_map_path is None
    assert placement_config.dynamic_eplb is True
    assert placement_config.num_redundant_experts == 8
    assert source.num_redundant_experts == 0


def test_ascend_unquantized_skips_upstream_modular_kernel_init():
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)

    assert method.maybe_make_prepare_finalize() is None


def test_process_weights_after_loading_uses_version_specific_layout(
    monkeypatch,
):
    method = _build_unquantized_method()
    layer = _build_weight_layer()
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()
    ascend_config = SimpleNamespace(enable_fused_mc2=False)

    monkeypatch.setattr(fused_moe_module, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(fused_moe_module, "maybe_trans_nz", lambda weight: weight)
    upstream_method_base = AscendUnquantizedFusedMoEMethod.__mro__[2]
    monkeypatch.setattr(
        upstream_method_base,
        "process_weights_after_loading",
        lambda self, layer: None,
        raising=False,
    )

    method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.w13_weight, original_w13.transpose(1, 2))
    torch.testing.assert_close(layer.w2_weight, original_w2.transpose(1, 2))
    assert layer.w13_weight.is_contiguous() is True
    assert layer.w2_weight.is_contiguous() is True


@pytest.mark.parametrize("moe_comm_type", [MoECommType.ALLGATHER, MoECommType.FUSED_MC2])
def test_unquantized_apply_builds_current_fused_experts_input(monkeypatch, moe_comm_type):
    method = _build_unquantized_method()
    layer = _build_apply_layer()
    hidden_states = torch.randn(2, 3, dtype=torch.float16)
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    routed_out = torch.ones_like(hidden_states)
    moe_comm_method = MagicMock()
    moe_comm_method.fused_experts.return_value = routed_out

    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=moe_comm_type, moe_comm_method=moe_comm_method),
    )
    monkeypatch.setattr(fused_moe_module, "get_moe_num_logical_experts", lambda *args, **kwargs: 4)
    monkeypatch.setattr(fused_moe_module, "get_forward_context", lambda: SimpleNamespace(input_ids=None))
    monkeypatch.setattr(fused_moe_module, "get_current_vllm_config", lambda: None)
    select_experts = MagicMock(return_value=(topk_weights, topk_ids))
    monkeypatch.setattr(fused_moe_module, "select_experts", select_experts)

    result = method.apply(
        layer=layer,
        x=hidden_states,
        use_grouped_topk=False,
        top_k=2,
        router_logits=torch.randn(2, 4),
        renormalize=True,
        num_experts=4,
        apply_router_weight_on_input=True,
        activation="gelu",
    )

    assert result is routed_out
    select_experts.assert_called_once()
    fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
    assert fused_input.hidden_states is hidden_states
    torch.testing.assert_close(fused_input.topk_weights, topk_weights.to(hidden_states.dtype))
    assert torch.equal(fused_input.topk_ids, topk_ids)
    assert fused_input.routing.apply_router_weight_on_input
    assert fused_input.activation == "gelu"
    assert fused_input.quant.quant_type == QuantType.NONE
    if moe_comm_type == MoECommType.FUSED_MC2:
        assert fused_input.weights.w1[0] is layer.w13_weight
        assert fused_input.weights.w2[0] is layer.w2_weight
    else:
        assert fused_input.weights.w1 is layer.w13_weight
        assert fused_input.weights.w2 is layer.w2_weight


@pytest.mark.parametrize(
    "moe_comm_type, flash_comm_v1_enabled, expected",
    [
        (MoECommType.ALLTOALL, False, True),
        (MoECommType.MC2, False, True),
        (MoECommType.FUSED_MC2, False, True),
        (MoECommType.ALLGATHER, False, False),
        (MoECommType.ALLGATHER, True, True),
    ],
)
def test_runner_reduction_contract(monkeypatch, moe_comm_type, flash_comm_v1_enabled, expected):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    shared_output = object()
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=moe_comm_type, flash_comm_v1_enabled=flash_comm_v1_enabled),
    )

    assert runner.use_dp_chunking is False
    assert runner._fused_output_is_reduced is expected
    assert runner._maybe_reduce_shared_expert_output(shared_output) is shared_output


class _Projection(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0 + 1.0, None


class _Gate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros((*hidden_states.shape[:-1], 1), dtype=hidden_states.dtype), None


@pytest.mark.parametrize("with_gate", [False, True])
def test_shared_experts_part2_applies_optional_gate(with_gate):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner._shared_experts = SimpleNamespace(
        act_fn=nn.Identity(),
        down_proj=_Projection(),
        expert_gate=_Gate() if with_gate else None,
    )
    hidden_states = torch.randn(3, 4)
    shared_gate_up = torch.randn(3, 4)

    output = runner._shared_experts_part2(hidden_states, shared_gate_up)

    expected = shared_gate_up * 2.0 + 1.0
    if with_gate:
        expected = expected * 0.5
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_shared_forward_impl_returns_current_runner_contract(monkeypatch, has_shared_experts):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner._shared_experts = object() if has_shared_experts else None
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)
    routed_out = torch.randn(2, 4)
    shared_out = torch.randn(2, 4)
    routed_result = SimpleNamespace(
        routed_out=routed_out,
        before_dispatch_evt=None,
        before_gmm2_evt=None,
        before_combine_evt=None,
        swiglu_limit=0.0,
    )
    runner.no_shared_forward_impl = MagicMock(return_value=routed_result)
    runner._forward_shared_experts = MagicMock(return_value=shared_out)
    current_stream = MagicMock()

    monkeypatch.setattr(AscendMoERunner, "is_internal_router", property(lambda _: False))
    monkeypatch.setattr(fused_moe_module.torch.npu, "current_stream", lambda: current_stream)

    result = runner.shared_forward_impl(hidden_states, router_logits)

    runner.no_shared_forward_impl.assert_called_once_with(
        hidden_states,
        router_logits,
        return_with_event=True,
    )
    if has_shared_experts:
        assert result[0] is shared_out
        assert result[1] is routed_out
        runner._forward_shared_experts.assert_called_once()
    else:
        assert result is routed_out
        runner._forward_shared_experts.assert_not_called()
