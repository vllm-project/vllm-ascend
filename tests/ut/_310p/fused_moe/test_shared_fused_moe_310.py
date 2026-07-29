# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vllm_ascend._310p.fused_moe import fused_moe as fused_moe_310_module
from vllm_ascend._310p.fused_moe.fused_moe import (
    AscendMoERunner310,
    AscendUnquantizedFusedMoEMethod310,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.quantization.quant_type import QuantType


def _build_runner() -> AscendMoERunner310:
    runner = AscendMoERunner310.__new__(AscendMoERunner310)
    nn.Module.__init__(runner)
    return runner


def _build_weight_layer():
    return SimpleNamespace(
        w13_weight=nn.Parameter(torch.randn(2, 3, 4)),
        w2_weight=nn.Parameter(torch.randn(2, 4, 3)),
    )


def test_runner_310_installs_specialized_unquantized_method_and_comm():
    runner = _build_runner()
    moe_config = MagicMock()
    runner.moe_config = moe_config
    runner._get_quant_type = MagicMock(return_value=QuantType.NONE)
    routed_experts = SimpleNamespace(quant_config=None, quant_method=None)
    quant_method = object()
    comm_method = object()

    with (
        patch.object(AscendMoERunner, "__init__", return_value=None) as parent_init,
        patch.object(fused_moe_310_module, "AscendUnquantizedFusedMoEMethod310", return_value=quant_method),
        patch.object(fused_moe_310_module, "AllGatherCommImpl310", return_value=comm_method),
        patch.dict(fused_moe_310_module._MoECommMethods, clear=False),
    ):
        AscendMoERunner310.__init__(
            runner,
            "model.layers.0.mlp",
            moe_config,
            MagicMock(),
            routed_experts,
        )

        assert routed_experts.quant_method is quant_method
        assert runner.quant_type == QuantType.NONE
        assert runner.multistream_overlap_shared_expert is False
        assert fused_moe_310_module._MoECommMethods[MoECommType.ALLGATHER] is comm_method
        parent_init.assert_called_once()


def test_process_weights_after_loading_310_uses_version_specific_layout(
    monkeypatch,
):
    method = AscendUnquantizedFusedMoEMethod310.__new__(AscendUnquantizedFusedMoEMethod310)
    method._maybe_pad_weight = MagicMock(side_effect=lambda weight: weight)
    layer = _build_weight_layer()
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()

    monkeypatch.setattr(fused_moe_310_module, "maybe_trans_nz", lambda weight: weight)
    monkeypatch.setattr(
        fused_moe_310_module.UnquantizedFusedMoEMethod,
        "process_weights_after_loading",
        lambda self, layer: None,
    )

    method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.w13_weight, original_w13.transpose(1, 2))
    torch.testing.assert_close(layer.w2_weight, original_w2.transpose(1, 2))
    assert layer.w13_weight.is_contiguous() is True
    assert layer.w2_weight.is_contiguous() is True


class _Projection(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0 + 1.0, None


class _Gate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros((*hidden_states.shape[:-1], 1), dtype=hidden_states.dtype), None


@pytest.mark.parametrize("with_gate", [False, True])
def test_shared_experts_part2_310_applies_optional_gate(with_gate):
    runner = _build_runner()
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
def test_shared_forward_impl_310_returns_current_runner_contract(monkeypatch, has_shared_experts):
    runner = _build_runner()
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

    monkeypatch.setattr(AscendMoERunner310, "is_internal_router", property(lambda _: False))
    monkeypatch.setattr(fused_moe_310_module.torch.npu, "current_stream", lambda: current_stream)

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
