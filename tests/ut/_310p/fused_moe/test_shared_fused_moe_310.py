# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vllm_ascend._310p.fused_moe import fused_moe as fused_moe_310_module
from vllm_ascend._310p.fused_moe.fused_moe import (
    AscendMoERunner310,
    AscendRoutedExperts310,
    AscendUnquantizedFusedMoEMethod310,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts, FusedMoEEvents


def _build_runner() -> AscendMoERunner310:
    runner = AscendMoERunner310.__new__(AscendMoERunner310)
    nn.Module.__init__(runner)
    runner.routed_input_transform = None
    runner.routed_output_transform = None
    return runner


def _build_weight_layer():
    return SimpleNamespace(
        w13_weight=nn.Parameter(torch.randn(2, 3, 4)),
        w2_weight=nn.Parameter(torch.randn(2, 4, 3)),
    )


def test_routed_experts_310_uses_parent_unquantized_method_during_init(monkeypatch):
    moe_config = MagicMock()
    parent_method = MagicMock()
    specialized_method = object()

    def parent_init(layer, *args, **kwargs):
        nn.Module.__init__(layer)
        layer.quant_config = None
        layer.moe_config = moe_config
        layer.quant_method = parent_method
        layer.custom_routing_function = None
        layer.e_score_correction_bias = None

    monkeypatch.setattr(fused_moe_310_module.AscendRoutedExperts, "__init__", parent_init)
    specialized_method_factory = MagicMock(return_value=specialized_method)
    monkeypatch.setattr(
        fused_moe_310_module,
        "AscendUnquantizedFusedMoEMethod310",
        specialized_method_factory,
    )

    routed_experts = AscendRoutedExperts310(tid2eid="tid2eid", n_shared_experts=3)

    assert routed_experts.quant_method is specialized_method
    specialized_method_factory.assert_called_once_with(moe_config)


@pytest.mark.parametrize("quant_config", [None, object()])
def test_routed_experts_310_replaces_only_unquantized_method_after_parent_init(monkeypatch, quant_config):
    moe_config = MagicMock()
    parent_method = object()
    specialized_method = object()
    init_kwargs = {}

    def parent_init(layer, *args, **kwargs):
        nn.Module.__init__(layer)
        init_kwargs.update(kwargs)
        layer.quant_config = quant_config
        layer.moe_config = moe_config
        layer.quant_method = parent_method
        layer.custom_routing_function = None
        layer.e_score_correction_bias = None

    specialized_method_factory = MagicMock(return_value=specialized_method)
    monkeypatch.setattr(fused_moe_310_module.AscendRoutedExperts, "__init__", parent_init)
    monkeypatch.setattr(
        fused_moe_310_module,
        "AscendUnquantizedFusedMoEMethod310",
        specialized_method_factory,
    )

    routed_experts = AscendRoutedExperts310(tid2eid="tid2eid", n_shared_experts=3)

    assert init_kwargs["tid2eid"] == "tid2eid"
    assert init_kwargs["n_shared_experts"] == 3
    if quant_config is None:
        assert routed_experts.quant_method is specialized_method
        specialized_method_factory.assert_called_once_with(moe_config)
    else:
        assert routed_experts.quant_method is parent_method
        specialized_method_factory.assert_not_called()


def test_runner_310_installs_specialized_comm():
    runner = _build_runner()
    moe_config = MagicMock()
    runner.moe_config = moe_config
    runner.gate = None
    routed_experts = SimpleNamespace(quant_config=None, quant_method=None)
    runner.ascend_shared_experts = SimpleNamespace(multistream_overlap=True)
    upstream_forward_entry = object()
    runner._select_forward = MagicMock(return_value=upstream_forward_entry)
    comm_method = object()

    with (
        patch.object(AscendMoERunner, "__init__", return_value=None) as parent_init,
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

        assert routed_experts.quant_method is None
        assert runner.ascend_shared_experts.multistream_overlap is False
        assert runner._forward_entry is upstream_forward_entry
        runner._select_forward.assert_called_once_with()
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


def test_unquantized_apply_310_uses_preselected_experts():
    method = AscendUnquantizedFusedMoEMethod310.__new__(AscendUnquantizedFusedMoEMethod310)
    expert_map = torch.tensor([0, 1], dtype=torch.int32)
    layer = SimpleNamespace(
        w13_weight=torch.randn(2, 4, 6),
        w2_weight=torch.randn(2, 6, 4),
        ascend_expert_map=expert_map,
        global_redundant_expert_num=0,
        ascend_mc2_mask=None,
        apply_router_weight_on_input=True,
        log2phy=None,
        ascend_pertoken_scale=None,
        activation="silu",
        swiglu_limit=0.0,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    hidden_states = torch.randn(3, 6)
    topk_weights = torch.rand(3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    expected_output = object()
    comm_method = MagicMock()
    comm_method.fused_experts.return_value = expected_output

    with patch.object(
        fused_moe_310_module,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_method=comm_method),
    ):
        output = method.apply(
            layer=layer,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

    assert output is expected_output
    fused_experts_input = comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
    assert fused_experts_input.topk_weights is topk_weights
    assert fused_experts_input.topk_ids is topk_ids
    assert fused_experts_input.routing.expert_map is expert_map
    assert fused_experts_input.routing.apply_router_weight_on_input is True


class _Projection(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0 + 1.0, None


class _Gate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros((*hidden_states.shape[:-1], 1), dtype=hidden_states.dtype), None


@pytest.mark.parametrize("with_gate", [False, True])
def test_shared_experts_part2_310_applies_optional_gate(with_gate):
    shared_experts_layer = SimpleNamespace(
        act_fn=nn.Identity(),
        down_proj=_Projection(),
        expert_gate=_Gate() if with_gate else None,
    )
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.layer = shared_experts_layer
    hidden_states = torch.randn(3, 4)
    shared_gate_up = torch.randn(3, 4)

    output = shared_experts.part2(hidden_states, shared_gate_up)

    expected = shared_gate_up * 2.0 + 1.0
    if with_gate:
        expected = expected * 0.5
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_forward_impl_310_returns_current_runner_contract(monkeypatch, has_shared_experts):
    runner = _build_runner()
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)
    routed_out = torch.randn(2, 4)
    shared_out = torch.randn(2, 4)
    ascend_shared_experts = SimpleNamespace(
        multistream_overlap=False,
        local_input_from_gathered=MagicMock(return_value=hidden_states),
        forward=MagicMock(return_value=shared_out),
    )
    routed_events = FusedMoEEvents(
        before_routed_experts=None,
        after_routed_experts=None,
        before_dispatch=None,
        before_gmm2=None,
        before_combine=None,
    )
    runner.routed_experts = SimpleNamespace(
        forward_impl=MagicMock(return_value=(routed_out, routed_events) if has_shared_experts else routed_out)
    )
    runner.ascend_shared_experts = ascend_shared_experts if has_shared_experts else None
    runner._sequence_parallel_context = MagicMock(return_value=nullcontext())
    current_stream = MagicMock()

    monkeypatch.setattr(AscendMoERunner310, "is_internal_router", property(lambda _: False))
    monkeypatch.setattr(fused_moe_310_module.torch.npu, "current_stream", lambda: current_stream)

    result = runner._forward_impl(hidden_states, router_logits, shared_experts_input=None)

    if has_shared_experts:
        runner.routed_experts.forward_impl.assert_called_once_with(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=None,
        )
        assert result[0] is shared_out
        assert result[1] is routed_out
        ascend_shared_experts.forward.assert_called_once()
    else:
        runner.routed_experts.forward_impl.assert_called_once_with(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=None,
        )
        assert result is routed_out
        ascend_shared_experts.forward.assert_not_called()
