# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe import routed_experts as routed_experts_module
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.routed_experts import (
    AscendRoutedExperts,
    AscendUnquantizedFusedMoEMethod,
    FusedMoEResult,
    make_eplb_placement_config,
    use_multistage_eplb_load,
)
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts
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
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        activation="gelu",
        apply_router_weight_on_input=True,
        ascend_expert_map=None,
        global_redundant_expert_num=0,
        log2phy=None,
        ascend_pertoken_scale=None,
        ascend_mc2_mask=None,
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


def test_ascend_routed_experts_uses_parent_unquantized_method_during_init(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.tid2eid = object()
    moe_config = MagicMock()
    parent_method = object()
    parent_get_quant_method = MagicMock(return_value=parent_method)
    monkeypatch.setattr(
        routed_experts_module.RoutedExperts,
        "_get_quant_method",
        parent_get_quant_method,
    )

    method = routed_experts._get_quant_method("model.layers.0.mlp", None, moe_config)

    assert method is parent_method
    parent_get_quant_method.assert_called_once_with("model.layers.0.mlp", None, moe_config)


@pytest.mark.parametrize("quant_config", [None, object()])
def test_ascend_routed_experts_replaces_only_unquantized_method_after_parent_init(monkeypatch, quant_config):
    moe_config = MagicMock()
    parent_method = object()
    ascend_method = object()
    init_methods = []

    def parent_init(layer, *args, **kwargs):
        nn.Module.__init__(layer)
        layer.quant_config = quant_config
        layer.moe_config = moe_config
        layer.quant_method = parent_method
        layer.custom_routing_function = None
        layer.e_score_correction_bias = None
        init_methods.append(layer.quant_method)

    ascend_method_factory = MagicMock(return_value=ascend_method)
    monkeypatch.setattr(routed_experts_module.RoutedExperts, "__init__", parent_init)
    monkeypatch.setattr(
        routed_experts_module,
        "AscendUnquantizedFusedMoEMethod",
        ascend_method_factory,
    )
    monkeypatch.setattr(AscendRoutedExperts, "init_eplb", lambda self, n_shared_experts: None)
    monkeypatch.setattr(
        routed_experts_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            ascend_compilation_config=SimpleNamespace(enable_static_kernel=False),
            enable_shared_expert_dp=False,
        ),
    )
    monkeypatch.setattr(
        routed_experts_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            use_v2_model_runner=False,
            model_config=SimpleNamespace(is_deepseek_mla=False),
        ),
    )

    routed_experts = AscendRoutedExperts(tid2eid="tid2eid")

    assert init_methods == [parent_method]
    if quant_config is None:
        assert routed_experts.quant_method is ascend_method
        ascend_method_factory.assert_called_once_with(moe_config, tid2eid="tid2eid")
    else:
        assert routed_experts.quant_method is parent_method
        ascend_method_factory.assert_not_called()


def test_ascend_routed_experts_passes_tid2eid_to_quant_config():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.tid2eid = object()
    quant_config = MagicMock()
    moe_config = MagicMock()

    method = routed_experts._get_quant_method("model.layers.0.mlp", quant_config, moe_config)

    assert method is quant_config.get_quant_method.return_value
    quant_config.get_quant_method.assert_called_once_with(
        routed_experts,
        "model.layers.0.mlp",
        tid2eid=routed_experts.tid2eid,
    )


def test_ascend_routed_experts_accepts_tid2eid_parameter_before_module_init(monkeypatch):
    tid2eid = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    parent_init = MagicMock(side_effect=RuntimeError("stop after parent init"))
    monkeypatch.setattr(routed_experts_module.RoutedExperts, "__init__", parent_init)

    with pytest.raises(RuntimeError, match="stop after parent init"):
        AscendRoutedExperts(tid2eid=tid2eid)

    parent_init.assert_called_once()


@pytest.mark.parametrize(("use_v2_model_runner", "legacy_init_calls"), [(True, 0), (False, 1)])
def test_ascend_routed_experts_initializes_only_matching_eplb_path(
    monkeypatch,
    use_v2_model_runner,
    legacy_init_calls,
):
    def parent_init(instance, *args, **kwargs):
        nn.Module.__init__(instance)
        instance.quant_config = object()
        instance.custom_routing_function = None
        instance.e_score_correction_bias = None

    init_eplb = MagicMock()
    monkeypatch.setattr(routed_experts_module.RoutedExperts, "__init__", parent_init)
    monkeypatch.setattr(AscendRoutedExperts, "init_eplb", init_eplb)
    monkeypatch.setattr(
        routed_experts_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            ascend_compilation_config=SimpleNamespace(enable_static_kernel=False),
            enable_shared_expert_dp=False,
        ),
    )
    monkeypatch.setattr(
        routed_experts_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            use_v2_model_runner=use_v2_model_runner,
            model_config=SimpleNamespace(is_deepseek_mla=False),
        ),
    )

    routed_experts = AscendRoutedExperts(n_shared_experts=2)

    assert routed_experts._use_v2_model_runner is use_v2_model_runner
    assert init_eplb.call_count == legacy_init_calls


def test_process_weights_after_loading_uses_version_specific_layout(
    monkeypatch,
):
    method = _build_unquantized_method()
    layer = _build_weight_layer()
    w13_loader = MagicMock()
    w2_loader = MagicMock()
    layer.w13_weight.weight_loader = w13_loader
    layer.w2_weight.weight_loader = w2_loader
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()
    ascend_config = SimpleNamespace(enable_fused_mc2=False)

    monkeypatch.setattr(routed_experts_module, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(routed_experts_module, "maybe_trans_nz", lambda weight: weight)
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
    assert layer.w13_weight.weight_loader is w13_loader
    assert layer.w2_weight.weight_loader is w2_loader


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
        routed_experts_module,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=moe_comm_type, moe_comm_method=moe_comm_method),
    )

    result = method.apply(
        layer=layer,
        x=hidden_states,
        topk_weights=topk_weights.to(hidden_states.dtype),
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )

    assert result is routed_out
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


def test_routed_experts_select_experts_validates_router_logits(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)
    routed_experts.router = SimpleNamespace(
        _select_experts=MagicMock(
            return_value=(
                torch.randn(2, 2, dtype=torch.float32),
                torch.randint(0, 3, (2, 2), dtype=torch.int64),
            )
        )
    )
    routed_experts.moe_config = SimpleNamespace(num_experts=4)
    routed_experts.global_redundant_expert_num = 0
    routed_experts.n_shared_experts = 0
    routed_experts.zero_expert_num = 0
    routed_experts.zero_expert_type = None
    monkeypatch.setattr(routed_experts_module, "get_forward_context", lambda: SimpleNamespace(input_ids=None))
    monkeypatch.setattr(routed_experts_module, "get_current_vllm_config", lambda: None)

    with pytest.raises(AssertionError, match="router_logits.shape\\[1\\]=3"):
        routed_experts._select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            enable_force_load_balance=False,
        )


@pytest.mark.parametrize("return_with_event", [False, True])
@pytest.mark.parametrize("v2_eplb", [False, True])
def test_routed_experts_forward_impl_runs_current_flow(monkeypatch, return_with_event, v2_eplb):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    hidden_states = torch.randn(2, 4)
    prepared_hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)
    prepared_router_logits = torch.randn(2, 3)
    routed_out = torch.randn(2, 4)
    finalized = torch.randn(2, 4)
    expert_load = torch.zeros(4, dtype=torch.int32)
    quant_method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    quant_method.apply = MagicMock(
        return_value=SimpleNamespace(
            routed_out=routed_out,
            expert_tokens=torch.tensor([3, 5]) if v2_eplb else None,
            group_list_type=1,
            before_dispatch_evt=None,
            before_gmm2_evt=None,
            before_combine_evt=None,
            swiglu_limit=0.0,
        )
    )
    routed_experts.enable_npugraph_ex_static_kernel = False
    routed_experts.enable_shared_expert_dp = False
    object.__setattr__(routed_experts, "quant_method", quant_method)
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    routed_experts.router = SimpleNamespace(
        _select_experts=MagicMock(return_value=(topk_weights, topk_ids)),
        eplb_state=SimpleNamespace(expert_load_view=expert_load) if v2_eplb else None,
    )
    routed_experts.top_k = 2
    routed_experts.renormalize = True
    routed_experts.use_grouped_topk = False
    routed_experts.moe_config = SimpleNamespace(num_experts=3, ep_rank=1, ep_size=2)
    routed_experts.ascend_expert_map = None
    routed_experts.topk_group = None
    routed_experts.num_expert_group = None
    routed_experts.custom_routing_function = None
    routed_experts.scoring_func = "softmax"
    routed_experts.routed_scaling_factor = 1.0
    routed_experts.e_score_correction_bias = None
    routed_experts.activation = "gelu"
    routed_experts.apply_router_weight_on_input = True
    routed_experts.log2phy = None
    routed_experts.global_redundant_expert_num = 0
    routed_experts.n_shared_experts = 0
    routed_experts._use_v2_model_runner = v2_eplb
    routed_experts.dynamic_eplb = False
    routed_experts.return_with_event = return_with_event
    moe_comm_method = MagicMock()
    moe_comm_method.prepare.return_value = SimpleNamespace(
        hidden_states=prepared_hidden_states,
        router_logits=prepared_router_logits,
        mc2_mask=None,
        padded_hidden_states_shape=torch.Size([2, 4]),
        pertoken_scale=None,
    )
    moe_comm_method.finalize.return_value = finalized
    monkeypatch.setattr(
        routed_experts_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            in_profile_run=False,
            moe_comm_method=moe_comm_method,
            flash_comm_v1_enabled=False,
            eplb_heat_collection_status=False,
        ),
    )
    monkeypatch.setattr(routed_experts_module, "get_forward_context", lambda: SimpleNamespace(all_moe_layers=None))
    monkeypatch.setattr(routed_experts_module, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(routed_experts_module, "get_moe_num_logical_experts", lambda *args, **kwargs: 3)

    result = routed_experts.forward_impl(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )

    if return_with_event:
        assert isinstance(result, FusedMoEResult)
        assert result.routed_out is finalized
    else:
        assert result is finalized
    moe_comm_method.prepare.assert_called_once_with(
        hidden_states=hidden_states,
        router_logits=router_logits,
        replace_allreduce=False,
        enable_shared_expert_dp=False,
        quant_type=QuantType.NONE,
    )
    quant_method.apply.assert_called_once()
    assert quant_method.apply.call_args.kwargs["layer"] is routed_experts
    assert quant_method.apply.call_args.kwargs["x"] is prepared_hidden_states
    torch.testing.assert_close(
        quant_method.apply.call_args.kwargs["topk_weights"],
        topk_weights.to(hidden_states.dtype),
    )
    assert torch.equal(quant_method.apply.call_args.kwargs["topk_ids"], topk_ids)
    routed_experts.router._select_experts.assert_called_once_with(
        hidden_states=prepared_hidden_states,
        router_logits=prepared_router_logits,
        input_ids=None,
    )
    expected_load = torch.tensor([0, 0, 3, 5], dtype=torch.int32) if v2_eplb else torch.zeros_like(expert_load)
    torch.testing.assert_close(expert_load, expected_load)
    moe_comm_method.finalize.assert_called_once_with(
        hidden_states=routed_out,
        reduce_results=False,
        padded_hidden_states_shape=torch.Size([2, 4]),
    )


class _Projection(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2.0 + 1.0, None


class _Gate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros((*hidden_states.shape[:-1], 1), dtype=hidden_states.dtype), None


@pytest.mark.parametrize("with_gate", [False, True])
def test_shared_experts_part2_applies_optional_gate(with_gate):
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
def test_forward_impl_returns_current_runner_contract(monkeypatch, has_shared_experts):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)
    routed_out = torch.randn(2, 4)
    shared_out = torch.randn(2, 4)
    ascend_shared_experts = SimpleNamespace(forward=MagicMock(return_value=shared_out))
    routed_result = SimpleNamespace(
        routed_out=routed_out,
        before_dispatch_evt=None,
        before_gmm2_evt=None,
        before_combine_evt=None,
        swiglu_limit=0.0,
    )
    runner.routed_experts = SimpleNamespace(
        forward_impl=MagicMock(return_value=routed_result if has_shared_experts else routed_out)
    )
    runner.ascend_shared_experts = ascend_shared_experts if has_shared_experts else None
    runner._sequence_parallel_context = MagicMock(return_value=nullcontext())
    current_stream = MagicMock()

    monkeypatch.setattr(AscendMoERunner, "is_internal_router", property(lambda _: False))
    monkeypatch.setattr(fused_moe_module.torch.npu, "current_stream", lambda: current_stream)

    result = runner._forward_impl(hidden_states, router_logits, shared_experts_input=None)

    if has_shared_experts:
        runner.routed_experts.forward_impl.assert_called_once_with(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        assert result[0] is shared_out
        assert result[1] is routed_out
        ascend_shared_experts.forward.assert_called_once()
    else:
        runner.routed_experts.forward_impl.assert_called_once_with(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        assert result is routed_out
        ascend_shared_experts.forward.assert_not_called()
