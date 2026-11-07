# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router import router_factory
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import CustomRoutingRouter
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import FusedTopKBiasRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
from vllm.model_executor.models.gpt_oss import GptOssRoutedExperts
from vllm.model_executor.models.transformers.moe import TransformersMoERunner, TransformersMoEState

from vllm_ascend import utils
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.ops.fused_moe import fused_moe, routed_experts
from vllm_ascend.ops.fused_moe.pluggable import AscendGptOssRoutedExperts, AscendTransformersMoERunner
from vllm_ascend.ops.fused_moe.router import eplb_router, fused_topk_router
from vllm_ascend.ops.fused_moe.router.pluggable import AscendBiasRouter, AscendGroupedRouter, AscendTopKRouter


@pytest.fixture
def moe_config(monkeypatch):
    saved_registry = op_registry_oot.copy()
    op_registry_oot.clear()
    monkeypatch.setattr(utils, "_ASCEND_CUSTOMOP_IS_REIGISTERED", False)
    monkeypatch.setattr(router_factory, "current_platform", SimpleNamespace(is_cpu=lambda: False))
    with patch.object(VllmConfig, "use_v2_model_runner", new_callable=PropertyMock, return_value=True):
        config = VllmConfig()
        with set_current_vllm_config(config):
            utils.register_ascend_customop(config)
            yield config
    op_registry_oot.clear()
    op_registry_oot.update(saved_registry)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, AscendTopKRouter),
        ({"use_grouped_topk": True, "num_expert_group": 2, "topk_group": 1}, AscendGroupedRouter),
        ({"e_score_correction_bias": torch.zeros(8)}, AscendBiasRouter),
    ],
)
def test_upstream_factory_selects_initialized_ascend_router(moe_config, kwargs, expected):
    router = router_factory.create_fused_moe_router(top_k=2, global_num_experts=8, **kwargs)
    assert type(router) is expected
    assert router.top_k == 2
    assert router.global_num_experts == 8
    assert isinstance(router, torch.nn.Module)
    assert router.eplb_state is None
    assert router.routing_method_type == router.kernel_routing_method_type


def test_factory_binding_is_unmodified(moe_config):
    import vllm.model_executor.layers.fused_moe as package
    import vllm.model_executor.layers.fused_moe.layer as layer

    assert issubclass(FusedMoERouter, PluggableLayer)
    assert package.FusedMoEFactory is layer.FusedMoEFactory is FusedMoEFactory
    assert FusedMoEFactory.__module__ == "vllm.model_executor.layers.fused_moe.layer"


def test_hash_and_vision_arguments_share_original_objects(moe_config):
    table = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32)
    bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
    vision_bias = torch.nn.Parameter(-bias.detach(), requires_grad=False)
    router = router_factory.create_fused_moe_router(
        top_k=2,
        global_num_experts=8,
        scoring_func="sqrtsoftplus",
        hash_indices_table=table,
        e_score_correction_bias=bias,
        bias_vl=vision_bias,
        image_sentinel_lo=129257,
    )
    assert isinstance(router, FusedTopKBiasRouter)
    assert router.tid2eid is router._hash_indices_table is table
    assert router.e_score_correction_bias is bias
    assert router.bias_vl is vision_bias
    assert router.image_sentinel_lo == 129257
    assert dict(router.named_parameters())["e_score_correction_bias"] is bias


def test_bias_dtype_conversion_keeps_gate_parameter(moe_config, monkeypatch):
    bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
    router = FusedTopKBiasRouter(top_k=2, global_num_experts=8, e_score_correction_bias=bias)
    calls = []

    def gating(logits, **kwargs):
        calls.append(kwargs)
        return torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.int32), None

    monkeypatch.setattr(fused_topk_router.torch_npu, "npu_moe_gating_top_k", gating, raising=False)
    router._compute_routing(torch.zeros(2, 128), torch.zeros(2, 8, dtype=torch.bfloat16), None)
    assert router.e_score_correction_bias is bias
    assert bias.dtype == torch.float32
    torch.testing.assert_close(calls[0]["bias"], bias.to(torch.bfloat16))


@pytest.mark.parametrize("scoring_func,renormalize", [("softmax", True), ("sigmoid", False)])
def test_routing_fallback_preserves_scores(moe_config, monkeypatch, scoring_func, renormalize):
    router = FusedTopKRouter(top_k=2, global_num_experts=8, scoring_func=scoring_func, renormalize=renormalize)
    monkeypatch.setattr(router, "is_fused_supported", lambda _: False)
    logits = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 7
    weights, ids = router.select_experts(torch.zeros(3, 128), logits)
    scores = logits.softmax(-1) if scoring_func == "softmax" else logits.sigmoid()
    expected_weights, expected_ids = scores.topk(2, dim=-1)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    torch.testing.assert_close(ids.long(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)


@pytest.mark.parametrize("use_allgather", [False, True])
def test_eplb_state_keeps_tensor_aliases_and_uses_ascend_mapping(moe_config, monkeypatch, use_allgather):
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_state.get_ep_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    state = EplbLayerState()
    state.expert_load_view = torch.zeros(8, dtype=torch.int32)
    state.logical_to_physical_map = torch.arange(8, dtype=torch.int64).reshape(8, 1)
    state.logical_replica_count = torch.ones(8, dtype=torch.int64)
    state.should_record_tensor = torch.tensor(True)
    state.num_unpadded_tokens_tensors = [torch.tensor(2)]
    router = FusedTopKRouter(top_k=2, global_num_experts=8, eplb_state=state)
    assert isinstance(router.eplb_state, AscendEplbLayerState)
    assert router.eplb_state.expert_load_view is state.expert_load_view
    assert router.eplb_state.logical_to_physical_map is state.logical_to_physical_map
    normal_table = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    allgather_table = normal_table.flip(-1)
    router.eplb_state.expert_replica_routing_table = normal_table
    router.eplb_state.allgather_expert_replica_routing_table = allgather_table
    monkeypatch.setattr(eplb_router, "use_allgather_eplb", lambda: use_allgather)
    ids = torch.tensor([[1, 3]], dtype=torch.int32)
    expected = torch.tensor([[4, 6]], dtype=torch.int32)
    with patch.object(torch.ops.vllm, "ascend_eplb_map_to_physical", return_value=expected) as mapping:
        assert router._apply_eplb_mapping(ids) is expected
    mapping.assert_called_once_with(ids, allgather_table if use_allgather else normal_table)


def test_eplb_missing_table_fails_before_routing(moe_config):
    state = AscendEplbLayerState()
    router = FusedTopKRouter(top_k=2, global_num_experts=8, eplb_state=state)
    assert router.eplb_state is state
    # State validation must still reject an uninitialized layer.
    with pytest.raises((ValueError, RuntimeError)):
        router._apply_eplb_mapping(torch.zeros(1, 2, dtype=torch.int32))


def test_explicit_custom_router_keeps_user_function_and_eplb(moe_config):
    function = lambda **kwargs: (torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32))
    state = AscendEplbLayerState()
    router = CustomRoutingRouter(top_k=2, global_num_experts=8, custom_routing_function=function, eplb_state=state)
    assert router.custom_routing_function is function
    assert router.eplb_state is state
    assert router._apply_eplb_mapping.__func__ is eplb_router.AscendEplbRouter._apply_eplb_mapping


@pytest.fixture
def factory_runtime(moe_config, monkeypatch, tmp_path):
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(1, 0, f"file://{tmp_path}/distributed", 0, "gloo")
    initialize_model_parallel(1, 1, backend="gloo")
    ascend_config = SimpleNamespace(
        mix_placement=False,
        ascend_compilation_config=SimpleNamespace(enable_static_kernel=False),
        eplb_config=SimpleNamespace(dynamic_eplb=False),
    )
    monkeypatch.setattr(routed_experts, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(fused_moe, "setup_moe_comm_method", lambda _: None)
    monkeypatch.setattr(fused_moe, "get_moe_comm_method", lambda *_: None)
    try:
        yield
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _make_moe(**kwargs):
    return FusedMoEFactory(
        num_experts=8,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        params_dtype=torch.bfloat16,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        prefix="migration",
        **kwargs,
    )


@pytest.mark.parametrize("enable_eplb", [False, True])
def test_default_factory_constructs_all_ascend_components(factory_runtime, enable_eplb):
    runner = _make_moe(enable_eplb=enable_eplb, num_redundant_experts=2 if enable_eplb else 0)
    assert isinstance(runner, fused_moe.AscendMoERunner)
    assert isinstance(runner.routed_experts, routed_experts.AscendRoutedExperts)
    assert type(runner.router) is AscendTopKRouter
    assert (runner.eplb_state is not None) == enable_eplb
    assert runner.routed_experts.w13_weight.shape[0] == (10 if enable_eplb else 8)


def test_model_components_keep_loader_state_and_explicit_router(factory_runtime):
    state = TransformersMoEState()
    router = FusedTopKRouter(top_k=2, global_num_experts=8)
    runner = _make_moe(
        routed_experts_cls=GptOssRoutedExperts,
        runner_cls=TransformersMoERunner,
        runner_args={"moe_state": state},
        router=router,
    )
    assert type(runner) is AscendTransformersMoERunner
    assert type(runner.routed_experts) is AscendGptOssRoutedExperts
    assert runner.router is router
    assert runner._moe_state is state
    weight = runner.routed_experts.w13_weight
    checkpoint = torch.arange(weight[0].numel(), dtype=torch.float32).to(weight.dtype)
    checkpoint = checkpoint.reshape(weight.shape[2], weight.shape[1])
    runner.routed_experts.weight_loader(weight, checkpoint, "w13_weight", "gpt_oss_w13", 0)
    torch.testing.assert_close(weight[0], checkpoint.t(), rtol=0, atol=0)


def test_factory_preserves_hash_and_vision_routing_objects(factory_runtime):
    table = torch.tensor([[0, 3], [2, 1]], dtype=torch.int32)
    bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32), requires_grad=False)
    vision_bias = torch.nn.Parameter(-bias.detach(), requires_grad=False)
    runner = _make_moe(
        hash_indices_table=table,
        scoring_func="sqrtsoftplus",
        e_score_correction_bias=bias,
        bias_vl=vision_bias,
        image_sentinel_lo=129257,
    )
    assert runner.router.tid2eid is table
    assert runner.router.e_score_correction_bias is bias
    assert runner.router.bias_vl is vision_bias
    assert runner.router.image_sentinel_lo == 129257
    assert runner.routed_experts.hash_indices_table is table


def test_dbrx_component_keeps_constructor_state_and_stacked_loader(factory_runtime):
    from transformers import DbrxConfig
    from vllm.model_executor.models.dbrx import DbrxExperts

    config = DbrxConfig(d_model=128, ffn_config={"ffn_hidden_size": 128})
    runner = _make_moe(routed_experts_cls=DbrxExperts, routed_experts_args={"config": config})
    experts = runner.routed_experts
    assert isinstance(experts, DbrxExperts)
    assert isinstance(experts, routed_experts.AscendRoutedExperts)
    assert experts.config is config
    checkpoint = torch.arange(8 * 128 * 128, dtype=torch.float32).to(torch.bfloat16)
    experts.weight_loader(experts.w13_weight, checkpoint, "experts.w1", "weight")
    torch.testing.assert_close(experts.w13_weight[:, :128], checkpoint.reshape(8, 128, 128), rtol=0, atol=0)


def test_unregistered_model_component_keeps_its_own_contract(factory_runtime):
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    class ModelExperts(RoutedExperts):
        def weight_loader(self, *args, **kwargs):
            return "model-loader"

    class ModelRunner(MoERunner):
        pass

    runner = _make_moe(routed_experts_cls=ModelExperts, runner_cls=ModelRunner)
    assert type(runner) is ModelRunner
    assert type(runner.routed_experts) is ModelExperts
    assert runner.routed_experts.weight_loader() == "model-loader"


@pytest.mark.parametrize("scheme", ["plain", "grouped", "bias"])
def test_310_router_replacements_initialize_the_selected_contract(moe_config, scheme):
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter

    from vllm_ascend._310p.fused_moe.pluggable_router import (
        AscendBiasRouter310,
        AscendGroupedRouter310,
        AscendTopKRouter310,
    )

    selected, replacement, kwargs = {
        "plain": (FusedTopKRouter, AscendTopKRouter310, {}),
        "grouped": (GroupedTopKRouter, AscendGroupedRouter310, {"num_expert_group": 2, "topk_group": 1}),
        "bias": (FusedTopKBiasRouter, AscendBiasRouter310, {"e_score_correction_bias": torch.zeros(8)}),
    }[scheme]
    op_registry_oot.pop(selected.__name__)
    selected.register_oot(replacement)
    router = selected(top_k=2, global_num_experts=8, **kwargs)
    assert type(router) is replacement
    assert isinstance(router, selected)
    assert router.top_k == 2
    assert router.global_num_experts == 8
    assert router.use_grouped_topk == (scheme == "grouped")
