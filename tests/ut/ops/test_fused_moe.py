# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SituAndMul

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe import routed_experts as routed_experts_module
from vllm_ascend.ops.fused_moe import shared_experts as shared_experts_module
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.routed_experts import (
    AscendRoutedExperts,
    AscendUnquantizedFusedMoEMethod,
    make_eplb_placement_config,
    use_multistage_eplb_load,
)
from vllm_ascend.ops.fused_moe.router import fused_topk_router as fused_topk_router_module
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.ops.fused_moe.shared_experts import (
    AscendSharedExperts,
    FusedMoEEvents,
    SharedExpertParallelMode,
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


def test_process_weights_after_loading_splits_lists_for_dynamic_eplb(monkeypatch):
    method = _build_unquantized_method(dynamic_eplb=True)
    layer = _build_weight_layer()
    num_experts = layer.w13_weight.shape[0]
    ascend_config = SimpleNamespace(enable_fused_mc2=1)

    monkeypatch.setattr(routed_experts_module, "is_mega_moe_supported", lambda: False)
    monkeypatch.setattr(routed_experts_module, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(routed_experts_module.torch_npu, "npu_format_cast", lambda weight, _: weight)
    monkeypatch.setattr(routed_experts_module.torch.npu, "empty_cache", lambda: None)
    upstream_method_base = AscendUnquantizedFusedMoEMethod.__mro__[2]
    monkeypatch.setattr(
        upstream_method_base,
        "process_weights_after_loading",
        lambda self, layer: None,
        raising=False,
    )

    method.process_weights_after_loading(layer)

    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")
    assert len(layer.w13_weight_list) == num_experts
    assert len(layer.w2_weight_list) == num_experts


def test_update_expert_map_updates_routed_experts_and_manager():
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    nn.Module.__init__(routed_experts)
    manager = SimpleNamespace(_expert_map=torch.tensor([0, 1, 2, -1], dtype=torch.int32))
    routed_experts.expert_map_manager = manager
    routed_experts.ascend_expert_map = torch.tensor([0, -1, 1], dtype=torch.int32)

    new_map = torch.tensor([0, 1, -1], dtype=torch.int32)
    routed_experts.update_expert_map(new_map)

    assert routed_experts.ascend_expert_map is new_map
    assert manager._expert_map is new_map


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
    "moe_comm_type, is_sequence_parallel, expected",
    [
        (MoECommType.ALLTOALL, False, True),
        (MoECommType.MC2, False, True),
        (MoECommType.FUSED_MC2, False, True),
        (MoECommType.ALLGATHER, False, False),
        (MoECommType.ALLGATHER, True, True),
    ],
)
def test_runner_reduction_contract(monkeypatch, moe_comm_type, is_sequence_parallel, expected):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.moe_config = SimpleNamespace(is_sequence_parallel=is_sequence_parallel)
    runner.ascend_shared_experts = SimpleNamespace(
        parallel_mode=MagicMock(return_value=SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP)
    )
    shared_output = object()
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(moe_comm_type=moe_comm_type),
    )

    assert runner.use_dp_chunking is False
    assert runner._fused_output_is_reduced is expected
    assert runner._maybe_reduce_shared_expert_output(shared_output) is shared_output


@pytest.mark.parametrize(
    ("is_sequence_parallel", "output_is_reduced", "should_reduce"),
    [
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (True, True, False),
    ],
)
def test_final_output_never_all_reduces_sequence_shards(
    monkeypatch,
    is_sequence_parallel,
    output_is_reduced,
    should_reduce,
):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.moe_config = SimpleNamespace(is_sequence_parallel=is_sequence_parallel)
    states = torch.ones(2, 4)
    reduced_states = states + 1
    all_reduce = MagicMock(return_value=reduced_states)
    monkeypatch.setattr(
        fused_moe_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )

    result = runner._maybe_reduce_final_output(
        states,
        trunc_size=None,
        output_is_reduced=output_is_reduced,
    )

    if should_reduce:
        assert result is reduced_states
        all_reduce.assert_called_once_with(states)
    else:
        assert result is states
        all_reduce.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "fused_output_is_reduced", "reduce_shared"),
    [
        (SharedExpertParallelMode.TENSOR_PARALLEL, False, False),
        (SharedExpertParallelMode.TENSOR_PARALLEL, True, True),
        (SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY, True, False),
        (SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY, True, False),
        (SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP, True, False),
    ],
)
def test_shared_output_reduction_depends_on_weight_layout(
    monkeypatch,
    mode,
    fused_output_is_reduced,
    reduce_shared,
):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.ascend_shared_experts = SimpleNamespace(parallel_mode=MagicMock(return_value=mode))
    shared_output = torch.ones(2, 4)
    reduced_output = shared_output + 1
    all_reduce = MagicMock(return_value=reduced_output)
    monkeypatch.setattr(
        fused_moe_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )

    result = runner._reduce_shared_output_if_needed(
        shared_output,
        fused_output_is_reduced,
    )

    if reduce_shared:
        assert result is reduced_output
        all_reduce.assert_called_once_with(shared_output)
    else:
        assert result is shared_output
        all_reduce.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "reduce_routed"),
    [
        (SharedExpertParallelMode.TENSOR_PARALLEL, False),
        (SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY, True),
        (SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP, False),
    ],
)
def test_local_shared_expert_dp_reduces_partial_routed_output(
    monkeypatch,
    mode,
    reduce_routed,
):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    runner.ascend_shared_experts = SimpleNamespace(parallel_mode=MagicMock(return_value=mode))
    runner.routed_output_transform = None
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=False,
        tp_size=2,
        ep_size=2,
    )
    routed_output = torch.ones(2, 4)
    reduced_output = routed_output + 1
    all_reduce = MagicMock(return_value=reduced_output)
    monkeypatch.setattr(
        fused_moe_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )

    result, result_is_reduced = runner._maybe_reduce_routed_output_before_transform(
        routed_output,
        False,
    )

    if reduce_routed:
        assert result is reduced_output
        assert result_is_reduced
        all_reduce.assert_called_once_with(routed_output)
    else:
        assert result is routed_output
        assert not result_is_reduced
        all_reduce.assert_not_called()


def test_routed_experts_select_experts_validates_router_logits(monkeypatch):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)
    input_ids = torch.tensor([11, 22])
    topk_weights = torch.randn(2, 2, dtype=torch.float32)
    topk_ids = torch.randint(0, 3, (2, 2), dtype=torch.int64)
    routed_experts.router = SimpleNamespace(_select_experts=MagicMock(return_value=(topk_weights, topk_ids)))
    routed_experts.moe_config = SimpleNamespace(num_experts=4)
    routed_experts.global_redundant_expert_num = 0
    routed_experts.n_shared_experts = 0
    routed_experts.log2phy = None
    monkeypatch.setattr(routed_experts_module, "get_forward_context", lambda: SimpleNamespace(input_ids=None))
    monkeypatch.setattr(routed_experts_module, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(routed_experts_module, "get_moe_num_logical_experts", lambda *args, **kwargs: 3)

    result_weights, result_ids = routed_experts._select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        enable_force_load_balance=False,
        input_ids=input_ids,
    )

    torch.testing.assert_close(result_weights, topk_weights.to(hidden_states.dtype))
    assert torch.equal(result_ids, topk_ids)
    assert routed_experts.router._select_experts.call_args.kwargs["input_ids"] is input_ids


def _build_routing_replay_experts(router, log2phy):
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.router = router
    routed_experts.moe_config = SimpleNamespace(num_experts=4)
    routed_experts.global_redundant_expert_num = 0
    routed_experts.n_shared_experts = 0
    routed_experts.log2phy = log2phy
    return routed_experts


def test_routing_replay_captures_logical_ids_before_ascend_mapping(monkeypatch):
    router = AscendGroupedTopKRouter(
        top_k=2,
        global_num_experts=4,
        num_expert_group=None,
        topk_group=None,
    )
    capturer = MagicMock()
    router.set_capture_fn(lambda topk_ids: capturer.capture(2, topk_ids))
    log2phy = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    routed_experts = _build_routing_replay_experts(router, log2phy)
    monkeypatch.setattr(
        routed_experts_module,
        "get_moe_num_logical_experts",
        lambda *args, **kwargs: 4,
    )
    hidden_states = torch.randn(2, 4)
    router_logits = torch.tensor(
        [[0.1, 0.9, 0.2, 0.8], [0.7, 0.2, 0.6, 0.1]],
        dtype=torch.float32,
    )

    _, physical_ids = routed_experts._select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        enable_force_load_balance=False,
    )

    logical_ids = capturer.capture.call_args.args[1]
    capturer.capture.assert_called_once()
    torch.testing.assert_close(physical_ids, log2phy[logical_ids])
    assert torch.all(logical_ids < log2phy.numel())


def test_routing_replay_disabled_keeps_ascend_routing_unchanged(monkeypatch):
    router = AscendGroupedTopKRouter(
        top_k=2,
        global_num_experts=4,
        num_expert_group=None,
        topk_group=None,
    )
    log2phy = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    routed_experts = _build_routing_replay_experts(router, log2phy)
    monkeypatch.setattr(
        routed_experts_module,
        "get_moe_num_logical_experts",
        lambda *args, **kwargs: 4,
    )
    hidden_states = torch.randn(2, 4)
    router_logits = torch.tensor(
        [[0.1, 0.9, 0.2, 0.8], [0.7, 0.2, 0.6, 0.1]],
        dtype=torch.float32,
    )

    _, physical_ids = routed_experts._select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        enable_force_load_balance=False,
    )

    assert router.capture_fn is None
    expected_logical_ids = torch.tensor([[1, 3], [0, 2]], dtype=torch.int64)
    torch.testing.assert_close(physical_ids, log2phy[expected_logical_ids])


def test_hash_router_uses_explicit_input_ids(monkeypatch):
    input_ids = torch.tensor([11, 22], dtype=torch.int32)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 4)
    topk_weights = torch.randn(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    prepare_finalize = SimpleNamespace(all_gather_input_id_with_dp_group=MagicMock(side_effect=lambda value: value))
    monkeypatch.setattr(
        fused_topk_router_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.ALLGATHER,
            moe_comm_method=SimpleNamespace(prepare_finalize=prepare_finalize),
        ),
    )
    hash_op = MagicMock(return_value=(topk_weights, topk_ids, None))
    monkeypatch.setattr(
        fused_topk_router_module.torch.ops._C_ascend,
        "moe_gating_top_k_hash",
        hash_op,
        raising=False,
    )
    router = AscendFusedTopKRouter(
        top_k=2,
        global_num_experts=4,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sqrtsoftplus",
        tid2eid=torch.ones(32, 4, dtype=torch.int32),
    )

    weights, ids = router._compute_routing(
        hidden_states,
        router_logits,
        torch.int32,
        input_ids=input_ids,
    )

    assert weights is topk_weights
    assert ids is topk_ids
    torch.testing.assert_close(hash_op.call_args.kwargs["input_ids"], input_ids.to(torch.int64))
    prepare_finalize.all_gather_input_id_with_dp_group.assert_called_once()

    with pytest.raises(ValueError, match="hash MoE routing requires input_ids"):
        router._compute_routing(hidden_states, router_logits, torch.int32)


def test_hash_router_chunks_unaligned_input_ids_for_sequence_parallel(monkeypatch):
    input_ids = torch.tensor([11, 22, 33, 44], dtype=torch.int32)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 4)
    topk_weights = torch.randn(2, 2)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    pad_and_split_input_ids = MagicMock(side_effect=lambda value: value)
    monkeypatch.setattr(
        fused_topk_router_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.MC2,
            moe_comm_method=SimpleNamespace(pad_and_split_input_ids=pad_and_split_input_ids),
        ),
    )
    sequence_parallel_chunk = MagicMock(side_effect=lambda value: value[:2].clone())
    monkeypatch.setattr(fused_topk_router_module, "sequence_parallel_chunk", sequence_parallel_chunk)
    hash_op = MagicMock(return_value=(topk_weights, topk_ids, None))
    monkeypatch.setattr(
        fused_topk_router_module.torch.ops._C_ascend,
        "moe_gating_top_k_hash",
        hash_op,
        raising=False,
    )
    router = AscendFusedTopKRouter(
        top_k=2,
        global_num_experts=4,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sqrtsoftplus",
        tid2eid=torch.ones(32, 4, dtype=torch.int32),
    )

    weights, ids = router._compute_routing(
        hidden_states,
        router_logits,
        torch.int32,
        input_ids=input_ids,
    )

    assert weights is topk_weights
    assert ids is topk_ids
    torch.testing.assert_close(
        hash_op.call_args.kwargs["input_ids"],
        input_ids[:2].to(torch.int64),
    )
    pad_and_split_input_ids.assert_called_once()
    sequence_parallel_chunk.assert_called_once()


@pytest.mark.parametrize("return_with_event", [False, True])
@pytest.mark.parametrize("v2_eplb", [False, True])
@pytest.mark.parametrize("is_sequence_parallel", [False, True])
def test_routed_experts_forward_impl_runs_current_flow(monkeypatch, return_with_event, v2_eplb, is_sequence_parallel):
    """Verify routed-expert forward preserves the current dispatch flow."""
    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    hidden_states = torch.randn(2, 4)
    prepared_hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)
    prepared_router_logits = torch.randn(2, 3)
    input_ids = torch.tensor([11, 22])
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
    routed_experts.moe_config = SimpleNamespace(
        num_experts=3,
        ep_rank=1,
        ep_size=2,
        is_sequence_parallel=is_sequence_parallel,
    )
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
            eplb_heat_collection_status=False,
        ),
    )
    monkeypatch.setattr(routed_experts_module, "get_forward_context", lambda: SimpleNamespace(all_moe_layers=None))
    monkeypatch.setattr(routed_experts_module, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(routed_experts_module, "get_moe_num_logical_experts", lambda *args, **kwargs: 3)

    result = routed_experts.forward_impl(
        hidden_states=hidden_states,
        router_logits=router_logits,
        input_ids=input_ids,
    )

    if return_with_event:
        assert isinstance(result, tuple)
        routed_output, fused_moe_events = result
        assert routed_output is finalized
        assert isinstance(fused_moe_events, FusedMoEEvents)
        assert fused_moe_events.before_routed_experts is None
        assert fused_moe_events.after_routed_experts is None
        assert fused_moe_events.before_dispatch is None
        assert fused_moe_events.before_gmm2 is None
        assert fused_moe_events.before_combine is None
    else:
        assert result is finalized
    moe_comm_method.prepare.assert_called_once_with(
        hidden_states=hidden_states,
        router_logits=router_logits,
        replace_allreduce=is_sequence_parallel,
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
        input_ids=input_ids,
    )
    expected_load = torch.zeros_like(expert_load)
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


def _make_quantized_situ_shared_experts(quant_type, gate_up_proj, down_proj):
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.layer = SimpleNamespace(
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
    )
    shared_experts.multistream_overlap = False
    shared_experts.quant_type = quant_type
    with set_current_vllm_config(VllmConfig()):
        shared_experts.situ_activation = SituAndMul(beta=4.0, linear_beta=25.0)
    shared_experts.lora_context = None
    shared_experts.parallel_mode = MagicMock(
        return_value=SharedExpertParallelMode.TENSOR_PARALLEL,
    )
    return shared_experts


def _make_shared_expert_events():
    event = MagicMock()
    return FusedMoEEvents(
        before_routed_experts=event,
        after_routed_experts=event,
        before_dispatch=event,
        before_gmm2=event,
        before_combine=event,
    )


def test_w8a8_shared_situ_uses_dequant_situ_quant(monkeypatch):
    gate_up_proj = SimpleNamespace(
        weight=torch.ones(4, 4, dtype=torch.int8),
        weight_scale=torch.ones(4),
        weight_scale_fp32=torch.ones(4),
    )
    down_proj = SimpleNamespace(
        weight=torch.ones(2, 4, dtype=torch.int8),
        weight_scale=torch.ones(2),
    )
    shared_experts = _make_quantized_situ_shared_experts(QuantType.W8A8, gate_up_proj, down_proj)
    hidden_states = torch.randn(2, 4, dtype=torch.bfloat16)
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.ones(2, 4, dtype=torch.int32)
    quantized_situ = torch.ones(2, 2, dtype=torch.int8)
    situ_scale = torch.ones(2)
    expected = torch.randn(2, 2, dtype=torch.bfloat16)
    dequant_situ_quant = MagicMock(return_value=(quantized_situ, situ_scale))

    monkeypatch.setattr(shared_experts_module, "has_lora", lambda _: False)
    monkeypatch.setattr(shared_experts_module, "npu_stream_switch", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(shared_experts_module, "shared_experts_calculation_stream", MagicMock())
    monkeypatch.setattr(shared_experts_module.torch.npu, "current_stream", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        shared_experts_module.torch_npu,
        "npu_dynamic_quant",
        MagicMock(return_value=(quantized_input, input_scale)),
        raising=False,
    )
    monkeypatch.setattr(
        shared_experts_module.torch_npu,
        "npu_quant_matmul",
        MagicMock(side_effect=[gate_up_out, expected]),
    )
    monkeypatch.setattr(
        shared_experts_module.torch.ops,
        "_C_ascend",
        SimpleNamespace(dequant_situ_quant=dequant_situ_quant),
    )

    output = shared_experts.forward(hidden_states, _make_shared_expert_events())

    assert output is expected
    situ_kwargs = dequant_situ_quant.call_args.kwargs
    assert situ_kwargs["x"] is gate_up_out
    assert situ_kwargs["beta"] == 4.0
    assert situ_kwargs["linear_beta"] == 25.0


def test_kimi_k3_w4a8_shared_situ_uses_projection_grouped_matmuls(monkeypatch):
    quantized_input = torch.ones(2, 4, dtype=torch.int8)
    input_scale = torch.ones(2)
    gate_up_out = torch.ones(2, 4, dtype=torch.bfloat16)
    quantized_situ = torch.ones(2, 2, dtype=torch.int8)
    situ_scale = torch.ones(2)
    expected = torch.randn(2, 2, dtype=torch.bfloat16)
    scoped_scheme = SimpleNamespace(is_kimi_k3_shared_expert_w4a8=True)
    gate_up_proj = MagicMock(return_value=(gate_up_out, None))
    gate_up_proj.quant_method = SimpleNamespace(quant_method=scoped_scheme)
    gate_up_proj.weight_scale = torch.ones(4, dtype=torch.int64)
    gate_up_proj.weight_scale_fp32 = torch.ones(4)
    down_proj = MagicMock(return_value=(expected, None))
    down_proj.quant_method = SimpleNamespace(quant_method=scoped_scheme)
    down_proj.weight_scale = torch.ones(2, dtype=torch.int64)
    shared_experts = _make_quantized_situ_shared_experts(QuantType.W4A8, gate_up_proj, down_proj)
    dequant_situ_quant = MagicMock(return_value=(quantized_situ, situ_scale))
    npu_quant_matmul = MagicMock()

    monkeypatch.setattr(shared_experts_module, "has_lora", lambda _: False)
    monkeypatch.setattr(shared_experts_module, "npu_stream_switch", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(shared_experts_module, "shared_experts_calculation_stream", MagicMock())
    monkeypatch.setattr(shared_experts_module.torch.npu, "current_stream", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        shared_experts_module.torch_npu,
        "npu_dynamic_quant",
        MagicMock(return_value=(quantized_input, input_scale)),
        raising=False,
    )
    monkeypatch.setattr(shared_experts_module.torch_npu, "npu_quant_matmul", npu_quant_matmul)
    monkeypatch.setattr(
        shared_experts_module.torch.ops,
        "_C_ascend",
        SimpleNamespace(dequant_situ_quant=dequant_situ_quant),
    )

    output = shared_experts.forward(torch.randn(2, 4, dtype=torch.bfloat16), _make_shared_expert_events())

    assert output is expected
    gate_args = gate_up_proj.call_args.args[0]
    down_args = down_proj.call_args.args[0]
    assert gate_args[0] is quantized_input
    assert gate_args[1] is input_scale
    assert down_args[0] is quantized_situ
    assert down_args[1] is situ_scale
    situ_kwargs = dequant_situ_quant.call_args.kwargs
    assert situ_kwargs["x"] is gate_up_out
    assert situ_kwargs["weight_scale"] is None
    assert situ_kwargs["activation_scale"] is None
    npu_quant_matmul.assert_not_called()


def test_w4a8_mxfp_shared_situ_uses_situ_mx_quant(monkeypatch):
    quantized_input = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    input_scale = torch.ones(2, 1)
    gate_up_out = torch.randn(2, 4, dtype=torch.bfloat16)
    quantized_situ = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    situ_scale = torch.ones(2, 1)
    expected = torch.randn(2, 2, dtype=torch.bfloat16)
    gate_up_proj = MagicMock(return_value=(gate_up_out, None))
    gate_up_proj.weight_scale = torch.ones(1)
    down_proj = MagicMock(return_value=(expected, None))
    down_proj.weight_scale = torch.ones(1)
    shared_experts = _make_quantized_situ_shared_experts(QuantType.W4A8MXFP, gate_up_proj, down_proj)
    situ_mx_quant = MagicMock(return_value=(quantized_situ, situ_scale))

    monkeypatch.setattr(shared_experts_module, "has_lora", lambda _: False)
    monkeypatch.setattr(shared_experts_module, "npu_stream_switch", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(shared_experts_module, "shared_experts_calculation_stream", MagicMock())
    monkeypatch.setattr(shared_experts_module.torch.npu, "current_stream", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        shared_experts_module.torch_npu,
        "npu_dynamic_mx_quant",
        MagicMock(return_value=(quantized_input, input_scale)),
        raising=False,
    )
    monkeypatch.setattr(
        shared_experts_module.torch.ops,
        "_C_ascend",
        SimpleNamespace(situ_mx_quant=situ_mx_quant),
    )

    output = shared_experts.forward(torch.randn(2, 4, dtype=torch.bfloat16), _make_shared_expert_events())

    assert output is expected
    situ_kwargs = situ_mx_quant.call_args.kwargs
    assert situ_kwargs["x"] is gate_up_out
    assert situ_kwargs["beta"] == 4.0
    assert situ_kwargs["linear_beta"] == 25.0
    assert situ_kwargs["dst_type"] == shared_experts_module.SITU_MX_DST_TYPE_E4M3FN


@pytest.mark.parametrize(
    (
        "weights_replicated",
        "native_sequence_parallel",
        "expected_mode",
    ),
    [
        (False, False, SharedExpertParallelMode.TENSOR_PARALLEL),
        (True, False, SharedExpertParallelMode.SHARED_EXPERT_DATA_PARALLEL_ONLY),
        # Decoupled: SP no longer implies weight replication.
        # SP+DP -> SEQUENCE_PARALLEL_SEDP (replicated weights, compute on the shard).
        # SP-only -> SEQUENCE_PARALLEL_ONLY (TP-sharded weights, gather then TP).
        (True, True, SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP),
        (False, True, SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY),
    ],
)
def test_shared_expert_parallel_mode_matches_activation_and_weight_layout(
    weights_replicated,
    native_sequence_parallel,
    expected_mode,
):
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.weights_replicated = weights_replicated
    shared_experts.moe_config = SimpleNamespace(
        is_sequence_parallel=native_sequence_parallel,
        tp_size=1,
        tp_rank=0,
        ep_size=2,
        tp_group=SimpleNamespace(world_size=2, rank_in_group=1),
    )

    assert shared_experts.parallel_mode() is expected_mode


@pytest.mark.parametrize("num_tokens", [0, 5])
def test_local_shared_expert_dp_pads_splits_gathers_and_unpads(num_tokens):
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    tp_group = MagicMock(world_size=2, rank_in_group=1)
    shared_experts.moe_config = SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        ep_size=2,
        tp_group=tp_group,
    )
    hidden_states = torch.arange(num_tokens * 2, dtype=torch.float32).view(
        num_tokens,
        2,
    )

    local_hidden_states, metadata = shared_experts._prepare_local_dp_input(hidden_states)
    pad_size = num_tokens % 2
    padded_hidden_states = F.pad(hidden_states, (0, 0, 0, pad_size))
    tp_group.all_gather.return_value = padded_hidden_states
    result = shared_experts._finalize_local_dp_output(
        local_hidden_states,
        metadata,
    )

    assert metadata == (num_tokens, pad_size)
    assert local_hidden_states.shape[0] == padded_hidden_states.shape[0] // 2
    torch.testing.assert_close(
        local_hidden_states,
        padded_hidden_states[local_hidden_states.shape[0] :],
    )
    torch.testing.assert_close(result, hidden_states)
    tp_group.all_gather.assert_called_once_with(local_hidden_states, dim=0)


@pytest.mark.parametrize("num_tokens", [4, 5])
def test_sequence_parallel_only_gathers_unpads_pads_and_reduce_scatters(monkeypatch, num_tokens):
    """SP-only (TP-sharded weights): all-gather + unpad the shard up front,
    pad + reduce-scatter the full output back to the SP shard layout."""
    tp_size = 2
    shard_size = (num_tokens + tp_size - 1) // tp_size
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.moe_config = SimpleNamespace(
        tp_group=SimpleNamespace(world_size=tp_size, rank_in_group=0),
    )
    hidden_shard = torch.randn(shard_size, 2)
    gathered = torch.randn(shard_size * tp_size, 2)
    all_gather = MagicMock(return_value=gathered)
    reduce_scatter = MagicMock(return_value=torch.zeros(shard_size, 2))
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_reduce_scatter", reduce_scatter)
    monkeypatch.setattr(shared_experts_module, "_EXTRA_CTX", SimpleNamespace(num_tokens=num_tokens))

    full_input = shared_experts._gather_sp_input(hidden_shard)
    all_gather.assert_called_once_with(hidden_shard, dim=0)
    torch.testing.assert_close(full_input, gathered[:num_tokens])

    full_output = torch.randn(num_tokens, 2)
    shared_experts._pad_and_reduce_scatter(full_output)
    reduce_scatter.assert_called_once()
    scattered = reduce_scatter.call_args.args[0]
    assert reduce_scatter.call_args.kwargs["dim"] == 0
    assert scattered.shape[0] == shard_size * tp_size
    torch.testing.assert_close(scattered[:num_tokens], full_output)
    if num_tokens % tp_size:
        torch.testing.assert_close(
            scattered[num_tokens:],
            torch.zeros(tp_size - num_tokens % tp_size, 2),
        )


@pytest.mark.parametrize(
    ("mode", "multistream_overlap", "starts_all_gather"),
    [
        (SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY, True, True),
        (SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY, False, False),
        (SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP, True, False),
    ],
)
def test_prepare_shared_expert_input_only_starts_sp_tp_all_gather(
    monkeypatch,
    mode,
    multistream_overlap,
    starts_all_gather,
):
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.multistream_overlap = multistream_overlap
    shared_experts.parallel_mode = MagicMock(return_value=mode)
    hidden_states = torch.randn(2, 4)
    gathered_states = torch.randn(4, 4)
    shared_experts._gather_sp_input = MagicMock(return_value=gathered_states)
    default_stream = MagicMock()
    auxiliary_stream = MagicMock()
    input_ready = MagicMock()
    all_gather_done = MagicMock()
    default_stream.record_event.return_value = input_ready
    auxiliary_stream.record_event.return_value = all_gather_done
    stream_state = {"current": default_stream}

    @contextmanager
    def switch_stream(stream, enabled):
        previous_stream = stream_state["current"]
        if enabled:
            stream_state["current"] = stream
        try:
            yield
        finally:
            stream_state["current"] = previous_stream

    monkeypatch.setattr(shared_experts_module, "npu_stream_switch", switch_stream)
    monkeypatch.setattr(shared_experts_module, "shared_experts_calculation_stream", lambda: auxiliary_stream)
    monkeypatch.setattr(shared_experts_module.torch.npu, "current_stream", lambda: stream_state["current"])

    result, done_event = shared_experts.prepare_input_before_routed_experts(hidden_states)

    if starts_all_gather:
        assert result is gathered_states
        assert done_event is all_gather_done
        default_stream.record_event.assert_called_once_with()
        auxiliary_stream.wait_event.assert_called_once_with(input_ready)
        shared_experts._gather_sp_input.assert_called_once_with(hidden_states)
        auxiliary_stream.record_event.assert_called_once_with()
    else:
        assert result is hidden_states
        assert done_event is None
        default_stream.record_event.assert_not_called()
        shared_experts._gather_sp_input.assert_not_called()


def test_sp_multistream_down_projection_and_reduce_scatter_wait_for_routed_finalize(
    monkeypatch,
):
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.layer = SimpleNamespace(
        gate_up_proj=SimpleNamespace(),
        down_proj=SimpleNamespace(),
    )
    shared_experts.multistream_overlap = True
    shared_experts.quant_type = QuantType.NONE
    shared_experts.lora_context = None
    shared_experts.parallel_mode = MagicMock(return_value=SharedExpertParallelMode.SEQUENCE_PARALLEL_ONLY)
    hidden_states = torch.randn(4, 4)
    part1_out = torch.randn(4, 8)
    shared_out = torch.randn(4, 4)
    reduced_out = torch.randn(2, 4)
    shared_experts.part1 = MagicMock(return_value=part1_out)
    shared_experts.part2 = MagicMock(return_value=shared_out)
    shared_experts._gather_sp_input = MagicMock()
    shared_experts._pad_and_reduce_scatter = MagicMock(return_value=reduced_out)
    default_stream = MagicMock()
    auxiliary_stream = MagicMock()
    stream_state = {"current": default_stream}
    events = FusedMoEEvents(
        before_routed_experts=MagicMock(),
        before_dispatch=MagicMock(),
        before_combine=MagicMock(),
        after_routed_finalize=MagicMock(),
    )

    @contextmanager
    def switch_stream(stream, enabled):
        previous_stream = stream_state["current"]
        if enabled:
            stream_state["current"] = stream
        try:
            yield
        finally:
            stream_state["current"] = previous_stream

    monkeypatch.setattr(shared_experts_module, "npu_stream_switch", switch_stream)
    monkeypatch.setattr(shared_experts_module, "shared_experts_calculation_stream", lambda: auxiliary_stream)
    monkeypatch.setattr(shared_experts_module.torch.npu, "current_stream", lambda: stream_state["current"])

    result = shared_experts.forward(
        hidden_states,
        events,
        input_is_gathered=True,
    )

    assert result is reduced_out
    shared_experts._gather_sp_input.assert_not_called()
    auxiliary_stream.wait_event.assert_any_call(events.after_routed_finalize)
    assert not any(
        event_wait.args == (events.before_combine,) for event_wait in auxiliary_stream.wait_event.call_args_list
    )
    shared_experts.part2.assert_called_once_with(hidden_states, part1_out)
    shared_experts._pad_and_reduce_scatter.assert_called_once_with(shared_out)
    default_stream.wait_stream.assert_called_once_with(auxiliary_stream)


def test_sequence_parallel_sedp_forward_skips_token_comms(monkeypatch):
    """SP+DP (replicated weights) computes directly on the SP shard without
    any token gather/scatter."""
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.layer = SimpleNamespace(
        gate_up_proj=SimpleNamespace(),
        down_proj=SimpleNamespace(),
    )  # no weight_scale -> dense part1/part2 path
    shared_experts.multistream_overlap = False
    shared_experts.quant_type = QuantType.NONE
    shared_experts.lora_context = None
    shared_experts.parallel_mode = MagicMock(return_value=SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP)
    hidden_states = torch.randn(2, 4)
    part1_out = torch.randn(2, 8)
    shared_out = torch.randn(2, 4)
    shared_experts.part1 = MagicMock(return_value=part1_out)
    shared_experts.part2 = MagicMock(return_value=shared_out)
    current_stream = MagicMock()
    events = SimpleNamespace(
        before_routed_experts=None,
        after_routed_experts=None,
        before_dispatch=None,
        before_gmm2=None,
        before_combine=None,
    )
    all_gather = MagicMock()
    reduce_scatter = MagicMock()
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_reduce_scatter", reduce_scatter)
    monkeypatch.setattr(shared_experts_module, "npu_stream_switch", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(shared_experts_module, "shared_experts_calculation_stream", MagicMock())
    monkeypatch.setattr(shared_experts_module.torch.npu, "current_stream", lambda: current_stream)

    output = shared_experts.forward(hidden_states, events)

    assert output is shared_out
    all_gather.assert_not_called()
    reduce_scatter.assert_not_called()
    shared_experts.part1.assert_called_once_with(hidden_states)
    shared_experts.part2.assert_called_once_with(hidden_states, part1_out)


def test_active_shared_expert_lora_uses_dense_wrappers(monkeypatch):
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.layer = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight_scale=torch.ones(1)),
        down_proj=SimpleNamespace(weight_scale=torch.ones(1)),
    )
    shared_experts.multistream_overlap = False
    shared_experts.quant_type = QuantType.W8A8
    shared_experts.parallel_mode = MagicMock(return_value=SharedExpertParallelMode.SEQUENCE_PARALLEL_SEDP)
    hidden_states = torch.randn(2, 4)
    part1_out = torch.randn(2, 8)
    shared_out = torch.randn(2, 4)
    shared_experts.part1 = MagicMock(return_value=part1_out)
    shared_experts.part2 = MagicMock(return_value=shared_out)
    current_stream = MagicMock()
    lora_context = SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=False))
    events = SimpleNamespace(
        before_routed_experts=None,
        after_routed_experts=None,
        before_dispatch=None,
        before_gmm2=None,
        before_combine=None,
    )

    monkeypatch.setattr(shared_experts_module, "npu_stream_switch", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(shared_experts_module, "shared_experts_calculation_stream", MagicMock())
    monkeypatch.setattr(shared_experts_module.torch.npu, "current_stream", lambda: current_stream)
    shared_experts.set_lora_context(lora_context)

    with patch.object(shared_experts_module.torch_npu, "npu_dynamic_quant", create=True) as dynamic_quant:
        output = shared_experts.forward(hidden_states, events)

    assert output is shared_out
    dynamic_quant.assert_not_called()
    shared_experts.part1.assert_called_once_with(hidden_states)
    shared_experts.part2.assert_called_once_with(hidden_states, part1_out)


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_set_lora_context_updates_experts(has_shared_experts):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace()
    shared_experts = MagicMock() if has_shared_experts else None
    runner.ascend_shared_experts = shared_experts
    lora_context = object()

    runner.set_lora_context(lora_context)

    assert runner.routed_experts._ascend_moe_lora_context is lora_context
    if shared_experts is not None:
        shared_experts.set_lora_context.assert_called_once_with(lora_context)


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_forward_impl_returns_current_runner_contract(monkeypatch, has_shared_experts):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 3)
    input_ids = torch.tensor([11, 22])
    routed_out = torch.randn(2, 4)
    shared_out = torch.randn(2, 4)
    ascend_shared_experts = SimpleNamespace(
        prepare_input_before_routed_experts=MagicMock(return_value=(hidden_states, None)),
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

    monkeypatch.setattr(AscendMoERunner, "is_internal_router", property(lambda _: False))
    monkeypatch.setattr(fused_moe_module.torch.npu, "current_stream", lambda: current_stream)

    result = runner._forward_impl(
        hidden_states,
        router_logits,
        shared_experts_input=None,
        input_ids=input_ids,
    )

    if has_shared_experts:
        ascend_shared_experts.prepare_input_before_routed_experts.assert_called_once_with(hidden_states)
        runner.routed_experts.forward_impl.assert_called_once_with(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        assert result[0] is shared_out
        assert result[1] is routed_out
        ascend_shared_experts.forward.assert_called_once()
    else:
        runner.routed_experts.forward_impl.assert_called_once_with(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        assert result is routed_out
        ascend_shared_experts.forward.assert_not_called()


def test_forward_impl_keeps_full_width_input_for_shared_experts(monkeypatch):
    runner = AscendMoERunner.__new__(AscendMoERunner)
    nn.Module.__init__(runner)
    routed_hidden_states = torch.randn(2, 4)
    shared_hidden_states = torch.randn(2, 8)
    router_logits = torch.randn(2, 3)
    routed_out = torch.randn(2, 4)
    shared_out = torch.randn(2, 8)
    routed_events = FusedMoEEvents(
        before_routed_experts=None,
        after_routed_experts=None,
        before_dispatch=None,
        before_gmm2=None,
        before_combine=None,
    )
    runner.routed_experts = SimpleNamespace(forward_impl=MagicMock(return_value=(routed_out, routed_events)))
    shared_input_all_gather_done = MagicMock()
    prepared_shared_hidden_states = torch.randn(4, 8)
    runner.ascend_shared_experts = SimpleNamespace(
        prepare_input_before_routed_experts=MagicMock(
            return_value=(prepared_shared_hidden_states, shared_input_all_gather_done),
        ),
        forward=MagicMock(return_value=shared_out),
    )
    runner._sequence_parallel_context = MagicMock(return_value=nullcontext())
    current_stream = MagicMock()

    monkeypatch.setattr(
        AscendMoERunner,
        "is_internal_router",
        property(lambda _: False),
    )
    monkeypatch.setattr(
        fused_moe_module.torch.npu,
        "current_stream",
        lambda: current_stream,
    )

    result = runner._forward_impl(
        routed_hidden_states,
        router_logits,
        shared_experts_input=shared_hidden_states,
    )

    runner.routed_experts.forward_impl.assert_called_once_with(
        hidden_states=routed_hidden_states,
        router_logits=router_logits,
        input_ids=None,
    )
    runner.ascend_shared_experts.prepare_input_before_routed_experts.assert_called_once_with(shared_hidden_states)
    current_stream.wait_event.assert_called_once_with(shared_input_all_gather_done)
    assert routed_events.after_routed_finalize is current_stream.record_event.return_value
    runner.ascend_shared_experts.forward.assert_called_once_with(
        prepared_shared_hidden_states,
        routed_events,
        input_is_gathered=True,
    )
    assert result[0] is shared_out
    assert result[1] is routed_out
