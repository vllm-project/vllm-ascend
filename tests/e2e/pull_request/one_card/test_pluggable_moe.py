# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from functools import partial
from typing import Any

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import create_forward_context, override_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
from vllm.model_executor.models.gpt_oss import GptOssRoutedExperts
from vllm.model_executor.models.transformers.moe import (
    TransformersMoERunner,
    TransformersMoEState,
    TransformersRoutedExperts,
)

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.eplb.eplb_state import AscendEplbLayerState
from vllm_ascend.distributed.eplb.tensor_ops import EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS
from vllm_ascend.ops.fused_moe.eplb import use_allgather_eplb
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.router.pluggable import AscendBiasRouter, AscendGroupedRouter, AscendTopKRouter
from vllm_ascend.ops.fused_moe.router.router_factory import create_ascend_fused_moe_router
from vllm_ascend.utils import register_ascend_customop


@pytest.fixture(scope="module")
def config():
    torch.npu.set_device(0)
    config = VllmConfig()
    with set_current_vllm_config(config):
        register_ascend_customop(config)
        yield config
    torch.npu.synchronize()


@pytest.mark.parametrize("scheme", ["plain", "grouped_bias", "bias"])
def test_registered_router_matches_ascend_kernel_path(config, scheme):
    kwargs: dict[str, Any] = {"top_k": 6, "global_num_experts": 256, "renormalize": True}
    expected_type = AscendTopKRouter
    if scheme != "plain":
        kwargs.update(
            scoring_func="sigmoid",
            e_score_correction_bias=torch.nn.Parameter(
                torch.linspace(-0.2, 0.2, 256, device="npu"), requires_grad=False
            ),
            routed_scaling_factor=1.5,
        )
        expected_type = AscendBiasRouter
    if scheme == "grouped_bias":
        kwargs.update(use_grouped_topk=True, num_expert_group=8, topk_group=4)
        expected_type = AscendGroupedRouter
    baseline = create_ascend_fused_moe_router(**kwargs)
    router = create_fused_moe_router(**kwargs)
    assert type(router) is expected_type
    hidden = torch.zeros(16, 128, dtype=torch.bfloat16, device="npu")
    logits = torch.linspace(-3, 3, 16 * 256, device="npu").reshape(16, 256)
    expected_weights, expected_ids = baseline.select_experts(hidden, logits)
    weights, ids = router.select_experts(hidden, logits)
    torch.npu.synchronize()
    torch.testing.assert_close(ids.cpu(), expected_ids.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=0, atol=0)
    if scheme != "plain":
        assert router.e_score_correction_bias is kwargs["e_score_correction_bias"]
    assert FusedMoEFactory.__module__ == "vllm.model_executor.layers.fused_moe.layer"


def test_bias_cast_keeps_shared_fp32_parameter_on_npu(config):
    bias = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 256, device="npu"), requires_grad=False)
    router = create_fused_moe_router(
        top_k=6, global_num_experts=256, scoring_func="sigmoid", e_score_correction_bias=bias
    )
    hidden = torch.zeros(16, 128, dtype=torch.bfloat16, device="npu")
    logits = torch.linspace(-3, 3, 16 * 256, device="npu").reshape(16, 256).to(torch.bfloat16)
    weights, ids = router.select_experts(hidden, logits)
    torch.npu.synchronize()
    assert router.e_score_correction_bias is bias
    assert dict(router.named_parameters())["e_score_correction_bias"] is bias
    assert bias.dtype == torch.float32
    assert weights.shape == ids.shape == (16, 6)
    assert torch.isfinite(weights).all().item()


@pytest.mark.parametrize("comm", [MoECommType.ALLTOALL, MoECommType.ALLGATHER])
def test_registered_router_observes_eplb_mapping_updates_on_npu(config, comm):
    state = AscendEplbLayerState()
    state.expert_load_view = torch.zeros(256, dtype=torch.int32, device="npu")
    state.logical_to_physical_map = torch.arange(256, device="npu").reshape(256, 1)
    state.logical_replica_count = torch.ones(256, dtype=torch.int64, device="npu")
    state.should_record_tensor = torch.tensor(True, device="npu")
    state.num_unpadded_tokens_tensors = [torch.tensor(16, device="npu")]
    state.expert_replica_routing_table = torch.arange(256, dtype=torch.int32, device="npu").repeat(
        EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS, 1
    )
    state.allgather_expert_replica_routing_table = state.expert_replica_routing_table.clone()
    router = create_fused_moe_router(top_k=6, global_num_experts=256, eplb_state=state)
    assert router.eplb_state is state
    hidden = torch.zeros(16, 128, dtype=torch.bfloat16, device="npu")
    logits = torch.linspace(-3, 3, 16 * 256, device="npu").reshape(16, 256)
    _, logical_ids = router._compute_routing(hidden, logits, torch.int32)
    with override_forward_context(create_forward_context(None, config)):
        _EXTRA_CTX.moe_comm_type = comm
        table = (
            state.allgather_expert_replica_routing_table if use_allgather_eplb() else state.expert_replica_routing_table
        )
        _, initial_ids = router.select_experts(hidden, logits)
    torch.testing.assert_close(initial_ids.cpu(), logical_ids.cpu(), rtol=0, atol=0)
    pointer = table.data_ptr()
    table.copy_((table + 1) % 256)
    with override_forward_context(create_forward_context(None, config)):
        _EXTRA_CTX.moe_comm_type = comm
        _, updated_ids = router.select_experts(hidden, logits)
    torch.npu.synchronize()
    assert table.data_ptr() == pointer
    torch.testing.assert_close(updated_ids.cpu(), (logical_ids.cpu() + 1) % 256, rtol=0, atol=0)


@pytest.fixture
def factory_runtime(config, tmp_path):
    init_distributed_environment(1, 0, f"file://{tmp_path}/distributed", 0, "hccl")
    initialize_model_parallel(1, 1, backend="hccl")
    init_ascend_config(config)
    try:
        yield config
    finally:
        config.compilation_config.static_forward_context.clear()
        config.compilation_config.static_all_moe_layers.clear()
        destroy_model_parallel()
        destroy_distributed_environment()


def _make_moe(prefix="migration", **kwargs):
    with torch.device("npu"):
        return FusedMoEFactory(
            num_experts=8,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            params_dtype=torch.bfloat16,
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            prefix=prefix,
            **kwargs,
        )


@pytest.mark.parametrize("enable_eplb", [False, True])
def test_default_moe_allocates_expert_capacity_on_npu(factory_runtime, enable_eplb):
    runner = _make_moe(enable_eplb=enable_eplb, num_redundant_experts=2 if enable_eplb else 0)
    assert isinstance(runner, AscendMoERunner)
    assert isinstance(runner.routed_experts, AscendRoutedExperts)
    assert type(runner.router) is AscendTopKRouter
    assert (runner.eplb_state is not None) == enable_eplb
    assert runner.routed_experts.w13_weight.device.type == "npu"
    assert runner.routed_experts.w13_weight.shape[0] == (10 if enable_eplb else 8)


def test_model_components_load_gpt_oss_weights_on_npu(factory_runtime):
    state = TransformersMoEState()
    runner = _make_moe(
        routed_experts_cls=GptOssRoutedExperts,
        runner_cls=TransformersMoERunner,
        runner_args={"moe_state": state},
    )
    assert isinstance(runner, TransformersMoERunner)
    assert isinstance(runner, AscendMoERunner)
    assert isinstance(runner.routed_experts, GptOssRoutedExperts)
    assert isinstance(runner.routed_experts, AscendRoutedExperts)
    assert runner._moe_state is state
    weight = runner.routed_experts.w13_weight
    checkpoint = torch.arange(weight[0].numel(), dtype=torch.float32, device="npu").to(weight.dtype)
    checkpoint = checkpoint.reshape(weight.shape[2], weight.shape[1])
    runner.routed_experts.weight_loader(weight, checkpoint, "w13_weight", "gpt_oss_w13", 0)
    torch.npu.synchronize()
    torch.testing.assert_close(weight[0].cpu(), checkpoint.t().cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("with_bias", [False, True])
def test_moe_forward_matches_explicit_ascend_router_on_npu(factory_runtime, with_bias):
    kwargs = {}
    if with_bias:
        kwargs = {
            "scoring_func": "sigmoid",
            "e_score_correction_bias": torch.nn.Parameter(
                torch.linspace(-0.2, 0.2, 8, device="npu"), requires_grad=False
            ),
        }
    router = create_ascend_fused_moe_router(top_k=2, global_num_experts=8, **kwargs)
    baseline = _make_moe(prefix="baseline", router=router, **kwargs)
    candidate = _make_moe(prefix="candidate", **kwargs)
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for name in ("w13_weight", "w2_weight"):
            weight = getattr(baseline.routed_experts, name)
            checkpoint = torch.randn(weight.shape, generator=generator).mul_(0.02).to(weight.dtype)
            weight.copy_(checkpoint.to("npu"))
            getattr(candidate.routed_experts, name).copy_(checkpoint.to("npu"))
        for runner in (baseline, candidate):
            runner.routed_experts.quant_method.process_weights_after_loading(runner.routed_experts)
        hidden = torch.randn((16, 128), generator=generator).to(device="npu", dtype=torch.bfloat16)
        logits = torch.randn((16, 8), generator=generator).to("npu")
        outputs = []
        forward_context = create_forward_context(None, factory_runtime)
        if forward_context.all_moe_layers is not None:
            assert forward_context.all_moe_layers == [baseline.layer_name, candidate.layer_name]
        with override_forward_context(forward_context):
            for runner in (baseline, candidate):
                _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
                _EXTRA_CTX.moe_comm_method = get_moe_comm_method(MoECommType.ALLGATHER, runner.moe_config)
                outputs.append(runner(hidden, logits))
        torch.npu.synchronize()
    assert torch.isfinite(outputs[1]).all().item()
    torch.testing.assert_close(outputs[1].cpu(), outputs[0].cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
def test_registered_hash_router_uses_explicit_input_ids_on_npu(factory_runtime, input_dtype):
    if not hasattr(torch.ops._C_ascend, "moe_gating_top_k_hash"):
        import vllm_ascend.vllm_ascend_C  # noqa: F401

    runner = _make_moe()
    table = (torch.arange(32 * 6, dtype=torch.int32).reshape(32, 6) % 256).to("npu")
    kwargs = {
        "top_k": 6,
        "global_num_experts": 256,
        "scoring_func": "sqrtsoftplus",
        "hash_indices_table": table,
        "routed_scaling_factor": 1.5,
    }
    baseline = create_ascend_fused_moe_router(**kwargs)
    router = create_fused_moe_router(**kwargs)
    assert router.tid2eid is table
    hidden = torch.zeros(16, 128, dtype=torch.bfloat16, device="npu")
    logits = torch.linspace(-3, 3, 16 * 256, device="npu").reshape(16, 256)
    input_ids = torch.arange(-1, 15, dtype=input_dtype, device="npu")
    results = []
    with override_forward_context(create_forward_context(None, factory_runtime)):
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
        _EXTRA_CTX.moe_comm_method = get_moe_comm_method(MoECommType.ALLGATHER, runner.moe_config)
        for selected in (baseline, router):
            results.append(selected.select_experts(hidden, logits, input_ids=input_ids))
    torch.npu.synchronize()
    weights, ids = results[1]
    expected_ids = table.cpu()[input_ids.cpu().clamp_min(0).long()]
    torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(ids.cpu(), results[0][1].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), results[0][0].cpu(), rtol=0, atol=0)
    scores = torch.nn.functional.softplus(logits.cpu()).sqrt().gather(1, expected_ids.long())
    expected_weights = scores / scores.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=1e-5, atol=1e-5)


def test_transformers_runner_preserves_precomputed_routing_on_npu(factory_runtime):
    from vllm.model_executor.models.transformers.moe import _hf_topk_router_precomputed

    baseline_state = TransformersMoEState()
    candidate_state = TransformersMoEState()
    baseline = _make_moe(
        prefix="baseline",
        routed_experts_cls=TransformersRoutedExperts,
        custom_routing_function=partial(_hf_topk_router_precomputed, moe_state=baseline_state),
    )
    candidate = _make_moe(
        prefix="candidate",
        routed_experts_cls=TransformersRoutedExperts,
        custom_routing_function=partial(_hf_topk_router_precomputed, moe_state=candidate_state),
        runner_cls=TransformersMoERunner,
        runner_args={"moe_state": candidate_state},
    )
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for name in ("w13_weight", "w2_weight"):
            weight = getattr(baseline.routed_experts, name)
            checkpoint = torch.randn(weight.shape, generator=generator).mul_(0.02).to(weight.dtype).to("npu")
            weight.copy_(checkpoint)
            getattr(candidate.routed_experts, name).copy_(checkpoint)
        for runner in (baseline, candidate):
            runner.routed_experts.quant_method.process_weights_after_loading(runner.routed_experts)
        hidden = torch.randn((16, 128), generator=generator).to(device="npu", dtype=torch.bfloat16)
        ids = (torch.arange(32).reshape(16, 2) % 8).to("npu")
        weights = torch.tensor([[0.25, 0.75]], device="npu").repeat(16, 1)
        baseline_state.topk_ids = ids.to(torch.int32)
        assert candidate_state.topk_ids is None
        with override_forward_context(create_forward_context(None, factory_runtime)):
            _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
            _EXTRA_CTX.moe_comm_method = get_moe_comm_method(MoECommType.ALLGATHER, baseline.moe_config)
            expected = baseline(hidden, weights)
            output = candidate(hidden, ids, weights)
        torch.npu.synchronize()
    assert candidate._moe_state is candidate_state
    assert candidate_state.topk_ids is not None
    torch.testing.assert_close(candidate_state.topk_ids.cpu(), ids.int().cpu(), rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
