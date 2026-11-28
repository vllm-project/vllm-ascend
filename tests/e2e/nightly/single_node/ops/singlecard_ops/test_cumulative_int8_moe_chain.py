# SPDX-License-Identifier: Apache-2.0
"""Exact local INT8 routing -> fused GMM1 -> GMM2 -> combine comparisons.

These tests supply single-rank topology and synthetic weights. All numerical
operators are real. They do not establish distributed or full-model accuracy.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.fused_moe import token_dispatcher as dispatchers
from vllm_ascend.ops.fused_moe.dataclass import token_dispatcher as dispatch_data
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import build_mlp_compute_input
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op

NZ_FORMAT = 29
SHAPES = [(8, 128, 128, 2), (256, 2048, 512, 8), (256, 2048, 256, 8)]
ROWS = [1, 7, 16, 129, 513]
PATTERNS = ["spread", "skewed", "last-experts"]


def make_layer(experts, hidden, intermediate, seed=17):
    generator = torch.Generator().manual_seed(seed)
    w1 = torch.randint(-8, 8, (experts, hidden, 2 * intermediate), generator=generator, dtype=torch.int8).npu()
    w2 = torch.randint(-8, 8, (experts, intermediate, hidden), generator=generator, dtype=torch.int8).npu()
    return SimpleNamespace(
        w13_weight=torch_npu.npu_format_cast(w1, NZ_FORMAT),
        w13_weight_scale_fp32=torch.full((experts, 2 * intermediate), 1 / 32, dtype=torch.float32, device="npu"),
        w2_weight=torch_npu.npu_format_cast(w2, NZ_FORMAT),
        w2_weight_scale=torch.full((experts, hidden), 1 / 32, dtype=torch.bfloat16, device="npu"),
        activation=MoEActivation.SILU,
    )


def make_inputs(rows, experts, hidden, topk, pattern, seed=23):
    generator = torch.Generator().manual_seed(seed)
    states = (torch.randn(rows, hidden, generator=generator) / 10).to(torch.bfloat16).npu()
    if pattern == "spread":
        ids = torch.rand(rows, experts, generator=generator).topk(topk, dim=-1).indices.int().npu()
    else:
        offset = experts - topk if pattern == "last-experts" else 0
        ids = (torch.arange(topk, dtype=torch.int32) + offset).expand(rows, -1).contiguous().npu()
    weights = torch.full((rows, topk), 1 / topk, dtype=torch.float32, device="npu")
    return states, ids, weights


def local_context():
    # Only unavailable distributed/forward context is supplied by this harness.
    # No numerical operator is replaced.
    return (
        patch.object(dispatchers, "get_ep_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0)),
        patch.object(w8a8_dynamic, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)),
        patch.object(dispatch_data, "enable_fusion_gmmswigluquant", return_value=True),
    )


def chain(layer, states, ids, weights, cumulative, prequantized=False, snapshots=True):
    dispatcher = dispatchers.TokenDispatcherWithAllGather(
        top_k=ids.shape[1],
        num_experts=layer.w13_weight_scale_fp32.shape[0],
        num_local_experts=layer.w13_weight_scale_fp32.shape[0],
    )
    scale = None
    if prequantized:
        states, scale = torch_npu.npu_dynamic_quant(states)
    source = build_fused_experts_input(
        hidden_states=states,
        topk_weights=weights,
        topk_ids=ids,
        layer=layer,
        quant_type=QuantType.W8A8,
        dynamic_eplb=False,
        activation=MoEActivation.SILU,
        pertoken_scale=scale,
    )
    request = dispatch_data.build_token_dispatch_input(fused_experts_input=source)
    assert request.prefer_cumulative_group_list
    if not cumulative:
        request = replace(request, prefer_cumulative_group_list=False)
    dispatched = dispatcher.token_dispatch(request)
    assert dispatched.group_list_type == (0 if cumulative else 1)
    mlp = build_mlp_compute_input(fused_experts_input=source, token_dispatch_output=dispatched, use_fusion_ops=True)
    values: dict[str, torch.Tensor] = {}
    if snapshots:
        values.update(
            routed=dispatched.hidden_states.clone(),
            indices=dispatched.combine_metadata.expanded_row_idx.clone(),
            prefix=(dispatched.group_list if cumulative else dispatched.group_list.cumsum(0)).clone(),
            route_scale=dispatched.dynamic_scale.clone(),
        )
    method = w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod.__new__(w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = False
    activation, activation_scale = method.apply_gmm1_act_quant(mlp)
    if snapshots:
        values.update(gmm1=activation.clone(), gmm1_scale=activation_scale.clone())
    expert_output = method.apply_gmm2(mlp, activation, activation_scale)
    if snapshots:
        values["gmm2"] = expert_output.clone()
    combined = dispatcher.token_combine(expert_output, dispatched.combine_metadata)
    if snapshots:
        values["combined"] = combined.clone()
        return values
    return combined


def assert_stages_equal(baseline, candidate):
    assert baseline.keys() == candidate.keys()
    for name in baseline:
        a, b = baseline[name], candidate[name]
        if a.is_floating_point():
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda message, stage=name: stage + ": " + message)


@pytest.fixture(scope="module", autouse=True)
def require_real_ops():
    if not get_current_hardware_profile().supports(HardwareCapability.MOE_INT8_CUMULATIVE_GROUP_LIST):
        pytest.skip("Cumulative local INT8 group lists are not enabled by this hardware profile")
    torch_npu.npu.config.allow_internal_format = True
    assert enable_custom_op()


@pytest.fixture(scope="module", params=SHAPES)
@torch.inference_mode()
def shape_layer(request):
    # The chain only reads weights; inputs and graph state stay test-local.
    experts, hidden, intermediate, _ = request.param
    return request.param, make_layer(experts, hidden, intermediate)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("prequantized", [False, True])
@torch.inference_mode()
def test_exact_complete_int8_chain(shape_layer, rows, pattern, prequantized):
    shape, layer = shape_layer
    experts, hidden, _, topk = shape
    states, ids, weights = make_inputs(rows, experts, hidden, topk, pattern)
    topology, forward, fusion = local_context()
    with topology, forward, fusion:
        baseline = chain(layer, states, ids, weights, cumulative=False, prequantized=prequantized)
        candidate = chain(layer, states, ids, weights, cumulative=True, prequantized=prequantized)
        torch.npu.synchronize()
        assert_stages_equal(baseline, candidate)


@torch.inference_mode()
def test_graph_replay_updates_routing_inputs(shape_layer):
    shape, layer = shape_layer
    experts, hidden, _, topk = shape
    states, ids, weights = make_inputs(7, experts, hidden, topk, "spread")
    topology, forward, fusion = local_context()
    with topology, forward, fusion:
        for _ in range(3):
            chain(layer, states, ids, weights, True, snapshots=False)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = chain(layer, states, ids, weights, True)
        for i, pattern in enumerate(PATTERNS):
            new_states, new_ids, new_weights = make_inputs(7, experts, hidden, topk, pattern, 37 + i)
            states.copy_(new_states)
            ids.copy_(new_ids)
            weights.copy_(new_weights)
            graph.replay()
            torch.npu.synchronize()
            expected = chain(layer, states, ids, weights, False)
            assert_stages_equal(expected, actual)
