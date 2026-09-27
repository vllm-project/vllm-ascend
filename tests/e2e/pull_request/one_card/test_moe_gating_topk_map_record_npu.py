# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.triton.moe_gating_topk_map_record import moe_gating_topk_map_record
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("tokens,experts,top_k", [(64, 8, 6), (65, 16, 8), (128, 16, 8), (512, 32, 8)])
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_gating_topk_map_record_matches_cann(tokens, experts, top_k, scoring, with_bias):
    enable_custom_op()
    torch.manual_seed(20260927)
    logits = torch.randn(tokens, experts, dtype=torch.float32, device="npu")
    bias = torch.randn(experts, dtype=torch.float32, device="npu") if with_bias else None
    row = torch.arange(1024, dtype=torch.int32, device="npu")[:, None]
    expert = torch.arange(experts, dtype=torch.int32, device="npu")[None, :]
    table = ((row + expert) % experts).contiguous()
    valid_tokens = tokens - tokens // 8
    enabled = torch.tensor(True, device="npu")
    load = torch.zeros(experts, dtype=torch.int32, device="npu")

    expected_weights, logical_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        logits,
        k=top_k,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=1,
        norm_type=0 if scoring == "softmax" else 1,
        out_flag=False,
        routed_scaling_factor=1.0,
        eps=1e-20,
        bias_opt=bias,
    )
    expected_ids = table[torch.arange(tokens, device="npu")[:, None] % 1024, logical_ids.long()]
    weights, ids = moe_gating_topk_map_record(
        logits, bias, table, load, enabled, valid_tokens, k=top_k, scoring=scoring
    )
    torch.npu.synchronize()
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-4, atol=1e-5)
    expected_load = torch.bincount(expected_ids[:valid_tokens].long().flatten(), minlength=experts).to(torch.int32)
    torch.testing.assert_close(load, expected_load, rtol=0, atol=0)


def test_gating_topk_map_record_uses_updated_routing_table_on_replay():
    enable_custom_op()
    experts = 16
    logits = torch.randn(64, experts, dtype=torch.float32, device="npu")
    table = torch.arange(experts, dtype=torch.int32, device="npu").expand(1024, -1).contiguous()
    load = torch.zeros(experts, dtype=torch.int32, device="npu")
    enabled = torch.tensor(False, device="npu")
    moe_gating_topk_map_record(logits, None, table, load, enabled, 64, k=8, scoring="softmax")
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, ids = moe_gating_topk_map_record(logits, None, table, load, enabled, 64, k=8, scoring="softmax")
    for shift in (1, 3):
        table.copy_((torch.arange(experts, dtype=torch.int32, device="npu") + shift) % experts)
        graph.replay()
        torch.npu.synchronize()
        expected_weights, logical_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
            logits,
            k=8,
            k_group=1,
            group_count=1,
            group_select_mode=1,
            renorm=1,
            norm_type=0,
            out_flag=False,
            routed_scaling_factor=1.0,
            eps=1e-20,
            bias_opt=None,
        )
        torch.testing.assert_close(ids, table[0, logical_ids.long()], rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(load, torch.zeros_like(load), rtol=0, atol=0)


def test_fused_router_falls_back_for_noncontiguous_inputs():
    class RecordingState:
        fused_record_allowed = True

    router = AscendFusedTopKRouter(top_k=8, global_num_experts=32, eplb_state=RecordingState())
    noncontiguous_logits = torch.randn(64, 64)[:, ::2]
    assert router._try_small_expert_fused_routing(noncontiguous_logits, None, 1, 1, 1) is None

    router.e_score_correction_bias = torch.randn(64)[::2]
    contiguous_logits = torch.randn(64, 32)
    assert router._try_small_expert_fused_routing(contiguous_logits, None, 1, 1, 1) is None
