# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.ops.triton.eplb import record_expert_tokens_triton
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


@pytest.mark.parametrize("tokens,experts,top_k", [(64, 8, 6), (128, 16, 8), (256, 32, 8), (65536, 16, 8)])
def test_grid_record_matches_moe_returned_counts(tokens, experts, top_k):
    enable_custom_op()
    logits = torch.randn(tokens, experts, dtype=torch.float32, device="npu")
    row = torch.arange(1024, dtype=torch.int32, device="npu")[:, None]
    expert = torch.arange(experts, dtype=torch.int32, device="npu")[None, :]
    table = ((row + expert) % experts).contiguous()
    enabled = torch.tensor(True, device="npu")
    initial_load = torch.arange(experts, dtype=torch.int32, device="npu")

    expected_weights, logical_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        logits,
        k=top_k,
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
    physical_ids = table[torch.arange(tokens, device="npu")[:, None] % 1024, logical_ids.long()]
    hidden = torch.zeros((tokens, 32), dtype=torch.bfloat16, device="npu")
    _, _, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
        hidden,
        physical_ids,
        active_num=tokens * top_k,
        expert_num=experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, experts],
    )
    expected_load = initial_load.clone()
    record_expert_tokens_triton(expert_tokens, expected_load, enabled, 1, 0)

    load = initial_load.clone()
    weights, ids = moe_gating_topk_map_record(logits, None, table, load, enabled, tokens, k=top_k, scoring="softmax")
    torch.npu.synchronize()
    torch.testing.assert_close(ids, physical_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(load, expected_load, rtol=0, atol=0)


def test_gating_topk_map_record_uses_updated_routing_table_on_replay():
    enable_custom_op()
    experts = 16
    logits = torch.randn(64, experts, dtype=torch.float32, device="npu")
    table = torch.arange(experts, dtype=torch.int32, device="npu").expand(1024, -1).contiguous()
    load = torch.zeros(experts, dtype=torch.int32, device="npu")
    enabled = torch.tensor(False, device="npu")
    valid_tokens = torch.tensor(64, dtype=torch.int32, device="npu")
    moe_gating_topk_map_record(logits, None, table, load, enabled, valid_tokens, k=8, scoring="softmax")
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, ids = moe_gating_topk_map_record(
            logits, None, table, load, enabled, valid_tokens, k=8, scoring="softmax"
        )
    expected_load = torch.zeros_like(load)
    for shift, record, valid_rows in ((1, True, 48), (3, True, 64), (2, False, 32)):
        table.copy_((torch.arange(experts, dtype=torch.int32, device="npu") + shift) % experts)
        enabled.fill_(record)
        valid_tokens.fill_(valid_rows)
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
        if record:
            expected_load += torch.bincount(ids[:valid_rows].long().flatten(), minlength=experts).to(torch.int32)
        torch.testing.assert_close(load, expected_load, rtol=0, atol=0)


def test_grid_record_matches_moe_local_physical_range():
    enable_custom_op()
    tokens, experts, top_k = 64, 16, 8
    local_start = experts
    logits = torch.randn(tokens, experts, dtype=torch.float32, device="npu")
    table = (torch.arange(experts, dtype=torch.int32, device="npu") + local_start).expand(1024, -1).contiguous()
    initial_load = torch.arange(2 * experts, dtype=torch.int32, device="npu")
    enabled = torch.tensor(True, device="npu")
    expected_weights, logical_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        logits,
        k=top_k,
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
    expected_ids = table[0, logical_ids.long()]
    hidden = torch.zeros((tokens, 32), dtype=torch.bfloat16, device="npu")
    _, _, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
        hidden,
        expected_ids,
        active_num=tokens * top_k,
        expert_num=2 * experts,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[local_start, local_start + experts],
    )
    expected_load = initial_load.clone()
    record_expert_tokens_triton(expert_tokens, expected_load, enabled, 1, local_start)
    load = initial_load.clone()
    weights, ids = moe_gating_topk_map_record(
        logits,
        None,
        table,
        load,
        enabled,
        tokens,
        k=top_k,
        scoring="softmax",
        local_expert_start=local_start,
        local_expert_count=experts,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(load, expected_load, rtol=0, atol=0)


def test_fused_router_falls_back_for_noncontiguous_inputs(monkeypatch):
    class RecordingState:
        fused_record_allowed = True
        local_expert_count = 32

    router = AscendFusedTopKRouter(top_k=8, global_num_experts=32, eplb_state=RecordingState())
    context = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
    monkeypatch.setattr("vllm_ascend.ops.fused_moe.router.fused_topk_router._EXTRA_CTX", context)
    noncontiguous_logits = torch.randn(64, 64)[:, ::2]
    assert router._try_small_expert_fused_routing(noncontiguous_logits, None, 1, 1, 1) is None

    router.e_score_correction_bias = torch.randn(64)[::2]
    contiguous_logits = torch.randn(64, 32)
    assert router._try_small_expert_fused_routing(contiguous_logits, None, 1, 1, 1) is None

    router.e_score_correction_bias = None
    router.eplb_state.local_expert_count = 16
    assert router._try_small_expert_fused_routing(contiguous_logits, None, 1, 1, 1) is None
    router.eplb_state.local_expert_count = 32
    context.moe_comm_type = MoECommType.ALLTOALL
    assert router._try_small_expert_fused_routing(contiguous_logits, None, 1, 1, 1) is None


def test_fused_router_dispatches_large_prefill_and_marks_record_active(monkeypatch):
    class RecordingState:
        fused_record_allowed = True
        fused_map_record_active = False
        local_expert_start = 0
        local_expert_count = 16
        expert_replica_routing_table = torch.zeros((1, 16), dtype=torch.int32)
        expert_load_view = torch.zeros(16, dtype=torch.int32)
        should_record_tensor = torch.tensor(True)

    state = RecordingState()
    router = AscendFusedTopKRouter(top_k=8, global_num_experts=16, eplb_state=state)
    context = SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER)
    monkeypatch.setattr("vllm_ascend.ops.fused_moe.router.fused_topk_router._EXTRA_CTX", context)
    calls = []

    def fake_map_record(logits, bias, table, load, record_enabled, valid_tokens, **kwargs):
        calls.append((logits.shape, valid_tokens, kwargs["k"]))
        return torch.empty((valid_tokens, 8)), torch.empty((valid_tokens, 8), dtype=torch.int32)

    monkeypatch.setattr(
        "vllm_ascend.ops.fused_moe.router.fused_topk_router.moe_gating_topk_map_record",
        fake_map_record,
    )
    logits = torch.empty((65536, 16))
    result = router._try_small_expert_fused_routing(logits, None, 1, 1, 1)
    assert result is not None
    assert result[0].shape == (65536, 8)
    assert result[1].shape == (65536, 8)
    assert state.fused_map_record_active
    assert calls == [((65536, 16), 65536, 8)]
