# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Single-NPU routing and Record regressions; no distributed collectives."""

import pytest
import torch
import torch_npu

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.eplb.tensor_ops import build_expert_replica_routing_table, count_allgather_tokens
from vllm_ascend.ops.fused_moe.eplb import map_to_physical_for_ascend, record_expert_tokens
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def require_gating_map():
    if not get_current_hardware_profile().supports(HardwareCapability.MOE_GATING_TOP_K_WITH_MAP):
        pytest.skip("GatingTopK+Map requires supported A3/A5 hardware")
    enable_custom_op()


def inputs(rows, experts, dtype, with_hash=False):
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(rows, experts, generator=generator).to(dtype).npu()
    bias = (torch.randn(experts, generator=generator) * 0.1).to(dtype).npu()
    # Mixed replica counts and shuffled placement avoid an identity-map false positive.
    counts = torch.arange(experts).remainder(3).add(1).int()
    placement = torch.full((experts, 3), -1, dtype=torch.int64)
    slots = torch.randperm(int(counts.sum()), generator=generator)
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        placement[expert, :count] = slots[offset : offset + count]
        offset += count
    table = build_expert_replica_routing_table(placement.npu(), counts.npu(), ep_rank=1)
    kwargs = dict(x=logits, k=6, bias=bias, group_count=1, k_group=1, norm_type=2, renorm=0, routed_scaling_factor=1.5)
    if with_hash:
        kwargs.update(
            input_ids=torch.arange(rows, device="npu").remainder(11),
            tid2eid=torch.stack([torch.randperm(experts, generator=generator)[:6] for _ in range(11)]).int().npu(),
        )
    return kwargs, table, offset


@pytest.mark.parametrize("rows,experts,with_hash", [(1, 384, False), (33, 513, True), (1025, 2048, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_gating_map_matches_separate_ops(rows, experts, with_hash, dtype, execution_mode):
    kwargs, table, physical_experts = inputs(rows, experts, dtype, with_hash)
    op = torch.ops._C_ascend.moe_gating_top_k_hash
    reference_weights, logical_ids, _ = op(**kwargs)
    logical_cpu = logical_ids.cpu().long()
    graph = None
    if execution_mode == "graph":
        for _ in range(3):
            op(**kwargs, expert_replica_routing_table=table)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            weights, ids, _ = op(**kwargs, expert_replica_routing_table=table)
    previous = None
    for update in (False, True):
        if update:
            # Preserve addresses to catch stale tables during graph replay too.
            table.copy_((table + 1).remainder(physical_experts))
        if graph is None:
            weights, ids, _ = op(**kwargs, expert_replica_routing_table=table)
        else:
            graph.replay()
        expected = table.cpu()[torch.arange(rows)[:, None] % table.shape[0], logical_cpu]
        separate = map_to_physical_for_ascend(logical_ids, table)
        torch.testing.assert_close(separate.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(weights, reference_weights, rtol=0, atol=0)
        assert (expected.long() != logical_cpu).any()
        if previous is not None:
            assert (expected != previous).all()
        previous = expected


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_gating_map_preserves_recorded_load(group_list_type):
    kwargs, table, physical_experts = inputs(17, 384, torch.float32)
    op = torch.ops._C_ascend.moe_gating_top_k_hash
    _, logical_ids, _ = op(**kwargs)
    separate = map_to_physical_for_ascend(logical_ids, table)
    _, fused, _ = op(**kwargs, expert_replica_routing_table=table)
    start, end = physical_experts // 2, physical_experts
    hidden = torch.ones(17, 32, dtype=torch.float16, device="npu")
    expected_ids = table.cpu()[torch.arange(17)[:, None], logical_ids.cpu().long()]
    torch.testing.assert_close(separate.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(fused.cpu(), expected_ids, rtol=0, atol=0)
    expected = torch.bincount(expected_ids.long().flatten(), minlength=physical_experts)
    buffers = []
    for ids in (separate, fused):
        _, _, counts, _ = torch_npu.npu_moe_init_routing_v2(
            hidden,
            ids,
            active_num=ids.numel(),
            expert_num=physical_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[start, end],
            quant_mode=-1,
        )
        counts = counts.long()
        torch.testing.assert_close(counts.cpu(), expected[start:end], rtol=0, atol=0)
        groups = counts.cumsum(0) if group_list_type == 0 else counts
        load = torch.zeros(physical_experts, dtype=torch.int64, device="npu")
        enabled = torch.zeros((), dtype=torch.int32, device="npu")
        for flag in (0, 1, 1, 0):
            enabled.fill_(flag)
            record_expert_tokens(groups, load, enabled, group_list_type, start)
        expected_load = torch.zeros_like(expected)
        expected_load[start:end] = 2 * expected[start:end]
        torch.testing.assert_close(load.cpu(), expected_load, rtol=0, atol=0)
        buffers.append(load)
        # Check the production AllGather counting helper excludes padding routes.
        mask = torch.arange(17, device="npu").remainder(3) != 0
        masked_counts = count_allgather_tokens(ids, mask, start, end - start)
        masked_expected = torch.bincount(ids.cpu()[mask.cpu()].long().flatten(), minlength=physical_experts)
        torch.testing.assert_close(masked_counts.cpu(), masked_expected[start:end], rtol=0, atol=0)
    torch.testing.assert_close(buffers[0], buffers[1], rtol=0, atol=0)
