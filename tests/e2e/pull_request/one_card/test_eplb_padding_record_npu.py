# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu

from vllm_ascend.ops.fused_moe.eplb import count_valid_expert_tokens, record_expert_tokens


@pytest.mark.parametrize("local_start", [0, 4])
def test_allgather_padding_counts_are_separate_from_native_group_list(local_start):
    ids = torch.tensor([[0, 4], [0, 4], [1, 5], [6, 6], [2, 7], [7, 7]], dtype=torch.int32, device="npu")
    x = torch.arange(6 * 32, dtype=torch.float32, device="npu").reshape(6, 32).to(torch.float16)
    mask = torch.tensor([True, False, True, False, True, False], device="npu")
    _, _, group_list, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        ids,
        active_num=12,
        expert_num=8,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[local_start, local_start + 4],
        quant_mode=-1,
    )
    raw_counts = torch.bincount(ids.cpu().flatten().long(), minlength=8)[local_start : local_start + 4]
    torch.testing.assert_close(group_list.cpu().long(), raw_counts, rtol=0, atol=0)
    expected = torch.bincount(ids.cpu()[mask.cpu()].flatten().long(), minlength=8)[local_start : local_start + 4]
    corrected = count_valid_expert_tokens(ids, mask, local_start, 4)
    torch.testing.assert_close(corrected.cpu(), expected, rtol=0, atol=0)
    assert not torch.equal(raw_counts, expected)
    # Recounting does not change native dispatch IDs or its GMM count tensor.
    torch.testing.assert_close(group_list.cpu().long(), raw_counts, rtol=0, atol=0)


def test_padding_counts_graph_replays_mutable_mask_and_record_flag():
    ids = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.int32, device="npu")
    mask = torch.ones(4, dtype=torch.bool, device="npu")
    enabled = torch.tensor(True, device="npu")
    initial = torch.arange(8, dtype=torch.int32, device="npu")
    load = initial.clone()

    def step():
        counts = count_valid_expert_tokens(ids, mask, 4, 4)
        record_expert_tokens(counts, load, enabled, 1, 4)

    step()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        step()
    for values, record_on in [([False] * 4, True), ([True, False, True, False], True), ([True] * 4, False)]:
        mask.copy_(torch.tensor(values, device="npu"))
        enabled.fill_(record_on)
        load.copy_(initial)
        graph.replay()
        expected = initial.cpu()
        if record_on:
            expected[4:] += torch.tensor(values, dtype=torch.int32)
        torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)
