# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU regression for vLLM #58784's prompt-tail draft placeholders."""

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.spec_decode.gather_draft_sampled import gather_draft_sampled
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


def test_gather_rejects_only_unproposed_draft_slots():
    init_device_properties_triton()
    rows_per_req = 434  # Cross the kernel's 1024-row block boundary.
    num_reqs = 3
    num_logits = num_reqs * rows_per_req
    logits_indices = torch.arange(3, num_logits + 3, dtype=torch.int64, device="npu")
    input_ids = torch.arange(num_logits + 3, dtype=torch.int64, device="npu") % 31
    input_ids[4] = 0  # A zero-filled draft slot must be treated as a placeholder.
    positions = torch.arange(num_logits + 3, dtype=torch.int64, device="npu") + 100
    local_pos = torch.arange(rows_per_req, dtype=torch.int32, device="npu").repeat(num_reqs)
    req_state_idx = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu").repeat_interleave(rows_per_req)
    # Requests 0 and 2 begin within prefill; request 1 begins after it.
    first_positions = [103 + i * rows_per_req for i in range(num_reqs)]
    prefill_len = torch.tensor(
        [first_positions[1], first_positions[2] + 1, first_positions[0] + 1],
        dtype=torch.int32,
        device="npu",
    )
    original_input_ids = input_ids.clone()

    draft_sampled, pos = gather_draft_sampled(
        input_ids, positions, logits_indices, req_state_idx, local_pos, prefill_len
    )
    expected = input_ids[logits_indices].clone()
    expected[1:rows_per_req] = -1
    expected[2 * rows_per_req + 1 :] = -1
    torch.testing.assert_close(draft_sampled, expected, rtol=0, atol=0)
    torch.testing.assert_close(pos, positions[logits_indices], rtol=0, atol=0)
    torch.testing.assert_close(input_ids, original_input_ids, rtol=0, atol=0)

    # The same draft slots become valid after the request passes prefill.
    prefill_len = torch.tensor(
        [first_positions[1], first_positions[2], first_positions[0]],
        dtype=torch.int32,
        device="npu",
    )
    draft_sampled, _ = gather_draft_sampled(input_ids, positions, logits_indices, req_state_idx, local_pos, prefill_len)
    torch.testing.assert_close(draft_sampled, input_ids[logits_indices], rtol=0, atol=0)


@pytest.mark.parametrize("first_token", [0, 5])
def test_unproposed_drafts_are_rejected_by_npu_sampler(first_token):
    """Check the gather result reaches rejection sampling without token-0 acceptance."""
    init_device_properties_triton()
    input_ids = torch.tensor([9, 0, 0, 8, 0, 0], dtype=torch.int64, device="npu")
    positions = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64, device="npu")
    logits_indices = torch.arange(6, dtype=torch.int64, device="npu")
    expanded_idx_mapping = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device="npu")
    expanded_local_pos = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device="npu")
    draft_sampled, pos = gather_draft_sampled(
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        torch.tensor([11, 20], dtype=torch.int32, device="npu"),
    )
    target_logits = torch.full((6, 64), -100.0, device="npu")
    target_logits[:, 0] = 0.0
    if first_token != 0:
        target_logits[0, 0] = -100.0
        target_logits[0, first_token] = 0.0
    sampled, num_sampled = rejection_sample(
        target_logits=target_logits,
        draft_logits=torch.zeros((2, 2, 64), device="npu"),
        draft_sampled=draft_sampled,
        cu_num_logits=torch.tensor([0, 3, 6], dtype=torch.int32, device="npu"),
        pos=pos,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=torch.ones(2, device="npu"),
        seed=torch.arange(2, dtype=torch.int64, device="npu"),
        num_speculative_steps=2,
    )
    assert num_sampled.tolist() == [1, 3]
    assert sampled[0, 0].item() == first_token
    assert sampled[1, :3].tolist() == [0, 0, 0]
