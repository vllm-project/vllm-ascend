# SPDX-License-Identifier: Apache-2.0
# Kernel source: vllm_ascend/ops/triton/v2/sample/fill_logprob_token_idx.py
# Coverage: _fill_logprob_token_ids_kernel
"""
Precision test for _fill_logprob_token_ids_kernel.

Kernel signature:
    _fill_logprob_token_ids_kernel(
        out_token_ids_ptr,           # [batch_size, 1 + num_cols] output token IDs
        out_token_ids_stride,        # stride(0) of out_token_ids
        out_valid_mask_ptr,          # [batch_size, 1 + num_cols] validity mask
        out_valid_mask_stride,       # stride(0) of valid_mask
        sampled_token_ids_ptr,       # [batch_size] sampled token IDs
        topk_indices_ptr,            # int32 [batch_size, NUM_TOPK] top-k indices
        topk_indices_stride,         # stride(0) of topk_indices
        expanded_idx_mapping_ptr,    # [batch_size] -> req_state_idx
        num_per_req_token_ids_ptr,   # [max_num_reqs] count of custom tokens
        per_req_token_ids_ptr,       # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
        per_req_token_ids_stride,    # stride(0) of per_req_token_ids
        NUM_TOPK: tl.constexpr,
        PADDED_COLS: tl.constexpr,
    )

Fills logprob token IDs matrix:
- Column 0: always the sampled token (always valid).
- Remaining columns: per-request custom token IDs if num_custom > 0,
  otherwise top-k indices.
"""

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample.fill_logprob_token_idx import _fill_logprob_token_ids_kernel

MAX_LOGPROB_TOKEN_IDS = 128


def _fill_logprob_token_ids_ref(
    batch_size,
    sampled_token_ids,
    topk_indices,
    expanded_idx_mapping,
    num_per_req_token_ids,
    per_req_token_ids,
    NUM_TOPK,
    PADDED_COLS,
):
    """CPU reference for _fill_logprob_token_ids_kernel."""
    out_token_ids = torch.zeros(batch_size, 1 + PADDED_COLS, dtype=torch.int64)
    out_valid_mask = torch.zeros(batch_size, 1 + PADDED_COLS, dtype=torch.bool)

    for b in range(batch_size):
        # Column 0: sampled token
        out_token_ids[b, 0] = sampled_token_ids[b].item()
        out_valid_mask[b, 0] = True

        req_state_idx = expanded_idx_mapping[b].item()
        num_custom = num_per_req_token_ids[req_state_idx].item()

        if num_custom > 0:
            # Override topk with per-request custom tokens
            for col in range(min(num_custom, PADDED_COLS)):
                out_token_ids[b, 1 + col] = per_req_token_ids[req_state_idx, col].item()
                out_valid_mask[b, 1 + col] = True
        else:
            # Fill with topk indices
            for col in range(min(NUM_TOPK, PADDED_COLS)):
                out_token_ids[b, 1 + col] = topk_indices[b, col].item()
                out_valid_mask[b, 1 + col] = True

    return out_token_ids, out_valid_mask


class TestFillLogprobTokenIdsKernel:
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("topk", [0, 3, 5])
    def test_custom_token_ids(self, batch_size, topk):
        """Test with per-request custom token IDs."""
        init_device_properties_triton()
        torch.manual_seed(42)
        device = "npu"

        num_reqs = 4
        PADDED_COLS = 16
        NUM_TOPK = topk

        sampled_token_ids = torch.randint(0, 1000, (batch_size,), dtype=torch.int64, device=device)
        # Production converts top-k IDs to int32 before launching this kernel.
        topk_indices = torch.randint(
            0,
            1000,
            (batch_size, max(NUM_TOPK, 1)),
            dtype=torch.int32,
            device=device,
        )
        # Deterministically include request 0, which owns the custom token IDs.
        expanded_idx_mapping = torch.arange(batch_size, dtype=torch.int32, device=device) % num_reqs

        # Some requests get custom token IDs, others don't
        num_per_req_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        per_req_token_ids = torch.zeros(num_reqs, MAX_LOGPROB_TOKEN_IDS, dtype=torch.int32, device=device)
        # Request 0 has custom tokens
        num_per_req_token_ids[0] = 3
        per_req_token_ids[0, 0] = 100
        per_req_token_ids[0, 1] = 200
        per_req_token_ids[0, 2] = 300

        out_token_ids = torch.zeros(batch_size, 1 + PADDED_COLS, dtype=torch.int64, device=device)
        out_valid_mask = torch.zeros(batch_size, 1 + PADDED_COLS, dtype=torch.bool, device=device)

        _fill_logprob_token_ids_kernel[(batch_size,)](
            out_token_ids,
            out_token_ids.stride(0),
            out_valid_mask,
            out_valid_mask.stride(0),
            sampled_token_ids,
            topk_indices,
            topk_indices.stride(0),
            expanded_idx_mapping,
            num_per_req_token_ids,
            per_req_token_ids,
            per_req_token_ids.stride(0),
            NUM_TOPK=NUM_TOPK,
            PADDED_COLS=PADDED_COLS,
        )
        torch.npu.synchronize()

        expected_ids, expected_mask = _fill_logprob_token_ids_ref(
            batch_size,
            sampled_token_ids.cpu(),
            topk_indices.cpu(),
            expanded_idx_mapping.cpu(),
            num_per_req_token_ids.cpu(),
            per_req_token_ids.cpu(),
            NUM_TOPK,
            PADDED_COLS,
        )

        assert torch.equal(out_token_ids.cpu(), expected_ids), (
            f"Token IDs do not match for batch_size={batch_size}, topk={topk}."
        )
        assert torch.equal(out_valid_mask.cpu(), expected_mask), (
            f"Valid mask do not match for batch_size={batch_size}, topk={topk}."
        )

    def test_no_custom_no_topk(self):
        """When both custom and topk are empty, only sampled token is written."""
        init_device_properties_triton()
        torch.manual_seed(42)
        device = "npu"

        batch_size = 2
        NUM_TOPK = 0
        PADDED_COLS = 8
        sampled_token_ids = torch.tensor([42, 77], dtype=torch.int64, device=device)
        topk_indices = torch.zeros(batch_size, 1, dtype=torch.int32, device=device)
        expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
        num_per_req_token_ids = torch.zeros(2, dtype=torch.int32, device=device)
        per_req_token_ids = torch.zeros(2, MAX_LOGPROB_TOKEN_IDS, dtype=torch.int32, device=device)

        out_token_ids = torch.zeros(batch_size, 1 + PADDED_COLS, dtype=torch.int64, device=device)
        out_valid_mask = torch.zeros(batch_size, 1 + PADDED_COLS, dtype=torch.bool, device=device)

        _fill_logprob_token_ids_kernel[(batch_size,)](
            out_token_ids,
            out_token_ids.stride(0),
            out_valid_mask,
            out_valid_mask.stride(0),
            sampled_token_ids,
            topk_indices,
            topk_indices.stride(0),
            expanded_idx_mapping,
            num_per_req_token_ids,
            per_req_token_ids,
            per_req_token_ids.stride(0),
            NUM_TOPK=NUM_TOPK,
            PADDED_COLS=PADDED_COLS,
        )
        torch.npu.synchronize()

        expected_ids, expected_mask = _fill_logprob_token_ids_ref(
            batch_size,
            sampled_token_ids.cpu(),
            topk_indices.cpu(),
            expanded_idx_mapping.cpu(),
            num_per_req_token_ids.cpu(),
            per_req_token_ids.cpu(),
            NUM_TOPK,
            PADDED_COLS,
        )

        torch.testing.assert_close(out_token_ids.cpu(), expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(out_valid_mask.cpu(), expected_mask, rtol=0, atol=0)
