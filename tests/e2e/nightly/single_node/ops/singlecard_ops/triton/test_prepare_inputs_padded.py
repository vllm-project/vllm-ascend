import gc

import pytest
import torch

from vllm_ascend.ops.triton.spec_decode.utils import (
    prepare_inputs_padded_kernel,
    prepare_next_token_padded_kernel,
)


def prepare_inputs_padded_ref(
    cu_num_draft_tokens,
    valid_sampled_tokens_count,
    query_start_loc,
):
    num_draft_tokens = torch.cat(
        [
            cu_num_draft_tokens[0:1],
            cu_num_draft_tokens[1:] - cu_num_draft_tokens[:-1],
        ]
    )

    num_rejected_tokens = torch.where(
        num_draft_tokens > 0,
        num_draft_tokens + 1 - valid_sampled_tokens_count,
        torch.zeros_like(num_draft_tokens),
    )

    token_indices_to_sample = query_start_loc[1:] - 1 - num_rejected_tokens

    return token_indices_to_sample.to(torch.int32)


@pytest.mark.parametrize("num_reqs", [1, 7, 32, 128, 2048])
def test_prepare_inputs_padded(num_reqs):
    device = "npu"
    torch.manual_seed(0)

    draft_lens = torch.randint(1, 6, (num_reqs,), device=device, dtype=torch.int32)

    cu_num_draft_tokens = torch.cumsum(draft_lens, dim=0).to(torch.int32)

    valid_sampled_tokens_count = torch.zeros_like(draft_lens)
    for i in range(num_reqs):
        valid_sampled_tokens_count[i] = torch.randint(0, draft_lens[i] + 2, (1,)).item()

    seq_lens = draft_lens + 1
    query_start_loc = torch.zeros(num_reqs + 1, device=device, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(seq_lens, dim=0)

    # Run PyTorch reference
    out_ref = prepare_inputs_padded_ref(cu_num_draft_tokens, valid_sampled_tokens_count, query_start_loc)

    # Run Triton kernel
    out_tri = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected_tokens = torch.empty(num_reqs, dtype=torch.int32, device=device)
    prepare_inputs_padded_kernel[(num_reqs,)](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        out_tri,
        num_rejected_tokens,
        num_reqs,
    )

    torch.testing.assert_close(out_tri, out_ref)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_prepare_next_token_padded():
    sampled_token_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, -1, -1, -1],
            [30, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device="npu",
    )
    backup_token_ids = torch.tensor([40, 41, 42, 43], dtype=torch.int32, device="npu")
    next_token_ids = torch.empty(4, dtype=torch.int32, device="npu")
    valid_sampled_tokens_count = torch.empty(4, dtype=torch.int32, device="npu")

    prepare_next_token_padded_kernel[(4,)](
        sampled_token_ids,
        backup_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        128000,
        6,
        4,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=8,
    )

    torch.testing.assert_close(
        next_token_ids,
        torch.tensor([15, 22, 30, 43], dtype=torch.int32, device="npu"),
    )
    torch.testing.assert_close(
        valid_sampled_tokens_count,
        torch.tensor([6, 3, 1, 0], dtype=torch.int32, device="npu"),
    )
