import gc

import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.spec_decode.utils import prepare_inputs_padded_kernel, prepare_next_token_ids_padded
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.spec_decode.llm_base_proposer import _PREPARE_INPUTS_BLOCK_SIZE as BLOCK_SIZE


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
    num_blocks_needed = triton.cdiv(num_reqs, BLOCK_SIZE)
    num_vector_core = get_vectorcore_num()
    grid_size = min(num_blocks_needed, num_vector_core)
    grid = (grid_size,)

    prepare_inputs_padded_kernel[grid](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        out_tri,
        num_rejected_tokens,
        num_reqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    torch.testing.assert_close(out_tri, out_ref)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_reqs", [1, 128])
def test_prepare_next_token_ids_padded(num_reqs):
    vocab_size, num_tokens = 1000, 5
    sampled = torch.randint(0, vocab_size, (num_reqs, num_tokens), device="npu")
    valid_lens = torch.arange(num_reqs, device="npu") % (num_tokens + 1)
    sampled[torch.arange(num_tokens, device="npu") >= valid_lens[:, None]] = -1
    backup = torch.arange(num_reqs, dtype=torch.int64, device="npu")
    discard = torch.arange(num_reqs, device="npu") % 3 == 0

    expected_count = ((sampled != -1) & (sampled < vocab_size)).sum(1)
    expected_count[discard] = 0
    expected = torch.where(
        expected_count > 0,
        sampled.gather(1, (expected_count - 1).clamp_min(0).unsqueeze(1)).squeeze(1),
        backup,
    )
    actual, actual_count = prepare_next_token_ids_padded(sampled, backup, discard, vocab_size)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_count, expected_count)
