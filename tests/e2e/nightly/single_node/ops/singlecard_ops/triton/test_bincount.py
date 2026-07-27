import pytest
import torch
from vllm.triton_utils import triton

from vllm_ascend.worker.v2.sample.penalties import _bincount_kernel


def torch_bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
):
    req_indices = expanded_idx_mapping
    prompt_bin_mask[req_indices] = 0
    output_bin_counts[req_indices] = 0

    for token_idx in range(expanded_idx_mapping.shape[0]):
        req_idx = expanded_idx_mapping[token_idx].item()

        p_len = prompt_len[req_idx].item()
        pref_len = prefill_len[req_idx].item()

        tokens = all_token_ids[req_idx]

        for pos in range(p_len):
            token = tokens[pos].item()
            bin_idx = token // 32
            bit_idx = token % 32
            prompt_bin_mask[req_idx, bin_idx] |= 1 << bit_idx

        for pos in range(p_len, pref_len):
            token = tokens[pos].item()
            output_bin_counts[req_idx, token] += 1


@pytest.mark.skip(reason="atomic_or operator hangs in current npu_ir version")
def test_bincount_kernel():
    """
    Compute the prompt binary mask and token bincount using the Triton kernel.

    Args:
        expanded_idx_mapping: Tensor containing the indices of requests to process.
        all_token_ids: Batch of input token IDs for all requests.
        prompt_len: Tensor storing the prompt length for each request.
        prefill_len: Tensor storing the prefill length for each request.
        prompt_bin_mask: Output binary mask tensor to mark prompt tokens.
        output_bin_counts: Output tensor to store token frequency counts.
        max_prefill_len: Maximum prefill length to limit kernel processing.
    """

    torch.manual_seed(42)

    expanded_idx_mapping = torch.tensor([63], dtype=torch.int32).npu()
    all_token_ids = torch.randint(
        low=0,
        high=10,
        size=(64, 40960),
        dtype=torch.int32,
    ).npu()

    prompt_len = torch.randint(
        low=0,
        high=10,
        size=(64,),
        dtype=torch.int32,
    ).npu()

    prefill_len = torch.randint(
        low=0,
        high=10,
        size=(64,),
        dtype=torch.int32,
    ).npu()

    prompt_bin_mask = torch.zeros(size=(64, 4748), dtype=torch.int32).npu()
    output_bin_counts = torch.zeros(size=(64, 151936), dtype=torch.int32).npu()

    ref_prompt_bin_mask = torch.zeros(size=(64, 4748), dtype=torch.int32).npu()
    ref_output_bin_counts = torch.zeros(size=(64, 151936), dtype=torch.int32).npu()

    max_prefill_len = 10

    prompt_bin_mask[expanded_idx_mapping] = 0
    output_bin_counts[expanded_idx_mapping] = 0
    num_tokens = expanded_idx_mapping.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)

    _bincount_kernel[(num_tokens, num_blocks)](
        expanded_idx_mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    torch_bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        ref_prompt_bin_mask,
        ref_output_bin_counts,
    )

    # ========== Verify results ==========
    assert torch.equal(prompt_bin_mask, ref_prompt_bin_mask), (
        f"prompt_bin_mask triton output differs from torch reference.\n"
        f"Max diff: {torch.max(torch.abs(prompt_bin_mask - ref_prompt_bin_mask))}\n"
        f"Mean diff: {torch.mean(torch.abs(prompt_bin_mask - ref_prompt_bin_mask))}"
    )

    assert torch.equal(output_bin_counts, ref_output_bin_counts), (
        f"output_bin_counts triton output differs from torch reference.\n"
        f"Max diff: {torch.max(torch.abs(output_bin_counts - ref_output_bin_counts))}\n"
        f"Mean diff: {torch.mean(torch.abs(output_bin_counts - ref_output_bin_counts))}"
    )
