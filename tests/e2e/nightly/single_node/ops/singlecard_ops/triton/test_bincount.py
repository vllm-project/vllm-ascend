import pytest
import torch

from vllm_ascend.worker.v2.sample.penalties import bincount

VOCAB_SIZE = 151936
MAX_NUM_REQS = 64


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


@pytest.mark.parametrize(
    "num_reqs, prompt_tokens, output_tokens",
    [
        # Single short request: the common case when one request is admitted.
        (1, 8, 4),
        # Prompt spanning several BLOCK_SIZE=1024 blocks.
        (1, 4096, 16),
        # Several requests admitted in the same step.
        (4, 2048, 32),
        # No generated tokens yet, so output_bin_counts must stay all zero.
        (2, 1024, 0),
    ],
)
def test_bincount(num_reqs, prompt_tokens, output_tokens):
    """The packed prompt bitmask and the output token counts must match torch."""
    torch.manual_seed(42)

    max_model_len = prompt_tokens + output_tokens + 1
    expanded_idx_mapping = torch.arange(num_reqs, dtype=torch.int32).npu()
    all_token_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(MAX_NUM_REQS, max_model_len),
        dtype=torch.int32,
    ).npu()

    prompt_len = torch.full((MAX_NUM_REQS,), prompt_tokens, dtype=torch.int32).npu()
    prefill_len = torch.full((MAX_NUM_REQS,), prompt_tokens + output_tokens, dtype=torch.int32).npu()

    num_words = (VOCAB_SIZE + 31) // 32
    prompt_bin_mask = torch.zeros(size=(MAX_NUM_REQS, num_words), dtype=torch.int32).npu()
    output_bin_counts = torch.zeros(size=(MAX_NUM_REQS, VOCAB_SIZE), dtype=torch.int32).npu()

    ref_prompt_bin_mask = torch.zeros(size=(MAX_NUM_REQS, num_words), dtype=torch.int32).npu()
    ref_output_bin_counts = torch.zeros(size=(MAX_NUM_REQS, VOCAB_SIZE), dtype=torch.int32).npu()

    bincount(
        expanded_idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        prompt_tokens + output_tokens,
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
        "prompt_bin_mask triton output differs from torch reference at "
        f"rows {torch.nonzero((prompt_bin_mask != ref_prompt_bin_mask).any(dim=1)).flatten().tolist()[:8]}"
    )

    assert torch.equal(output_bin_counts, ref_output_bin_counts), (
        "output_bin_counts triton output differs from torch reference at "
        f"rows {torch.nonzero((output_bin_counts != ref_output_bin_counts).any(dim=1)).flatten().tolist()[:8]}"
    )
