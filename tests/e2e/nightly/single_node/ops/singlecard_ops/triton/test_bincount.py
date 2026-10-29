import pytest
import torch

from vllm_ascend.worker.v2.sample.penalties import apply_penalties, bincount

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


def _make_inputs(idx_mapping, prompt_tokens, output_tokens, vocab_size, seed=42):
    """Build kernel inputs; token ids are drawn over the whole vocabulary."""
    torch.manual_seed(seed)
    max_model_len = prompt_tokens + output_tokens + 1
    expanded_idx_mapping = torch.tensor(idx_mapping, dtype=torch.int32).npu()
    all_token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(MAX_NUM_REQS, max_model_len),
        dtype=torch.int32,
    ).npu()
    prompt_len = torch.full((MAX_NUM_REQS,), prompt_tokens, dtype=torch.int32).npu()
    prefill_len = torch.full((MAX_NUM_REQS,), prompt_tokens + output_tokens, dtype=torch.int32).npu()
    num_words = (vocab_size + 31) // 32
    return expanded_idx_mapping, all_token_ids, prompt_len, prefill_len, num_words


@pytest.mark.parametrize(
    "idx_mapping, prompt_tokens, output_tokens",
    [
        # Single request placed at the end of the batch: token_idx 0 must not be
        # confused with req_state_idx 63 anywhere in the kernels.
        ([63], 8, 4),
        # Several requests in a scrambled, non-contiguous order.
        ([63, 7, 31, 2], 2048, 32),
        # Prompt spanning several BLOCK_SIZE=1024 blocks.
        ([17], 4096, 16),
        # No generated tokens yet, so output_bin_counts must stay all zero.
        ([40, 9], 1024, 0),
    ],
)
@pytest.mark.parametrize(
    "vocab_size",
    [
        151936,  # divisible by 32
        151000,  # not divisible by 32: exercises the tail-word masking
        5121,  # small and not divisible by 32
    ],
)
def test_bincount(idx_mapping, prompt_tokens, output_tokens, vocab_size):
    """The packed prompt bitmask and the output token counts must match torch."""
    expanded_idx_mapping, all_token_ids, prompt_len, prefill_len, num_words = _make_inputs(
        idx_mapping, prompt_tokens, output_tokens, vocab_size
    )

    # Poison the outputs so a row or word that is never written is caught
    # instead of silently matching a zero-initialised reference.
    prompt_bin_mask = torch.full((MAX_NUM_REQS, num_words), -1, dtype=torch.int32).npu()
    output_bin_counts = torch.full((MAX_NUM_REQS, vocab_size), -1, dtype=torch.int32).npu()
    ref_prompt_bin_mask = torch.full((MAX_NUM_REQS, num_words), -1, dtype=torch.int32).npu()
    ref_output_bin_counts = torch.full((MAX_NUM_REQS, vocab_size), -1, dtype=torch.int32).npu()

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

    touched = expanded_idx_mapping.long()
    assert torch.equal(prompt_bin_mask[touched], ref_prompt_bin_mask[touched]), (
        f"prompt_bin_mask differs from torch reference for rows {idx_mapping} at vocab_size={vocab_size}"
    )
    assert torch.equal(output_bin_counts[touched], ref_output_bin_counts[touched]), (
        f"output_bin_counts differs from torch reference for rows {idx_mapping}"
    )
    # Rows outside expanded_idx_mapping must be left untouched.
    untouched = torch.ones(MAX_NUM_REQS, dtype=torch.bool)
    untouched[torch.tensor(idx_mapping)] = False
    untouched = untouched.npu()
    assert torch.all(prompt_bin_mask[untouched] == -1), "bincount wrote outside expanded_idx_mapping"
    assert torch.all(output_bin_counts[untouched] == -1), "bincount wrote outside expanded_idx_mapping"


def torch_apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
):
    vocab_size = logits.shape[1]
    bit_index = torch.arange(vocab_size, device=logits.device)
    for token_idx in range(expanded_idx_mapping.shape[0]):
        req = expanded_idx_mapping[token_idx].item()
        rep = repetition_penalty[req].item()
        freq = frequency_penalty[req].item()
        presence = presence_penalty[req].item()
        if rep == 1.0 and freq == 0.0 and presence == 0.0:
            continue
        counts = output_bin_counts[req].to(torch.float32)
        out_mask = counts != 0
        row = logits[token_idx].to(torch.float32)
        if rep != 1.0:
            words = prompt_bin_mask[req][bit_index // 32]
            prompt_mask = ((words >> (bit_index % 32)) & 1) != 0
            scale = torch.where(prompt_mask | out_mask, rep, 1.0)
            row = row * torch.where(row > 0, 1.0 / scale, scale)
        row = row - freq * counts
        row = row - presence * out_mask.to(torch.float32)
        logits[token_idx] = row


@pytest.mark.parametrize("vocab_size", [5121, 151936])
def test_bincount_then_apply_penalties(vocab_size):
    """bincount feeds apply_penalties: check the chained result, not just bincount."""
    idx_mapping = [63, 7, 31, 2]
    prompt_tokens, output_tokens = 512, 24
    expanded_idx_mapping, all_token_ids, prompt_len, prefill_len, num_words = _make_inputs(
        idx_mapping, prompt_tokens, output_tokens, vocab_size, seed=7
    )

    prompt_bin_mask = torch.full((MAX_NUM_REQS, num_words), -1, dtype=torch.int32).npu()
    output_bin_counts = torch.full((MAX_NUM_REQS, vocab_size), -1, dtype=torch.int32).npu()
    ref_prompt_bin_mask = torch.full((MAX_NUM_REQS, num_words), -1, dtype=torch.int32).npu()
    ref_output_bin_counts = torch.full((MAX_NUM_REQS, vocab_size), -1, dtype=torch.int32).npu()

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

    num_tokens = len(idx_mapping)
    torch.manual_seed(11)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32).npu()
    ref_logits = logits.clone()
    # No draft tokens, so every logits row is the last position of its request.
    expanded_local_pos = torch.zeros(num_tokens, dtype=torch.int32).npu()
    token_ids = all_token_ids[expanded_idx_mapping.long()]

    repetition_penalty = torch.full((MAX_NUM_REQS,), 1.3, dtype=torch.float32).npu()
    frequency_penalty = torch.full((MAX_NUM_REQS,), 0.4, dtype=torch.float32).npu()
    presence_penalty = torch.full((MAX_NUM_REQS,), 0.2, dtype=torch.float32).npu()

    apply_penalties(
        logits,
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    )
    torch_apply_penalties(
        ref_logits,
        expanded_idx_mapping,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        ref_prompt_bin_mask,
        ref_output_bin_counts,
    )

    torch.testing.assert_close(logits, ref_logits, rtol=1e-5, atol=1e-5)
