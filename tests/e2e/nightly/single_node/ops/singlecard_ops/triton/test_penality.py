import gc
import pytest
import torch

from vllm_ascend.worker.v2.sample.penalties import apply_penalties

DTYPES = [torch.bfloat16, torch.float16]
NUM_TOKENS = [2, 4, 8]
VOCAB_SIZE = [151936]
NUM_STATUS = [1, 4, 8, 16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
NUM_SPECULATIVE_TOKENS = [0, 1, 3]
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


def pytorch_apply_penalties(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    num_speculative_tokens: int,
) -> torch.Tensor:
    """
    Pytorch equivalent implementation
    """
    num_tokens, vocab_size = logits.shape
    device = logits.device
    dtype = logits.dtype

    logits_float = logits.float()

    num_status = prompt_bin_mask.shape[0]
    num_packed = prompt_bin_mask.shape[1]

    prompt_masks_unpacked = torch.zeros(
        num_status, vocab_size, dtype=torch.bool,
        device=device
    )

    for state_idx in range(num_status):
        for packed_idx in range(num_packed):
            packed_val = prompt_bin_mask[state_idx, packed_idx].item()
            start_idx = packed_idx * 32
            end_idx = min(start_idx + 32, vocab_size)

            for bit_pos in range(end_idx - start_idx):
                if (packed_val >> bit_pos) & 1:
                    prompt_masks_unpacked[state_idx, start_idx + bit_pos] = True

    for token_idx in range(num_tokens):
        req_state_idx = idx_mapping[token_idx].item()
        
        rep_penalty = repetition_penalty[req_state_idx].item()
        freq_penalty = frequency_penalty[req_state_idx].item()
        pres_penalty = presence_penalty[req_state_idx].item()

        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty

        if not use_penalty:
            continue

        current_prompt_mask = prompt_masks_unpacked[req_state_idx]
        base_output_counts = output_bin_counts[req_state_idx]
        
        # Compute cumulative draft counts
        pos = expanded_local_pos[token_idx].item()
        start_idx_in_batch = token_idx - pos
        draft_counts = torch.zeros(vocab_size, device=device, dtype=torch.int32)
        
        for prev_pos in range(num_speculative_tokens):
            if prev_pos < pos:
                prev_token = token_ids[start_idx_in_batch + prev_pos + 1].item()
                draft_counts[prev_token] += 1
        
        # Total counts = base output counts + cumulative draft counts
        total_output_counts = base_output_counts + draft_counts
        output_bin_mask = total_output_counts > 0

        if use_rep_penalty:
            scale = torch.ones(vocab_size, device=device)
            mask = current_prompt_mask | output_bin_mask
            scale[mask] = rep_penalty

            pos_mask = logits_float[token_idx] > 0
            scale_factor = torch.where(pos_mask, 1.0 / scale, scale)
            logits_float[token_idx] *= scale_factor

        if use_freq_penalty:
            logits_float[token_idx] -= freq_penalty * total_output_counts.float()

        if use_pres_penalty:
            logits_float[token_idx] -= pres_penalty * output_bin_mask.float()

    return logits_float.to(dtype)


def create_test_data(
    num_tokens: int = 8,
    vocab_size: int = 51200,
    num_status: int = 16,
    num_speculative_tokens: int = 3,
    device: str = "npu",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """Create test data for penalties"""
    torch.manual_seed(seed)

    logits = torch.randn(num_tokens, vocab_size, device=device, dtype=dtype)

    repetition_penalty = torch.ones(num_status, device=device, dtype=torch.float32)
    for i in range(num_status):
        if torch.rand(1) > 0.3:
            repetition_penalty[i] = torch.rand(1, device=device).item() * 0.8 + 0.6

    frequency_penalty = torch.zeros(num_status, device=device, dtype=torch.float32)
    for i in range(num_status):
        if torch.rand(1) > 0.5:
            frequency_penalty[i] = torch.rand(1, device=device).item() * 0.2

    presence_penalty = torch.zeros(num_status, device=device, dtype=torch.float32)
    for i in range(num_status):
        if torch.rand(1) > 0.5:
            presence_penalty[i] = torch.rand(1, device=device).item() * 0.2

    idx_mapping = torch.randint(
        0, num_status, (num_tokens,), device=device,
        dtype=torch.int32
    )
    
    # Create token_ids for speculative decoding
    token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int32)
    
    # Create expanded_local_pos (position within speculative decoding window)
    expanded_local_pos = torch.zeros(num_tokens, device=device, dtype=torch.int32)
    for i in range(num_tokens):
        expanded_local_pos[i] = torch.randint(
            0, num_speculative_tokens + 1, (1,)
        ).item()

    num_packed = (vocab_size + 31) // 32
    prompt_bin_mask = torch.zeros(num_status, num_packed, device=device, dtype=torch.int32)

    for state_idx in range(num_status):
        num_tokens_in_prompt = max(1, vocab_size // 20)
        prompt_tokens = torch.randperm(vocab_size)[:num_tokens_in_prompt]

        for token_id in prompt_tokens:
            packed_idx = token_id // 32
            bit_pos = token_id % 32
            prompt_bin_mask[state_idx, packed_idx] |= (1 << bit_pos)

    output_bin_counts = torch.zeros(num_status, vocab_size, device=device, dtype=torch.int32)
    for state_idx in range(num_status):
        num_output_tokens = max(1, vocab_size // 20)
        output_tokens = torch.randint(0, vocab_size,
                                      (num_output_tokens, ))
        counts = torch.randint(1, 10, (num_output_tokens,))

        for token, count in zip(output_tokens, counts):
            output_bin_counts[state_idx, token] = count

    return (
        logits,
        idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
        num_speculative_tokens,
    )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("num_status", NUM_STATUS)
@pytest.mark.parametrize("num_speculative_tokens", NUM_SPECULATIVE_TOKENS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_apply_penalties(
    num_tokens,
    vocab_size,
    num_status,
    num_speculative_tokens,
    dtype,
    seed,
    device
):
    (
        logits_triton,
        idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
        num_spec_tokens,
    ) = create_test_data(
        num_tokens=num_tokens,
        vocab_size=vocab_size,
        num_status=num_status,
        num_speculative_tokens=num_speculative_tokens,
        device=device,
        dtype=dtype,
        seed=seed
    )

    logits_pytorch = logits_triton.clone()

    apply_penalties(
        logits_triton,
        idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
        num_spec_tokens,
    )

    logits_pytorch_result = pytorch_apply_penalties(
        logits_pytorch,
        idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
        num_spec_tokens,
    )

    atol = DEFAULT_ATOL
    rtol = DEFAULT_RTOL
    if dtype == torch.bfloat16:
        atol = 1e-02
        rtol = 1e-02
    assert torch.allclose(logits_triton, logits_pytorch_result, atol=atol, rtol=rtol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
