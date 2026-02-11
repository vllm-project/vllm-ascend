import pytest
import torch

from vllm_ascend.worker.v2.sample.penalties import apply_penalties_and_temperature

DTYPES = [torch.bfloat16, torch.float16]
NUM_REQS = [2, 4, 8]
VOCAB_SIZE = [151936]
NUM_STATUS = [1, 4, 8, 16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


class SamplingMetadata:
    def __init__(self,
                 repetition_penalty: torch.Tensor,
                 frequency_penalty: torch.Tensor,
                 presence_penalty: torch.Tensor,
                 temperature: torch.Tensor,
                 idx_mapping: torch.Tensor,
                 prompt_bin_mask: torch.Tensor,
                 output_bin_counts: torch.Tensor):
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.idx_mapping = idx_mapping
        self.prompt_bin_mask = prompt_bin_mask
        self.output_bin_counts = output_bin_counts


def pytorch_apply_penalties_and_temperature(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """
    Pytorch equivalent implementation
    """
    num_reqs, vocab_size = logits.shape
    device = logits.device
    dtype = logits.dtype

    logits_float = logits.float()

    repetition_penalty = sampling_metadata.repetition_penalty
    frequency_penalty = sampling_metadata.frequency_penalty
    presence_penalty = sampling_metadata.presence_penalty
    temperature = sampling_metadata.temperature
    idx_mapping = sampling_metadata.idx_mapping
    prompt_bin_mask = sampling_metadata.prompt_bin_mask
    output_bin_counts = sampling_metadata.output_bin_counts

    temperature = torch.where(temperature == 0.0, torch.ones_like(temperature), temperature)

    num_status = prompt_bin_mask.shape[0]
    num_packed = prompt_bin_mask.shape[1]

    prompt_masks_unpacked = torch.zeros(num_status, vocab_size, dtype=torch.bool, device=device)

    for state_idx in range(num_status):
        for packed_idx in range(num_packed):
            packed_val = prompt_bin_mask[state_idx, packed_idx].item()
            start_idx = packed_idx * 32
            end_idx = min(start_idx + 32, vocab_size)

            for bit_pos in range(end_idx - start_idx):
                if (packed_val >> bit_pos) & 1:
                    prompt_masks_unpacked[state_idx, start_idx + bit_pos] = True

    for batch_idx in range(num_reqs):
        req_state_idx = idx_mapping[batch_idx].item()

        rep_penalty = repetition_penalty[batch_idx].item()
        freq_penalty = frequency_penalty[batch_idx].item()
        pres_penalty = presence_penalty[batch_idx].item()
        temp = temperature[batch_idx].item()

        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        use_penalty = (use_rep_penalty or use_freq_penalty) or use_pres_penalty
        use_temperature = temp != 1.0

        if not (use_penalty or use_temperature):
            continue

        current_prompt_mask = prompt_masks_unpacked[req_state_idx]
        current_output_counts = output_bin_counts[req_state_idx]
        output_bin_mask = current_output_counts > 0

        if use_rep_penalty:
            scale = torch.ones(vocab_size, device=device)
            mask = current_prompt_mask | output_bin_mask
            scale[mask] = rep_penalty

            pos_mask = logits_float[batch_idx] > 0
            scale_factor = torch.where(pos_mask, 1.0 / scale, scale)
            logits_float[batch_idx] *= scale_factor

        if use_freq_penalty:
            logits_float[batch_idx] -= freq_penalty * current_output_counts.float()

        if use_pres_penalty:
            logits_float[batch_idx] -= pres_penalty * output_bin_mask.float()

        if use_temperature:
            logits_float[batch_idx] /= temp

    return logits_float.to(dtype)


def create_test_data(
    num_reqs: int = 8,
    vocab_size: int = 51200,
    num_status: int = 16,
    device: str = "npu",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """Create test data for penalties and temperature"""
    torch.manual_seed(seed)

    logits = torch.randn(num_reqs, vocab_size, device=device, dtype=dtype)

    repetiton_penalty = torch.ones(num_reqs, device=device, dtype=torch.float32)
    for i in range(num_reqs):
        if torch.rand(1) > 0.3:
            repetiton_penalty[i] = torch.rand(1, device=device).item() * 0.8 + 0.6

    frequency_penalty = torch.zeros(num_reqs, device=device, dtype=torch.float32)
    for i in range(num_reqs):
        if torch.rand(1) > 0.5:
            frequency_penalty[i] = torch.rand(1, device=device).item() * 0.2

    presence_penalty = torch.zeros(num_reqs, device=device, dtype=torch.float32)
    for i in range(num_reqs):
        if torch.rand(1) > 0.5:
            presence_penalty[i] = torch.rand(1, device=device).item() * 0.2

    temperature = torch.ones(num_reqs, device=device, dtype=torch.float32)
    for i in range(num_reqs):
        if torch.rand(1) > 0.2:
            presence_penalty[i] = torch.rand(1, device=device).item() * 1.8 + 0.2

    idx_mapping = torch.randint(0, num_status, (num_reqs,), device=device, dtype=torch.int32)

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
        output_tokens = torch.randint(0, vocab_size, (num_output_tokens, ))
        counts = torch.randint(1, 10, (num_output_tokens,))

        for token, count in zip(output_tokens, counts):
            output_bin_counts[state_idx, token] = count

    sampling_metadata = SamplingMetadata(
        repetition_penalty=repetiton_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        temperature=temperature,
        idx_mapping=idx_mapping,
        prompt_bin_mask=prompt_bin_mask,
        output_bin_counts=output_bin_counts
    )

    return logits, sampling_metadata


@pytest.mark.parametrize("num_reqs", NUM_REQS)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZE)
@pytest.mark.parametrize("num_status", NUM_STATUS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_apply_penalties_and_temperature(
    num_reqs,
    vocab_size,
    num_status,
    dtype,
    seed,
    device
):
    logits_triton, sampling_metadata = create_test_data(
        num_reqs=num_reqs,
        vocab_size=vocab_size,
        num_status=num_status,
        device=device,
        dtype=dtype,
        seed=seed
    )

    logits_pytorch = logits_triton.clone()

    apply_penalties_and_temperature(logits_triton, sampling_metadata)

    logits_pytorch_result = pytorch_apply_penalties_and_temperature(logits_pytorch,
                                                                    sampling_metadata)

    atol = DEFAULT_ATOL
    rtol = DEFAULT_RTOL
    if dtype == torch.bfloat16:
        atol = 1e-02
        rtol = 1e-02
    assert torch.allclose(logits_triton, logits_pytorch_result, atol=atol, rtol=rtol)

