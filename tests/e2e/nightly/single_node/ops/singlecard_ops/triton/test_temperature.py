import random

import pytest
import torch

from vllm_ascend.worker.v2.sample.gumbel import apply_temperature

# Common vocab sizes from mainstream models
VOCAB_SIZES = [
    32000,  # LLaMA / LLaMA2 / Mistral
    50257,  # GPT-2
    65024,  # ChatGLM
    128256,  # LLaMA3
    151936,  # Qwen2
]


def torch_apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """Pure PyTorch reference implementation of temperature scaling.

    Args:
        logits: Tensor of shape (num_tokens, vocab_size) containing the logits.
        expanded_idx_mapping: Tensor containing the mapping from token index
            to request index of tensor temperature.
        temperature: Tensor containing the temperature value for each request.
    """
    for token_idx in range(logits.shape[0]):
        req_state_idx = expanded_idx_mapping[token_idx].item()
        temp = temperature[req_state_idx].item()
        if temp == 0.0 or temp == 1.0:
            continue
        logits[token_idx] = logits[token_idx].float() / temp


@pytest.mark.parametrize(
    "num_tokens, vocab_size",
    [(random.randint(1, 64), vocab_size) for vocab_size in VOCAB_SIZES],
)
def test_temperature_kernel(num_tokens, vocab_size):
    """
    Test the Triton _temperature_kernel against a pure PyTorch reference.

    The kernel divides logits by per-request temperature values. Tokens
    whose temperature is 0.0 or 1.0 are skipped (logits unchanged).

    Args:
        num_tokens: Number of tokens (rows) in the logits tensor.
        vocab_size: Vocabulary size (columns) in the logits tensor.
    """
    torch.manual_seed(42)

    # Build input tensors
    logits_triton = torch.randn((num_tokens, vocab_size), dtype=torch.float32).npu()
    logits_ref = logits_triton.clone()

    num_requests = num_tokens
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32).npu()

    # Include edge cases: 0.0 and 1.0 should leave logits unchanged
    temperature = torch.rand(num_requests, dtype=torch.float32).npu()
    temperature = temperature * 1.8 + 0.2  # range [0.2, 2.0]
    if num_requests >= 3:
        temperature[0] = 0.0
        temperature[1] = 1.0

    # ========== Run Triton kernel ==========
    apply_temperature(logits_triton, expanded_idx_mapping, temperature)

    # ========== Run PyTorch reference ==========
    torch_apply_temperature(logits_ref, expanded_idx_mapping, temperature)

    # ========== Verify results ==========
    assert torch.allclose(logits_triton, logits_ref, atol=1e-4, rtol=1e-5), (
        f"Triton temperature kernel output differs from torch reference.\n"
        f"Max diff: {torch.max(torch.abs(logits_triton - logits_ref))}\n"
        f"Mean diff: {torch.mean(torch.abs(logits_triton - logits_ref).float())}"
    )
