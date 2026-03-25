# SPDX-License-Identifier: Apache-2.0
# Minimal verification script for model_executor-style apply_penalties Triton migration.
# Compares vllm_ascend.ops.triton.penalty.apply_penalties_triton with PyTorch reference.
# Requires NPU and Triton-Ascend. Run in Ascend container for full validation.

import gc
import pytest
import torch

from vllm.model_executor.layers.utils import apply_penalties as ref_apply_penalties
from vllm.model_executor.layers.utils import get_token_bin_counts_and_mask


def _pt_apply_penalties(
    logits,
    prompt_tokens_tensor,
    output_tokens_tensor,
    presence_penalties,
    frequency_penalties,
    repetition_penalties,
):
    """Reference: model_executor apply_penalties (PyTorch)."""
    return ref_apply_penalties(
        logits,
        prompt_tokens_tensor,
        output_tokens_tensor,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


@pytest.mark.parametrize("num_seqs", [1, 8, 32, 128])
@pytest.mark.parametrize("vocab_size", [5120, 151936])
@pytest.mark.parametrize("max_prompt_len", [32, 128])
@pytest.mark.parametrize("max_output_len", [16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_apply_penalties_triton_vs_ref(
    num_seqs,
    vocab_size,
    max_prompt_len,
    max_output_len,
    dtype,
    device="npu",
    seed=42,
):
    """Compare Triton apply_penalties with PyTorch reference on NPU."""
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.ops.triton.penalty import apply_penalties_triton

    init_device_properties_triton()
    torch.manual_seed(seed)

    # Create test data
    logits_ref = torch.randn(num_seqs, vocab_size, device=device, dtype=dtype)
    logits_triton = logits_ref.clone()

    prompt_tokens = torch.randint(
        0, vocab_size, (num_seqs, max_prompt_len), device=device, dtype=torch.int64
    )
    # Pad with vocab_size (invalid token)
    pad_mask = torch.rand(num_seqs, max_prompt_len, device=device) > 0.7
    prompt_tokens[pad_mask] = vocab_size

    output_tokens = torch.randint(
        0, vocab_size, (num_seqs, max_output_len), device=device, dtype=torch.int64
    )
    pad_mask_out = torch.rand(num_seqs, max_output_len, device=device) > 0.8
    output_tokens[pad_mask_out] = vocab_size

    presence_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.2
    frequency_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.2
    repetition_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.4 + 1.0

    # Reference (PyTorch - runs on NPU via fallback)
    _pt_apply_penalties(
        logits_ref,
        prompt_tokens,
        output_tokens,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )

    # Triton
    apply_penalties_triton(
        logits_triton,
        prompt_tokens,
        output_tokens,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(logits_triton.float(), logits_ref.float(), atol=atol, rtol=rtol), (
        f"Max diff: {(logits_triton.float() - logits_ref.float()).abs().max().item()}"
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_get_token_bin_counts_and_mask_triton(
    num_seqs=2,
    vocab_size=1024,
    seq_len=64,
    device="npu",
    seed=42,
):
    """Compare get_token_bin_counts_and_mask Triton with PyTorch."""
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.ops.triton.bincount import get_token_bin_counts_and_mask_triton

    init_device_properties_triton()
    torch.manual_seed(seed)

    tokens = torch.randint(
        0, vocab_size + 1, (num_seqs, seq_len), device=device, dtype=torch.int64
    )

    ref_bin_counts, ref_mask = get_token_bin_counts_and_mask(
        tokens, vocab_size, num_seqs
    )
    triton_bin_counts, triton_mask = get_token_bin_counts_and_mask_triton(
        tokens, vocab_size, num_seqs
    )

    assert (triton_bin_counts.long() == ref_bin_counts).all(), "bin_counts mismatch"
    assert (triton_mask == ref_mask).all(), "mask mismatch"
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
