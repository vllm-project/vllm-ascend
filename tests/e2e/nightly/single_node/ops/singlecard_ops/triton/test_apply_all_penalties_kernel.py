# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Compare apply_all_penalties_kernel with a PyTorch OpenAI-style penalty reference.
# Isolates the kernel from token_bin_counts_and_mask_kernel by feeding synthetic
# prompt/output masks and output bin counts. Requires NPU and Triton-Ascend.

import gc

import pytest
import torch

from vllm_ascend.ops.triton.penalty import _apply_all_penalties_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# (num_seqs, vocab_size, dtype, mask_mode)
#
# mask_mode:
#   mixed        — random prompt and output presence (~10% of vocab each);
#                  output_bin_counts is 1–4 where output_mask is True.
#   none         — no tokens seen; logits must stay unchanged.
#   prompt_only  — only prompt_mask is set (repetition penalty, no freq/presence).
#   output_only  — only output_mask / output_bin_counts are set (all three penalties).
PENALTY_KERNEL_CASES = [
    pytest.param(1, 2048, torch.float16, "mixed", id="single-seq-block-aligned"),
    pytest.param(4, 5120, torch.float16, "mixed", id="vocab-tail-tile"),
    pytest.param(8, 32000, torch.bfloat16, "mixed", id="llama-bf16"),
    pytest.param(3, 151936, torch.float16, "mixed", id="qwen-fp16"),
    pytest.param(8, 5120, torch.float32, "mixed", id="fp32"),
    pytest.param(4, 5120, torch.float16, "none", id="no-tokens-seen"),
    pytest.param(4, 5120, torch.float16, "prompt_only", id="prompt-only"),
    pytest.param(4, 5120, torch.float16, "output_only", id="output-only"),
]


def _make_penalty_inputs(
    num_seqs: int,
    vocab_size: int,
    dtype: torch.dtype,
    mask_mode: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.randn(num_seqs, vocab_size, device=device, dtype=dtype)
    prompt_mask = torch.zeros(num_seqs, vocab_size, dtype=torch.bool, device=device)
    output_mask = torch.zeros(num_seqs, vocab_size, dtype=torch.bool, device=device)
    output_bin_counts = torch.zeros(num_seqs, vocab_size, dtype=torch.int32, device=device)

    if mask_mode in ("mixed", "prompt_only"):
        prompt_mask = torch.rand(num_seqs, vocab_size, device=device) > 0.9
    if mask_mode in ("mixed", "output_only"):
        output_mask = torch.rand(num_seqs, vocab_size, device=device) > 0.9
        output_bin_counts = torch.where(
            output_mask,
            torch.randint(1, 5, (num_seqs, vocab_size), device=device, dtype=torch.int32),
            output_bin_counts,
        )

    presence_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.2
    frequency_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.2
    repetition_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.4 + 1.0
    return (
        logits,
        prompt_mask,
        output_mask,
        output_bin_counts,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def torch_apply_all_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """OpenAI-style repetition / frequency / presence, matching apply_all_penalties_kernel."""
    out = logits.float()
    seen = prompt_mask | output_mask
    penalty_factor = torch.where(
        seen,
        repetition_penalties[:, None].to(out.dtype),
        torch.ones((), device=out.device, dtype=out.dtype),
    )
    out = torch.where(out > 0, out / penalty_factor, out * penalty_factor)
    out = out - frequency_penalties[:, None].to(out.dtype) * output_bin_counts.float()
    out = out - presence_penalties[:, None].to(out.dtype) * output_mask.float()
    return out


@pytest.mark.parametrize("num_seqs,vocab_size,dtype,mask_mode", PENALTY_KERNEL_CASES)
@torch.inference_mode()
def test_apply_all_penalties_kernel(
    num_seqs,
    vocab_size,
    dtype,
    mask_mode,
    device="npu",
    seed=42,
):
    """Compare apply_all_penalties_kernel with a PyTorch penalty reference."""
    init_device_properties_triton()
    torch.manual_seed(seed)

    (
        logits,
        prompt_mask,
        output_mask,
        output_bin_counts,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    ) = _make_penalty_inputs(num_seqs, vocab_size, dtype, mask_mode, device)
    logits_triton = logits.clone()
    ref = torch_apply_all_penalties(
        logits,
        prompt_mask,
        output_mask,
        output_bin_counts,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )

    _apply_all_penalties_triton(
        logits_triton,
        prompt_mask,
        output_mask,
        output_bin_counts,
        repetition_penalties,
        frequency_penalties,
        presence_penalties,
    )
    torch.npu.synchronize()

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(logits_triton.float(), ref, atol=atol, rtol=rtol), (
        f"Max diff: {(logits_triton.float() - ref).abs().max().item()}"
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
