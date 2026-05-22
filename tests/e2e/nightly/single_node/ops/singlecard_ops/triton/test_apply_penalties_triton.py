# SPDX-License-Identifier: Apache-2.0
# Compare vllm_ascend.sample.penalties.apply_all_penalties (Triton-Ascend) with
# vllm.v1.sample.ops.penalties.apply_all_penalties (PyTorch via model_executor).
# Requires NPU and Triton-Ascend.

import gc

import pytest
import torch
from vllm.v1.sample.ops.penalties import apply_all_penalties as v1_apply_all_penalties

from vllm_ascend.sample.penalties import apply_all_penalties as ascend_apply_all_penalties

# Same scenario grid as test_apply_penalties_model_executor (equivalence + boundaries).
APPLY_PENALTY_CASES = [
    pytest.param(0, 0, "mixed", id="empty-both"),
    pytest.param(0, 16, "mixed", id="empty-prompt"),
    pytest.param(32, 0, "mixed", id="empty-output"),
    pytest.param(1, 1, "mixed", id="single-token-each"),
    pytest.param(32, 16, "mixed", id="typical-small"),
    pytest.param(128, 64, "mixed", id="typical-large"),
    pytest.param(128, 64, "all_padding", id="all-padding"),
]


def _make_tokens(
    num_seqs: int,
    seq_len: int,
    vocab_size: int,
    mode: str,
    device: str,
) -> torch.Tensor:
    if mode == "all_padding":
        return torch.full((num_seqs, seq_len), vocab_size, device=device, dtype=torch.int64)
    if seq_len == 0:
        return torch.empty((num_seqs, 0), device=device, dtype=torch.int64)
    tokens = torch.randint(0, vocab_size, (num_seqs, seq_len), device=device, dtype=torch.int64)
    pad_mask = torch.rand(num_seqs, seq_len, device=device) > 0.7
    tokens[pad_mask] = vocab_size
    return tokens


@pytest.mark.parametrize("num_seqs", [1, 8, 32, 128])
@pytest.mark.parametrize("vocab_size", [5120, 151936])
@pytest.mark.parametrize(
    "max_prompt_len,max_output_len,token_mode",
    APPLY_PENALTY_CASES,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_apply_all_penalties_v1_vs_ascend(
    num_seqs,
    vocab_size,
    max_prompt_len,
    max_output_len,
    token_mode,
    dtype,
    device="npu",
    seed=42,
):
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    torch.manual_seed(seed)

    logits_v1 = torch.randn(num_seqs, vocab_size, device=device, dtype=dtype)
    logits_ascend = logits_v1.clone()

    prompt_tokens = _make_tokens(num_seqs, max_prompt_len, vocab_size, token_mode, device)
    output_tokens = _make_tokens(num_seqs, max_output_len, vocab_size, token_mode, device)
    output_token_ids = [row.tolist() for row in output_tokens.cpu()]

    presence_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.2
    frequency_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.2
    repetition_penalties = torch.rand(num_seqs, device=device, dtype=torch.float32) * 0.4 + 1.0

    v1_apply_all_penalties(
        logits_v1,
        prompt_tokens,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        output_token_ids,
    )
    ascend_apply_all_penalties(
        logits_ascend,
        prompt_tokens,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        output_token_ids,
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(logits_ascend.float(), logits_v1.float(), atol=atol, rtol=rtol), (
        f"Max diff: {(logits_ascend.float() - logits_v1.float()).abs().max().item()}"
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
