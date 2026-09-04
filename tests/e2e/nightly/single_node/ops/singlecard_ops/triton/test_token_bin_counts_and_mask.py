# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Compare get_token_bin_counts_and_mask_triton (token_bin_counts_and_mask_kernel)
# with a PyTorch scatter_add_ reference. Requires NPU and Triton-Ascend.

import gc

import pytest
import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.ops.triton.bincount import get_token_bin_counts_and_mask_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# TODO(realliujiaxu): delete this after `enable_reduce_sample` is removed
init_ascend_config(VllmConfig())

# Qwen-style vocab and profiler / penalties-path shapes.
# (num_seqs, seq_len, vocab_size, token_mode)
#
# token_mode:
#   mixed        — random valid token ids in [0, vocab_size), then ~30% of
#                  positions replaced with padding sentinel vocab_size.
#                  Simulates a padded prompt/output history.
#                  seq_len==0 still yields an empty tensor regardless of mode.
#   all_padding  — every position is vocab_size (no valid tokens). Kernel
#                  must ignore them and return zero counts / False mask.
BINCOUNT_CASES = [
    pytest.param(3, 128, 151936, "mixed", id="profiler-prompt"),
    pytest.param(3, 64, 151936, "mixed", id="profiler-output"),
    pytest.param(1, 128, 151936, "mixed", id="qwen-single-seq"),
    pytest.param(8, 257, 151936, "mixed", id="multi-seq-block-tail"),
    pytest.param(32, 0, 151936, "mixed", id="empty-seq"),
    pytest.param(8, 64, 151936, "all_padding", id="all-padding"),
    pytest.param(32, 256, 5120, "mixed", id="small-vocab-block-aligned"),
]


def _make_tokens(
    num_seqs: int,
    seq_len: int,
    vocab_size: int,
    mode: str,
    device: str,
) -> torch.Tensor:
    # Padding sentinel is vocab_size; kernel ignores ids outside [0, vocab_size).
    if mode == "all_padding":
        return torch.full((num_seqs, seq_len), vocab_size, device=device, dtype=torch.int64)
    if seq_len == 0:
        return torch.empty((num_seqs, 0), device=device, dtype=torch.int64)
    # mixed: valid ids plus random padding (pad_mask True ≈ 30% of positions).
    tokens = torch.randint(0, vocab_size, (num_seqs, seq_len), device=device, dtype=torch.int64)
    pad_mask = torch.rand(num_seqs, seq_len, device=device) > 0.7
    tokens[pad_mask] = vocab_size
    return tokens


def torch_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    tp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match upstream scatter_add_ over vocab_size+1, then slice off the pad bin."""
    n_rows, n_cols = tokens.shape
    bin_counts = torch.zeros((n_rows, vocab_size), dtype=torch.int32, device=tokens.device)
    if n_rows == 0 or n_cols == 0:
        return bin_counts, bin_counts > 0

    vocab_start = tp_rank * vocab_size
    local = tokens - vocab_start
    valid = (tokens >= vocab_start) & (local < vocab_size)
    padded = torch.where(valid, local, torch.full_like(local, vocab_size))
    counts = torch.zeros((n_rows, vocab_size + 1), dtype=torch.int32, device=tokens.device)
    counts.scatter_add_(1, padded.to(torch.int64), torch.ones_like(padded, dtype=torch.int32))
    bin_counts = counts[:, :vocab_size]
    return bin_counts, bin_counts > 0


@pytest.mark.parametrize("num_seqs,seq_len,vocab_size,token_mode", BINCOUNT_CASES)
@torch.inference_mode()
def test_token_bin_counts_and_mask_kernel(num_seqs, seq_len, vocab_size, token_mode, device="npu", seed=42):
    """Compare token_bin_counts_and_mask_kernel with a PyTorch scatter_add_ reference."""
    init_device_properties_triton()
    torch.manual_seed(seed)

    tokens = _make_tokens(num_seqs, seq_len, vocab_size, token_mode, device)
    bin_counts, mask = get_token_bin_counts_and_mask_triton(tokens, vocab_size, num_seqs)
    ref_counts, ref_mask = torch_token_bin_counts_and_mask(tokens, vocab_size)

    assert bin_counts.dtype == torch.int32
    assert mask.dtype == torch.bool
    assert torch.equal(bin_counts, ref_counts), (
        f"bin_counts differs from scatter_add_ reference. "
        f"Max abs diff: {(bin_counts - ref_counts).abs().max().item()}"
    )
    assert torch.equal(mask, ref_mask), "mask differs from (bin_counts > 0) reference"
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
