# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Triton-Ascend implementation of get_token_bin_counts_and_mask.
# Migrated from model_executor/layers/utils.get_token_bin_counts_and_mask.
# Reference: https://github.com/vllm-project/vllm-ascend/pull/6979

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def token_bin_counts_and_mask_kernel(
    tokens_ptr,
    tokens_batch_stride,
    tokens_seq_stride,
    batch_size,
    seq_len,
    vocab_size,
    bin_counts_ptr,
    counts_batch_stride,
    counts_vocab_stride,
    SEQ_BLOCK: tl.constexpr,
):
    """Count token occurrences per batch row.

    2D tiling:
      - axis=0: core/program group dimension
      - axis=1: block id dimension

    We linearize (batch_idx, seq_block_id) into a single global block id and
    distribute blocks across all programs to improve utilization when
    batch_size is small but seq_len is large (typical prefill).

    Tokens with value >= vocab_size (e.g. padding) are skipped.
    """
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    progs = tl.num_programs(axis=0)

    n_seq_blocks = tl.cdiv(seq_len, SEQ_BLOCK)
    linear_block = pid1 * progs + pid0
    total_blocks = batch_size * n_seq_blocks
    if linear_block >= total_blocks:
        return

    batch_idx = linear_block // n_seq_blocks
    seq_block_id = linear_block - batch_idx * n_seq_blocks
    seq_start = seq_block_id * SEQ_BLOCK

    batch_tokens_start = tokens_ptr + batch_idx * tokens_batch_stride
    batch_counts_start = bin_counts_ptr + batch_idx * counts_batch_stride

    pos_offsets = seq_start + tl.arange(0, SEQ_BLOCK)
    pos_mask = pos_offsets < seq_len
    token = tl.load(
        batch_tokens_start + pos_offsets * tokens_seq_stride,
        mask=pos_mask,
        other=vocab_size,  # force invalid
    )
    # Only count valid token ids in [0, vocab_size). Padding must use id >= vocab_size
    # (see vLLM apply_penalties contract); those positions are masked out here.
    token_in_range = (token >= 0) & (token < vocab_size) & pos_mask
    count_ptr = batch_counts_start + token * counts_vocab_stride
    tl.atomic_add(count_ptr, 1, mask=token_in_range)


def get_token_bin_counts_and_mask_triton(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-Ascend implementation of token bin counting.

    Args:
        tokens: [num_seqs, seq_len] tensor of token IDs. Padding value
            should be vocab_size and will be ignored.
        vocab_size: Vocabulary size.
        num_seqs: If provided, asserts tokens.shape[0] == num_seqs.

    Returns:
        bin_counts: [num_seqs, vocab_size] int32 counts.
        mask: [num_seqs, vocab_size] bool, True where count > 0.
    """
    n_rows, n_cols = tokens.shape
    if num_seqs is not None and num_seqs > 0:
        assert n_rows == num_seqs, f"tokens rows must match num_seqs: tokens.shape[0]={n_rows}, num_seqs={num_seqs}"
    n_rows = num_seqs if num_seqs is not None else n_rows

    # seq_len == 0 is valid for empty decode history; return directly.
    if n_cols == 0:
        bin_counts = torch.zeros((n_rows, vocab_size), dtype=torch.int32, device=tokens.device)
        return bin_counts, bin_counts > 0

    core_num = get_vectorcore_num()

    bin_counts = torch.zeros((n_rows, vocab_size), dtype=torch.int32, device=tokens.device)
    if not tokens.is_contiguous():
        tokens = tokens.contiguous()

    # 2D grid: (progs, blocks_per_prog_group)
    # Keep axis-0 bounded by vector core count, and distribute (batch, seq_block)
    # blocks across all programs to increase utilization when n_rows is small.
    SEQ_BLOCK = 256
    n_seq_blocks = triton.cdiv(n_cols, SEQ_BLOCK)
    total_blocks = n_rows * n_seq_blocks
    progs = min(core_num, total_blocks)
    grid = (progs, triton.cdiv(total_blocks, progs))

    token_bin_counts_and_mask_kernel[grid](
        tokens,
        tokens.stride(0),
        tokens.stride(1),
        n_rows,
        n_cols,
        vocab_size,
        bin_counts,
        bin_counts.stride(0),
        bin_counts.stride(1),
        SEQ_BLOCK=SEQ_BLOCK,
    )
    return bin_counts, bin_counts > 0
