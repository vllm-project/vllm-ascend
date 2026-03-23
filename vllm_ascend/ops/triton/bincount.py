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
    BIG_CORE_NUM: tl.constexpr,
    BIG_ROW_BLOCK_SIZE: tl.constexpr,
):
    """Count token occurrences per batch row. Tokens with value >= vocab_size
    (e.g. padding) are skipped. Uses atomic_add for robust accumulation.
    """
    pid = tl.program_id(axis=0)
    SMALL_ROW_BLOCK_SIZE = BIG_ROW_BLOCK_SIZE - 1
    if pid < BIG_CORE_NUM:
        row_block_size = BIG_ROW_BLOCK_SIZE
        row_start_idx = pid * BIG_ROW_BLOCK_SIZE
    else:
        row_block_size = SMALL_ROW_BLOCK_SIZE
        row_start_idx = (
            BIG_CORE_NUM * BIG_ROW_BLOCK_SIZE
            + (pid - BIG_CORE_NUM) * SMALL_ROW_BLOCK_SIZE
        )

    row_end_idx = tl.minimum(row_start_idx + row_block_size, batch_size)
    for batch_idx in range(row_start_idx, row_end_idx):
        batch_tokens_start = tokens_ptr + batch_idx * tokens_batch_stride
        batch_counts_start = bin_counts_ptr + batch_idx * counts_batch_stride

        for pos in range(seq_len):
            token = tl.load(batch_tokens_start + pos * tokens_seq_stride)
            if token >= 0 and token < vocab_size:
                count_ptr = batch_counts_start + token * counts_vocab_stride
                tl.atomic_add(count_ptr, 1)


def get_token_bin_counts_and_mask_triton(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-Ascend implementation of token bin counting.

    Args:
        tokens: [num_seqs, seq_len] tensor of token IDs. Padding value
            should be >= vocab_size and will be ignored.
        vocab_size: Vocabulary size.
        num_seqs: If provided, asserts tokens.shape[0] == num_seqs.

    Returns:
        bin_counts: [num_seqs, vocab_size] int32 counts.
        mask: [num_seqs, vocab_size] bool, True where count > 0.
    """
    core_num = get_vectorcore_num()
    n_rows, n_cols = tokens.shape
    if num_seqs is not None and num_seqs > 0:
        assert n_rows == num_seqs, (
            f"tokens rows must match num_seqs: "
            f"tokens.shape[0]={n_rows}, num_seqs={num_seqs}"
        )
    n_rows = num_seqs if num_seqs is not None else n_rows

    bin_counts = torch.zeros(
        (n_rows, vocab_size), dtype=torch.int32, device=tokens.device
    )
    if not tokens.is_contiguous():
        tokens = tokens.contiguous()
    if not bin_counts.is_contiguous():
        bin_counts = bin_counts.contiguous()

    big_row_block_size = triton.cdiv(n_rows, core_num)
    big_core_num = max(
        0,
        min(
            core_num - (big_row_block_size * core_num - n_rows),
            core_num,
        ),
    )
    grid = (min(n_rows, core_num),)

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
        BIG_CORE_NUM=big_core_num,
        BIG_ROW_BLOCK_SIZE=big_row_block_size,
    )
    return bin_counts, bin_counts > 0
