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
from vllm.distributed.parallel_state import get_tp_group
from vllm.triton_utils import tl, triton

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["batch_size", "seq_len"])
def token_bin_counts_and_mask_kernel(
    tokens_ptr,
    tokens_batch_stride,
    tokens_seq_stride,
    batch_size,
    seq_len,
    vocab_size,
    bin_counts_ptr,
    tp_rank,
    counts_batch_stride,
    counts_vocab_stride,
    total_blocks,
    SEQ_BLOCK: tl.constexpr,
):
    """Count token occurrences per batch row.

    1D grid with grid-stride loop: each program processes blocks at
    stride=num_programs to stay within the Triton-Ascend coreDim
    limit (65535) while distributing work evenly across cores.
    """
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)

    vocab_start_idx = tp_rank * vocab_size
    n_seq_blocks = tl.cdiv(seq_len, SEQ_BLOCK)

    for linear_block in tl.range(pid, total_blocks, num_progs):
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
            other=vocab_size + vocab_start_idx,
        )

        local_token = token - vocab_start_idx
        token_in_range = pos_mask & (token >= vocab_start_idx) & (local_token < vocab_size)

        safe_local_token = tl.where(token_in_range, local_token, 0)
        count_ptr = batch_counts_start + safe_local_token * counts_vocab_stride
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

    if n_rows == 0 or n_cols == 0:
        bin_counts = torch.zeros((n_rows, vocab_size), dtype=torch.int32, device=tokens.device)
        return bin_counts, bin_counts > 0

    core_num = get_vectorcore_num()

    bin_counts = torch.zeros((n_rows, vocab_size), dtype=torch.int32, device=tokens.device)
    if not tokens.is_contiguous():
        tokens = tokens.contiguous()

    # 1D grid: distribute all (batch, seq_block) work items across
    # vector cores via a loop inside the kernel.  This avoids the
    # Triton-Ascend grid-size limit of 65535.
    SEQ_BLOCK = 256
    n_seq_blocks = triton.cdiv(n_cols, SEQ_BLOCK)
    total_blocks = n_rows * n_seq_blocks
    grid_size = min(core_num, total_blocks)

    if get_ascend_config().enable_reduce_sample:
        tp_group = get_tp_group()
        tp_rank = tp_group.rank_in_group
    else:
        tp_rank = 0
    token_bin_counts_and_mask_kernel[(grid_size,)](
        tokens,
        tokens.stride(0),
        tokens.stride(1),
        n_rows,
        n_cols,
        vocab_size,
        bin_counts,
        tp_rank,
        bin_counts.stride(0),
        bin_counts.stride(1),
        total_blocks,
        SEQ_BLOCK=SEQ_BLOCK,
        multibuffer=False,
    )
    return bin_counts, bin_counts > 0
