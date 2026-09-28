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

from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.triton_utils import tl, triton

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(
    do_not_specialize=[
        "tokens_batch_stride",
        "batch_size",
        "seq_len",
        "total_blocks",
    ]
)
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


class TokenBinCountsAndMaskKernel(VllmTritonJitKernel["TokenBinCountsAndMaskKernel.CompileKey"]):
    SEQ_BLOCK = 256
    kernel = token_bin_counts_and_mask_kernel

    @dataclass(frozen=True)
    class CompileKey:
        seq_block: int

    def dispatch(self, *, seq_block: int) -> CompileKey:
        return self.CompileKey(seq_block=seq_block)

    def get_warmup_keys(self) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(seq_block=self.SEQ_BLOCK)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(
            tokens=TritonWarmupTensor(torch.int64, shape=(1, 1), strides=(1, 1)),
            bin_counts=TritonWarmupTensor(torch.int32, shape=(1, 1), strides=(1, 1)),
            tokens_batch_stride=1,
            tokens_seq_stride=1,
            batch_size=1,
            seq_len=1,
            vocab_size=1,
            tp_rank=0,
            counts_batch_stride=1,
            counts_vocab_stride=1,
            total_blocks=1,
            grid_size=1,
        )

    @kernel_launcher
    def __call__(
        self,
        tokens: torch.Tensor,
        bin_counts: torch.Tensor,
        *,
        tokens_batch_stride: int,
        tokens_seq_stride: int,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        tp_rank: int,
        counts_batch_stride: int,
        counts_vocab_stride: int,
        total_blocks: int,
        grid_size: int,
    ) -> LaunchSpec:
        return (grid_size,), dict(SEQ_BLOCK=self.SEQ_BLOCK)


_TOKEN_BIN_COUNTS_AND_MASK_KERNEL = TokenBinCountsAndMaskKernel()


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
    _TOKEN_BIN_COUNTS_AND_MASK_KERNEL(
        tokens=tokens,
        bin_counts=bin_counts,
        tokens_batch_stride=tokens.stride(0),
        tokens_seq_stride=tokens.stride(1),
        batch_size=n_rows,
        seq_len=n_cols,
        vocab_size=vocab_size,
        tp_rank=tp_rank,
        counts_batch_stride=bin_counts.stride(0),
        counts_vocab_stride=bin_counts.stride(1),
        total_blocks=total_blocks,
        grid_size=grid_size,
    )
    return bin_counts, bin_counts > 0
