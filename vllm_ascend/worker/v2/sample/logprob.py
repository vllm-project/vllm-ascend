# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/logprob.py.
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

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        se += tl.sum(e)                                                                                                                                         
    lse = tl.log(se)

    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)

    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - lse - max_val
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


def compute_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE= 12944,
        PADDED_TOPK=max(triton.next_power_of_2(num_logprobs), 2),
        multibuffer=False,
    )
    return logprobs


@triton.jit(do_not_specialize=["batch_size", "rows_per_core"])
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    batch_size,
    rows_per_core,
    BLOCK_SIZE: tl.constexpr,
):
    core_id = tl.program_id(0)

    start_row = core_id * rows_per_core
    end_row = start_row + rows_per_core

    for req_idx in range(start_row, end_row):
        if req_idx < batch_size:
            row_ptr = logits_ptr + req_idx * logits_stride

            token_id = tl.load(token_ids_ptr + req_idx)
            x = tl.load(row_ptr + token_id)

            n_vec = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            for i in range(0, vocab_size, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
                n_vec += (logits > x).to(tl.int32)
            n = tl.sum(n_vec)
            tl.store(output_ptr + req_idx, n)


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape
    if num_logprobs == 0:
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
    else:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        logprob_token_ids = torch.cat((sampled_token_ids.unsqueeze(-1), topk_indices), dim=1)

    # NOTE(woosuk): Here, to save GPU memory, we do not materialize the full
    # logprobs tensor. Instead, we only compute and return the logprobs of
    # the topk + 1 tokens.
    logprobs = compute_token_logprobs(logits, logprob_token_ids)
    token_ranks = torch.empty(
        batch_size,
        dtype=torch.int64,
        device=logits.device,
    )

    vec_core = get_vectorcore_num()
    NUM_CORES = min(batch_size, vec_core)

    rows_per_core = triton.cdiv(batch_size, NUM_CORES)
    BLOCK_SIZE = 8192
    grid = (NUM_CORES,)
    _ranks_kernel[grid](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        batch_size,
        rows_per_core,
        BLOCK_SIZE=BLOCK_SIZE,
        multibuffer=False,
    )

    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
    )
