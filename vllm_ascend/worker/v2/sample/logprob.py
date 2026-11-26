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
from vllm.logger import logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample import logprob as vllm_logprob

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_RANK_BLOCK_SIZE = 8192
_MIN_RANKED_ONLINE_VOCAB_SIZE = 65536
_MIN_TOKEN_ONLINE_VOCAB_SIZE = 128256
_ONLINE_BLOCK_SIZE = 12288


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
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
        max_val = tl.max(tl.maximum(logits, max_val, propagate_nan=tl.PropagateNan.ALL))
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
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk_ids_stride + k_offset, mask=k_mask, other=0)

    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - lse - max_val
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _online_logprobs_kernel(
    output_ptr,
    rank_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    token_ids_stride,
    sampled_token_ids_ptr,
    vocab_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
    WITH_RANK: tl.constexpr,
    RANK_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    logits_row = logits_ptr + row * logits_stride
    lanes = tl.arange(0, BLOCK_SIZE)
    row_max = float("-inf")
    denominator = 0.0

    if WITH_RANK:
        sampled_id = tl.load(sampled_token_ids_ptr + row)
        sampled_logit = tl.load(logits_row + sampled_id).to(tl.float32)
        rank_lanes = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

    for offset in range(0, vocab_size, BLOCK_SIZE):
        columns = offset + lanes
        valid = columns < vocab_size
        logits = tl.load(logits_row + columns, valid, other=float("-inf")).to(tl.float32)
        tile_max = tl.max(tl.maximum(logits, float("-inf"), propagate_nan=tl.PropagateNan.ALL))
        next_max = tl.maximum(row_max, tile_max, propagate_nan=tl.PropagateNan.ALL)
        # A fully masked prefix contributes zero until the first finite logit.
        shift = tl.where(next_max == float("-inf"), 0.0, next_max)
        denominator = denominator * tl.exp(row_max - shift) + tl.sum(tl.exp(logits - shift))
        row_max = next_max

        if WITH_RANK:
            rank_lanes += ((logits >= sampled_logit) & valid).to(tl.int32)

    if WITH_RANK:
        # Upstream counts padded -inf lanes in its 8192-column tiles.
        rank_tail = (RANK_BLOCK_SIZE - vocab_size % RANK_BLOCK_SIZE) % RANK_BLOCK_SIZE
        padded_rank = tl.where(sampled_logit == float("-inf"), rank_tail, 0)
        tl.store(rank_ptr + row, tl.sum(rank_lanes) + padded_rank)

    slots = tl.arange(0, PADDED_TOPK)
    valid_slots = slots < topk
    token_ids = tl.load(token_ids_ptr + row * token_ids_stride + slots, valid_slots, other=0)
    selected_logits = tl.load(logits_row + token_ids, valid_slots, other=0).to(tl.float32)
    tl.store(output_ptr + row * topk + slots, selected_logits - tl.log(denominator) - row_max, valid_slots)


def _compute_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_ranks: torch.Tensor | None = None,
    sampled_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    if batch_size == 0 or num_logprobs == 0:
        return logprobs

    token_ids = token_ids.to(torch.int64)
    padded_topk = max(triton.next_power_of_2(num_logprobs), 2)
    with_rank = token_ranks is not None
    min_online_vocab_size = _MIN_RANKED_ONLINE_VOCAB_SIZE if with_rank else _MIN_TOKEN_ONLINE_VOCAB_SIZE
    if vocab_size >= min_online_vocab_size:
        _online_logprobs_kernel[(batch_size,)](
            logprobs,
            token_ranks,
            logits,
            logits.stride(0),
            token_ids,
            token_ids.stride(0),
            sampled_token_ids,
            vocab_size,
            num_logprobs,
            BLOCK_SIZE=_ONLINE_BLOCK_SIZE,
            PADDED_TOPK=padded_topk,
            WITH_RANK=with_rank,
            RANK_BLOCK_SIZE=_RANK_BLOCK_SIZE,
            multibuffer=False,
        )
        return logprobs

    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        token_ids.stride(0),
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=12944,
        PADDED_TOPK=padded_topk,
        multibuffer=False,
    )
    if token_ranks is not None:
        num_cores = min(batch_size, get_vectorcore_num())
        _ranks_kernel[(num_cores,)](
            token_ranks,
            logits,
            logits.stride(0),
            sampled_token_ids,
            vocab_size,
            batch_size,
            triton.cdiv(batch_size, num_cores),
            BLOCK_SIZE=_RANK_BLOCK_SIZE,
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
                n_vec += (logits >= x).to(tl.int32)
            n = tl.sum(n_vec)
            tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return _compute_logprobs(logits, token_ids)


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
    logprob_token_ids_state=None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size = logits.shape[0]

    if max_per_req_token_ids != 0:
        logger.warning_once("Custom logprob_token_ids is not supported yet. Falling back to default logprob_token_ids.")

    logprob_token_ids = sampled_token_ids.unsqueeze(-1)
    if num_logprobs > 0:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
    token_ranks = torch.empty(
        batch_size,
        dtype=torch.int64,
        device=logits.device,
    )

    logprobs = _compute_logprobs(logits, logprob_token_ids, token_ranks, sampled_token_ids)

    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state=None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    if logits_mode or max_per_req_token_ids != 0:
        return vllm_logprob.compute_topk_scores(
            logits,
            num_logprobs,
            sampled_token_ids,
            cu_num_logits,
            logprob_token_ids_state,
            expanded_idx_mapping,
            max_per_req_token_ids,
            logits_mode,
        )
    if isinstance(cu_num_logits, torch.Tensor):
        result = compute_topk_logprobs(logits, num_logprobs, sampled_token_ids)
        return result._replace(cu_num_generated_tokens_tensor=cu_num_logits)
    return compute_topk_logprobs(logits, num_logprobs, sampled_token_ids, cu_num_logits)
