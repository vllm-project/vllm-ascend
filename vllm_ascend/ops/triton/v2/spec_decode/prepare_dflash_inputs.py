# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_MAX_CONTEXT_BLOCK_SIZE = 256
_QUERY_BLOCK_SIZE = 16
_SAMPLE_BLOCK_SIZE = 16


@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
    SAMPLE_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    workers_per_req = tl.num_programs(1)

    global_worker_idx = req_idx * workers_per_req + worker_idx
    total_workers = num_reqs * workers_per_req

    # Context
    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    ctx_base = num_ctx // workers_per_req
    ctx_extra = num_ctx % workers_per_req
    ctx_begin = worker_idx * ctx_base + tl.minimum(worker_idx, ctx_extra)
    ctx_count = ctx_base + tl.where(worker_idx < ctx_extra, 1, 0)

    ctx_lane = tl.arange(0, BLOCK_SIZE)
    ctx_mask = ctx_lane < ctx_count
    ctx_pos_idx = ctx_start + ctx_begin + ctx_lane

    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=ctx_mask, other=0)
    ctx_block_num = tl.minimum(ctx_pos // block_size, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=ctx_mask,
        other=0,
    ).to(tl.int64)
    ctx_slot = ctx_block_id * block_size + ctx_pos % block_size

    tl.store(out_context_positions_ptr + ctx_pos_idx, ctx_pos, mask=ctx_mask)
    tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, ctx_slot, mask=ctx_mask)

    # Request state shared by query/sample.
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    # Query
    base_queries_per_worker = num_query_per_req // workers_per_req
    extra_queries = num_query_per_req % workers_per_req
    query_begin = worker_idx * base_queries_per_worker + tl.minimum(worker_idx, extra_queries)
    query_count = base_queries_per_worker + tl.where(worker_idx < extra_queries, 1, 0)

    query_lane = tl.arange(0, QUERY_BLOCK_SIZE)
    query_mask = query_lane < query_count
    query_off = query_begin + query_lane
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off

    # query_off == 0 is always owned by worker 0 under quotient/remainder splitting.
    bonus_token = 0
    if worker_idx == 0:
        num_sampled = tl.load(num_sampled_ptr + req_idx)
        if num_sampled > 0:
            bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
        else:
            bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    input_id = tl.where(query_off == 0, bonus_token, parallel_drafting_token_id)

    q_block_num = tl.minimum(query_pos // block_size, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=query_mask,
        other=0,
    ).to(tl.int64)
    q_slot = q_block_id * block_size + query_pos % block_size

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=query_mask)
    tl.store(out_query_positions_ptr + query_idx, tl.minimum(query_pos, max_model_len - 1), mask=query_mask)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=query_mask)

    # Sample
    base_samples_per_worker = num_speculative_steps // workers_per_req
    extra_samples = num_speculative_steps % workers_per_req
    sample_begin = worker_idx * base_samples_per_worker + tl.minimum(worker_idx, extra_samples)
    sample_count = base_samples_per_worker + tl.where(worker_idx < extra_samples, 1, 0)

    sample_lane = tl.arange(0, SAMPLE_BLOCK_SIZE)
    sample_mask = sample_lane < sample_count
    sample_local = sample_begin + sample_lane
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    sample_query_off = sample_local + sample_off
    sample_mask &= sample_query_off < num_query_per_req

    sample_idx = req_idx * num_speculative_steps + sample_local
    sample_query_idx = query_base + sample_query_off
    sample_query_pos = last_valid_pos + 1 + sample_query_off
    sampled_pos = sample_query_pos + 1 if SAMPLE_FROM_ANCHOR else sample_query_pos

    tl.store(out_sample_indices_ptr + sample_idx, sample_query_idx, mask=sample_mask)
    tl.store(out_sample_pos_ptr + sample_idx, sampled_pos, mask=sample_mask)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=sample_mask)

    # Per-request scalar state.
    if worker_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)
        tl.store(out_temperature_ptr + req_state_idx, tl.load(temperature_ptr + req_state_idx))
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))

    # Graph-safety padding. Each range is balanced across the full launch grid.
    last_query_end = num_reqs * num_query_per_req

    # query_start_loc: [num_reqs, max_num_reqs + 1)
    qs_pad_count = max_num_reqs + 1 - num_reqs
    qs_base = qs_pad_count // total_workers
    qs_extra = qs_pad_count % total_workers
    qs_begin = num_reqs + global_worker_idx * qs_base + tl.minimum(global_worker_idx, qs_extra)
    qs_count = qs_base + tl.where(global_worker_idx < qs_extra, 1, 0)
    qs_end = qs_begin + qs_count

    for i in range(qs_begin, qs_end, 16):
        qs_off = i + tl.arange(0, 16)
        tl.store(out_query_start_loc_ptr + qs_off, last_query_end, mask=qs_off < qs_end)

    # seq_lens: [num_reqs, max_num_reqs)
    seq_pad_count = max_num_reqs - num_reqs
    seq_base = seq_pad_count // total_workers
    seq_extra = seq_pad_count % total_workers
    seq_begin = num_reqs + global_worker_idx * seq_base + tl.minimum(global_worker_idx, seq_extra)
    seq_count = seq_base + tl.where(global_worker_idx < seq_extra, 1, 0)
    seq_end = seq_begin + seq_count

    for i in range(seq_begin, seq_end, 16):
        seq_off = i + tl.arange(0, 16)
        tl.store(out_seq_lens_ptr + seq_off, 0, mask=seq_off < seq_end)

    # Sample buffers: [num_reqs * steps, max_num_reqs * steps)
    sample_pad_start = num_reqs * num_speculative_steps
    sample_pad_count = (max_num_reqs - num_reqs) * num_speculative_steps
    sp_base = sample_pad_count // total_workers
    sp_extra = sample_pad_count % total_workers
    sp_begin = sample_pad_start + global_worker_idx * sp_base + tl.minimum(global_worker_idx, sp_extra)
    sp_count = sp_base + tl.where(global_worker_idx < sp_extra, 1, 0)
    sp_end = sp_begin + sp_count

    for i in range(sp_begin, sp_end, 64):
        sp_off = i + tl.arange(0, 64)
        sp_mask = sp_off < sp_end
        tl.store(out_sample_indices_ptr + sp_off, 0, mask=sp_mask)
        tl.store(out_sample_pos_ptr + sp_off, 0, mask=sp_mask)
        tl.store(out_sample_idx_mapping_ptr + sp_off, -1, mask=sp_mask)

    # query_slot_mapping: [num_reqs * query_per_req, max_num_tokens)
    q_pad_start = num_reqs * num_query_per_req
    q_pad_count = max_num_tokens - q_pad_start
    qp_base = q_pad_count // total_workers
    qp_extra = q_pad_count % total_workers
    qp_begin = q_pad_start + global_worker_idx * qp_base + tl.minimum(global_worker_idx, qp_extra)
    qp_count = qp_base + tl.where(global_worker_idx < qp_extra, 1, 0)
    qp_end = qp_begin + qp_count

    for i in range(qp_begin, qp_end, 256):
        qp_off = i + tl.arange(0, 256)
        tl.store(out_query_slot_mapping_ptr + qp_off, PAD_SLOT_ID, mask=qp_off < qp_end)


def prepare_dflash_inputs_triton(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0

    max_target_query_len = int(input_batch.num_scheduled_tokens.max())

    init_device_properties_triton()
    vectorcore_count = get_vectorcore_num()

    workers_for_parallelism = max(1, vectorcore_count // num_reqs)
    workers_for_context = max(1, triton.cdiv(max_target_query_len, _MAX_CONTEXT_BLOCK_SIZE))
    workers_for_query = max(1, triton.cdiv(num_query_per_req, _QUERY_BLOCK_SIZE))
    workers_for_sample = max(1, triton.cdiv(num_speculative_steps, _SAMPLE_BLOCK_SIZE))
    workers_per_req = max(
        workers_for_parallelism,
        workers_for_context,
        workers_for_query,
        workers_for_sample,
    )

    max_ctx_per_worker = triton.cdiv(max_target_query_len, workers_per_req)
    block_size_kernel = min(
        _MAX_CONTEXT_BLOCK_SIZE,
        triton.next_power_of_2(max(2, max_ctx_per_worker)),
    )

    _prepare_dflash_inputs_kernel[(num_reqs, workers_per_req)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_temperature,
        input_seeds,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=block_size_kernel,
        QUERY_BLOCK_SIZE=_QUERY_BLOCK_SIZE,
        SAMPLE_BLOCK_SIZE=_SAMPLE_BLOCK_SIZE,
    )
