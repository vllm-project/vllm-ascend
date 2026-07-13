# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
)

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper


class AscendDFlashSpeculator(DFlashSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)
        if vllm_version_is("0.24.0"):
            self.context_slot_mapping = torch.zeros(
                self.max_num_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self._context_slot_mappings = torch.zeros(
                len(self.draft_kv_cache_group_ids),
                self.max_num_tokens,
                dtype=torch.int32,
                device=self.device,
            )

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        with build_attn_metadata_wrapper():
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                num_tokens_across_dp,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
            )


@triton.jit
def _prepare_dflash_inputs_kernel_ascend(
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
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Block table for slot mapping lookup.
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
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)

    if block_idx > 0:
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    nrejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - nrejected

    nsampled = tl.load(num_sampled_ptr + req_idx)
    if nsampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    # --- Context positions / slots ---
    for j in range(0, num_ctx):
        ctx_pos_idx = ctx_start + j
        ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx)
        ctx_block_num = ctx_pos // block_size
        ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
        ctx_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + ctx_block_num).to(tl.int64)
        ctx_slot = ctx_block_id * block_size + (ctx_pos % block_size)
        tl.store(out_context_positions_ptr + ctx_pos_idx, ctx_pos)
        tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, ctx_slot)

    # --- Query positions / input_ids / slots ---
    for q_off in range(0, num_query_per_req):
        query_pos = last_valid_pos + 1 + q_off
        query_idx = query_base + q_off
        if q_off == 0:
            input_id = bonus_token
        else:
            input_id = parallel_drafting_token_id

        q_block_num = query_pos // block_size
        q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
        q_block_id = tl.load(block_table_ptr + req_idx * block_table_stride + q_block_num).to(tl.int64)
        q_slot = q_block_id * block_size + (query_pos % block_size)

        tl.store(out_input_ids_ptr + query_idx, input_id)
        clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
        tl.store(out_query_positions_ptr + query_idx, clamped_query_pos)
        tl.store(out_query_slot_mapping_ptr + query_idx, q_slot)

    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    # --- Sample indices / positions / idx_mapping ---
    for s_off in range(sample_off, num_query_per_req):
        sample_idx = req_idx * num_speculative_steps + (s_off - sample_off)
        query_idx = query_base + s_off
        query_pos = last_valid_pos + 1 + s_off
        sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
        tl.store(out_sample_indices_ptr + sample_idx, query_idx)
        tl.store(out_sample_pos_ptr + sample_idx, sample_pos)
        tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx)

    tl.store(out_query_start_loc_ptr + req_idx, query_base)
    # seq_lens is the absolute sequence length the draft attention
    # reads up to (context + query), not just the count of accepted
    # tokens this step.
    tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)

    if req_idx == num_reqs - 1:
        # Pad per-request buffers to max_num_reqs for CUDA graph safety.
        last_query_end = num_reqs * num_query_per_req
        for i in range(num_reqs, max_num_reqs + 1):
            tl.store(out_query_start_loc_ptr + i, last_query_end)
        for i in range(num_reqs, max_num_reqs):
            tl.store(out_seq_lens_ptr + i, 0)
        # Padded sample slots point at query index 0 (a valid row in
        # last_hidden_states) so CG replay never reads OOB. Padded sample
        # idx mappings point to -1, which is ignored during sampling.
        pad_start = num_reqs * num_speculative_steps
        pad_end = max_num_reqs * num_speculative_steps
        for i in range(pad_start, pad_end):
            tl.store(out_sample_indices_ptr + i, 0)
            tl.store(out_sample_pos_ptr + i, 0)
            tl.store(out_sample_idx_mapping_ptr + i, -1)
        # Pad query slot mappings past num_query_tokens with PAD so the
        # captured CG sees PAD slots (no K/V write) for replay sizes
        # larger than the current request count.
        q_pad_start = num_reqs * num_query_per_req
        for i in range(q_pad_start, max_num_tokens):
            tl.store(out_query_slot_mapping_ptr + i, PAD_SLOT_ID)


def prepare_dflash_inputs_ascend(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int | None = None,
    sample_from_anchor: bool = False,
) -> None:
    """Launch the Ascend kernel for the v0.24 DFlash caller."""
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    block_size_triton = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, block_size_triton)
    if max_model_len is None:
        # v0.24 did not clamp query positions in this kernel.
        max_model_len = 2**31 - 1
    _prepare_dflash_inputs_kernel_ascend[(num_reqs, num_blocks)](
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
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
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
        BLOCK_SIZE=block_size_triton,
    )
