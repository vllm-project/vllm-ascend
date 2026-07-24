# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
)

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper

logger = logging.getLogger(__name__)


class AscendDFlashSpeculator(DFlashSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # we need to update full graph params in run_fullgraph,
        # so create a stream to update full graph params.
        cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode.has_full_cudagraphs():
            self.update_stream: torch.npu.Stream = torch.npu.Stream()

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # npu needs attn_backends to update full graph params in run_fullgraph.
        attn_backends: dict[str, type[AttentionBackend]] = {}
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

        self.attn_backends = attn_backends

    # NOTE: upstream vLLM named this to _build_draft_attn_metadatas;
    # keep the current name for now as upstream may change it again.
    def build_draft_attn_metadatas(self, num_reqs_padded):
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        with build_attn_metadata_wrapper():
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs_padded,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                causal=self._group_causal,
            )
        return [attn_metadata]

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
