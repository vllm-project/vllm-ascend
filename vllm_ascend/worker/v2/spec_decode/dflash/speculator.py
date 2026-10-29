# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

<<<<<<< HEAD
from vllm_ascend.utils import vllm_version_is
=======
from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton
>>>>>>> ff37dbaab5c58d567daaecfbb0337447e7e90274
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
)


def prepare_dflash_inputs_factory(
    kv_cache_block_size: int,
    block_tables: Any | None = None,
) -> Callable[..., None]:
    # vLLM main (#57169) dropped ``kernel_block_sizes`` from ``BlockTables``
    # and the DFlash caller now forwards ``BlockTables.block_sizes`` (physical
    # KV blocks) as ``block_size``, while ``AscendBlockTables`` keeps the block
    # table kernel-block addressed (a physical block is re-expanded into
    # ``blocks_per_kv_block`` kernel rows) and the Ascend prep indexes it at
    # kernel granularity. DCP ownership still uses the physical size, so only
    # ``block_size`` is restored, per group.
    kernel_block_sizes = getattr(block_tables, "kernel_block_sizes", None)
    input_block_tables = getattr(block_tables, "input_block_tables", None)
    if block_tables is not None and (kernel_block_sizes is None or input_block_tables is None):
        raise RuntimeError(
            "DFlash on vLLM main needs AscendBlockTables.kernel_block_sizes and "
            "input_block_tables to restore the kernel block size"
        )
    # The upstream caller passes ``input_block_tables[gid]`` verbatim, so the
    # per-group kernel block size is looked up by tensor identity. Callers
    # that do not supply Ascend block tables keep the physical size forwarded
    # by the DFlash caller.
    kernel_block_size_by_table = (
        {id(table): int(size) for table, size in zip(input_block_tables, kernel_block_sizes)}
        if kernel_block_sizes is not None and input_block_tables is not None
        else {}
    )

    def prepare_with_block_size(
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
        cp_rank: int,
        cp_size: int,
        cp_interleave: int,
        parallel_drafting_token_id: int,
        num_query_per_req: int,
        num_speculative_steps: int,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        sample_from_anchor: bool = False,
        num_reqs_override: int | None = None,
    ) -> None:
        kernel_block_size = kernel_block_size_by_table.get(id(block_table))
        if kernel_block_size is None:
            if kernel_block_size_by_table:
                raise RuntimeError("DFlash received a block table unknown to AscendBlockTables")
            # No Ascend block tables were supplied; keep the caller's size.
            kernel_block_size = block_size
        prepare_dflash_inputs(
            input_buffers,
            query_slot_mapping,
            context_positions,
            context_slot_mapping,
            sample_indices,
            sample_pos,
            sample_idx_mapping,
            temperature,
            seeds,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            input_temperature,
            input_seeds,
            block_table,
            kernel_block_size,
            cp_rank,
            cp_size,
            cp_interleave,
            parallel_drafting_token_id,
            num_query_per_req,
            num_speculative_steps,
            max_num_reqs,
            max_num_tokens,
            max_model_len,
            sample_from_anchor,
            num_reqs_override=num_reqs_override,
            kv_cache_block_size=kv_cache_block_size,
        )

    return prepare_with_block_size


class AscendDFlashSpeculator(LmheadTPDraftSamplingMixin, DFlashSpeculator):
    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        with disable_profiling_chunk_for_draft(self.vllm_config):
            return super().load_draft_model(target_model, target_attn_layer_names)

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        with build_attn_metadata_wrapper():
            # vLLM main (#56181) replaced _build_draft_attn_metadata with
            # _build_uniform_attn_metadata (BatchExecutionDescriptor).
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens_padded,
                num_reqs=num_reqs_padded,
            )
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=self.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return [attn_metadata]

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Rebuild ``actual_seq_lengths_q`` from the padded request count,
        mirroring Eagle's ``_update_decode_attn_metadata``.

        Upstream ``Speculator._build_draft_attn_metadata`` clamps
        ``query_start_loc`` at the real ``num_reqs`` to keep the cumulative
        series non-decreasing, so when a batch is padded to a capture size
        (``num_reqs_padded > num_reqs``) the cumulative query lengths stop at
        ``num_reqs * num_query_per_req`` instead of ``num_tokens_padded``. The
        Ascend FIA operator requires, in TND layout, that the last element of
        ``actual_seq_lengths_q`` equals the query token count of the graph
        being replayed; otherwise tiling fails with
        ``queryT != last element of actualSequenceLengthQ``.
        """
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._lmhead_tp_validate_draft_sampling()

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

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
        dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(  # type: ignore[call-arg]
            self.vllm_config.cache_config.block_size,
            block_tables,
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
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        self.input_batch = input_batch
<<<<<<< HEAD
        sync_state = num_tokens_across_dp if vllm_version_is("0.28.0") else dp_sync
=======
        sync_state = dp_sync_state
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Remove this guard once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
>>>>>>> ff37dbaab5c58d567daaecfbb0337447e7e90274
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
                sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )


<<<<<<< HEAD
# main2main compat: upstream ``_prepare_dflash_inputs_kernel`` added four
# ``temperature``/``seeds`` parameters and corresponding stores (see
# vllm-project/vllm#50000). Ascend keeps its own kernel for NPU, matching
# the upstream parameter layout.

if vllm_version_is("0.28.0"):

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
        # Copy sampling state (added upstream in vllm-project/vllm#50000).
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(
            out_seeds_ptr + req_state_idx,
            tl.load(seeds_ptr + req_state_idx),
        )

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
else:
    # main2main compat: upstream ``_prepare_dflash_inputs_kernel`` added
    # ``cp_rank``/``CP_SIZE``/``CP_INTERLEAVE`` for DCP support (see
    # vllm-project/vllm#52188). The extra parameters are unused: Ascend does not
    # run dflash with DCP (CP_SIZE is always 1, where upstream ``cp_local_slot``
    # yields the same slots as this kernel).
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
        # Block table for slot mapping lookup.
        block_table_ptr,
        block_table_stride,
        # Scalars
        parallel_drafting_token_id,
        block_size,
=======
def prepare_dflash_inputs(
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
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
    num_reqs_override: int | None = None,
    *,
    kv_cache_block_size: int,
) -> None:
    prepare_dflash_inputs_triton(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        block_size,
        cp_rank,
        cp_size,
        cp_interleave,
        parallel_drafting_token_id,
>>>>>>> ff37dbaab5c58d567daaecfbb0337447e7e90274
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
<<<<<<< HEAD
        cp_rank,
        SAMPLE_FROM_ANCHOR: tl.constexpr,
        PAD_SLOT_ID: tl.constexpr,
        CP_SIZE: tl.constexpr,
        CP_INTERLEAVE: tl.constexpr,
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
        # Copy sampling state (added upstream in vllm-project/vllm#50000).
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(
            out_seeds_ptr + req_state_idx,
            tl.load(seeds_ptr + req_state_idx),
        )

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
=======
        sample_from_anchor,
        num_reqs_override=num_reqs_override,
        kv_cache_block_size=kv_cache_block_size,
    )
>>>>>>> ff37dbaab5c58d567daaecfbb0337447e7e90274
