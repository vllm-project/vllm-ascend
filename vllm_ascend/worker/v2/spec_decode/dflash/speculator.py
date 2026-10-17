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

from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
)

# Upstream uses the attention kernel block size for DCP ownership, which is
# incorrect when physical KV blocks are larger than kernel blocks. Bind the
# physical size so ownership uses KV cache blocks while slot lookup uses the
# kernel-sized block table supplied by the upstream caller.
if vllm_version_is("0.31.0"):

    def prepare_dflash_inputs_factory(kv_cache_block_size: int) -> Callable[..., None]:
        # v0.31.0's DFlash caller already forwards the kernel block size, so the
        # Ascend block table is indexed with the right granularity.
        def prepare_with_block_size(*args: Any, **kwargs: Any) -> None:
            prepare_dflash_inputs(*args, **kwargs, kv_cache_block_size=kv_cache_block_size)

        return prepare_with_block_size

else:

    def prepare_dflash_inputs_factory(  # type: ignore[misc]
        kv_cache_block_size: int,
        block_tables: Any | None = None,
    ) -> Callable[..., None]:
        # vLLM main (#57169) dropped ``kernel_block_sizes`` from ``BlockTables``
        # and the DFlash caller now forwards ``BlockTables.block_sizes``
        # (physical KV blocks) as the ``block_size`` argument, while
        # ``AscendBlockTables`` keeps the block table kernel-block addressed by
        # re-expanding its rows. Restore the kernel block size per group.
        kernel_block_size_by_ptr: dict[int, int] = {}
        kernel_block_sizes = getattr(block_tables, "kernel_block_sizes", None)
        input_block_tables = getattr(block_tables, "input_block_tables", None)
        if kernel_block_sizes is not None and input_block_tables is not None:
            kernel_block_size_by_ptr = {
                table.data_ptr(): int(size) for table, size in zip(input_block_tables, kernel_block_sizes)
            }

        def prepare_with_block_size(*args: Any, **kwargs: Any) -> None:
            # ``prepare_dflash_inputs`` takes ``block_table`` then ``block_size``
            # right after ``input_seeds``; upstream passes them positionally.
            if kernel_block_size_by_ptr and len(args) >= 18:
                block_table = args[16]
                if isinstance(block_table, torch.Tensor):
                    kernel_block_size = kernel_block_size_by_ptr.get(block_table.data_ptr())
                    if kernel_block_size is not None and kernel_block_size != args[17]:
                        args = (*args[:17], kernel_block_size, *args[18:])
            prepare_dflash_inputs(*args, **kwargs, kv_cache_block_size=kv_cache_block_size)

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
        if vllm_version_is("0.31.0"):
            dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(
                self.vllm_config.cache_config.block_size
            )
        else:
            dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(  # type: ignore[call-arg]
                self.vllm_config.cache_config.block_size,
                block_tables,
            )

    # vLLM #60754 renamed the DP-sync argument to `dp_sync_state` on main, and
    # vLLM #60335 added `num_speculative_tokens`. Define the per-version public
    # signature and body.
    if vllm_version_is("0.31.0"):

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
            dp_sync: Any = None,
            dummy_run: bool = False,
            skip_attn_for_dummy_run: bool = False,
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
            is_profile: bool = False,
        ) -> torch.Tensor:
            return self._propose_impl(
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
                dp_sync,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
            )

        def _propose_impl(
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
            dp_sync: Any,
            dummy_run: bool,
            skip_attn_for_dummy_run: bool,
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
            is_profile: bool,
        ) -> torch.Tensor:
            self.input_batch = input_batch
            sync_state = dp_sync
            if dummy_run and skip_attn_for_dummy_run:
                # Profiling runs the draft with its own query token count, which
                # can differ from the target batch. Let forward_context coordinate
                # the actual draft counts instead of reusing the target DP state.
                # TODO: Remove this guard once main2main includes upstream vLLM
                # #54856 (facd9a74a1), which resets the profiling DP counts.
                sync_state = None
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
                )

    else:

        def propose(  # type: ignore[misc]
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
            return self._propose_impl(  # type: ignore[call-arg]
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
                dp_sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile,
                num_speculative_tokens,
            )

        def _propose_impl(  # type: ignore[misc]
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
            dp_sync: Any,
            dummy_run: bool,
            skip_attn_for_dummy_run: bool,
            mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
            is_profile: bool,
            num_speculative_tokens: int | None,
        ) -> torch.Tensor:
            self.input_batch = input_batch
            sync_state = dp_sync
            if dummy_run and skip_attn_for_dummy_run:
                # Profiling runs the draft with its own query token count, which
                # can differ from the target batch. Let forward_context coordinate
                # the actual draft counts instead of reusing the target DP state.
                # TODO: Remove this guard once main2main includes upstream vLLM
                # #54856 (facd9a74a1), which resets the profiling DP counts.
                sync_state = None
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
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        sample_from_anchor,
        kv_cache_block_size=kv_cache_block_size,
    )
