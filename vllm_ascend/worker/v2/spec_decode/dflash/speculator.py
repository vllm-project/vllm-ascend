# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import logging
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
from vllm_ascend.utils import lmhead_tp_enable, vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
)

logger = logging.getLogger(__name__)

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
                kv_cache_block_size=kv_cache_block_size,
            )

        return prepare_with_block_size


class AscendDFlashSpeculator(LmheadTPDraftSamplingMixin, DFlashSpeculator):
    # Set by load_draft_model: True when the loaded draft mounts the target's
    # lm_head object (the DFlash/DFlash2 drafter checkpoints ship no lm_head
    # weights, so the loader shares the target's; VL targets keep the head
    # nested, e.g. language_model.lm_head). In that case load_draft_model
    # DECOUPLES the draft onto a shallow copy of the head: same weight
    # tensors and quant method, separate object. The draft-side capacity
    # attributes (dyn/static) land on the copy -- the target's own head
    # object must stay clean, or the target's sample() resolves the
    # draft-scale dyn (rows > dyn -> ValueError, the Q6 crash). The copy
    # also keeps the head-level row pad for the draft calls: the draft head
    # rows are the REAL num_reqs * num_query_per_req per rank (NOT the
    # padded-batch rows -- the first NPU retest of the skip-based fix
    # logged rows=28/21/14 on an 18-req batch and hung HCCL after a few
    # minutes of desynced all_gathers). Default False ("not shared / not
    # yet loaded"): _run_model can never run before load_draft_model.
    _lmhead_tp_head_is_shared = False

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        with disable_profiling_chunk_for_draft(self.vllm_config):
            model = super().load_draft_model(target_model, target_attn_layer_names)
        # The draft head is shared when it is literally one of the target
        # model's submodules: VL targets keep lm_head nested (e.g.
        # language_model.lm_head) with no top-level attribute, so a
        # top-level identity check would miss the sharing.
        draft_head = getattr(model, "lm_head", None)
        self._lmhead_tp_head_is_shared = draft_head is not None and any(
            module is draft_head for module in target_model.modules()
        )
        if self._lmhead_tp_head_is_shared:
            # Decouple the head OBJECT while sharing the weight tensors:
            # inject/publish land on the copy (restoring the draft-side row
            # pad), and the target's head object never carries draft
            # capacity. copy.copy on an nn.Module shares the parameter/
            # buffer tensors by reference; the plain capacity attributes
            # written later live in the copy's own __dict__.
            model.lm_head = copy.copy(draft_head)
            if lmhead_tp_enable():
                logger.info(
                    "lmhead TP: the DFlash draft shares the target LM head "
                    "weights; a decoupled head object carries the draft-side "
                    "capacity so the target's head object stays clean."
                )
        # Generic DFlash draft models never pass lmhead_tp_capacity to the
        # head constructor; attach the static bound post-load so the
        # head-level pad has a fallback for steps without a DP sync.
        self._lmhead_tp_inject_static_capacity(model)
        return model

    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        """Run the DFlash draft backbone and publish the draft LM-head
        capacity before this step's head call. DFlash drafts
        ``num_query_per_req`` query tokens per request and the head samples
        ``num_speculative_steps`` of them, so the synced max query rows
        already cover the head rows and are the capacity."""
        output = super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._lmhead_tp_publish_draft_capacity(num_tokens_across_dp, tokens_per_req=self.num_query_per_req)
        return output

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
            self.input_batch = input_batch
            sync_state = dp_sync_state
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
