# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Callable
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper

logger = logging.getLogger(__name__)


class AscendDFlashSpeculator(DFlashSpeculator):
    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        with build_attn_metadata_wrapper():
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
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

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
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
        self.input_batch = input_batch
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


# Keep upstream ABI compatibility here. The Triton kernel and Ascend launch
# policy live in ops/triton/v2/spec_decode/prepare_dflash_inputs.py.
prepare_dflash_inputs: Callable[..., None]

if vllm_version_is("0.27.1"):
    prepare_dflash_inputs = prepare_dflash_inputs_triton

else:
    # Main-to-main compatibility only. Upstream extended prepare_dflash_inputs
    # for DSpark + DCP in vllm-project/vllm#52188. vLLM-Ascend has not adapted
    # those semantics yet, so keep using the existing Ascend DFlash behavior.
    logger.warning(
        "The upstream prepare_dflash_inputs ABI includes DSpark + DCP support "
        "introduced by vllm-project/vllm#52188, which is not adapted in "
        "vLLM-Ascend yet. Falling back to the existing Ascend DFlash "
        "implementation; cp_rank, cp_size and cp_interleave are ignored."
    )

    def _prepare_dflash_inputs_main_compat(
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
        # cp_rank/cp_size/cp_interleave are intentionally ignored until
        # DSpark + DCP is adapted in vLLM-Ascend.
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
            parallel_drafting_token_id,
            num_query_per_req,
            num_speculative_steps,
            max_num_reqs,
            max_num_tokens,
            max_model_len,
            sample_from_anchor,
        )

    prepare_dflash_inputs = _prepare_dflash_inputs_main_compat
