# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/default.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, ring_state_update_skipped
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.kvpp import KVPPRuntime
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext, AscendPCPManager


class AscendModelState(DefaultModelState):
    """Model state for Ascend NPUs."""

    pcp_manager: "AscendPCPManager | None" = None
    pcp_context: "AscendPCPAttentionContext | None" = None
    kvpp_runtime: "KVPPRuntime | None" = None
    kvpp_is_dummy_run: bool = False

    def _get_engram_device_inputs(self, input_batch: AscendInputBatch) -> dict[str, torch.Tensor]:
        """Device request coordinates for upstream NgramHashState."""
        layer_name = getattr(self.model, "engram_cache_layer_name", None)
        kv_cache_config = getattr(self, "kv_cache_config", None)
        if layer_name is None or kv_cache_config is None:
            return {}
        if self.kvpp_is_dummy_run or ring_state_update_skipped():
            return {}
        group_id = next(
            (
                group_id
                for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
                if layer_name in group.layer_names
            ),
            None,
        )
        if group_id is None:
            return {}
        if self.pcp_context is not None:
            batch = self.pcp_context.global_batch
            block_tables = self.pcp_context.global_block_tables
            slot_mappings = self.pcp_context.global_slot_mappings
        else:
            batch = input_batch
            block_tables = getattr(self, "block_tables", None)
            slot_mappings = getattr(self, "slot_mappings", None)
        if block_tables is None or slot_mappings is None or group_id >= len(block_tables):
            return {}
        return {
            "query_start_loc": batch.query_start_loc,
            "slot_mapping": slot_mappings[group_id],
            "block_table": block_tables[group_id][: batch.num_reqs],
        }

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        prepare_engram_inputs = getattr(self.model, "prepare_engram_inputs", None)
        if prepare_engram_inputs is None:
            return model_inputs
        num_tokens = input_batch.num_tokens_after_padding
        model_inputs.update(
            prepare_engram_inputs(
                input_batch.input_ids[:num_tokens],
                input_batch.positions[:num_tokens],
                num_tokens,
                **self._get_engram_device_inputs(input_batch),
            )
        )
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        prepare_engram_graph_inputs = getattr(self.model, "prepare_engram_graph_inputs", None)
        if prepare_engram_graph_inputs is not None:
            model_inputs.update(prepare_engram_graph_inputs(num_tokens))
        return model_inputs

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        """Override prepare_attn method because `build_attn_metadata` is different from vllm."""
        # vLLM #50945 adds this contract; Ascend still disables DBO.
        assert ubatch_idx == 0, "DBO is not supported on Ascend"
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = input_batch.num_reqs_after_padding
        else:
            # Piecewise cudagraphs and eager use the actual request count.
            num_reqs = input_batch.num_reqs

        if cudagraph_mode == CUDAGraphMode.FULL or self.vllm_config.parallel_config.prefill_context_parallel_size > 1:
            # PCP pads each rank to the largest rank-local token count even
            # during eager prefill, so token-shaped metadata must match the
            # padded model input.
            num_input_tokens = input_batch.num_tokens_after_padding
        else:
            num_input_tokens = input_batch.num_tokens

        num_actual_reqs = input_batch.num_reqs
        num_actual_tokens = input_batch.num_tokens
        if self.kvpp_runtime is not None and self.kvpp_runtime.scheduler is not None:
            # PCP-local offsets include earlier chunks of this same forward.
            # Use prior-forward history shared by every PCP x TP group member.
            history_batch = (
                self.pcp_manager.global_batch
                if self.pcp_manager is not None and not self.kvpp_is_dummy_run
                else input_batch
            )
            self.kvpp_runtime.prepare_forward(
                not self.kvpp_is_dummy_run
                and bool(np.any(history_batch.num_computed_tokens_np[: history_batch.num_reqs] > 0))
            )
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        is_prefilling = torch.from_numpy(input_batch.is_prefilling_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        pcp_context = (
            self.pcp_manager.build_attention_context(input_batch, block_tables, slot_mappings)
            if self.pcp_manager is not None
            else None
        )
        # attn_metadata is needed when update_full_graph_params, but no way can get it now.
        # Temporarily store it in model_state.
        self.block_tables = block_tables
        self.slot_mappings = slot_mappings
        self.kv_cache_config = kv_cache_config
        self.pcp_context = pcp_context
        self.attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_tokens=num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            is_prefilling=is_prefilling,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            parallel_config=self.vllm_config.parallel_config,
            # extra attributes for ascend npus.
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            pcp_context=pcp_context,
            for_cudagraph_capture=for_capture,
            # Same wiring as model_runner_v1
            full_graph_mode=cudagraph_mode == CUDAGraphMode.FULL,
        )
        return self.attn_metadata
