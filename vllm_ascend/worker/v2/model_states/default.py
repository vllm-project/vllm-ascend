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

from collections.abc import Callable, Collection
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import get_prebound_copy_sfa_slots
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import CopySfaRequestStates
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_sparse_kv_offload_manager,
    update_sparse_kv_offload_metadata,
)
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, ring_state_update_skipped
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

if TYPE_CHECKING:
    from vllm_ascend.worker.device_metadata import TargetDeviceMetadata
    from vllm_ascend.worker.v2.kvpp import KVPPRuntime
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext, AscendPCPManager


class AscendModelState(DefaultModelState):
    """Model state for Ascend NPUs."""

    pcp_manager: "AscendPCPManager | None" = None
    pcp_context: "AscendPCPAttentionContext | None" = None
    kvpp_runtime: "KVPPRuntime | None" = None
    kvpp_is_dummy_run: bool = False
    device_metadata: "TargetDeviceMetadata | None" = None

    def finish_execution(self, *, failed: bool) -> None:
        """Join auxiliary metadata work even if input preparation failed."""
        if self.device_metadata is not None:
            self.device_metadata.finish()

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._offload_req_ids: CpuGpuBuffer | None = None
        self._offload_token_to_req: CpuGpuBuffer | None = None
        self._offload_pool_slots: CpuGpuBuffer | None = None
        self._offload_pool_active: CpuGpuBuffer | None = None
        self._offload_request_states: CopySfaRequestStates | None = None
        self._offload_live_req_ids: Collection[str] = ()
        self._offload_draft_attn_groups: list[list[AttentionGroup]] = []
        self._offload_draft_layer_names: set[str] = set()
        sparse_cfg = get_ascend_config().sparse_kv_offload_config
        if sparse_cfg.enabled:
            pin_memory = is_pin_memory_available()
            self._offload_req_ids = CpuGpuBuffer(
                vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int64,
                device=device,
                pin_memory=pin_memory,
            )
            self._offload_token_to_req = CpuGpuBuffer(
                vllm_config.scheduler_config.max_num_batched_tokens,
                dtype=torch.int32,
                device=device,
                pin_memory=pin_memory,
            )
            if sparse_cfg.use_fused_copy_sfa:
                pool_capacity = vllm_config.scheduler_config.max_num_seqs + 2
                self._offload_pool_slots = CpuGpuBuffer(
                    pool_capacity, dtype=torch.int32, device=device, pin_memory=pin_memory
                )
                self._offload_pool_active = CpuGpuBuffer(
                    pool_capacity, dtype=torch.bool, device=device, pin_memory=pin_memory
                )
                self._offload_request_states = CopySfaRequestStates()

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        if self._offload_request_states is not None:
            self._offload_request_states.remove_request(req_id)

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
        block_tables: tuple[torch.Tensor, ...] | None
        slot_mappings: torch.Tensor | None
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
        model_inputs.update(self.prepare_engram_inputs(input_batch, req_states))
        return model_inputs

    def prepare_engram_inputs(self, input_batch, req_states) -> dict[str, Any]:
        """Model-specific history providers override this single preparation hook."""
        prepare_engram_inputs = getattr(self.model, "prepare_engram_inputs", None)
        if prepare_engram_inputs is None:
            return {}
        num_tokens = input_batch.num_tokens_after_padding
        return prepare_engram_inputs(
            input_batch.input_ids[:num_tokens],
            input_batch.positions[:num_tokens],
            num_tokens,
            **self._get_engram_device_inputs(input_batch),
        )

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        model_inputs.update(self.prepare_engram_dummy_inputs(num_reqs, num_tokens))
        return model_inputs

    def prepare_engram_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        prepare_engram_graph_inputs = getattr(self.model, "prepare_engram_graph_inputs", None)
        if prepare_engram_graph_inputs is not None:
            return prepare_engram_graph_inputs(num_tokens)
        return {}

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
        offload_req_ids = getattr(self, "_offload_req_ids", None)
        offload_token_to_req = getattr(self, "_offload_token_to_req", None)
        if offload_req_ids is not None:
            assert offload_token_to_req is not None
            update_sparse_kv_offload_metadata(
                input_batch.num_tokens,
                input_batch.num_reqs,
                num_input_tokens,
                num_reqs,
                input_batch.req_ids,
                input_batch.query_start_loc_np,
                offload_req_ids,
                offload_token_to_req,
            )
        pool_slots = getattr(self, "_offload_pool_slots", None)
        pool_active = getattr(self, "_offload_pool_active", None)
        request_states = getattr(self, "_offload_request_states", None)
        restore_copy_sfa_tails = False
        if request_states is not None:
            assert pool_slots is not None and pool_active is not None
            offload_dummy = getattr(input_batch, "is_dummy", False)
            histories = [
                history
                for groups in [*attn_groups, *getattr(self, "_offload_draft_attn_groups", [])]
                for group in groups
                for builder in group.metadata_builders
                if (history := getattr(builder, "lim_last_cache", None)) is not None
            ]
            restore_copy_sfa_tails, dense_fills, _ = request_states.prepare(
                req_ids=input_batch.req_ids[: input_batch.num_reqs],
                live_req_ids=self._offload_live_req_ids,
                slots=pool_slots.np,
                active=pool_active.np,
                prebound_slots=get_prebound_copy_sfa_slots() if not offload_dummy else {},
                computed_tokens=input_batch.num_computed_tokens_np if not offload_dummy else None,
                padded_reqs=num_reqs,
                block_size=self.vllm_config.cache_config.block_size,
                hot_tokens=get_ascend_config().sparse_kv_offload_config.topk_buffer_size,
                dummy=offload_dummy,
                lim_cache_histories=histories,
            )
            if dense_fills:
                get_sparse_kv_offload_manager().dense_fill_copy_sfa_rows(
                    dense_fills,
                    block_size=self.vllm_config.cache_config.block_size,
                    block_table=block_tables[0],
                )
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
        build_metadata: Callable[..., Any] = build_attn_metadata
        if self.device_metadata is not None:
            build_metadata = partial(self.device_metadata.run_build, build_attn_metadata)
        self.attn_metadata = build_metadata(
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
            req_ids_tensor=offload_req_ids.gpu[:num_reqs] if offload_req_ids is not None else None,
            token_to_req=offload_token_to_req.gpu[:num_input_tokens] if offload_token_to_req is not None else None,
            offload_dummy=getattr(input_batch, "is_dummy", False),
            req_topk_buffer_slots=pool_slots.cpu[:num_reqs] if pool_slots is not None else None,
            req_topk_buffer_active=pool_active.cpu[:num_reqs] if pool_active is not None else None,
            copy_sfa_restore_tails=restore_copy_sfa_tails,
            draft_layer_names=getattr(self, "_offload_draft_layer_names", None),
        )
        if restore_copy_sfa_tails:
            manager = get_sparse_kv_offload_manager()
            for metadata in self.attn_metadata.values():
                if getattr(metadata, "copy_sfa_copy_src_offsets", None) is not None:
                    manager.restore_copy_sfa_tails(metadata)
                    break
        return self.attn_metadata
