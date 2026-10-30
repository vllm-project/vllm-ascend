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
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import get_prebound_copy_sfa_slots
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import CopySfaRequestStates
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_sparse_kv_offload_manager,
    update_sparse_kv_offload_metadata,
)
from vllm_ascend.utils import enable_dsa_cp, should_skip_allreduce_across_dp_group
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, ring_state_update_skipped
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class ReplayAttnMetadata(ModelSpecificAttnMetadata):
    """Pass V4.1 replay boundaries to the SWA metadata builder."""

    def __init__(self, replay_start: torch.Tensor):
        self.replay_start = replay_start

    def get_extra_common_attn_kwargs(self, cache_group_idx: int, num_reqs: int) -> dict[str, Any]:
        del cache_group_idx
        return {"replay_start": self.replay_start[:num_reqs]}


if TYPE_CHECKING:
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata, AscendDSAV41MetadataBuilder
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.models.deepseek_v41.decoder_replay_layers import AscendDecoderReplayLayers
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
        self._initialize_decoder_replay()

    def _initialize_decoder_replay(self) -> None:
        self.decoder_replay: AscendDecoderReplayLayers | None = getattr(self.model, "decoder_replay_layers", None)
        if self.decoder_replay is None:
            return
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        max_capture = self.vllm_config.compilation_config.max_cudagraph_capture_size or 0
        capacity = max(self.max_num_tokens, max_capture)
        capacity = ((capacity + tp_size - 1) // tp_size) * tp_size
        self._decoder_replay_rows = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self._decoder_replay_positions = torch.empty_like(self._decoder_replay_rows)
        self._decoder_replay_offsets = CpuGpuBuffer(
            self.max_num_reqs + 1, dtype=torch.int32, device=self.device, pin_memory=is_pin_memory_available()
        )
        self._decoder_replay_dropped_before = CpuGpuBuffer(
            self.max_num_reqs + 1, dtype=torch.int32, device=self.device, pin_memory=is_pin_memory_available()
        )
        self._decoder_replay_start = torch.empty(self.max_num_reqs, dtype=torch.int32, device=self.device)
        self._decoder_replay_zero_start = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        self._decoder_replay_slot_mappings: torch.Tensor | None = None
        self._decoder_replay_source_slot_mappings: torch.Tensor | None = None
        self._decoder_replay_builders: list[tuple[int, list[str], AscendDSAV41MetadataBuilder]] = []
        self._decoder_replay_common_metadata: dict[int, AscendCommonAttentionMetadata] = {}
        self._decoder_replay_metadata: dict[str, AscendDSAV41Metadata] | None = None
        self._decoder_replay_num_tokens = 0
        self._decoder_replay_num_actual_tokens = 0
        self._decoder_replay_counts: torch.Tensor | None = None
        self._decoder_replay_counts_device: torch.Tensor | None = None
        self.prompt_logprobs_worker = None

    def _initialize_decoder_replay_metadata(self, attn_groups, kv_cache_config) -> None:
        from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder

        assert self.decoder_replay is not None
        self._decoder_replay_slot_mappings = torch.empty(
            (len(kv_cache_config.kv_cache_groups), self._decoder_replay_rows.numel()),
            dtype=torch.int64,
            device=self.device,
        )
        self._decoder_replay_source_slot_mappings = torch.empty_like(self._decoder_replay_slot_mappings)
        self._decoder_replay_builders.clear()
        for group_id, groups in enumerate(attn_groups):
            for group in groups:
                if not isinstance(group.kv_cache_spec, CircularBufferSpec):
                    builder_cls = group.backend.get_builder_cls()
                    if issubclass(builder_cls, AscendDSAV41MetadataBuilder):
                        builder = builder_cls(group.kv_cache_spec, group.layer_names, self.vllm_config, self.device)
                        self._decoder_replay_builders.append((group_id, group.layer_names, builder))
        from vllm_ascend.ops.triton.decoder_replay import gather_replay_batch

        gather_replay_batch(
            self._decoder_replay_offsets.gpu,
            self._decoder_replay_dropped_before.gpu,
            torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device),
            self._decoder_replay_zero_start,
            torch.zeros(self._decoder_replay_rows.numel(), dtype=torch.int64, device=self.device),
            self._decoder_replay_source_slot_mappings,
            self._decoder_replay_rows,
            self._decoder_replay_offsets.gpu,
            self._decoder_replay_positions,
            self._decoder_replay_slot_mappings,
            self._decoder_replay_start,
            self._decoder_replay_rows.numel(),
            self.decoder_replay.window,
            warmup=True,
        )

    def _prepare_decoder_replay(self, input_batch, cudagraph_mode, attn_groups, kv_cache_config) -> None:
        layers = self.decoder_replay
        if layers is None:
            return
        layers.rows = layers.context_factory = None
        if self.vllm_config.parallel_config.data_parallel_size > 1 and should_skip_allreduce_across_dp_group(
            self.vllm_config
        ):
            return
        if not self._decoder_replay_builders:
            self._initialize_decoder_replay_metadata(attn_groups, kv_cache_config)
        if cudagraph_mode != CUDAGraphMode.NONE or getattr(input_batch, "is_dummy", False):
            return

        common_by_group = self._decoder_replay_common_metadata
        real_batch = input_batch.num_reqs > 0 and bool(common_by_group)
        trims = False
        kept_lens = np.empty(0, dtype=np.int32)
        self._decoder_replay_num_actual_tokens = input_batch.num_tokens
        if real_batch:
            num_reqs = input_batch.num_reqs
            offsets = input_batch.query_start_loc_np[: num_reqs + 1]
            query_lens = np.diff(offsets)
            worker = self.prompt_logprobs_worker
            keep_rows = np.zeros(num_reqs, dtype=bool)
            if worker is not None:
                keep_rows = worker.uses_prompt_logprobs[input_batch.idx_mapping_np[:num_reqs]]
            prefills = input_batch.is_prefilling_np[:num_reqs]
            kept_lens = np.where(prefills & ~keep_rows, np.minimum(query_lens, layers.window), query_lens)
            trims = bool((kept_lens < query_lens).any())
            self._decoder_replay_num_actual_tokens = int(kept_lens.sum())
            kept_offsets = self._decoder_replay_offsets.np[: num_reqs + 1]
            kept_offsets[0] = 0
            np.cumsum(kept_lens, out=kept_offsets[1:])
            self._decoder_replay_dropped_before.np[: num_reqs + 1] = offsets - kept_offsets

        self._decoder_replay_num_tokens = self._decoder_replay_num_actual_tokens
        if enable_dsa_cp():
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            self._decoder_replay_num_tokens = ((self._decoder_replay_num_tokens + tp_size - 1) // tp_size) * tp_size
        self._decoder_replay_counts = None
        self._decoder_replay_counts_device = None
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        if dp_size > 1:
            dp = get_dp_group()
            agreed = torch.zeros((2, dp_size), dtype=torch.int32)
            agreed[:, dp.rank_in_group] = torch.tensor([trims, self._decoder_replay_num_tokens], dtype=torch.int32)
            with skip_dp_coordination() if should_skip_allreduce_across_dp_group(self.vllm_config) else nullcontext():
                if not should_skip_allreduce_across_dp_group(self.vllm_config):
                    dist.all_reduce(agreed, group=dp.cpu_group)
            trims = bool(agreed[0].any())
            self._decoder_replay_counts = agreed[1]
            self._decoder_replay_counts_device = agreed[1].to(self.device, non_blocking=True)
        if not trims:
            return
        if real_batch:
            self._decoder_replay_metadata = self._build_decoder_replay_metadata(input_batch, kept_lens)
        else:
            num_tokens = input_batch.num_tokens_after_padding
            torch.arange(num_tokens, out=self._decoder_replay_rows[:num_tokens])
            self._decoder_replay_rows[num_tokens : self._decoder_replay_num_tokens].zero_()
            self._decoder_replay_metadata = self.attn_metadata
        layers.rows = self._decoder_replay_rows[: self._decoder_replay_num_tokens]
        layers.num_actual_rows = self._decoder_replay_num_actual_tokens
        layers.context_factory = self._decoder_replay_context

    def _build_decoder_replay_metadata(self, input_batch, kept_lens):
        from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
        from vllm_ascend.ops.triton.decoder_replay import gather_replay_batch

        assert self.decoder_replay is not None
        assert self._decoder_replay_source_slot_mappings is not None
        assert self._decoder_replay_slot_mappings is not None
        num_reqs = len(kept_lens)
        num_tokens = self._decoder_replay_num_tokens
        self._decoder_replay_dropped_before.copy_to_gpu(num_reqs + 1)
        source_slots = self._decoder_replay_source_slot_mappings[:, : input_batch.num_tokens]
        for group_id in range(len(self.block_tables)):
            source_slots[group_id].copy_(self.slot_mappings[group_id][: input_batch.num_tokens])
        slots = self._decoder_replay_slot_mappings[:, :num_tokens]
        original = next(iter(self._decoder_replay_common_metadata.values()))
        gather_replay_batch(
            original.query_start_loc[: num_reqs + 1],
            self._decoder_replay_dropped_before.gpu[: num_reqs + 1],
            original.seq_lens[:num_reqs],
            original.replay_start if original.replay_start is not None else self._decoder_replay_zero_start[:num_reqs],
            original.positions,
            source_slots,
            self._decoder_replay_rows,
            self._decoder_replay_offsets.gpu,
            self._decoder_replay_positions,
            slots,
            self._decoder_replay_start,
            num_tokens,
            self.decoder_replay.window,
        )
        common = original.replace(
            query_start_loc=self._decoder_replay_offsets.gpu[: num_reqs + 1],
            query_start_loc_cpu=self._decoder_replay_offsets.cpu[: num_reqs + 1],
            seq_lens=original.seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=self._decoder_replay_num_actual_tokens,
            num_input_tokens=num_tokens,
            positions=self._decoder_replay_positions[:num_tokens],
            positions_cpu=None,
            max_query_len=int(kept_lens.max()),
            is_prefilling=original.is_prefilling[:num_reqs],
            actual_seq_lengths_q=self._decoder_replay_offsets.np[1 : num_reqs + 1].tolist(),
        )
        rope = get_cos_and_sin_dsa(common.positions, use_cache=False)
        metadata = {}
        batch_shared: dict[str, Any] = {}
        group_shared: dict[int, dict[str, Any]] = {}
        for group_id, layer_names, builder in self._decoder_replay_builders:
            group_common = common.replace(
                block_table_tensor=self.block_tables[group_id][:num_reqs],
                slot_mapping=slots[group_id],
            )
            value = builder.build(
                0,
                group_common,
                num_actual_reqs=num_reqs,
                common_v41_batch_metadata=batch_shared,
                common_v41_metadata=group_shared.setdefault(group_id, {}),
                rope_views=rope,
                replay_start=self._decoder_replay_start[:num_reqs],
                replay_seq_lens=original.seq_lens[:num_reqs],
                replay_max_seq_len=int(kept_lens.max()) + self.decoder_replay.window - 1,
            )
            metadata.update({name: value for name in layer_names})
        return metadata

    @contextmanager
    def _decoder_replay_context(self):
        outer = get_forward_context()
        assert self._decoder_replay_slot_mappings is not None
        slot_mapping = {
            name: self._decoder_replay_slot_mappings[group_id, : self._decoder_replay_num_tokens]
            for group_id, names, _ in self._decoder_replay_builders
            for name in names
        }
        with set_forward_context(
            self._decoder_replay_metadata,
            self.vllm_config,
            num_tokens=self._decoder_replay_num_tokens,
            num_tokens_across_dp=self._decoder_replay_counts_device,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=slot_mapping,
            is_padding=outer.is_padding,
        ):
            yield

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

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states) -> torch.Tensor:
        # MRV2 prepares MM embeddings before entering ForwardContext. Upstream
        # slices input_ids to valid local rows to match the image mask, leaving
        # different input lengths on PCP ranks. Publish the common padded length
        # (including graph padding) before that slice, and overwrite it each batch.
        set_embedding_num_tokens_padded = getattr(self.model, "set_embedding_num_tokens_padded", None)
        if set_embedding_num_tokens_padded is not None:
            set_embedding_num_tokens_padded(input_batch.num_tokens_after_padding)
        return super().prepare_inputs_embeds(scheduled_encoder_inputs, input_batch, req_states)

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
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
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

        # TODO: Remove the PIECEWISE dummy condition once the common metadata
        # path handles padded attention inputs separately from logical queries.
        if (
            cudagraph_mode == CUDAGraphMode.FULL
            or self.vllm_config.parallel_config.prefill_context_parallel_size > 1
            or (cudagraph_mode == CUDAGraphMode.PIECEWISE and input_batch.is_dummy)
        ):
            # Token-shaped metadata must match the padded model inputs.
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
        self._decoder_replay_common_metadata = {}
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
            model_specific_attn_metadata=model_specific_attn_metadata,
            req_ids_tensor=offload_req_ids.gpu[:num_reqs] if offload_req_ids is not None else None,
            token_to_req=offload_token_to_req.gpu[:num_input_tokens] if offload_token_to_req is not None else None,
            offload_dummy=getattr(input_batch, "is_dummy", False),
            req_topk_buffer_slots=pool_slots.cpu[:num_reqs] if pool_slots is not None else None,
            req_topk_buffer_active=pool_active.cpu[:num_reqs] if pool_active is not None else None,
            copy_sfa_restore_tails=restore_copy_sfa_tails,
            draft_layer_names=getattr(self, "_offload_draft_layer_names", None),
            common_metadata_callback=(
                lambda group_id, common: self._decoder_replay_common_metadata.__setitem__(group_id, common)
                if self.decoder_replay is not None
                else None
            ),
        )
        self._prepare_decoder_replay(input_batch, cudagraph_mode, attn_groups, kv_cache_config)
        if restore_copy_sfa_tails:
            manager = get_sparse_kv_offload_manager()
            for metadata in self.attn_metadata.values():
                if getattr(metadata, "copy_sfa_copy_src_offsets", None) is not None:
                    manager.restore_copy_sfa_tails(metadata)
                    break
        return self.attn_metadata
