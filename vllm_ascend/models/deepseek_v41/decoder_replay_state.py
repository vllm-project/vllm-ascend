# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import get_dp_group, get_pcp_group, get_tp_group, tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.utils.math_utils import round_up
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import get_mc2_mask, get_mrv2_in_profile_run
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata, AscendDSAV41MetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.ops.triton.decoder_replay import gather_replay_batch
from vllm_ascend.utils import enable_dsa_cp, should_skip_allreduce_across_dp_group
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from .decoder_replay_layers import AscendDecoderReplayLayers

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


@dataclass(frozen=True)
class DecoderReplayInputs:
    batch: AscendInputBatch
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    attn_groups: list[list[AttentionGroup]]
    encoder_replay_start: torch.Tensor
    pcp_context: "AscendPCPAttentionContext | None" = None


class DecoderReplayState:
    def __init__(self, config: VllmConfig, layers: AscendDecoderReplayLayers, device: torch.device) -> None:
        self.config = config
        self.layers = layers
        num_reqs = config.scheduler_config.max_num_seqs
        local_reqs = num_reqs * (2 if getattr(config.parallel_config, "prefill_context_parallel_size", 1) > 1 else 1)
        capacity = round_up(config.scheduler_config.max_num_batched_tokens, config.parallel_config.tensor_parallel_size)
        self.keeps_rows = np.zeros(num_reqs, dtype=np.bool_)
        self.rows = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.positions = torch.zeros_like(self.rows)
        self.offsets = torch.zeros(local_reqs + 1, dtype=torch.int32, device=device)
        self.dropped_before = CpuGpuBuffer(local_reqs + 1, dtype=torch.int32, device=device, pin_memory=False)
        self.replay_start = torch.zeros(local_reqs, dtype=torch.int32, device=device)
        self._pcp_replay_start = CpuGpuBuffer(local_reqs, dtype=torch.int32, device=device, pin_memory=False)
        self.slots: torch.Tensor | None = None
        self.builders: list[tuple[int, list[str], AscendDSAV41MetadataBuilder]] = []
        self.metadata: dict[str, AscendDSAV41Metadata] = {}
        self.num_tokens = 0
        self.num_actual_tokens = 0
        self.dp_tokens: torch.Tensor | None = None
        self.replay_pcp_context: AscendPCPAttentionContext | None = None

    def _pcp_kept_lengths(self, inputs: DecoderReplayInputs) -> tuple[np.ndarray, int, torch.Tensor] | None:
        context = inputs.pcp_context
        if context is None:
            return None
        batch = inputs.batch
        global_batch = context.global_batch
        num_global_reqs = global_batch.num_reqs
        global_lens = np.diff(global_batch.query_start_loc_np[: num_global_reqs + 1])
        trim = (
            global_batch.is_prefilling_np[:num_global_reqs]
            & ~self.keeps_rows[global_batch.idx_mapping_np[:num_global_reqs]]
        )
        global_kept = np.where(trim, np.minimum(global_lens, self.layers.window), global_lens)
        if not (global_kept < global_lens).any():
            return None
        num_padded = int(global_kept.sum())
        if num_padded >= batch.num_tokens_after_padding:
            return None

        starts = np.zeros(self.keeps_rows.size, dtype=np.int32)
        starts[global_batch.idx_mapping_np[:num_global_reqs]] = np.where(
            global_kept < global_lens,
            global_batch.seq_lens_np[:num_global_reqs] - self.layers.window,
            0,
        )
        local_starts = starts[batch.idx_mapping_np[: batch.num_reqs]]
        local_lens = np.diff(batch.query_start_loc_np[: batch.num_reqs + 1])
        kept_lens = np.clip(batch.seq_lens_np[: batch.num_reqs] - local_starts, 0, local_lens)
        self._pcp_replay_start.np[: batch.num_reqs] = local_starts
        self._pcp_replay_start.copy_to_gpu(batch.num_reqs)
        replay_start = torch.maximum(
            inputs.encoder_replay_start[: batch.num_reqs],
            self._pcp_replay_start.gpu[: batch.num_reqs],
        )
        return kept_lens, num_padded, replay_start

    def _build_replay_pcp_context(
        self, inputs: DecoderReplayInputs, global_kept_lens: np.ndarray
    ) -> "AscendPCPAttentionContext":
        context = inputs.pcp_context
        assert context is not None and context.local_token_indices is not None
        global_batch = context.global_batch
        num_reqs = global_batch.num_reqs
        global_offsets = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(global_kept_lens, out=global_offsets[1:])
        source_rows = np.concatenate(
            [
                np.arange(
                    global_batch.query_start_loc_np[req + 1] - kept,
                    global_batch.query_start_loc_np[req + 1],
                    dtype=np.int64,
                )
                for req, kept in enumerate(global_kept_lens)
            ]
        )
        global_rows = torch.as_tensor(source_rows, device=self.rows.device)
        local_rows = context.local_token_indices.index_select(0, self.rows[: self.num_actual_tokens])
        padded_rows = torch.full(
            (self.num_tokens,), global_batch.num_tokens, dtype=torch.int64, device=self.rows.device
        )
        padded_rows[: self.num_actual_tokens].copy_(local_rows)
        gathered_rows = get_pcp_group().all_gather(padded_rows.contiguous(), dim=0)
        order = torch.argsort(gathered_rows)
        restore_idx = order[torch.searchsorted(gathered_rows[order], global_rows)]
        # replace keeps the runtime subclass, while mypy needs the base dataclass
        # to resolve its inherited fields.
        compact_batch = cast(
            AscendInputBatch,
            replace(
                cast(InputBatch, global_batch),
                num_tokens=int(global_kept_lens.sum()),
                num_tokens_after_padding=int(global_kept_lens.sum()),
                num_scheduled_tokens=global_kept_lens,
                num_computed_tokens_np=global_batch.seq_lens_np[:num_reqs] - global_kept_lens,
                query_start_loc_np=global_offsets,
                query_start_loc=torch.as_tensor(global_offsets, device=self.rows.device),
                input_ids=global_batch.input_ids.index_select(0, global_rows),
                positions=global_batch.positions.index_select(0, global_rows),
                max_query_len=int(global_kept_lens.max(initial=0)),
            ),
        )
        return replace(
            context,
            global_batch=compact_batch,
            global_slot_mappings=context.global_slot_mappings.index_select(1, global_rows),
            hidden_restore_idx=restore_idx,
            padded_gather_idx=None,
            gathered_kv_write_mask=None,
        )

    def _initialize(self, inputs: DecoderReplayInputs) -> None:
        self.slots = torch.empty(
            (inputs.slot_mappings.shape[0], self.rows.numel()), dtype=torch.int64, device=self.rows.device
        )
        with set_current_vllm_config(self.config):
            for group_id, groups in enumerate(inputs.attn_groups):
                for group in groups:
                    if is_circular_kv_cache_spec(group.kv_cache_spec):
                        continue
                    builder_cls = group.backend.get_builder_cls()
                    if issubclass(builder_cls, AscendDSAV41MetadataBuilder):
                        builder = builder_cls(group.kv_cache_spec, group.layer_names, self.config, self.rows.device)
                        builder.prepare_source_rope()
                        self.builders.append((group_id, group.layer_names, builder))
        self._gather(inputs, warmup=True)

    def _gather(self, inputs: DecoderReplayInputs, *, warmup: bool = False) -> None:
        batch = inputs.batch
        assert self.slots is not None
        source_slots = inputs.slot_mappings
        if inputs.pcp_context is not None:
            pcp_group = get_pcp_group()
            source_slots = source_slots.chunk(pcp_group.world_size, dim=1)[pcp_group.rank_in_group]
        gather_replay_batch(
            batch.query_start_loc[: batch.num_reqs + 1],
            self.dropped_before.gpu[: batch.num_reqs + 1],
            batch.seq_lens,
            inputs.encoder_replay_start,
            batch.positions,
            source_slots,
            self.rows,
            self.offsets,
            self.positions,
            self.slots,
            self.replay_start,
            self.num_tokens,
            self.layers.window,
            warmup=warmup,
        )

    def prepare(self, inputs: DecoderReplayInputs, mode: CUDAGraphMode, *, for_capture: bool = False) -> None:
        self.layers.rows = None
        self.layers.context_factory = None
        self.layers.num_padded_rows = None
        self.replay_pcp_context = None
        if self.slots is None:
            self._initialize(inputs)
        if for_capture or mode != CUDAGraphMode.NONE or get_mrv2_in_profile_run():
            return
        parallel = self.config.parallel_config
        if parallel.data_parallel_size > 1 and should_skip_allreduce_across_dp_group(self.config):
            return
        batch = inputs.batch
        num_reqs = batch.num_reqs
        query_lens = np.diff(batch.query_start_loc_np[: num_reqs + 1])
        pcp_replay = self._pcp_kept_lengths(inputs)
        if pcp_replay is None:
            trim_requests = batch.is_prefilling_np[:num_reqs] & ~self.keeps_rows[batch.idx_mapping_np[:num_reqs]]
            if batch.is_dummy or inputs.pcp_context is not None:
                trim_requests = np.zeros(num_reqs, dtype=np.bool_)
            kept_lens = np.where(trim_requests, np.minimum(query_lens, self.layers.window), query_lens)
            trims = bool((kept_lens < query_lens).any())
            replay_inputs = inputs
        else:
            kept_lens, num_padded, replay_start = pcp_replay
            trims = True
            replay_inputs = replace(inputs, encoder_replay_start=replay_start)
        self.num_actual_tokens = int(kept_lens.sum())
        self.num_tokens = self.num_actual_tokens
        if pcp_replay is not None:
            self.num_tokens = num_padded
        elif enable_dsa_cp():
            self.num_tokens = round_up(self.num_tokens, parallel.tensor_parallel_size)
        self.dp_tokens = None
        if parallel.data_parallel_size > 1:
            agreed = torch.zeros((2, parallel.data_parallel_size), dtype=torch.int32)
            agreed[:, get_dp_group().rank_in_group] = torch.tensor([trims, self.num_tokens])
            if not should_skip_dp_coordination():
                dist.all_reduce(agreed, group=get_dp_group().cpu_group)
            trims = bool(agreed[0].any())
            self.dp_tokens = agreed[1]
        if not trims:
            return
        offsets = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(kept_lens, out=offsets[1:])
        dropped = batch.query_start_loc_np[: num_reqs + 1] - offsets
        self.dropped_before.np[: num_reqs + 1] = dropped
        self.dropped_before.copy_to_gpu(num_reqs + 1)
        self._gather(replay_inputs)
        if pcp_replay is not None:
            assert inputs.pcp_context is not None
            global_batch = inputs.pcp_context.global_batch
            global_lens = np.diff(global_batch.query_start_loc_np[: global_batch.num_reqs + 1])
            global_trim = (
                global_batch.is_prefilling_np[: global_batch.num_reqs]
                & ~self.keeps_rows[global_batch.idx_mapping_np[: global_batch.num_reqs]]
            )
            global_kept = np.where(global_trim, np.minimum(global_lens, self.layers.window), global_lens)
            self.replay_pcp_context = self._build_replay_pcp_context(inputs, global_kept)
            self.layers.num_padded_rows = self.num_tokens
        self.metadata = self._build_metadata(replay_inputs, offsets)
        self.layers.rows = self.rows[: self.num_actual_tokens if pcp_replay is not None else self.num_tokens]
        self.layers.num_actual_rows = self.num_actual_tokens
        self.layers.context_factory = self.forward_context

    def _build_metadata(self, inputs: DecoderReplayInputs, offsets: np.ndarray) -> dict[str, AscendDSAV41Metadata]:
        batch = inputs.batch
        num_reqs = batch.num_reqs
        assert self.slots is not None
        positions = self.positions[: self.num_tokens]
        slot_mappings = self.slots
        if self.replay_pcp_context is not None:
            slot_mappings = get_pcp_group().all_gather(self.slots[:, : self.num_tokens].contiguous(), dim=1)
        rope = get_cos_and_sin_dsa(positions, use_cache=False)
        batch_shared: dict[str, Any] = {}
        group_shared: dict[int, dict[str, Any]] = {}
        metadata: dict[str, AscendDSAV41Metadata] = {}
        for group_id, layer_names, builder in self.builders:
            common = AscendCommonAttentionMetadata(
                query_start_loc=self.offsets[: num_reqs + 1],
                query_start_loc_cpu=torch.from_numpy(offsets),
                seq_lens=batch.seq_lens[:num_reqs],
                seq_lens_cpu=torch.from_numpy(batch.seq_lens_np[:num_reqs]),
                num_reqs=num_reqs,
                num_actual_tokens=self.num_actual_tokens,
                num_input_tokens=self.num_tokens,
                max_query_len=max(int(np.diff(offsets).max()), batch.max_query_len or 0),
                max_seq_len=self.config.model_config.max_model_len,
                block_table_tensor=inputs.block_tables[group_id][:num_reqs],
                slot_mapping=slot_mappings[group_id],
                positions=positions,
                attn_state=batch.attn_state,
                is_prefilling=torch.from_numpy(batch.is_prefilling_np[:num_reqs]),
                replay_start=self.replay_start[:num_reqs],
            )
            build_kwargs = {}
            if self.replay_pcp_context is not None:
                build_kwargs = {"pcp_context": self.replay_pcp_context, "pcp_cache_group_idx": group_id}
            value = builder.build(
                0,
                common,
                num_actual_reqs=num_reqs,
                common_v41_batch_metadata=batch_shared,
                common_v41_metadata=group_shared.setdefault(group_id, {}),
                rope_views=rope,
                **build_kwargs,
            )
            for task in builder.take_device_metadata_tasks():
                task.run()
            for name in layer_names:
                metadata[name] = value
        return metadata

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        outer = get_forward_context()
        mask = get_mc2_mask()
        saved_mask = mask.clone() if mask is not None else None
        try:
            with set_forward_context(
                self.metadata,
                self.config,
                num_tokens=self.num_tokens,
                num_tokens_across_dp=self.dp_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                context = get_forward_context()
                context.additional_kwargs["eplb_heat_collection_status"] = outer.additional_kwargs.get(
                    "eplb_heat_collection_status", False
                )
                replay_mask = context.additional_kwargs.get("mc2_mask")
                if replay_mask is not None:
                    replay_mask[self.num_actual_tokens :].fill_(False)
                if outer.is_padding is not None:
                    padding = outer.is_padding
                    if self.layers.sequence_parallel:
                        padding = tensor_model_parallel_all_gather(padding, dim=0)
                    context.is_padding = padding.index_select(0, self.rows[: self.num_tokens])
                    context.is_padding[self.num_actual_tokens :].fill_(True)
                    if self.layers.sequence_parallel:
                        tp = get_tp_group()
                        context.is_padding = context.is_padding.chunk(tp.world_size)[tp.rank_in_group]
                yield
        finally:
            if saved_mask is not None:
                mask.copy_(saved_mask)
