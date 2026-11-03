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
from vllm_ascend.utils import should_skip_allreduce_across_dp_group
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


@dataclass(frozen=True)
class _DecoderReplayPlan:
    # Retained query token counts for each request on this rank.
    kept_lens: np.ndarray
    # PCP replay may replace encoder_replay_start.
    replay_inputs: DecoderReplayInputs
    # Execution length, including padding rather than only valid tokens.
    num_tokens: int
    # PCP ranks may need a compact layout even when they retain no valid tokens.
    requires_compaction: bool
    # Whether PCP replay is active, not merely whether a PCP context exists.
    uses_pcp_replay: bool


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
        self.padding = torch.ones(capacity, dtype=torch.bool, device=device)
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
        if context is None or inputs.batch.is_dummy:
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
        # Default to the original batch so no replay state leaks across steps.
        self.layers.rows = None
        self.layers.context_factory = None
        self.layers.num_padded_rows = None
        self.replay_pcp_context = None
        self.dp_tokens = None
        self.metadata = {}

        # Initialize before early exits: capture and profile may also need warmup.
        if self.slots is None:
            self._initialize(inputs)
        if for_capture or mode != CUDAGraphMode.NONE or get_mrv2_in_profile_run():
            return

        # Compaction requires coordinated execution lengths across DP replicas.
        parallel = self.config.parallel_config
        if parallel.data_parallel_size > 1 and (
            should_skip_allreduce_across_dp_group(self.config) or should_skip_dp_coordination()
        ):
            return

        plan = self._build_replay_plan(inputs)
        self.num_actual_tokens = int(plan.kept_lens.sum())
        self.num_tokens = plan.num_tokens

        # Dummy batches have no valid tokens but may still need collective padding.
        if inputs.batch.is_dummy:
            self.num_actual_tokens = 0

        # No replica needs compaction; keep the original execution context.
        any_replica_compacts = self._coordinate_replay_across_dp(plan)
        if not any_replica_compacts:
            return

        # A peer compacted its batch; only update the cross-DP token counts locally.
        # Keep rows unset to reuse attention metadata, including dummy PCP contexts
        # that need not provide token ownership indices.
        if not plan.requires_compaction:
            self.layers.context_factory = self.forward_context
            return

        # This rank needs a compact batch and matching attention metadata.
        self._prepare_compact_replay(inputs, plan)

    def _build_replay_plan(self, inputs: DecoderReplayInputs) -> _DecoderReplayPlan:
        batch = inputs.batch
        num_reqs = batch.num_reqs
        query_lens = np.diff(batch.query_start_loc_np[: num_reqs + 1])
        pcp_replay = self._pcp_kept_lengths(inputs)
        if pcp_replay is not None:
            kept_lens, num_padded, replay_start = pcp_replay
            # PCP retains owners of the global tail window; empty ranks still participate.
            return _DecoderReplayPlan(
                kept_lens=kept_lens,
                replay_inputs=replace(inputs, encoder_replay_start=replay_start),
                num_tokens=num_padded,
                requires_compaction=True,
                uses_pcp_replay=True,
            )

        # Trim only prefills that do not need full outputs. Map batch indices to request slots.
        request_indices = batch.idx_mapping_np[:num_reqs]
        trim_requests = batch.is_prefilling_np[:num_reqs] & ~self.keeps_rows[request_indices]
        if batch.is_dummy or inputs.pcp_context is not None:
            # Without a PCP replay plan, do not independently trim each rank's local tail.
            trim_requests = np.zeros(num_reqs, dtype=np.bool_)
        kept_lens = np.where(trim_requests, np.minimum(query_lens, self.layers.window), query_lens)
        requires_compaction = bool((kept_lens < query_lens).any())

        if not requires_compaction:
            # Preserve collective width, including padding on empty PCP ranks and idle DP replicas.
            num_tokens = batch.num_tokens_after_padding
        else:
            num_tokens = int(kept_lens.sum())

        return _DecoderReplayPlan(
            kept_lens=kept_lens,
            replay_inputs=inputs,
            num_tokens=num_tokens,
            requires_compaction=requires_compaction,
            uses_pcp_replay=False,
        )

    def _coordinate_replay_across_dp(self, plan: _DecoderReplayPlan) -> bool:
        dp_size = self.config.parallel_config.data_parallel_size
        if dp_size == 1:
            return plan.requires_compaction

        dp_group = get_dp_group()
        # Each rank writes its own column; reduction shares all replicas' values.
        # Row 0 records compaction flags; row 1 records padded execution lengths.
        replica_info = torch.zeros((2, dp_size), dtype=torch.int32)
        replica_info[:, dp_group.rank_in_group] = torch.tensor(
            [plan.requires_compaction, plan.num_tokens], dtype=torch.int32
        )
        dist.all_reduce(replica_info, group=dp_group.cpu_group)
        self.dp_tokens = replica_info[1]
        return bool(replica_info[0].any())

    def _prepare_compact_replay(self, inputs: DecoderReplayInputs, plan: _DecoderReplayPlan) -> None:
        batch = inputs.batch
        num_reqs = batch.num_reqs
        # Convert retained lengths to query boundaries: [1, 6, 2] becomes [0, 1, 7, 9].
        offsets = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(plan.kept_lens, out=offsets[1:])
        # Cumulative dropped rows at each request boundary describe the new layout to the kernel.
        dropped = batch.query_start_loc_np[: num_reqs + 1] - offsets
        self.dropped_before.np[: num_reqs + 1] = dropped
        self.dropped_before.copy_to_gpu(num_reqs + 1)

        # Prepare row indices, positions, slot mappings, and replay starts.
        # The layer adapter gathers hidden states and executes the decoder later.
        self._gather(plan.replay_inputs)
        if plan.uses_pcp_replay:
            assert inputs.pcp_context is not None
            global_batch = inputs.pcp_context.global_batch
            num_global_reqs = global_batch.num_reqs
            global_lens = np.diff(global_batch.query_start_loc_np[: num_global_reqs + 1])
            global_request_indices = global_batch.idx_mapping_np[:num_global_reqs]
            global_trim = global_batch.is_prefilling_np[:num_global_reqs] & ~self.keeps_rows[global_request_indices]
            global_kept = np.where(global_trim, np.minimum(global_lens, self.layers.window), global_lens)
            # Update PCP's global batch and cross-rank restore indices to match the compact layout.
            self.replay_pcp_context = self._build_replay_pcp_context(inputs, global_kept)
            self.layers.num_padded_rows = plan.num_tokens

        self.metadata = self._build_metadata(plan.replay_inputs, offsets)
        # PCP selects valid rows and pads downstream; other paths may include padding row indices.
        num_selected_rows = self.num_actual_tokens if plan.uses_pcp_replay else plan.num_tokens
        self.layers.rows = self.rows[:num_selected_rows]
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
                pcp_context=self.replay_pcp_context,
                pcp_cache_group_idx=group_id if self.replay_pcp_context is not None else None,
            )
            value = builder.build(
                0,
                common,
                num_actual_reqs=num_reqs,
                common_v41_batch_metadata=batch_shared,
                common_v41_metadata=group_shared.setdefault(group_id, {}),
                rope_views=rope,
            )
            for task in builder.take_device_metadata_tasks():
                task.run()
            for name in layer_names:
                metadata[name] = value
        return metadata

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        outer = get_forward_context()
        compact = self.layers.rows is not None
        padding = outer.is_padding
        full_padding = padding
        if compact or padding is None:
            if padding is not None and self.layers.sequence_parallel:
                padding = tensor_model_parallel_all_gather(padding, dim=0)
            replay_padding = self.padding[: self.num_tokens]
            replay_padding.fill_(True)
            if padding is None:
                replay_padding[: self.num_actual_tokens].fill_(False)
            else:
                replay_padding[: self.num_actual_tokens].copy_(
                    padding.index_select(0, self.rows[: self.num_actual_tokens])
                )
            padding = replay_padding
            full_padding = replay_padding
            if self.layers.sequence_parallel:
                tp = get_tp_group()
                padding = padding.chunk(tp.world_size)[tp.rank_in_group]
        mask = get_mc2_mask()
        saved_mask = mask.clone() if mask is not None else None
        try:
            with set_forward_context(
                self.metadata if compact else outer.attn_metadata,
                self.config,
                num_tokens=self.num_tokens,
                num_tokens_across_dp=self.dp_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=None if compact else outer.slot_mapping,
                is_padding=padding,
            ):
                context = get_forward_context()
                context.additional_kwargs["eplb_heat_collection_status"] = outer.additional_kwargs.get(
                    "eplb_heat_collection_status", False
                )
                replay_mask = context.additional_kwargs.get("mc2_mask")
                if replay_mask is not None:
                    replay_mask[self.num_actual_tokens :].fill_(False)
                    # Preserve masked rows in untrimmed batches as well as the
                    # new suffix padding. MC2 masks use the pre-SP layout.
                    if self.layers.sequence_parallel and not compact and outer.is_padding is not None:
                        full_padding = tensor_model_parallel_all_gather(padding, dim=0)
                    assert full_padding is not None
                    replay_mask[: self.num_tokens].logical_and_(~full_padding)
                yield
        finally:
            if saved_mask is not None:
                mask.copy_(saved_mask)
