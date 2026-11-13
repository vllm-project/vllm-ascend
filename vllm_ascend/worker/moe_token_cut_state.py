# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import MoETokenCutRuntimeConfig
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.moe_token_cut import (
    MAX_MOE_TOKEN_CUT_BATCH_SIZE,
    MoETokenCutConfig,
    MoETokenCutOutput,
    apply_moe_token_cut,
)
from vllm_ascend.worker.moe_token_cut_native_rebatch import (
    NativeRebatchRows,
    NativeRebatchStore,
    NativeRestoreContext,
    NativeResumeCheckpoint,
    validate_rebatch_operator,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.moe_token_cut_deferred import MoETokenDeferredStore
    from vllm_ascend.worker.moe_token_cut_restore import (
        MoETokenDeferredBatch,
        MoETokenRestorePlan,
    )


@dataclass(slots=True)
class MoETokenCutShadowStats:
    """Host-side diagnostics for the current model-runner process."""

    forwards: int = 0
    moe_layers: int = 0
    skipped_layers: int = 0
    last_skip_reason: str | None = None


@dataclass(slots=True)
class MoETokenCutCaptureBuffers:
    """Fixed-capacity device buffers for a future rebatch/restore backend."""

    hidden_states: torch.Tensor
    residuals: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    compacted_hidden_states: torch.Tensor
    compacted_topk_ids: torch.Tensor
    compacted_topk_weights: torch.Tensor
    cut_idx: torch.Tensor
    active_mask: torch.Tensor
    prefix_index: torch.Tensor
    actual_bs: torch.Tensor
    no_cut_rank_num: torch.Tensor


@dataclass(frozen=True, slots=True)
class MoETokenCutLayerCapture:
    """One layer's fixed-shape handoff to the future restore backend."""

    layer_idx: int
    request_ids: tuple[str, ...]
    hidden_states: torch.Tensor
    residuals: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    compacted_hidden_states: torch.Tensor
    compacted_topk_ids: torch.Tensor
    compacted_topk_weights: torch.Tensor
    cut_idx: torch.Tensor
    active_mask: torch.Tensor
    prefix_index: torch.Tensor
    actual_bs: torch.Tensor
    no_cut_rank_num: torch.Tensor


@dataclass(slots=True)
class MoETokenResumeCheckpoint:
    """Post-Router state and KV addressing parked across model forwards."""

    layer_idx: int
    hidden_state: torch.Tensor
    residual: torch.Tensor
    position: torch.Tensor
    topk_ids: torch.Tensor | None = None
    topk_weights: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    seq_len: torch.Tensor | None = None
    seq_len_cpu: int | None = None
    block_table: torch.Tensor | None = None


class MoETokenCutState:
    """Runner-owned lifecycle for the experimental token-cut path.

    The state is reset at every model forward. Shadow mode executes the cut
    decision and retains its typed result for inspection. Capture mode also
    snapshots restore inputs. Handoff/apply modes union deferred decisions for
    the post-forward Worker/Scheduler retry path. Apply mode additionally lets
    the MoE adapter consume compacted dispatch input and persists post-attention
    rows so the retry forward can refill at the same layer's MoE.
    """

    def __init__(self, config: MoETokenCutRuntimeConfig, num_layers: int, *, data_parallel_size: int = 1):
        if data_parallel_size < 1:
            raise ValueError("data_parallel_size must be positive")
        self.config = config
        if config.enabled and config.rebatch_backend == "native":
            validate_rebatch_operator()
        self.num_layers = num_layers
        self.data_parallel_size = data_parallel_size
        self.policy = MoETokenCutConfig(
            level_limit=config.level_limit,
            local_cut_count=config.local_cut_count,
            cut_count_threshold=config.cut_count_threshold,
            small_expert_token_threshold=config.small_expert_token_threshold,
            send_aiv_num=config.send_aiv_num,
            recv_aiv_num=config.recv_aiv_num,
        )
        self.stats = MoETokenCutShadowStats()
        self.active = False
        self.last_output: MoETokenCutOutput | None = None
        self.capture_buffers: MoETokenCutCaptureBuffers | None = None
        self.captured_layers: set[int] = set()
        self.request_ids_by_layer: dict[int, tuple[str, ...]] = {}
        self._request_ids: tuple[str, ...] = ()
        self._current_request_ids: tuple[str, ...] = ()
        self._model_input_row_count = 0
        self._layer_inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending_layer_idx: int | None = None
        self._global_deferred_mask: torch.Tensor | None = None
        self._layer_deferred_masks: dict[int, torch.Tensor] = {}
        self._layer_routing_snapshots: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self._layer_attention_snapshots: dict[
            int,
            dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
        ] = {}
        self._resume_checkpoints: dict[str, MoETokenResumeCheckpoint | NativeResumeCheckpoint] = {}
        self._native_rebatch_stores: dict[int, NativeRebatchStore] = {}
        self._native_restore_contexts: dict[int, NativeRestoreContext] = {}
        self._native_staged_restores: dict = {}
        self._native_combined_work: dict = {}
        self._native_layer_positions: dict = {}
        self._native_layer_attention_metadata: dict = {}
        self._native_attention_result = None
        self.native_attention_propagations = 0
        self._native_rebatch_stream = None
        self.native_combined_calls = 0
        self.native_overlap_launches = 0
        self._restored_request_ids: set[str] = set()
        self.allow_cut_this_forward = False
        self.allow_restore_this_forward = False
        self._cut_control_source = "config_step_window"
        self._restore_control_source = "config_step_window"
        self.apply_compaction_this_forward = False
        # Supplied kernel uses layer_idx only as its HCCL mailbox generation.
        # A real layer index repeats next forward and may be unavailable on an
        # idle rank. Count agreed native invocations on EVERY participating rank.
        self._native_invocation_id = 0
        self.model_graph_frame = None

    def begin_forward(
        self,
        *,
        is_decode: bool,
        is_profile_run: bool,
        is_draft_model: bool,
        num_actual_tokens: int,
        request_ids: tuple[str, ...] = (),
        allow_cut: bool | None = None,
        allow_restore: bool | None = None,
    ) -> None:
        """Start one runner forward and clear tensor references from the last one."""
        if self._native_combined_work or self._native_staged_restores:
            raise RuntimeError("previous combined token-cut layer has not been consumed")
        self._native_layer_positions.clear()
        self._native_layer_attention_metadata.clear()
        self._native_attention_result = None
        for name, value in (("allow_cut", allow_cut), ("allow_restore", allow_restore)):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"MoE token-cut {name} must be a bool")
        scheduled_checkpoints = set(request_ids).intersection(self._resume_checkpoints)
        if scheduled_checkpoints and (not is_decode or is_profile_run or is_draft_model):
            raise RuntimeError(
                "MoE token-cut retry requires a pure single-token decode batch; "
                "a saved cross-layer checkpoint cannot be recomputed as prefill"
            )
        self.last_output = None
        self.captured_layers.clear()
        self.request_ids_by_layer.clear()
        self._layer_inputs.clear()
        self._pending_layer_idx = None
        self._global_deferred_mask = None
        self._layer_deferred_masks.clear()
        self._layer_routing_snapshots.clear()
        self._layer_attention_snapshots.clear()
        self._restored_request_ids.clear()
        self._request_ids = request_ids
        self._current_request_ids = request_ids
        self.active = bool(
            self.config.enabled
            and is_decode
            and not is_profile_run
            and not is_draft_model
            and 0 < num_actual_tokens <= MAX_MOE_TOKEN_CUT_BATCH_SIZE
        )
        if self.active:
            self.stats.forwards += 1
        # Count once per eligible model forward, never once per MoE layer or
        # native call. A retry with every row parked still counts as a step.
        self.allow_cut_this_forward = self._resolve_step_control("cut", allow_cut)
        self.allow_restore_this_forward = self._resolve_step_control("restore", allow_restore)
        self._cut_control_source = "forward_override" if allow_cut is not None else "config_step_window"
        self._restore_control_source = "forward_override" if allow_restore is not None else "config_step_window"
        self.apply_compaction_this_forward = bool(self.allow_cut_this_forward and self.config.mode == "apply")

    def _resolve_step_control(self, operation: str, override: bool | None) -> bool:
        if not self.active:
            return False
        if override is not None:
            return override
        start = getattr(self.config, f"{operation}_step_start")
        end = getattr(self.config, f"{operation}_step_end")
        step = self.stats.forwards
        return bool(getattr(self.config, f"allow_{operation}") and start <= step and (end is None or step <= end))

    def get_step_controls(self) -> dict[str, Any]:
        """Host-only snapshot; permissions are not evidence of an actual cut."""
        return {
            "decode_step": self.stats.forwards if self.active else None,
            "active": self.active,
            "allow_cut": self.allow_cut_this_forward,
            "allow_restore": self.allow_restore_this_forward,
            "cut_control_source": self._cut_control_source,
            "restore_control_source": self._restore_control_source,
        }

    def set_layer_inputs(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residuals: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> None:
        """Publish the post-attention normalization boundary for one layer."""
        if not self.active or self.config.mode not in ("capture", "apply"):
            return
        self._layer_inputs[layer_idx] = (hidden_states, residuals)
        self._pending_layer_idx = layer_idx
        if positions is not None:
            self._native_layer_positions[layer_idx] = positions

    @property
    def cross_layer_active(self) -> bool:
        return self.active and self.config.mode == "apply"

    @property
    def native_rebatch(self) -> bool:
        return self.config.rebatch_backend == "native"

    @property
    def native_combined(self) -> bool:
        return self.native_rebatch and self.config.native_combined

    @property
    def current_request_ids(self) -> tuple[str, ...]:
        return self._current_request_ids

    def prepare_model_rows(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove retry rows that already have a saved cross-layer state."""
        if not self.cross_layer_active:
            return hidden_states, positions
        if positions.ndim != 1:
            raise RuntimeError("MoE token-cut cross-layer refill requires 1-D positions")
        if hidden_states.shape[0] < len(self._request_ids):
            raise RuntimeError("MoE token-cut model input has fewer rows than requests")
        self._model_input_row_count = hidden_states.shape[0]
        if not self.allow_restore_this_forward:
            parked_mask = torch.tensor(
                [request_id in self._resume_checkpoints for request_id in self._request_ids],
                dtype=torch.bool,
                device=hidden_states.device,
            )
            # Keep scheduled parked rows empty at sampling and roll back their
            # scheduler progress again while restoration is disabled.
            if any(request_id in self._resume_checkpoints for request_id in self._request_ids):
                self._global_deferred_mask = parked_mask
        kept_rows = [
            index for index, request_id in enumerate(self._request_ids) if request_id not in self._resume_checkpoints
        ]
        self._current_request_ids = tuple(self._request_ids[index] for index in kept_rows)
        indices = torch.tensor(kept_rows, dtype=torch.long, device=hidden_states.device)
        return (
            hidden_states.index_select(0, indices),
            positions.index_select(0, indices.to(device=positions.device)),
        )

    def original_row_indices(self, device: torch.device) -> torch.Tensor:
        """Map the current compact row order back to this forward's batch."""
        row_by_request = {request_id: index for index, request_id in enumerate(self._request_ids)}
        try:
            rows = [row_by_request[request_id] for request_id in self._current_request_ids]
        except KeyError as error:
            raise RuntimeError("restored request is absent from the scheduled batch") from error
        return torch.tensor(rows, dtype=torch.long, device=device)

    def capture_layer_attention_metadata(
        self,
        layer_idx: int,
        per_layer_metadata: Any,
        layer_name: str,
    ) -> None:
        """Snapshot KV addressing for rows that may be cut at this layer."""
        if not self.cross_layer_active:
            return
        if not isinstance(per_layer_metadata, dict):
            raise RuntimeError("MoE token-cut requires per-layer attention metadata")
        metadata = per_layer_metadata[layer_name]
        if self.native_combined:
            self._native_layer_attention_metadata[layer_idx] = metadata
        if self._native_attention_result is not None:
            metadata = self._native_attention_result.project(metadata, self._current_request_ids)
            lengths = self._native_attention_result.cpu_seq_lens.tolist()
            self._layer_attention_snapshots[layer_idx] = {
                rid: (
                    metadata.slot_mapping[row],
                    self._native_attention_result.seq_lens[row],
                    metadata.block_tables[row],
                    lengths[row],
                )
                for row, rid in enumerate(self._current_request_ids)
            }
            return
        slot_mapping = getattr(metadata, "slot_mapping", None)
        seq_lens = getattr(metadata, "seq_lens", None)
        block_tables = getattr(metadata, "block_tables", None)
        if not all(isinstance(value, torch.Tensor) for value in (slot_mapping, seq_lens, block_tables)):
            raise RuntimeError("MoE token-cut requires slot_mapping, seq_lens, and block_tables")
        if self.native_rebatch:
            # v0.27 eager FIA keeps lengths on CPU. Native rebatch consumes
            # device tensors, including fresh rows mixed with cached restores.
            # Transfer the vector once without changing the Attention metadata.
            seq_lens = seq_lens.to(device=slot_mapping.device)
        rows = self.original_row_indices(slot_mapping.device)
        seq_lens_cpu = getattr(metadata, "seq_lens_cpu", None)
        seq_lens_list = getattr(metadata, "seq_lens_list", None)
        snapshots: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
        for current_row, request_id in enumerate(self._current_request_ids):
            checkpoint = self._resume_checkpoints.get(request_id)
            if self.native_rebatch and request_id in self._restored_request_ids and checkpoint is not None:
                snapshots[request_id] = (
                    checkpoint.slot_mapping,
                    checkpoint.seq_len,
                    checkpoint.block_table,
                    checkpoint.seq_len_cpu,
                )
                continue
            original_row = rows[current_row]
            original_row_cpu = self._request_ids.index(request_id)
            if isinstance(seq_lens_cpu, torch.Tensor):
                seq_len_cpu = int(seq_lens_cpu[original_row_cpu])
            elif seq_lens_list is not None:
                seq_len_cpu = int(seq_lens_list[original_row_cpu])
            else:
                raise RuntimeError("MoE token-cut requires CPU sequence lengths")
            tensors = (
                slot_mapping[original_row],
                seq_lens[original_row.to(device=seq_lens.device)],
                block_tables[original_row.to(device=block_tables.device)],
            )
            snapshots[request_id] = (
                *(value if self.native_rebatch else value.clone() for value in tensors),
                seq_len_cpu,
            )
        self._layer_attention_snapshots[layer_idx] = snapshots

    def project_attention_metadata(
        self,
        per_layer_metadata: Any,
        layer_name: str,
        device: torch.device,
    ) -> Any:
        """Project current rows and restore parked KV addressing values."""
        from vllm_ascend.worker.moe_token_cut_rebatch import (
            project_layer_attention_metadata,
        )

        if self._native_attention_result is not None:
            projected = dict(per_layer_metadata)
            projected[layer_name] = self._native_attention_result.project(
                per_layer_metadata[layer_name], self._current_request_ids
            )
            self.native_attention_propagations += 1
            return projected

        projected = project_layer_attention_metadata(
            per_layer_metadata,
            layer_name,
            self.original_row_indices(device),
        )
        metadata = projected[layer_name]
        for row, request_id in enumerate(self._current_request_ids):
            checkpoint = self._resume_checkpoints.get(request_id)
            if checkpoint is None or request_id not in self._restored_request_ids:
                continue
            if (
                checkpoint.slot_mapping is None
                or checkpoint.seq_len is None
                or checkpoint.seq_len_cpu is None
                or checkpoint.block_table is None
            ):
                raise RuntimeError(f"request {request_id!r} has an incomplete attention checkpoint")
            metadata.slot_mapping[row].copy_(checkpoint.slot_mapping)
            metadata.seq_lens[row].copy_(checkpoint.seq_len)
            metadata.block_tables[row].copy_(checkpoint.block_table)
            if isinstance(getattr(metadata, "seq_lens_cpu", None), torch.Tensor):
                metadata.seq_lens_cpu[row].fill_(checkpoint.seq_len_cpu)
            if getattr(metadata, "seq_lens_list", None) is not None:
                metadata.seq_lens_list[row] = checkpoint.seq_len_cpu
        return projected

    def publish_native_attention_result(self, layer_idx, request_ids, rows):
        """Called only after rebatch completion, before the next Attention."""
        from vllm_ascend.worker.moe_token_cut_rebatch import NativeAttentionResult

        source = self._native_layer_attention_metadata.pop(layer_idx, None)
        if source is None:
            raise RuntimeError("native rebatch completed without source Attention metadata")
        n = len(request_ids)
        self._native_attention_result = NativeAttentionResult(
            tuple(request_ids), source, rows.slots[:n], rows.seq_lens[:n], rows.block_tables[:n]
        )

    def register_native_attention_source(self, layer_idx, per_layer_metadata, layer_name):
        # Even an all-parked rank skips Attention but can restore at this MoE.
        # Record the target KV layout independently of executing Attention.
        if not isinstance(per_layer_metadata, dict) or layer_name not in per_layer_metadata:
            raise RuntimeError("native rebatch requires the target layer's Attention metadata")
        self._native_layer_attention_metadata[layer_idx] = per_layer_metadata[layer_name]

    def apply_saved_routing(
        self,
        token_dispatch_input: MoETokenDispatchInput,
        *,
        rank_id: int,
        rank_num: int,
    ) -> MoETokenDispatchInput:
        """Replace freshly selected routes with saved post-Router values."""
        if not self.cross_layer_active or not self._restored_request_ids:
            return token_dispatch_input
        layer_idx = self._pending_layer_idx
        if layer_idx is None:
            return token_dispatch_input
        local_request_ids = self._align_request_ids(
            target_rows=token_dispatch_input.hidden_states.shape[0],
            rank_id=rank_id,
            rank_num=rank_num,
        )
        restored_rows = [
            row for row, request_id in enumerate(local_request_ids) if request_id in self._restored_request_ids
        ]
        if not restored_rows:
            return token_dispatch_input
        topk_ids = token_dispatch_input.topk_ids.clone()
        topk_weights = token_dispatch_input.topk_weights.clone()
        for row in restored_rows:
            request_id = local_request_ids[row]
            checkpoint = self._resume_checkpoints[request_id]
            if checkpoint.layer_idx != layer_idx:
                continue
            if checkpoint.topk_ids is None or checkpoint.topk_weights is None:
                raise RuntimeError(f"request {request_id!r} has no saved routing on EP rank {rank_id}")
            topk_ids[row].copy_(checkpoint.topk_ids)
            topk_weights[row].copy_(checkpoint.topk_weights)
        return replace(
            token_dispatch_input,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )

    def fresh_router_row_indices(
        self,
        row_count: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Rows whose gate logits still need to be computed at this MoE."""
        if not self.cross_layer_active or not self._restored_request_ids:
            return None
        layer_idx = self._pending_layer_idx
        if layer_idx is None:
            return None
        fresh_rows = [
            row
            for row, request_id in enumerate(self._current_request_ids)
            if not (
                request_id in self._restored_request_ids and self._resume_checkpoints[request_id].layer_idx == layer_idx
            )
        ]
        fresh_rows.extend(range(len(self._current_request_ids), row_count))
        return torch.tensor(fresh_rows, dtype=torch.long, device=device)

    def saved_routing_partition(
        self,
        row_count: int,
        device: torch.device,
        *,
        rank_id: int,
        rank_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Split a prepared MC2 shard into fresh and saved Router rows.

        Saved values are the post-Router values consumed by dispatch, including
        logical-to-physical expert mapping and any shared-expert columns.
        """
        if not self.cross_layer_active or not self._restored_request_ids:
            return None
        layer_idx = self._pending_layer_idx
        if layer_idx is None:
            return None
        local_request_ids = self._align_request_ids(
            target_rows=row_count,
            rank_id=rank_id,
            rank_num=rank_num,
        )
        saved_rows: list[int] = []
        saved_ids: list[torch.Tensor] = []
        saved_weights: list[torch.Tensor] = []
        for row, request_id in enumerate(local_request_ids):
            if request_id not in self._restored_request_ids:
                continue
            checkpoint = self._resume_checkpoints[request_id]
            if checkpoint.layer_idx != layer_idx:
                continue
            if checkpoint.topk_ids is None or checkpoint.topk_weights is None:
                raise RuntimeError(
                    f"request {request_id!r} has no saved routing on EP rank "
                    f"{rank_id}; MC2 shard ownership changed while it was parked"
                )
            saved_rows.append(row)
            saved_ids.append(checkpoint.topk_ids.to(device=device))
            saved_weights.append(checkpoint.topk_weights.to(device=device))
        if not saved_rows:
            return None
        saved_row_set = set(saved_rows)
        fresh_rows = [row for row in range(row_count) if row not in saved_row_set]
        return (
            torch.tensor(fresh_rows, dtype=torch.long, device=device),
            torch.tensor(saved_rows, dtype=torch.long, device=device),
            torch.stack(saved_ids),
            torch.stack(saved_weights),
        )

    def capture_layer_routing(
        self,
        layer_idx: int,
        token_dispatch_input: MoETokenDispatchInput,
        *,
        rank_id: int,
        rank_num: int,
    ) -> None:
        """Snapshot post-Router values in the local MC2 shard."""
        local_request_ids = self._align_request_ids(
            target_rows=token_dispatch_input.hidden_states.shape[0],
            rank_id=rank_id,
            rank_num=rank_num,
        )
        routing_snapshots = self._layer_routing_snapshots.setdefault(layer_idx, {})
        for row, request_id in enumerate(local_request_ids):
            if request_id:
                routing_snapshots[request_id] = (
                    token_dispatch_input.topk_ids[row]
                    if self.native_rebatch
                    else token_dispatch_input.topk_ids[row].clone(),
                    token_dispatch_input.topk_weights[row]
                    if self.native_rebatch
                    else token_dispatch_input.topk_weights[row].clone(),
                )

    def get_resume_checkpoint(self, request_id: str) -> MoETokenResumeCheckpoint | NativeResumeCheckpoint | None:
        """Return one checkpoint for diagnostics and invariant tests."""
        return self._resume_checkpoints.get(request_id)

    def restore_before_moe(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residuals: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Insert rows saved at ``layer_idx`` immediately before that layer's MoE."""
        if not self.cross_layer_active or not self.allow_restore_this_forward:
            return hidden_states, residuals, positions
        current = dict(zip(self._current_request_ids, range(len(self._current_request_ids))))
        scheduled = set(self._request_ids)
        resume_ids = [
            request_id
            for request_id, checkpoint in self._resume_checkpoints.items()
            if request_id in scheduled and checkpoint.layer_idx == layer_idx
        ]
        if not resume_ids:
            return hidden_states, residuals, positions

        if self.native_rebatch:
            if self.native_combined:
                from vllm_ascend.worker.moe_token_cut_combined import stage_restore

                return stage_restore(self, layer_idx, hidden_states, residuals, positions, resume_ids)
            return self._restore_native(layer_idx, hidden_states, residuals, positions, resume_ids)

        merged_ids = tuple(
            request_id for request_id in self._request_ids if request_id in current or request_id in resume_ids
        )
        hidden_rows = []
        residual_rows = []
        position_rows = []
        for request_id in merged_ids:
            row = current.get(request_id)
            if row is not None:
                hidden_rows.append(hidden_states[row])
                residual_rows.append(residuals[row])
                position_rows.append(positions[row])
            else:
                checkpoint = self._resume_checkpoints[request_id]
                hidden_rows.append(checkpoint.hidden_state)
                residual_rows.append(checkpoint.residual)
                position_rows.append(checkpoint.position)
                self._restored_request_ids.add(request_id)
                # This row skipped the current layer's Attention. If cut is
                # also enabled, re-parking it must retain the original KV map.
                if (
                    checkpoint.slot_mapping is not None
                    and checkpoint.seq_len is not None
                    and checkpoint.block_table is not None
                    and checkpoint.seq_len_cpu is not None
                ):
                    snapshots = self._layer_attention_snapshots.setdefault(layer_idx, {})
                    snapshots[request_id] = (
                        checkpoint.slot_mapping,
                        checkpoint.seq_len,
                        checkpoint.block_table,
                        checkpoint.seq_len_cpu,
                    )
        self._current_request_ids = merged_ids
        return (
            torch.stack(hidden_rows),
            torch.stack(residual_rows),
            torch.stack(position_rows),
        )

    def compact_after_moe(
        self,
        layer_idx: int,
        moe_output: torch.Tensor,
        pre_moe_hidden_states: torch.Tensor,
        residuals: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Park newly deferred rows and remove them from following layers."""
        if not self.cross_layer_active:
            return moe_output, residuals, positions
        if layer_idx in self._native_combined_work:
            from vllm_ascend.worker.moe_token_cut_combined import finish_combined

            return finish_combined(self, layer_idx, moe_output)
        if layer_idx in self._native_staged_restores:
            raise RuntimeError("reserved restore rows did not reach combined token-cut before dispatch")
        deferred_mask = self._layer_deferred_masks.pop(layer_idx, None)
        if deferred_mask is None:
            return moe_output, residuals, positions
        original_rows = self.original_row_indices(deferred_mask.device)
        current_deferred = deferred_mask.index_select(0, original_rows).to(dtype=torch.bool)
        deferred_rows = torch.nonzero(current_deferred, as_tuple=False).flatten()
        if deferred_rows.numel() == 0:
            return moe_output, residuals, positions

        if self.native_rebatch:
            return self._park_native(
                layer_idx, moe_output, pre_moe_hidden_states, residuals, positions, current_deferred
            )

        deferred_host_rows = deferred_rows.to(device="cpu").tolist()
        routing_snapshots = self._layer_routing_snapshots.get(layer_idx, {})
        attention_snapshots = self._layer_attention_snapshots.get(layer_idx, {})
        for row in deferred_host_rows:
            request_id = self._current_request_ids[row]
            routing = routing_snapshots.get(request_id)
            attention = attention_snapshots.get(request_id)
            if attention is None:
                raise RuntimeError(f"request {request_id!r} has no Attention/KV snapshot at MoE layer {layer_idx}")
            self._resume_checkpoints[request_id] = MoETokenResumeCheckpoint(
                layer_idx=layer_idx,
                hidden_state=pre_moe_hidden_states[row].clone(),
                residual=residuals[row].clone(),
                position=positions[row].clone(),
                topk_ids=None if routing is None else routing[0],
                topk_weights=None if routing is None else routing[1],
                slot_mapping=attention[0],
                seq_len=attention[1],
                block_table=attention[2],
                seq_len_cpu=attention[3],
            )
            self._restored_request_ids.discard(request_id)

        kept_mask = ~current_deferred
        kept_rows = torch.nonzero(kept_mask, as_tuple=False).flatten()
        old_request_ids = self._current_request_ids
        self._current_request_ids = tuple(
            request_id
            for request_id, is_deferred in zip(old_request_ids, current_deferred.to(device="cpu").tolist())
            if not is_deferred
        )
        self._layer_routing_snapshots.pop(layer_idx, None)
        self._layer_attention_snapshots.pop(layer_idx, None)
        return (
            moe_output.index_select(0, kept_rows.to(device=moe_output.device)),
            residuals.index_select(0, kept_rows.to(device=residuals.device)),
            positions.index_select(0, kept_rows.to(device=positions.device)),
        )

    def finalize_model_rows(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Scatter completed rows back to the Scheduler's original row order."""
        if not self.cross_layer_active:
            return hidden_states
        output_rows = max(self._model_input_row_count, len(self._request_ids))
        output = hidden_states.new_zeros((output_rows, *hidden_states.shape[1:]))
        if self._current_request_ids:
            rows = self.original_row_indices(hidden_states.device)
            output.index_copy_(0, rows, hidden_states)
        completed_restores = self._restored_request_ids.intersection(self._current_request_ids)
        for request_id in completed_restores:
            self._resume_checkpoints.pop(request_id, None)
        self._restored_request_ids.difference_update(completed_restores)
        self._native_attention_result = None
        self._native_layer_attention_metadata.clear()
        return output

    def _native_rows(self, layer_idx, hidden_states, residuals, positions, *, routing=True):
        request_ids = self._current_request_ids
        attention = self._layer_attention_snapshots.get(layer_idx, {})
        routes = self._layer_routing_snapshots.get(layer_idx, {})
        store = self._native_rebatch_stores.get(layer_idx)
        if request_ids and any(rid not in attention for rid in request_ids):
            raise RuntimeError("native rebatch requires Attention/KV metadata for every current row")
        if routing and any(rid not in routes for rid in request_ids):
            raise RuntimeError("native rebatch requires TP=1 local routing for every current row")
        n = len(request_ids)
        if routing:
            ids = torch.stack([routes[rid][0] for rid in request_ids])
            weights = torch.stack([routes[rid][1] for rid in request_ids])
        else:
            # Fresh rows are routed later in layer.mlp. Cached rows receive
            # their saved routes from the native restore output.
            ids = store.rows.topk_ids.new_zeros((n, store.rows.topk_ids.shape[1]))
            weights = store.rows.topk_weights.new_zeros(ids.shape)
        if request_ids:
            slots = torch.stack([attention[rid][0] for rid in request_ids])
            seq_lens = torch.stack([attention[rid][1] for rid in request_ids])
            tables = torch.stack([attention[rid][2] for rid in request_ids])
        else:
            slots = store.rows.slots[:0]
            seq_lens = store.rows.seq_lens[:0]
            tables = store.rows.block_tables[:0]
        return NativeRebatchRows(hidden_states, residuals, ids, weights, positions, slots, seq_lens, tables)

    def _park_native(self, layer_idx, moe_output, hidden_states, residuals, positions, deferred):
        old_ids = self._current_request_ids
        dropped = tuple(rid for rid, cut in zip(old_ids, deferred.cpu().tolist()) if cut)
        rows = self._native_rows(layer_idx, hidden_states, residuals, positions)
        store = self._native_rebatch_stores.get(layer_idx)
        context = self._native_restore_contexts.get(layer_idx)
        if context is None:
            raise RuntimeError("native rebatch cannot park rows without the originating token-cut EP context")
        if store is None:
            store = NativeRebatchStore(rows, restore_context=context)
            self._native_rebatch_stores[layer_idx] = store
        elif store.restore_context != context:
            raise RuntimeError("native rebatch EP context changed while the layer store exists")
        output = store.park(old_ids, rows, ~deferred, dropped)
        attention = self._layer_attention_snapshots[layer_idx]
        for rid in dropped:
            self._resume_checkpoints[rid] = NativeResumeCheckpoint(layer_idx, rid, store, attention[rid][3])
            self._restored_request_ids.discard(rid)
        self._current_request_ids = tuple(rid for rid in old_ids if rid not in set(dropped))
        kept = torch.nonzero(~deferred, as_tuple=False).flatten()
        self._layer_routing_snapshots.pop(layer_idx, None)
        self._layer_attention_snapshots.pop(layer_idx, None)
        # Combine mode caches dropped rows but does NOT compact live hidden or
        # residual rows, and never writes out_rebatch_residuals in this kernel.
        return moe_output.index_select(0, kept), residuals.index_select(0, kept), output.positions[: len(kept)]

    def _restore_native(self, layer_idx, hidden_states, residuals, positions, resume_ids):
        store = self._native_rebatch_stores[layer_idx]
        old_ids = self._current_request_ids
        rows = self._native_rows(layer_idx, hidden_states, residuals, positions, routing=False)
        cpu_lengths = {rid: self._resume_checkpoints[rid].seq_len_cpu for rid in resume_ids}
        output = store.restore(old_ids, rows, tuple(resume_ids))
        # The kernel appends restored rows. Convert back to Scheduler row order
        # and retain views of this invocation's output for later-layer metadata
        # and saved routing. No remaining checkpoint points to packed cache slots.
        appended_ids = old_ids + tuple(resume_ids)
        for i, rid in enumerate(resume_ids, start=len(old_ids)):
            checkpoint = MoETokenResumeCheckpoint(
                layer_idx,
                output.hidden[i],
                output.residual[i],
                output.positions[i],
                output.topk_ids[i],
                output.topk_weights[i],
                output.slots[i],
                output.seq_lens[i],
                cpu_lengths[rid],
                output.block_tables[i],
            )
            self._resume_checkpoints[rid] = checkpoint
            self._restored_request_ids.add(rid)
            self._layer_attention_snapshots.setdefault(layer_idx, {})[rid] = (
                checkpoint.slot_mapping,
                checkpoint.seq_len,
                checkpoint.block_table,
                checkpoint.seq_len_cpu,
            )
        merged_ids = tuple(rid for rid in self._request_ids if rid in appended_ids)
        indices = torch.tensor([appended_ids.index(rid) for rid in merged_ids], device=hidden_states.device)
        self._current_request_ids = merged_ids
        return (
            output.hidden.index_select(0, indices),
            output.residual.index_select(0, indices),
            output.positions.index_select(0, indices),
        )

    def get_layer_capture(self, layer_idx: int) -> MoETokenCutLayerCapture | None:
        """Return the stable device-buffer view for a captured sparse layer."""
        buffers = self.capture_buffers
        if buffers is None or layer_idx not in self.captured_layers:
            return None
        request_ids = self.request_ids_by_layer[layer_idx]
        batch_size = len(request_ids)
        return MoETokenCutLayerCapture(
            layer_idx=layer_idx,
            request_ids=request_ids,
            hidden_states=buffers.hidden_states[layer_idx, :batch_size],
            residuals=buffers.residuals[layer_idx, :batch_size],
            topk_ids=buffers.topk_ids[layer_idx, :batch_size],
            topk_weights=buffers.topk_weights[layer_idx, :batch_size],
            compacted_hidden_states=buffers.compacted_hidden_states[layer_idx, :batch_size],
            compacted_topk_ids=buffers.compacted_topk_ids[layer_idx, :batch_size],
            compacted_topk_weights=buffers.compacted_topk_weights[layer_idx, :batch_size],
            cut_idx=buffers.cut_idx[layer_idx, :batch_size],
            active_mask=buffers.active_mask[layer_idx, :batch_size],
            prefix_index=buffers.prefix_index[layer_idx],
            actual_bs=buffers.actual_bs[layer_idx],
            no_cut_rank_num=buffers.no_cut_rank_num[layer_idx],
        )

    def iter_layer_captures(self) -> Iterator[MoETokenCutLayerCapture]:
        """Yield captures in decoder-layer order for validation or restore."""
        for layer_idx in sorted(self.captured_layers):
            capture = self.get_layer_capture(layer_idx)
            if capture is not None:
                yield capture

    def build_restore_plan(self, layer_idx: int) -> MoETokenRestorePlan | None:
        """Synchronously validate one capture and construct its reference plan."""
        capture = self.get_layer_capture(layer_idx)
        if capture is None:
            return None
        from vllm_ascend.worker.moe_token_cut_restore import build_restore_plan

        return build_restore_plan(capture)

    def build_local_deferred_mask(
        self,
        layer_idx: int,
    ) -> torch.Tensor | None:
        """Map one layer's local decision into original Worker row order.

        This remains device-side. The caller may all-reduce the mask across the
        MC2 group before synchronizing it at sampling bookkeeping.
        """
        capture = self.get_layer_capture(layer_idx)
        if capture is None:
            return None
        from vllm_ascend.worker.moe_token_cut_collective import (
            build_local_deferred_mask,
        )

        return build_local_deferred_mask(
            self._request_ids,
            capture.request_ids,
            capture.cut_idx,
        )

    def stage_deferred_rows(
        self,
        layer_idx: int,
        store: MoETokenDeferredStore,
    ) -> MoETokenDeferredBatch | None:
        """Materialize and persist one captured layer's deferred request rows.

        This is an explicit diagnostic synchronization point. The production
        path will replace it with a device-side handoff before enabling Runner
        or Scheduler mutation.
        """
        capture = self.get_layer_capture(layer_idx)
        if capture is None:
            return None
        from vllm_ascend.worker.moe_token_cut_restore import (
            build_restore_plan,
            materialize_deferred_batch,
        )

        plan = build_restore_plan(capture)
        deferred = materialize_deferred_batch(capture, plan)
        store.enqueue(deferred)
        return deferred

    def run_shadow(
        self,
        token_dispatch_input: MoETokenDispatchInput,
        *,
        group_ep: str,
        rank_id: int,
        rank_num: int,
        expert_num: int,
        process_group: Any | None = None,
    ) -> MoETokenCutOutput | None:
        """Execute one shadow cut decision without changing dispatch data."""
        if self.model_graph_frame is not None:
            return self.model_graph_frame.dispatch(
                token_dispatch_input, group_ep=group_ep, rank_id=rank_id,
                rank_num=rank_num, expert_num=expert_num,
            )
        if self.config.graph_mode == "full_decode":
            # Prefill/profile fallback never creates eager checkpoints inside
            # graph-owned caches. The runner agrees this branch across DP.
            return None
        if self.native_combined:
            from vllm_ascend.worker.moe_token_cut_combined import run_combined

            return run_combined(
                self,
                token_dispatch_input,
                group_ep=group_ep,
                rank_id=rank_id,
                rank_num=rank_num,
                expert_num=expert_num,
                process_group=process_group,
                apply_op=apply_moe_token_cut,
            )
        local_cut_enabled = self.active and self.allow_cut_this_forward
        if self.data_parallel_size == 1 and not local_cut_enabled:
            return None
        if not self.config.enabled:
            return None
        skip_reason = None
        if token_dispatch_input.routing.pertoken_scale is not None:
            skip_reason = "per-token quantization scale is not supported"
        batch_size = token_dispatch_input.hidden_states.shape[0]
        if batch_size > MAX_MOE_TOKEN_CUT_BATCH_SIZE:
            skip_reason = "padded token count exceeds the kernel limit"
        if rank_num <= 0 or expert_num % rank_num != 0:
            skip_reason = "experts are not evenly partitioned across EP ranks"
        if self.native_rebatch and (
            token_dispatch_input.hidden_states.dtype != torch.bfloat16
            or token_dispatch_input.topk_weights.dtype != torch.bfloat16
            or token_dispatch_input.hidden_states.shape[1] > 4096
        ):
            skip_reason = "native rebatch requires BF16 hidden/routing weights and hidden_size <= 4096"
        kernel_input = token_dispatch_input
        if self.data_parallel_size > 1:
            weights_dtype = token_dispatch_input.topk_weights.dtype
            hidden_dtype = token_dispatch_input.hidden_states.dtype
            if not (
                (weights_dtype == torch.float32 and hidden_dtype == torch.bfloat16)
                or (weights_dtype in (torch.float16, torch.bfloat16) and hidden_dtype == weights_dtype)
            ):
                skip_reason = "token-cut hidden and routing dtypes are not supported"
            if process_group is None:
                raise RuntimeError("DP token-cut requires the MC2 process group on every rank")
            from vllm_ascend.worker.moe_token_cut_collective import agree_token_cut_execution

            if not agree_token_cut_execution(
                requested=local_cut_enabled,
                compatible=skip_reason is None,
                device=token_dispatch_input.hidden_states.device,
                process_group=process_group,
            ):
                if local_cut_enabled:
                    self._skip(skip_reason or "another DP rank has unsupported token-cut inputs")
                return None
            # Non-cutting ranks still contribute their REAL routing counts.
            # The native adapter protects them with a local no-delete threshold.
        elif skip_reason is not None:
            self._skip(skip_reason)
            return None

        layer_idx = self._pending_layer_idx
        layer_inputs = self._layer_inputs.get(layer_idx)
        native_residuals = (
            None
            if layer_inputs is None
            else self._align_layer_tensor(layer_inputs[1], target_rows=batch_size, rank_id=rank_id, rank_num=rank_num)
        )
        native_invocation_id = self._native_invocation_id
        self._native_invocation_id += 1
        output = apply_moe_token_cut(
            kernel_input,
            group_ep=group_ep,
            rank_id=rank_id,
            rank_num=rank_num,
            expert_num=expert_num,
            expert_per_rank=expert_num // rank_num,
            config=self.policy,
            layer_idx=native_invocation_id,
            residuals=native_residuals,
            allow_cut=local_cut_enabled,
        )
        if self.native_rebatch and layer_idx is not None:
            self._native_restore_contexts[layer_idx] = NativeRestoreContext(
                group_ep, rank_id, rank_num, expert_num, self.policy.send_aiv_num, self.policy.recv_aiv_num
            )
        if not local_cut_enabled:
            # The protected kernel output must not replace live prefill/decode
            # inputs, nor create checkpoints or scheduler handoffs on this rank.
            return None
        self.last_output = output
        self.stats.moe_layers += 1
        self.stats.last_skip_reason = None
        layer_idx = self._pending_layer_idx
        if self.config.mode == "apply" and layer_idx is not None:
            self.capture_layer_routing(
                layer_idx,
                token_dispatch_input,
                rank_id=rank_id,
                rank_num=rank_num,
            )
        if self.config.mode == "capture":
            self._capture_layer(
                token_dispatch_input,
                output,
                rank_id=rank_id,
                rank_num=rank_num,
            )
        if self.config.mode in ("handoff", "apply"):
            if self.config.mode == "apply" and not self.apply_compaction_this_forward:
                self._layer_inputs.pop(layer_idx, None)
                self._pending_layer_idx = None
                return output
            if process_group is None:
                raise RuntimeError("MoE token-cut handoff mode requires the MC2 process group")
            from vllm_ascend.worker.moe_token_cut_collective import (
                all_reduce_deferred_mask,
                build_local_deferred_mask,
            )

            local_request_ids = self._align_request_ids(
                target_rows=batch_size,
                rank_id=rank_id,
                rank_num=rank_num,
            )
            local_mask = build_local_deferred_mask(
                self._request_ids,
                local_request_ids,
                output.cut_idx[:batch_size],
            )
            # DP ranks own disjoint batches. Only TP shards of the same batch
            # may union decisions by row; DP>1 is currently restricted to TP=1.
            global_mask = (
                local_mask if self.data_parallel_size > 1 else all_reduce_deferred_mask(local_mask, process_group)
            )
            if self._global_deferred_mask is None:
                self._global_deferred_mask = global_mask
            else:
                self._global_deferred_mask.logical_or_(global_mask)
            if layer_idx is not None:
                self._layer_deferred_masks[layer_idx] = global_mask.clone()
            self._layer_inputs.pop(layer_idx, None)
            self._pending_layer_idx = None
        return output

    def take_global_deferred_mask(self) -> torch.Tensor | None:
        """Take and clear the EP-wide union produced by handoff mode."""
        deferred_mask = self._global_deferred_mask
        self._global_deferred_mask = None
        return deferred_mask

    def discard_requests(self, request_ids: set[str]) -> None:
        """Forget saved checkpoints for requests retired by the Scheduler."""
        for work in self._native_combined_work.values():
            work.wait()
        for store in self._native_rebatch_stores.values():
            store.remove(request_ids)
        for request_id in request_ids:
            self._resume_checkpoints.pop(request_id, None)
            self._restored_request_ids.discard(request_id)

    def _capture_layer(
        self,
        token_dispatch_input: MoETokenDispatchInput,
        output: MoETokenCutOutput,
        *,
        rank_id: int,
        rank_num: int,
    ) -> None:
        layer_idx = self._pending_layer_idx
        layer_inputs = self._layer_inputs.pop(layer_idx, None)
        self._pending_layer_idx = None
        if layer_idx is None or layer_inputs is None:
            self._skip("Qwen3-MoE post-attention layer boundary was not captured")
            return
        if not 0 <= layer_idx < self.num_layers:
            self._skip(f"layer index {layer_idx} is outside the configured model")
            return

        layer_hidden_states, layer_residuals = layer_inputs
        hidden_states = token_dispatch_input.hidden_states
        batch_size, hidden_size = hidden_states.shape
        if layer_hidden_states.shape[1] != hidden_size:
            raise RuntimeError("MoE token-cut layer boundary hidden size does not match dispatch input")
        residuals = self._align_layer_tensor(
            layer_residuals,
            target_rows=batch_size,
            rank_id=rank_id,
            rank_num=rank_num,
        )
        request_ids = self._align_request_ids(
            target_rows=batch_size,
            rank_id=rank_id,
            rank_num=rank_num,
        )
        topk = token_dispatch_input.topk_ids.shape[1]
        buffers = self.capture_buffers
        if buffers is None:
            buffers = self._allocate_capture_buffers(
                hidden_states,
                residuals,
                token_dispatch_input.topk_ids,
                token_dispatch_input.topk_weights,
            )
            self.capture_buffers = buffers
        elif (
            buffers.hidden_states.shape[2] != hidden_size
            or buffers.topk_ids.shape[2] != topk
            or buffers.hidden_states.dtype != hidden_states.dtype
            or buffers.topk_weights.dtype != token_dispatch_input.topk_weights.dtype
        ):
            raise RuntimeError("MoE token-cut capture shape or dtype changed after buffer allocation")

        buffers.hidden_states[layer_idx, :batch_size].copy_(hidden_states)
        buffers.residuals[layer_idx, :batch_size].copy_(residuals)
        buffers.topk_ids[layer_idx, :batch_size].copy_(token_dispatch_input.topk_ids)
        buffers.topk_weights[layer_idx, :batch_size].copy_(token_dispatch_input.topk_weights)
        buffers.compacted_hidden_states[layer_idx, :batch_size].copy_(output.dispatch_input.hidden_states)
        buffers.compacted_topk_ids[layer_idx, :batch_size].copy_(output.dispatch_input.topk_ids)
        buffers.compacted_topk_weights[layer_idx, :batch_size].copy_(output.dispatch_input.topk_weights)
        buffers.cut_idx[layer_idx].zero_()
        buffers.cut_idx[layer_idx, :batch_size].copy_(output.cut_idx[:batch_size])
        buffers.active_mask[layer_idx].zero_()
        active_mask = output.dispatch_input.routing.mc2_mask
        if active_mask is None:
            raise RuntimeError("MoE token-cut output is missing the compacted active mask")
        buffers.active_mask[layer_idx, :batch_size].copy_(active_mask[:batch_size])
        buffers.prefix_index[layer_idx].zero_()
        buffers.prefix_index[layer_idx].copy_(output.prefix_index)
        buffers.actual_bs[layer_idx].copy_(output.actual_bs.reshape(1))
        buffers.no_cut_rank_num[layer_idx].copy_(output.no_cut_rank_num.reshape(1))
        self.captured_layers.add(layer_idx)
        self.request_ids_by_layer[layer_idx] = request_ids

    def _align_layer_tensor(
        self,
        tensor: torch.Tensor,
        *,
        target_rows: int,
        rank_id: int,
        rank_num: int,
    ) -> torch.Tensor:
        if self.data_parallel_size > 1:
            if tensor.shape[0] > target_rows:
                raise RuntimeError("DP-local layer boundary has more rows than the MC2 input")
            return torch.nn.functional.pad(tensor, (0, 0, 0, target_rows - tensor.shape[0]))
        if tensor.shape[0] == target_rows:
            return tensor
        total_rows = target_rows * rank_num
        if tensor.shape[0] > total_rows:
            raise RuntimeError("MoE token-cut layer boundary has more rows than the MC2 input")
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, total_rows - tensor.shape[0]))
        return torch.tensor_split(padded, rank_num, dim=0)[rank_id]

    def _align_request_ids(
        self,
        *,
        target_rows: int,
        rank_id: int,
        rank_num: int,
    ) -> tuple[str, ...]:
        if self.data_parallel_size > 1:
            if len(self._current_request_ids) > target_rows:
                raise RuntimeError("DP-local request order has more rows than the MC2 input")
            return self._current_request_ids + ("",) * (target_rows - len(self._current_request_ids))
        if len(self._current_request_ids) == target_rows:
            return self._current_request_ids
        total_rows = target_rows * rank_num
        if len(self._current_request_ids) > total_rows:
            raise RuntimeError("MoE token-cut request order has more rows than the MC2 input")
        padded = self._current_request_ids + ("",) * (total_rows - len(self._current_request_ids))
        start = rank_id * target_rows
        return padded[start : start + target_rows]

    def _allocate_capture_buffers(
        self,
        hidden_states: torch.Tensor,
        residuals: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> MoETokenCutCaptureBuffers:
        hidden_shape = (self.num_layers, MAX_MOE_TOKEN_CUT_BATCH_SIZE, hidden_states.shape[1])
        topk_shape = (self.num_layers, MAX_MOE_TOKEN_CUT_BATCH_SIZE, topk_ids.shape[1])
        return MoETokenCutCaptureBuffers(
            hidden_states=torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device),
            residuals=torch.empty(hidden_shape, dtype=residuals.dtype, device=residuals.device),
            topk_ids=torch.empty(topk_shape, dtype=topk_ids.dtype, device=topk_ids.device),
            topk_weights=torch.empty(topk_shape, dtype=topk_weights.dtype, device=topk_weights.device),
            compacted_hidden_states=torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device),
            compacted_topk_ids=torch.empty(topk_shape, dtype=topk_ids.dtype, device=topk_ids.device),
            compacted_topk_weights=torch.empty(topk_shape, dtype=topk_weights.dtype, device=topk_weights.device),
            cut_idx=torch.zeros(
                (self.num_layers, MAX_MOE_TOKEN_CUT_BATCH_SIZE),
                dtype=torch.int8,
                device=hidden_states.device,
            ),
            active_mask=torch.zeros(
                (self.num_layers, MAX_MOE_TOKEN_CUT_BATCH_SIZE),
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            prefix_index=torch.zeros(
                (self.num_layers, MAX_MOE_TOKEN_CUT_BATCH_SIZE),
                dtype=torch.int32,
                device=hidden_states.device,
            ),
            actual_bs=torch.zeros((self.num_layers, 1), dtype=torch.int32, device=hidden_states.device),
            no_cut_rank_num=torch.zeros((self.num_layers, 1), dtype=torch.int32, device=hidden_states.device),
        )

    def _skip(self, reason: str) -> None:
        self.stats.skipped_layers += 1
        self.stats.last_skip_reason = reason
        logger.warning_once("MoE token-cut shadow decision skipped: %s.", reason)