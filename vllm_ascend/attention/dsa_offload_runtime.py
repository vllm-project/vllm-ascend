# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime ordering for the independently installed recipes DSA operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_ascend.attention.dsa_offload_state import (
    DsaLayerWorkspace,
    DsaResidentState,
)
from vllm_ascend.ops.dsa_offload import (
    DsaOffloadConfig,
    dsa_install,
    dsa_plan,
    dsa_serve,
)


@dataclass(frozen=True)
class DsaGroupSpec:
    """One full-indexer owner and the following shared-indexer layers."""

    group_id: int
    owner_layer: int
    layers: tuple[int, ...]
    group_kind: int = 0

    def __post_init__(self) -> None:
        if self.group_id < 0 or self.owner_layer < 0:
            raise ValueError("DSA group and owner IDs must be non-negative")
        if not self.layers or self.layers[0] != self.owner_layer:
            raise ValueError("DSA group layers must start with the owner layer")
        if len(self.layers) != len(set(self.layers)):
            raise ValueError("DSA group contains duplicate layers")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("DSA layer IDs must be non-negative")


@dataclass(frozen=True)
class DsaGroupMetadata:
    """Mutable operator tensors shared by every layer in one DSA group."""

    pool_ids: torch.Tensor
    id_to_slot: torch.Tensor
    lru_counter: torch.Tensor

    def __post_init__(self) -> None:
        for name, tensor in (
            ("pool_ids", self.pool_ids),
            ("id_to_slot", self.id_to_slot),
            ("lru_counter", self.lru_counter),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.dtype != torch.int32 or tensor.ndim != 2:
                raise ValueError(f"{name} must be a 2-D int32 tensor")
        batch = self.pool_ids.shape[0]
        if self.id_to_slot.shape[0] != batch or self.lru_counter.shape[0] != batch:
            raise ValueError("DSA group metadata must use one common batch capacity")


@dataclass(frozen=True)
class DsaGroupPlan:
    group_id: int
    batch_size: int
    config: DsaOffloadConfig
    plan: torch.Tensor
    install_records: torch.Tensor
    selection_kv_actual_seq: torch.Tensor


@dataclass
class _ActiveGroup:
    state: DsaGroupPlan
    next_layer_index: int = 0


def build_dsa_group_specs(
    indexer_types: Sequence[str] | None,
    num_hidden_layers: int,
) -> tuple[DsaGroupSpec, ...]:
    """Build owner/shared groups from normalized backend layer metadata.

    ``full``/``shared`` remains accepted for compatibility with existing
    model configs and is normalized by the attention backend.
    """

    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if indexer_types is None:
        return tuple(
            DsaGroupSpec(layer, layer, (layer,))
            for layer in range(num_hidden_layers)
        )
    if isinstance(indexer_types, (str, bytes)) or not isinstance(
        indexer_types, Sequence
    ):
        raise TypeError("indexer_types must be a sequence of 'full'/'shared' values")
    if len(indexer_types) < num_hidden_layers:
        raise ValueError("indexer_types is shorter than num_hidden_layers")

    groups: list[DsaGroupSpec] = []
    owner: int | None = None
    layers: list[int] = []

    def close_group() -> None:
        nonlocal owner, layers
        if owner is None:
            return
        groups.append(
            DsaGroupSpec(
                group_id=len(groups),
                owner_layer=owner,
                layers=tuple(layers),
            )
        )

    for layer_id, indexer_type in enumerate(indexer_types[:num_hidden_layers]):
        normalized = str(indexer_type).lower()
        if normalized == "full":
            close_group()
            owner = layer_id
            layers = [layer_id]
        elif normalized == "shared":
            if owner is None:
                raise ValueError("A shared DSA layer cannot precede its owner")
            layers.append(layer_id)
        else:
            raise ValueError(f"Unsupported indexer type: {indexer_type!r}")
    close_group()
    if not groups:
        raise ValueError("No DSA groups were produced")
    return tuple(groups)


def _tensor_storage_bytes(tensor: torch.Tensor, seen: set[int]) -> int:
    """Count backing storage once, including expanded metadata views."""
    storage = tensor.untyped_storage()
    storage_id = int(storage.data_ptr())
    if storage_id in seen:
        return 0
    seen.add(storage_id)
    return int(storage.nbytes())


class DsaOffloadRuntime:
    """Enforce recipes Plan/Serve/Install ordering for a Decode step.

    Operator implementation and registration stay in the separately built
    ``custom_ops`` package. This class owns only tensor identity and call
    ordering. MTP layers must not be included in ``groups`` or ``workspaces``.
    """

    def __init__(
        self,
        groups: Sequence[DsaGroupSpec],
        group_metadata: Mapping[int, DsaGroupMetadata],
        resident_state: DsaResidentState,
    ) -> None:
        self.groups = {group.group_id: group for group in groups}
        if len(self.groups) != len(groups):
            raise ValueError("Duplicate DSA group ID")
        self.group_metadata = dict(group_metadata)
        if self.group_metadata.keys() != self.groups.keys():
            raise ValueError("Every DSA group requires exactly one metadata set")
        self.resident_state = resident_state
        self._layer_to_group: dict[int, int] = {}
        for group in groups:
            for layer_id in group.layers:
                if layer_id in self._layer_to_group:
                    raise ValueError(f"DSA layer {layer_id} belongs to two groups")
                resident_state.workspace(layer_id)
                self._layer_to_group[layer_id] = group.group_id
        workspace_layers = set(resident_state.workspaces)
        if workspace_layers != set(self._layer_to_group):
            raise ValueError("DSA workspaces and group layers do not match")
        self._active: dict[int, _ActiveGroup] = {}
        self._step_open = False
        self._completed_groups: set[int] = set()
        self._pending_row_moves: list[tuple[int, int]] = []
        self._pending_invalidated_rows: set[int] = set()
        seen_storages: set[int] = set()
        self.workspace_bytes = 0
        for workspace in resident_state.workspaces.values():
            for tensor in (
                workspace.resident_kv_cache,
                workspace.resident_k_rope,
                workspace.selection_kv_cache,
                workspace.selection_k_rope,
                workspace.selection_block_table,
                workspace.selection_query_lens,
                workspace.selection_default_indices,
            ):
                if tensor is not None:
                    self.workspace_bytes += _tensor_storage_bytes(tensor, seen_storages)
        for metadata in self.group_metadata.values():
            for tensor in (metadata.pool_ids, metadata.id_to_slot, metadata.lru_counter):
                self.workspace_bytes += _tensor_storage_bytes(tensor, seen_storages)

    def queue_row_changes(
        self,
        moves: Sequence[tuple[int, int]],
        invalidated: Sequence[int],
    ) -> None:
        """Queue InputBatch row changes until the next DSA Plan boundary."""
        self._pending_row_moves.extend(moves)
        self._pending_invalidated_rows.update(invalidated)

    def _apply_pending_row_changes(self) -> None:
        if not (self._pending_row_moves or self._pending_invalidated_rows):
            return
        invalidated = set(self._pending_invalidated_rows)
        # Resident payload is physically indexed by the InputBatch row in every
        # layer workspace. Moving only the shared metadata would make the new
        # row describe stale payload from its previous occupant. Moving all
        # per-layer resident tensors is both expensive and unnecessary: cold
        # invalidate every row touched by a condense/reorder and let DsaInstall
        # repopulate it from Host Main KV in the next step. This also avoids
        # ambiguous snapshot semantics when several row moves are queued before
        # one Plan boundary.
        for source, destination in self._pending_row_moves:
            invalidated.add(source)
            invalidated.add(destination)
        for metadata in self.group_metadata.values():
            for row in invalidated:
                metadata.pool_ids[row].fill_(-1)
                metadata.id_to_slot[row].fill_(-1)
                metadata.lru_counter[row].zero_()
        self._pending_row_moves.clear()
        self._pending_invalidated_rows.clear()

    def group_for_layer(self, layer_id: int) -> DsaGroupSpec:
        try:
            return self.groups[self._layer_to_group[layer_id]]
        except KeyError as error:
            raise KeyError(f"Unknown DSA layer {layer_id}") from error

    def prepare_step(self) -> None:
        """Submit row metadata updates before eager execution or graph replay.

        DSA Install, these updates, and the next DSA Plan all use the current
        NPU stream, which gives eager and ACL Graph one ordering contract.
        """

        self.resident_state.wait_previous_install()
        self._apply_pending_row_changes()

    def finish_step(self) -> None:
        """Record completion outside the eager/ACL Graph model wrapper."""

        event = torch.npu.Event()
        event.record()
        self.resident_state.record_final_install_event(event)

    def begin_step(self) -> None:
        if self._active or self._step_open:
            raise RuntimeError("Cannot begin a DSA step before the previous step finishes")
        self.prepare_step()
        self._completed_groups.clear()
        self._step_open = True

    def plan_group(
        self,
        group_id: int,
        selection_topk_indices: torch.Tensor,
        full_kv_actual_seq: torch.Tensor,
        *,
        config: DsaOffloadConfig,
    ) -> DsaGroupPlan:
        if not self._step_open:
            self.begin_step()
        if group_id in self._completed_groups:
            raise RuntimeError(f"DSA group {group_id} already completed in this step")
        if group_id in self._active:
            raise RuntimeError(f"DSA group {group_id} already has an active plan")
        group = self.groups[group_id]
        metadata = self.group_metadata[group_id]
        batch_size = selection_topk_indices.shape[0]
        plan, install_records, actual_seq = dsa_plan(
            selection_topk_indices,
            full_kv_actual_seq,
            metadata.pool_ids[:batch_size],
            metadata.id_to_slot[:batch_size],
            metadata.lru_counter[:batch_size],
            config=config,
            group_id=group.group_id,
            owner_layer=group.owner_layer,
            group_kind=group.group_kind,
        )
        state = DsaGroupPlan(
            group_id,
            batch_size,
            config,
            plan,
            install_records,
            actual_seq,
        )
        self._active[group_id] = _ActiveGroup(state)
        return state

    def serve_layer(
        self,
        layer_id: int,
        full_kv_cache: torch.Tensor,
        full_k_rope: torch.Tensor,
        full_kv_block_table: torch.Tensor,
        full_kv_actual_seq: torch.Tensor,
        *,
        config: DsaOffloadConfig,
    ) -> tuple[DsaLayerWorkspace, DsaGroupPlan]:
        group_id = self._layer_to_group[layer_id]
        group = self.groups[group_id]
        active = self._active.get(group_id)
        if active is None:
            raise RuntimeError(f"DSA group {group_id} has no active plan")
        expected_layer = group.layers[active.next_layer_index]
        if layer_id != expected_layer:
            raise RuntimeError(
                f"DSA group {group_id} expected layer {expected_layer}, got {layer_id}"
            )
        if config != active.state.config:
            raise RuntimeError("DSA Serve config differs from the active group plan")
        workspace = self.resident_state.workspace(layer_id)
        batch_size = active.state.batch_size
        selection_rows = config.selection_rows(batch_size)
        selection_blocks = config.selection_block_count(batch_size)
        dsa_serve(
            active.state.plan,
            full_kv_cache,
            full_k_rope,
            workspace.resident_kv_cache[:batch_size],
            workspace.resident_k_rope[:batch_size],
            workspace.selection_kv_cache[:selection_blocks],
            workspace.selection_k_rope[:selection_blocks],
            full_kv_block_table=full_kv_block_table[:batch_size],
            full_kv_actual_seq=full_kv_actual_seq[:batch_size],
            config=config,
        )
        return workspace, active.state

    def install_after_layer(
        self,
        layer_id: int,
        *,
        config: DsaOffloadConfig,
    ) -> None:
        group_id = self._layer_to_group[layer_id]
        group = self.groups[group_id]
        active = self._active.get(group_id)
        if active is None:
            raise RuntimeError(f"DSA group {group_id} has no active plan")
        expected_layer = group.layers[active.next_layer_index]
        if layer_id != expected_layer:
            raise RuntimeError(
                f"DSA group {group_id} expected install for layer "
                f"{expected_layer}, got {layer_id}"
            )
        if config != active.state.config:
            raise RuntimeError("DSA Install config differs from the active group plan")
        metadata = self.group_metadata[group_id]
        if layer_id != group.owner_layer:
            self._install_layer(
                layer_id,
                active.state,
                metadata,
                config,
                metadata_update=0,
            )
        if layer_id == group.layers[-1]:
            self._install_layer(
                group.owner_layer,
                active.state,
                metadata,
                config,
                metadata_update=1,
            )
            self._active.pop(group_id)
            self._completed_groups.add(group_id)
            if self._completed_groups == set(self.groups):
                self._step_open = False
        else:
            active.next_layer_index += 1

    def _install_layer(
        self,
        layer_id: int,
        plan_state: DsaGroupPlan,
        metadata: DsaGroupMetadata,
        config: DsaOffloadConfig,
        *,
        metadata_update: int,
    ) -> None:
        workspace = self.resident_state.workspace(layer_id)
        batch_size = plan_state.batch_size
        selection_rows = config.selection_rows(batch_size)
        selection_blocks = config.selection_block_count(batch_size)
        dsa_install(
            plan_state.install_records,
            workspace.selection_kv_cache[:selection_blocks],
            workspace.selection_k_rope[:selection_blocks],
            workspace.selection_block_table[:selection_rows],
            workspace.resident_kv_cache[:batch_size],
            workspace.resident_k_rope[:batch_size],
            metadata.pool_ids[:batch_size],
            metadata.id_to_slot[:batch_size],
            metadata.lru_counter[:batch_size],
            config=config,
            metadata_update=metadata_update,
        )
