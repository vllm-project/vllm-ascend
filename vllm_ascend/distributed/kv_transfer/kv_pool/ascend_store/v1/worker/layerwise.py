"""Common Worker route for layer-restricted Backend sessions."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, TypeAlias

from vllm.logger import logger

from ...attention_fence import reset_attention_compute_start_gate
from ..projection import GVALayerwiseProjection, LayerwiseProjection, LayerwiseProjectionBinder
from ..projection.layerwise.gva import gva_local_keys, gva_lookup_keys
from ..projection.layerwise.key_range import key_range_local_keys, key_range_lookup_keys
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import CheckpointStoreCommand, LoadCommand, StoreCommand
from ..runtime.backend import GVABackendIO, KeyRangeBackendIO
from ..runtime.batch import KeyAxes, KVGroupBatch, KVTransferBatch
from ..runtime.bulk import (
    BlockRows,
    StoreCandidateRows,
    load_block_rows,
    lookup_chunk_rows,
    store_candidate_rows,
)
from ..runtime.evidence import LayerStoreResult, LoadCompletion, StoreCompletion
from ..runtime.resources import KVPoolResources
from ..timeline.layerwise_load import LayerwiseLoadTimeline
from ..timeline.layerwise_store import LayerwiseStoreTimeline
from ..timeline.store_batch import StoreBatch
from ..topology import KVPoolTopology
from .base import KVPoolWorker, _store_command_completions
from .state import (
    KVPoolStepContext,
    StoreCandidates,
    StoreGroupCandidates,
    concat_uint64,
    empty_candidate_rows,
    empty_rows,
    merge_key_axes,
    readonly_indices,
    select_store_candidate_objects,
    store_candidate_keys,
)

_LayerwiseBackendIO: TypeAlias = GVABackendIO | KeyRangeBackendIO
_StorePreparationResult: TypeAlias = tuple[KVTransferBatch | None, tuple[StoreCompletion, ...]]


class LayerwiseWorker(KVPoolWorker):
    """Own the shared Layerwise request, session, and per-layer lifecycle."""

    def __init__(
        self,
        topology: KVPoolTopology,
        projection_binder: LayerwiseProjectionBinder,
        resources: KVPoolResources,
        backend_io: _LayerwiseBackendIO,
        *,
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._layerwise_projection_binder = projection_binder
        self._layerwise_projection: LayerwiseProjection | None = None
        self._layerwise_backend_io = backend_io
        self._layerwise_store_leader = _is_store_leader(topology)
        load_timeline = LayerwiseLoadTimeline(
            topology,
            layerwise_prefetch_layers,
            backend_io.initialize_thread,
            start_gate_factory,
            backend_io.load_layer,
            backend_io.start_load_sessions,
            backend_io.prepare_load_layers,
            backend_io.finish_load_sessions,
        )
        store_timeline = (
            LayerwiseStoreTimeline(
                topology,
                backend_io.initialize_thread,
                self._prepare_layerwise_store,
                self._store_layer,
                backend_io.start_store_sessions,
                backend_io.prepare_store_layers,
                backend_io.commit_store_sessions,
                backend_io.revoke_store_sessions,
            )
            if store_enabled
            else None
        )
        self._layerwise_load_timeline = load_timeline
        self._layerwise_store_timeline = store_timeline
        super().__init__(
            topology,
            resources,
            backend_io,
            load_timeline,
            store_timeline,
            source_ready_event_factory=source_ready_event_factory,
        )

    def _bind_projection(self, registration: dict[str, Any]) -> None:
        projection = self._layerwise_projection_binder.bind(**registration)
        self._bind_backend_projection(projection)
        self._layerwise_projection = projection

    @property
    def _bound_layerwise_projection(self) -> LayerwiseProjection:
        if self._layerwise_projection is None:
            raise RuntimeError("Layerwise projection is unavailable before cache registration")
        return self._layerwise_projection

    @abstractmethod
    def _bind_backend_projection(self, projection: LayerwiseProjection) -> None:
        """Bind the concrete projection to the selected Backend I/O adapter."""

    def lookup(self, request: LookupRequest) -> LookupResult:
        projection = self._bound_layerwise_projection
        if request.transfer_group_ids != projection.group_ids:
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured groups {projection.group_ids}"
            )
        masks = projection.reachability.select_for_lookup(request.block_hashes, request.query_range)
        observations = []
        for group_id, mask in zip(projection.group_ids, masks, strict=True):
            group = projection.groups[group_id]
            starts, counts, hashes = lookup_chunk_rows(
                request.query_range.end_token,
                request.block_hashes,
                block_size=group.block_size,
                hash_block_size=group.hash_block_size,
                start_token=request.query_range.start_token,
                mask=mask,
            )
            keys = self._lookup_keys(projection, group_id, hashes)
            try:
                lookup_keys = [key for axis in keys for key in axis]
                observed = self._backend_io.exists(lookup_keys)
            except Exception as error:
                logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                return LookupResult(0)
            row_count = len(hashes)
            available = tuple(
                all(observed[axis_index * row_count + row_index] for axis_index in range(len(keys)))
                for row_index in range(row_count)
            )
            observations.append(((starts, counts, hashes), available))
        reachable = projection.reachability.resolve_available_end(
            request.query_range,
            request.block_hashes,
            observations,
        )
        return LookupResult(reachable.end_token, reachable.tail_key_boundaries)

    @staticmethod
    def _lookup_keys(projection: LayerwiseProjection, group_id: int, hashes) -> KeyAxes:
        if isinstance(projection, GVALayerwiseProjection):
            return gva_lookup_keys(projection.groups[group_id], hashes)
        return key_range_lookup_keys(projection.groups[group_id], hashes)

    @staticmethod
    def _local_keys(projection: LayerwiseProjection, group_id: int, hashes) -> KeyAxes:
        if isinstance(projection, GVALayerwiseProjection):
            return gva_local_keys(projection.groups[group_id], hashes)
        return key_range_local_keys(projection.groups[group_id], hashes)

    def _build_load_batch(self, commands: tuple[LoadCommand, ...]) -> KVTransferBatch:
        projection = self._bound_layerwise_projection
        masks_by_request = tuple(
            projection.reachability.select_for_load(command.block_hashes, command.load_range) for command in commands
        )
        rows_by_request: list[dict[int, BlockRows]] = []
        for command, masks in zip(commands, masks_by_request, strict=True):
            boundaries = {item.group_id: item.boundary_token for item in command.tail_key_boundaries}
            if len(boundaries) != len(command.tail_key_boundaries):
                raise ValueError("Load contains duplicate tail-key boundaries for one cache group")
            unknown = set(boundaries).difference(projection.group_ids)
            if unknown:
                raise ValueError(f"Load tail-key boundaries contain unknown cache groups {sorted(unknown)}")
            rows_by_request.append(
                {
                    group_id: load_block_rows(
                        command.load_range.end_token,
                        command.block_hashes,
                        command.block_ids_by_group[group_id],
                        block_size=projection.groups[group_id].block_size,
                        hash_block_size=projection.groups[group_id].hash_block_size,
                        start_token=command.load_range.start_token,
                        mask=mask,
                        tail_boundary_token=boundaries.get(group_id),
                    )
                    for group_id, mask in zip(projection.group_ids, masks, strict=True)
                }
            )
        return self._assemble_load_batch(tuple(command.request_id for command in commands), rows_by_request)

    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> StoreCandidates:
        projection = self._bound_layerwise_projection
        rows_by_group: list[list[StoreCandidateRows]] = [[] for _group_id in projection.group_ids]
        keys_by_group: list[list[KeyAxes]] = [[] for _group_id in projection.group_ids]
        for command in commands:
            for group_index, (group_id, rows) in enumerate(
                zip(projection.group_ids, self._store_candidate_rows(command), strict=True)
            ):
                rows_by_group[group_index].append(rows)
                keys_by_group[group_index].append(self._local_keys(projection, group_id, rows[1]))
        return StoreCandidates(
            tuple(
                StoreGroupCandidates(group_id, tuple(rows), merge_key_axes(group_id, keys_by_group[group_index]))
                for group_index, (group_id, rows) in enumerate(zip(projection.group_ids, rows_by_group, strict=True))
            )
        )

    def _store_candidate_rows(self, command: StoreCommand) -> tuple[StoreCandidateRows, ...]:
        projection = self._bound_layerwise_projection
        if isinstance(command, CheckpointStoreCommand):
            raise ValueError("Layerwise transfer does not support state checkpoint Store")

        if not self._layerwise_store_leader:
            return tuple(empty_candidate_rows() for _group_id in projection.group_ids)
        masks = projection.reachability.select_for_store(
            command.block_hashes,
            command.store_range,
            command.num_prompt_tokens,
        )
        return tuple(
            store_candidate_rows(
                command.store_range.end_token,
                command.block_hashes,
                command.block_ids_by_group[group_id],
                block_size=projection.groups[group_id].block_size,
                hash_block_size=projection.groups[group_id].hash_block_size,
                start_token=command.store_range.start_token,
                mask=mask,
            )
            for group_id, mask in zip(projection.group_ids, masks, strict=True)
        )

    def _assemble_load_batch(
        self,
        request_ids: tuple[str, ...],
        rows_by_request: list[dict[int, BlockRows]],
    ) -> KVTransferBatch:
        projection = self._bound_layerwise_projection
        groups = []
        for group_id in projection.group_ids:
            block_parts = []
            count_parts = []
            request_splits = [0]
            for rows_by_group in rows_by_request:
                rows = rows_by_group.get(group_id, empty_rows())
                counts, _, block_ids = rows
                block_parts.append(block_ids)
                count_parts.append(counts)
                request_splits.append(request_splits[-1] + len(block_ids))
            key_axes = merge_key_axes(
                group_id,
                [
                    self._local_keys(projection, group_id, rows_by_group.get(group_id, empty_rows())[1])
                    for rows_by_group in rows_by_request
                ],
            )
            topology = self._groups[group_id]
            groups.append(
                KVGroupBatch(
                    group_id,
                    concat_uint64(block_parts),
                    concat_uint64(count_parts),
                    key_axes,
                    readonly_indices(request_splits),
                    tuple(layer.physical_layer_id for layer in topology.layers),
                    projection.groups[group_id].object_size,
                )
            )
        return KVTransferBatch(request_ids, tuple(groups))

    def _prepare_layerwise_store(self, commands: tuple[StoreCommand, ...]) -> _StorePreparationResult:
        try:
            candidates = self._build_store_candidates(commands)
            candidate_keys = store_candidate_keys(candidates)
            accepted, claim_once = self._admitted_store_keys(candidate_keys)
            if accepted is not None and not accepted:
                return None, _store_command_completions(commands)
            selected_objects = (
                None
                if accepted is None
                else select_store_candidate_objects(candidate_keys, accepted, claim_once=claim_once)
            )
            selected = self._materialize_store_candidates(
                commands,
                candidates,
                selected_objects,
                prepare_layerwise=True,
            )
        except Exception as error:
            return None, _store_command_completions(commands, error)
        return selected, ()

    def _store_group_layout(self, group_id: int) -> tuple[int, tuple[tuple[int, ...], ...] | None]:
        return self._bound_layerwise_projection.groups[group_id].object_size, None

    @property
    def _transfer_group_count(self) -> int:
        return len(self._bound_layerwise_projection.group_ids)

    def _prepare_step_store(self, commands: tuple[StoreCommand, ...]) -> None:
        timeline = self._layerwise_store_timeline
        if timeline is None:
            raise RuntimeError("Layerwise Store is disabled")
        timeline.prepare(commands)

    def _submit_step_store(self, context: KVPoolStepContext, *, fence: bool) -> None:
        self._finalize_layerwise_store(context)
        if fence:
            self.fence_previous_store()

    def _finalize_layerwise_store(self, context: KVPoolStepContext) -> None:
        if self._pending_store_batch is not None:
            raise RuntimeError("Previous Store invocation has not reached its fence")
        timeline = self._layerwise_store_timeline
        if timeline is None:
            raise RuntimeError("Layerwise Store is disabled")
        self._pending_store_batch = timeline.finalize()
        context.store_submitted = True

    def _prepare_store_close(self) -> StoreBatch | None:
        timeline = self._layerwise_store_timeline
        return timeline.prepare_close() if timeline is not None else None

    def _submit_store_layer(self, layer_name: str) -> None:
        timeline = self._layerwise_store_timeline
        if timeline is not None:
            timeline.submit_layer(layer_name, self._record_source_ready)

    def _wait_for_layer_load(self, layer_name: str) -> tuple[LoadCompletion, ...]:
        return self._layerwise_load_timeline.wait_for_layer(layer_name)

    def _store_layer(self, source_ready_event: Any, layer_id: int) -> LayerStoreResult:
        try:
            source_ready_event.synchronize()
        except Exception as error:
            return LayerStoreResult(None, True, error)
        return self._layerwise_backend_io.store_layer(layer_id)


def _is_store_leader(topology: KVPoolTopology) -> bool:
    """Resolve the rank that owns each Layerwise object before request traffic."""

    if topology.dcp_size <= 1:
        return topology.tp_rank % topology.put_step == 0
    dcp_rank = topology.transfer_groups[0].key_metadata.dcp_rank
    head = topology.tp_rank // topology.put_step
    peers = tuple(
        rank
        for rank in range(head * topology.put_step, (head + 1) * topology.put_step)
        if rank % topology.dcp_size == dcp_rank
    )
    return not peers or topology.tp_rank == min(peers)


__all__ = ("LayerwiseWorker",)
