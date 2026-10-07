"""Bulk Worker routes with fixed synchronous or asynchronous Load timelines."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

from vllm.logger import logger

from ...projection import (
    BulkProjection,
    BulkProjectionBinder,
    ConsumerPipelineBulkProjection,
    HybridBulkProjection,
    OrdinaryBulkProjection,
    TPMismatchBulkProjection,
)
from ...projection.bulk.common import bulk_arguments
from ...projection.bulk.consumer_pipeline import (
    consumer_pipeline_load_keys,
    consumer_pipeline_load_ranges,
    consumer_pipeline_lookup_keys,
    consumer_pipeline_store_keys,
    consumer_pipeline_store_ranges,
)
from ...projection.bulk.hybrid import (
    hybrid_bulk_ranges,
    hybrid_load_keys,
    hybrid_lookup_keys,
    hybrid_store_keys,
)
from ...projection.bulk.ordinary import (
    ordinary_bulk_ranges,
    ordinary_load_keys,
    ordinary_lookup_keys,
    ordinary_store_keys,
)
from ...projection.bulk.tp_mismatch import (
    tp_mismatch_bulk_ranges,
    tp_mismatch_load_keys,
    tp_mismatch_lookup_keys,
    tp_mismatch_store_keys,
)
from ...protocol.lookup import LookupRequest, LookupResult
from ...protocol.transfer import CheckpointStoreCommand, LoadCommand, StoreCommand
from ...timeline.asynchronous_load import AsynchronousLoadTimeline
from ...timeline.asynchronous_store import AsynchronousStoreTimeline
from ...timeline.synchronous_load import SynchronousLoadTimeline
from ...topology import KVPoolTopology
from ..base import KVPoolWorker, _failed_store_completions, _store_command_completions
from ..io import BackendIO
from ..io.arguments import BulkBackendArguments
from ..io.io import _batch_sources, _failed_load_completions
from ..resources import KVPoolResources
from ..transfer.batch import KeyAxes, KVGroupBatch, KVTransferBatch
from ..transfer.evidence import LoadCompletion, StoreCompletion, TransferEvidence
from ..transfer.rows import (
    BlockRows,
    StoreCandidateRows,
    boundary_hash,
    fine_lookup_chunk_rows,
    load_block_rows,
    lookup_chunk_rows,
    select_store_writer_rows,
    store_candidate_rows,
)
from ..transfer.state import (
    KVPoolStepContext,
    StoreCandidates,
    StoreGroupCandidates,
    concat_uint64,
    empty_candidate_rows,
    merge_key_axes,
    readonly_indices,
    select_store_candidate_objects,
    store_candidate_keys,
)


class BulkWorker(KVPoolWorker):
    """Own one bound Bulk projection and its common Store execution route."""

    def __init__(
        self,
        topology: KVPoolTopology,
        projection_binder: BulkProjectionBinder,
        resources: KVPoolResources,
        *,
        store_enabled: bool = True,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._bulk_projection_binder = projection_binder
        self._bulk_projection: BulkProjection | None = None
        backend_io = BackendIO(resources.backend, resources.backend_spec)
        self._bulk_backend_io = backend_io
        load_timeline = self._create_load_timeline(backend_io)
        store_timeline = (
            AsynchronousStoreTimeline(backend_io.initialize_thread, self._execute_bulk_store) if store_enabled else None
        )
        self._asynchronous_store_timeline = store_timeline
        super().__init__(
            topology,
            resources,
            backend_io,
            load_timeline,
            store_timeline,
            source_ready_event_factory=source_ready_event_factory,
        )

    @abstractmethod
    def _create_load_timeline(self, backend_io: BackendIO) -> SynchronousLoadTimeline | AsynchronousLoadTimeline:
        """Create the leaf Worker's fixed Bulk Load timeline."""

    def _bind_projection(self, registration: dict[str, Any]) -> None:
        self._bulk_projection = self._bulk_projection_binder.bind(**registration)

    @property
    def _bound_bulk_projection(self) -> BulkProjection:
        if self._bulk_projection is None:
            raise RuntimeError("Concrete Bulk projection is unavailable before cache registration")
        return self._bulk_projection

    def lookup(self, request: LookupRequest) -> LookupResult:
        projection = self._bound_bulk_projection
        if isinstance(projection, HybridBulkProjection):
            return self._lookup_hybrid(projection, request)
        if request.transfer_group_ids != (projection.group_id,):
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured group {(projection.group_id,)}"
            )
        starts, counts, hashes = lookup_chunk_rows(
            request.query_range.end_token,
            request.block_hashes,
            block_size=projection.block_size,
            hash_block_size=projection.hash_block_size,
            start_token=request.query_range.start_token,
        )
        keys = self._lookup_keys(projection, hashes)
        try:
            observed = self._backend_io.exists([key for axis in keys for key in axis])
        except Exception as error:
            logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
            return LookupResult(0)
        row_count = len(hashes)
        available = tuple(
            all(observed[axis_index * row_count + row_index] for axis_index in range(len(keys)))
            for row_index in range(row_count)
        )
        reachable = projection.reachability.resolve_available_end(
            request.query_range,
            request.block_hashes,
            (((starts, counts, hashes), available),),
        )
        return LookupResult(reachable.end_token, reachable.tail_key_boundaries)

    def _lookup_hybrid(self, projection: HybridBulkProjection, request: LookupRequest) -> LookupResult:
        if request.transfer_group_ids != projection.group_ids:
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured groups {projection.group_ids}"
            )
        masks = projection.reachability.select_for_lookup(request.block_hashes, request.query_range)
        if len(masks) != len(projection.groups):
            raise RuntimeError("Hybrid Lookup masks do not match the configured transfer groups")
        observations = []
        for group in projection.groups:
            mask = masks[group.dense_index]
            row_projection = fine_lookup_chunk_rows if projection.fine_grained_lookup else lookup_chunk_rows
            starts, counts, hashes = row_projection(
                request.query_range.end_token,
                request.block_hashes,
                block_size=group.block_size,
                hash_block_size=group.hash_block_size,
                start_token=request.query_range.start_token,
                mask=mask,
            )
            keys = hybrid_lookup_keys(group, hashes)
            try:
                observed = self._backend_io.exists([key for axis in keys for key in axis])
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
    def _lookup_keys(projection: BulkProjection, hashes) -> KeyAxes:
        if isinstance(projection, TPMismatchBulkProjection):
            return tp_mismatch_lookup_keys(projection, hashes)
        if isinstance(projection, ConsumerPipelineBulkProjection):
            return consumer_pipeline_lookup_keys(projection, hashes)
        assert isinstance(projection, OrdinaryBulkProjection)
        return ordinary_lookup_keys(projection, hashes)

    def _build_load_batch(self, commands: tuple[LoadCommand, ...]) -> KVTransferBatch:
        projection = self._bound_bulk_projection
        if isinstance(projection, HybridBulkProjection):
            return self._build_hybrid_load_batch(projection, commands)
        rows_by_request: list[BlockRows] = []
        keys_by_request: list[KeyAxes] = []
        for command in commands:
            boundaries = {item.group_id: item.boundary_token for item in command.tail_key_boundaries}
            if len(boundaries) != len(command.tail_key_boundaries):
                raise ValueError("Load contains duplicate tail-key boundaries for one cache group")
            unknown = set(boundaries).difference((projection.group_id,))
            if unknown:
                raise ValueError(f"Load tail-key boundaries contain unknown cache groups {sorted(unknown)}")
            rows = load_block_rows(
                command.load_range.end_token,
                command.block_hashes,
                command.block_ids_by_group[projection.group_id],
                block_size=projection.block_size,
                hash_block_size=projection.hash_block_size,
                start_token=command.load_range.start_token,
                tail_boundary_token=boundaries.get(projection.group_id),
            )
            rows_by_request.append(rows)
            keys_by_request.append(self._load_keys(projection, rows[1]))

        block_parts = [rows[2] for rows in rows_by_request]
        count_parts = [rows[0] for rows in rows_by_request]
        request_splits = [0]
        for rows in rows_by_request:
            request_splits.append(request_splits[-1] + len(rows[2]))
        group = KVGroupBatch(
            projection.group_id,
            concat_uint64(block_parts),
            concat_uint64(count_parts),
            merge_key_axes(projection.group_id, keys_by_request),
            readonly_indices(request_splits),
            projection.physical_layer_ids,
            projection.object_size,
        )
        return KVTransferBatch(tuple(command.request_id for command in commands), (group,))

    def _build_hybrid_load_batch(
        self,
        projection: HybridBulkProjection,
        commands: tuple[LoadCommand, ...],
    ) -> KVTransferBatch:
        masks_by_request = tuple(
            projection.reachability.select_for_load(command.block_hashes, command.load_range) for command in commands
        )
        rows_by_request: list[dict[int, BlockRows]] = []
        for command, masks in zip(commands, masks_by_request, strict=True):
            if len(masks) != len(projection.groups):
                raise RuntimeError("Hybrid Load masks do not match the configured transfer groups")
            boundaries = {item.group_id: item.boundary_token for item in command.tail_key_boundaries}
            if len(boundaries) != len(command.tail_key_boundaries):
                raise ValueError("Load contains duplicate tail-key boundaries for one cache group")
            unknown = set(boundaries).difference(projection.group_ids)
            if unknown:
                raise ValueError(f"Load tail-key boundaries contain unknown cache groups {sorted(unknown)}")
            rows_by_request.append(
                {
                    group.group_id: load_block_rows(
                        command.load_range.end_token,
                        command.block_hashes,
                        command.block_ids_by_group[group.group_id],
                        block_size=group.block_size,
                        hash_block_size=group.hash_block_size,
                        start_token=command.load_range.start_token,
                        tail_boundary_token=boundaries.get(group.group_id),
                        mask=masks[group.dense_index],
                        minimum_block_id=group.minimum_block_id,
                    )
                    for group in projection.groups
                }
            )

        groups = []
        for group_projection in projection.groups:
            block_parts = []
            count_parts = []
            keys_by_request = []
            request_splits = [0]
            for rows_by_group in rows_by_request:
                counts, hashes, block_ids = rows_by_group[group_projection.group_id]
                block_parts.append(block_ids)
                count_parts.append(counts)
                keys_by_request.append(hybrid_load_keys(group_projection, hashes))
                request_splits.append(request_splits[-1] + len(block_ids))
            groups.append(
                KVGroupBatch(
                    group_projection.group_id,
                    concat_uint64(block_parts),
                    concat_uint64(count_parts),
                    merge_key_axes(group_projection.group_id, keys_by_request),
                    readonly_indices(request_splits),
                    group_projection.physical_layer_ids,
                    group_projection.object_size,
                )
            )
        return KVTransferBatch(tuple(command.request_id for command in commands), tuple(groups))

    @staticmethod
    def _load_keys(projection: BulkProjection, hashes) -> KeyAxes:
        if isinstance(projection, TPMismatchBulkProjection):
            return tp_mismatch_load_keys(projection, hashes)
        if isinstance(projection, ConsumerPipelineBulkProjection):
            return consumer_pipeline_load_keys(projection, hashes)
        assert isinstance(projection, OrdinaryBulkProjection)
        return ordinary_load_keys(projection, hashes)

    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> StoreCandidates:
        projection = self._bound_bulk_projection
        if isinstance(projection, HybridBulkProjection):
            return self._build_hybrid_store_candidates(projection, commands)
        rows_by_request: list[StoreCandidateRows] = []
        keys_by_request: list[KeyAxes] = []
        for command in commands:
            if isinstance(command, CheckpointStoreCommand):
                raise ValueError("State checkpoint Store belongs to the deferred align-state route")
            rows = store_candidate_rows(
                command.store_range.end_token,
                command.block_hashes,
                command.block_ids_by_group[projection.group_id],
                block_size=projection.block_size,
                hash_block_size=projection.hash_block_size,
                start_token=command.store_range.start_token,
            )
            rows = select_store_writer_rows(
                self._topology,
                rows,
                tp_mismatch=isinstance(projection, TPMismatchBulkProjection),
            )
            rows_by_request.append(rows)
            keys_by_request.append(self._store_keys(projection, rows[1]))
        return StoreCandidates(
            (
                StoreGroupCandidates(
                    projection.group_id,
                    tuple(rows_by_request),
                    merge_key_axes(projection.group_id, keys_by_request),
                ),
            )
        )

    def _build_hybrid_store_candidates(
        self,
        projection: HybridBulkProjection,
        commands: tuple[StoreCommand, ...],
    ) -> StoreCandidates:
        rows_by_group: dict[int, list[StoreCandidateRows]] = {group_id: [] for group_id in projection.group_ids}
        keys_by_group: dict[int, list[KeyAxes]] = {group_id: [] for group_id in projection.group_ids}
        for command in commands:
            if isinstance(command, CheckpointStoreCommand):
                request_rows = self._hybrid_checkpoint_rows(projection, command)
            else:
                masks = projection.reachability.select_for_store(
                    command.block_hashes,
                    command.store_range,
                    command.num_prompt_tokens,
                )
                if len(masks) != len(projection.groups):
                    raise RuntimeError("Hybrid Store masks do not match the configured transfer groups")
                request_rows = {}
                for group in projection.groups:
                    mask = masks[group.dense_index]
                    if group.uses_align_state:
                        rows = empty_candidate_rows()
                    else:
                        rows = store_candidate_rows(
                            command.store_range.end_token,
                            command.block_hashes,
                            command.block_ids_by_group[group.group_id],
                            block_size=group.block_size,
                            hash_block_size=group.hash_block_size,
                            start_token=command.store_range.start_token,
                            mask=mask,
                            minimum_block_id=group.minimum_block_id,
                        )
                        rows = select_store_writer_rows(
                            self._topology,
                            rows,
                            tp_mismatch=False,
                            align_state=False,
                        )
                    request_rows[group.group_id] = rows

            for group in projection.groups:
                rows = request_rows.get(group.group_id, empty_candidate_rows())
                rows_by_group[group.group_id].append(rows)
                keys_by_group[group.group_id].append(hybrid_store_keys(group, rows[1]))

        return StoreCandidates(
            tuple(
                StoreGroupCandidates(
                    group.group_id,
                    tuple(rows_by_group[group.group_id]),
                    merge_key_axes(group.group_id, keys_by_group[group.group_id]),
                )
                for group in projection.groups
            )
        )

    def _hybrid_checkpoint_rows(
        self,
        projection: HybridBulkProjection,
        command: CheckpointStoreCommand,
    ) -> dict[int, StoreCandidateRows]:
        policy = projection.checkpoint_policy
        if policy is None:
            raise ValueError("State checkpoint Store requires an align-state cache group")
        source_group_ids = tuple(source.group_id for source in command.sources)
        if len(source_group_ids) != len(set(source_group_ids)):
            raise ValueError("State checkpoint Store contains duplicate cache-group sources")
        invalid_sources = set(source_group_ids).difference(policy.state_group_ids)
        if invalid_sources:
            raise ValueError(f"State checkpoint Store contains invalid source groups {sorted(invalid_sources)}")
        boundaries = {source.boundary_token for source in command.sources}
        if len(boundaries) != 1:
            raise ValueError("State checkpoints for one request must share one token boundary")
        boundary = next(iter(boundaries))
        checkpoint_hash = boundary_hash(boundary, command.block_hashes, self._topology.hash_block_size)

        rows_by_group: dict[int, StoreCandidateRows] = {}
        accepted_state_groups = []
        for source in command.sources:
            group = projection.group(source.group_id)
            if source.block_id < group.minimum_block_id:
                continue
            accepted_state_groups.append(group)
            rows = ([group.block_size], [checkpoint_hash], [source.block_id])
            rows_by_group[group.group_id] = select_store_writer_rows(
                self._topology,
                rows,
                tp_mismatch=False,
                align_state=True,
            )

        if accepted_state_groups and any(boundary % group.block_size for group in accepted_state_groups):
            for group_id in policy.companion_group_ids:
                group = projection.group(group_id)
                last_block_index = (boundary + group.block_size - 1) // group.block_size - 1
                first_block_index = min(command.published_store_end_token // group.block_size, last_block_index)
                block_ids = command.block_ids_by_group[group.group_id]
                logical_count = last_block_index + 1
                block_offset = max(logical_count - len(block_ids), 0)
                counts: list[int] = []
                hashes = []
                selected_block_ids: list[int] = []
                for block_index in range(first_block_index, logical_count):
                    local_index = block_index - block_offset
                    if local_index < 0 or local_index >= len(block_ids):
                        continue
                    block_id = block_ids[local_index]
                    if block_id < group.minimum_block_id:
                        continue
                    block_end = min((block_index + 1) * group.block_size, boundary)
                    counts.append(group.block_size)
                    hashes.append(boundary_hash(block_end, command.block_hashes, group.hash_block_size))
                    selected_block_ids.append(block_id)
                rows_by_group[group_id] = select_store_writer_rows(
                    self._topology,
                    (counts, hashes, selected_block_ids),
                    tp_mismatch=False,
                    align_state=False,
                )
        return rows_by_group

    @staticmethod
    def _store_keys(projection: BulkProjection, hashes) -> KeyAxes:
        if isinstance(projection, TPMismatchBulkProjection):
            return tp_mismatch_store_keys(projection, hashes)
        if isinstance(projection, ConsumerPipelineBulkProjection):
            return consumer_pipeline_store_keys(projection, hashes)
        assert isinstance(projection, OrdinaryBulkProjection)
        return ordinary_store_keys(projection, hashes)

    def _store_group_layout(self, group_id: int) -> tuple[int, tuple[tuple[int, ...], ...] | None]:
        projection = self._bound_bulk_projection
        if isinstance(projection, HybridBulkProjection):
            group = projection.group(group_id)
            return group.object_size, tuple(axis.physical_layer_ids for axis in group.store_axes)
        physical_layer_ids_by_axis = (
            tuple(axis.physical_layer_ids for axis in projection.store_axes)
            if isinstance(projection, ConsumerPipelineBulkProjection)
            else None
        )
        return projection.object_size, physical_layer_ids_by_axis

    @property
    def _transfer_group_count(self) -> int:
        projection = self._bound_bulk_projection
        return len(projection.group_ids) if isinstance(projection, HybridBulkProjection) else 1

    def _execute_bulk_load(
        self,
        batch: KVTransferBatch,
        layer_id: int | None = None,
    ) -> tuple[LoadCompletion, ...]:
        if layer_id is not None:
            raise ValueError("Bulk Load does not select an individual layer")
        if self._bulk_projection is None:
            raise RuntimeError("Bulk Load requires explicit Bulk projection")
        try:
            arguments = self._materialize_concrete_bulk_arguments(batch, store=False)
        except Exception as error:
            logger.error(
                "Bulk Load materialization failed for requests %s. type=%s, error=%s",
                batch.request_ids,
                type(error).__name__,
                error,
            )
            return _failed_load_completions(batch, None)
        return self._backend_io.load_materialized(batch, arguments)

    def _execute_bulk_store(
        self, commands: tuple[StoreCommand, ...], source_ready_event: Any
    ) -> tuple[StoreCompletion, ...]:
        selected: KVTransferBatch | None = None
        try:
            candidates = self._build_store_candidates(commands)
            candidate_keys = store_candidate_keys(candidates)
            accepted, claim_once = self._admitted_store_keys(candidate_keys)
            if accepted is not None and not accepted:
                return _store_command_completions(commands)
            selected_objects = (
                None
                if accepted is None
                else select_store_candidate_objects(candidate_keys, accepted, claim_once=claim_once)
            )
            selected = self._materialize_store_candidates(commands, candidates, selected_objects)
            if selected is None:
                return _store_command_completions(commands)
            source_ready_event.synchronize()
        except Exception as error:
            if selected is None:
                return _store_command_completions(commands, error)
            evidence = tuple(TransferEvidence(source, None, True) for source in _batch_sources(selected, None))
            return _failed_store_completions(selected, evidence, error)
        if self._bulk_projection is None:
            raise RuntimeError("Bulk Store requires explicit Bulk projection")
        try:
            arguments = self._materialize_concrete_bulk_arguments(selected, store=True)
        except Exception as error:
            evidence = tuple(TransferEvidence(source, None, True) for source in _batch_sources(selected, None))
            return _failed_store_completions(selected, evidence, error)
        return self._backend_io.store_materialized(selected, arguments)

    def _materialize_concrete_bulk_arguments(
        self,
        batch: KVTransferBatch,
        *,
        store: bool,
    ) -> BulkBackendArguments:
        projection = self._bound_bulk_projection
        keys: list[str] = []
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        if isinstance(projection, HybridBulkProjection):
            if tuple(group.group_id for group in batch.groups) != projection.group_ids:
                raise RuntimeError("Hybrid Bulk execution groups do not match the bound transfer projection")
            for group in batch.groups:
                ranges = hybrid_bulk_ranges(
                    projection.group(group.group_id),
                    group.block_ids,
                    group.token_counts,
                    group.selected_objects,
                    store=store,
                )
                group_keys, group_addresses, group_sizes = bulk_arguments(
                    group.key_axes,
                    ranges,
                    group.selected_objects,
                )
                keys.extend(group_keys)
                addresses.extend(group_addresses)
                sizes.extend(group_sizes)
        else:
            if len(batch.groups) != 1 or batch.groups[0].group_id != projection.group_id:
                raise RuntimeError("Single-group Bulk execution does not match its bound cache group")
            group = batch.groups[0]
            if isinstance(projection, TPMismatchBulkProjection):
                ranges = tp_mismatch_bulk_ranges(
                    projection,
                    group.block_ids,
                    group.token_counts,
                    group.selected_objects,
                )
            elif isinstance(projection, ConsumerPipelineBulkProjection):
                range_projection = consumer_pipeline_store_ranges if store else consumer_pipeline_load_ranges
                ranges = range_projection(
                    projection,
                    group.block_ids,
                    group.token_counts,
                    group.selected_objects,
                )
            else:
                assert isinstance(projection, OrdinaryBulkProjection)
                ranges = ordinary_bulk_ranges(
                    projection,
                    group.block_ids,
                    group.token_counts,
                    group.selected_objects,
                )
            keys, addresses, sizes = bulk_arguments(group.key_axes, ranges, group.selected_objects)
        return BulkBackendArguments(
            keys,
            addresses,
            sizes,
            _batch_sources(batch, None),
        )

    def _submit_step_store(self, context: KVPoolStepContext, *, fence: bool) -> None:
        del fence
        if self._pending_store_batch is not None:
            raise RuntimeError("Previous Store invocation has not reached its fence")
        timeline = self._asynchronous_store_timeline
        if timeline is None:
            raise RuntimeError("Bulk Store is disabled")
        self._pending_store_batch = timeline.submit(
            context.step.store.commands,
            self._record_source_ready(),
        )
        context.store_submitted = True


class SynchronousBulkWorker(BulkWorker):
    """Run Bulk Load on the caller while retaining asynchronous Store."""

    def _create_load_timeline(self, backend_io: BackendIO) -> SynchronousLoadTimeline:
        del backend_io
        timeline = SynchronousLoadTimeline(self._execute_bulk_load)
        self._synchronous_load_timeline = timeline
        return timeline


class AsynchronousBulkWorker(BulkWorker):
    """Run Bulk Load and Store on Worker-owned execution threads."""

    def _create_load_timeline(self, backend_io: BackendIO) -> AsynchronousLoadTimeline:
        timeline = AsynchronousLoadTimeline(backend_io.initialize_thread, self._execute_bulk_load)
        self._asynchronous_load_timeline = timeline
        return timeline

    def _before_load_submit(self, batch: KVTransferBatch) -> None:
        overlapping = self._pending_load_request_ids.intersection(batch.request_ids)
        if overlapping:
            raise RuntimeError(f"Request already has a pending asynchronous Load: {sorted(overlapping)}")
        self._pending_load_request_ids.update(batch.request_ids)

    def _load_submit_failed(self, batch: KVTransferBatch) -> None:
        self._pending_load_request_ids.difference_update(batch.request_ids)

    def _accept_load_completion(self, completion: LoadCompletion) -> None:
        if completion.request_id not in self._pending_load_request_ids:
            raise RuntimeError(f"Load completion has no pending execution: {completion.request_id}")
        self._pending_load_request_ids.remove(completion.request_id)


__all__ = (
    "AsynchronousBulkWorker",
    "BulkWorker",
    "SynchronousBulkWorker",
)
