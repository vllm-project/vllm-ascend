"""Execute Bulk transfers with construction-time Load mode and independent I/O executors."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from concurrent.futures import BrokenExecutor, CancelledError, Future
from typing import Any

from vllm.logger import logger

from ...infrastructure.configuration.topology import KVPoolTopology
from ...infrastructure.coordinator import Coordinator, checkpoint_companion_indices
from ...infrastructure.execution import Executor
from ...infrastructure.mapping import (
    LocalTable,
    build_lookup_keys,
    build_mappings,
    map_keys,
    select_layers,
)
from ...infrastructure.protocol.messages import CheckpointStoreCommand, StoreCommand
from ...infrastructure.resources import KVPoolResources
from ..base import KVPoolStepContext, KVPoolWorker, PendingStore
from ..transfer.batch import (
    KeysByPartition,
    KVTransferBatch,
    StoreGroupTransferCandidates,
    StoreTransferCandidates,
    select_store_transfer_candidates,
    store_transfer_candidate_keys,
)
from ..transfer.completion import (
    LoadCompletion,
    StoreCompletion,
    TransferEvidence,
    _batch_sources,
    _failed_load_completions,
    _store_command_completions,
)
from ..transfer.rows import (
    StoreTransferRecords,
    boundary_hash,
    empty_store_transfer_records,
    merge_keys_by_partition,
    store_transfer_records,
)
from .transfer import (
    BulkBackendArguments,
    _bulk_load_completions,
    _empty_store_completions,
    _failed_store_completions,
    _store_completions,
    bind_bulk_sizes,
    bind_store_writer_selection,
    materialize_bulk_arguments,
)


class BulkWorker(KVPoolWorker):
    """Own Bulk business operations, pending results and independent Load/Store execution."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: KVPoolResources,
        *,
        coordinator: Coordinator,
        kv_cache_layout: str = "NHD",
        store_enabled: bool = True,
        load_async: bool = False,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._companion_group_ids = tuple(
            group.group_id for group in topology.transfer_groups if not group.uses_align_state
        )
        self._store_writer_selection = {
            group.group_id: bind_store_writer_selection(
                topology, tp_mismatch=topology.tp_partition.tp_mismatch, align_state=group.uses_align_state
            )
            for group in topology.transfer_groups
        }
        self._store_executor = Executor("KVPoolStoreExecutor", resources.backend.set_device) if store_enabled else None
        self._load_executor = Executor("KVPoolLoadExecutor", resources.backend.set_device) if load_async else None
        self._pending_loads: deque[Future[tuple[LoadCompletion, ...]]] = deque()
        self._load_batches: dict[Future[tuple[LoadCompletion, ...]], KVTransferBatch] = {}
        self._pending_load_request_ids: set[str] = set()
        super().__init__(
            topology,
            resources,
            executors=tuple(
                executor for executor in (self._store_executor, self._load_executor) if executor is not None
            ),
            store_enabled=store_enabled,
            coordinator=coordinator,
            kv_cache_layout=kv_cache_layout,
            source_ready_event_factory=source_ready_event_factory,
        )

    def _bind_mappings(self) -> None:
        # Resources retains the caller's tensors through executor drain and unregister.
        kv_caches = self._resources.kv_caches
        if kv_caches is None:
            raise RuntimeError("KV tensors must be retained before binding Bulk mappings")
        self._local: dict[int, LocalTable] = {}
        physical_layer_ids_by_partition = {}
        for group_id, group in self._groups.items():
            keys, local, _ = build_mappings(
                self._topology, kv_caches, group_id=group_id, kv_cache_layout=self._kv_cache_layout
            )
            if not any(len(segments["bases"]) for segments in local.values()):
                raise ValueError("KV cache group has no registered memory entries")
            self._keys[group_id] = keys
            self._lookup_key_parts[group_id] = build_lookup_keys(self._topology, group_id)
            self._local[group_id] = local
            layers_by_pp = select_layers(group, self._topology.consumer_pipeline_partitions)
            physical_layer_ids_by_partition[group_id] = tuple(
                tuple(layer.physical_layer_id for layer in layers_by_pp[pp_rank]) for pp_rank, _, _ in keys
            )
        self._bulk_sizes = bind_bulk_sizes(self._local)
        self._bulk_object_layouts = {
            group_id: (sum(sum(lengths) for lengths in sizes), physical_layer_ids_by_partition[group_id])
            for group_id, sizes in self._bulk_sizes.items()
        }

    def _group_object_layout(self, group_id: int) -> tuple[int, tuple[tuple[int, ...], ...]]:
        return self._bulk_object_layouts[group_id]

    def _submit_load(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        if self._load_executor is None:
            return self._execute_bulk_load(batch)
        overlapping = self._pending_load_request_ids.intersection(batch.request_ids)
        if overlapping:
            raise RuntimeError(f"Request already has a pending asynchronous Load: {sorted(overlapping)}")
        self._pending_load_request_ids.update(batch.request_ids)
        try:
            future = self._load_executor.submit(self._execute_bulk_load, batch)
        except BaseException:
            self._pending_load_request_ids.difference_update(batch.request_ids)
            raise
        self._load_batches[future] = batch
        self._pending_loads.append(future)
        return ()

    def _collect_load_completions(self) -> tuple[LoadCompletion, ...]:
        completed: list[LoadCompletion] = []
        while self._pending_loads and self._pending_loads[0].done():
            future = self._pending_loads.popleft()
            batch = self._load_batches.pop(future)
            completed.extend(self._load_batch_completions(future, batch))
        return tuple(completed)

    def _load_batch_completions(
        self, future: Future[tuple[LoadCompletion, ...]], batch: KVTransferBatch
    ) -> tuple[LoadCompletion, ...]:
        try:
            return future.result()
        except (BrokenExecutor, CancelledError):
            raise
        except Exception:
            logger.exception("Bulk Load failed for requests %s", batch.request_ids)
            return _failed_load_completions(batch, None)

    def _accept_load_completion(self, completion: LoadCompletion) -> None:
        if completion.request_id not in self._pending_load_request_ids:
            raise RuntimeError(f"Load completion has no pending execution: {completion.request_id}")
        self._pending_load_request_ids.remove(completion.request_id)

    def _submit_step_store(self, context: KVPoolStepContext, *, fence: bool) -> None:
        del fence
        if self._store_executor is None:
            raise RuntimeError("Bulk Store is disabled")
        commands = context.step.store.commands
        future = self._store_executor.submit(self._execute_bulk_store, commands, self._record_source_ready())
        self._track_store_batch(
            PendingStore(
                future,
                frozenset(command.request_id for command in commands),
                frozenset(command.store_job_id for command in commands if command.store_job_id is not None),
            )
        )
        context.store_submitted = True

    def _collect_store_completions(self, batch: PendingStore, *, wait: bool) -> tuple[StoreCompletion, ...] | None:
        if not wait and not batch.future.done():
            return None
        return batch.future.result()

    def _check_execution_results(self) -> None:
        # Observe task failures at shutdown without consuming pending Load notifications.
        for future in self._pending_loads:
            self._load_batch_completions(future, self._load_batches[future])

    def _build_store_candidates(self, commands: tuple[StoreCommand, ...]) -> StoreTransferCandidates:
        transfer_records_by_group: dict[int, list[StoreTransferRecords]] = {group_id: [] for group_id in self._groups}
        keys_by_group: dict[int, list[KeysByPartition]] = {group_id: [] for group_id in self._groups}
        for command in commands:
            if isinstance(command, CheckpointStoreCommand):
                request_rows = self._checkpoint_rows(command)
                for group_id in self._groups:
                    rows = request_rows.get(group_id)
                    if rows is None:
                        rows = empty_store_transfer_records()
                    transfer_records_by_group[group_id].append(rows)
                    keys_by_group[group_id].append(
                        tuple(map_keys(parts, rows[1]) for parts in self._keys[group_id].values())
                    )
                continue

            masks = self._coordinator.select_for_store(
                command.start_token, command.end_token, command.num_prompt_tokens
            )
            for (group_id, group), mask in zip(self._groups.items(), masks, strict=True):
                if group_id in self._state_group_ids:
                    rows = empty_store_transfer_records()
                else:
                    rows = store_transfer_records(
                        command.end_token,
                        command.block_hashes,
                        command.block_ids_by_group[group_id],
                        block_size=group.block_size,
                        hash_block_size=self._topology.hash_block_size,
                        start_token=command.start_token,
                        mask=mask,
                    )
                    selection = self._store_writer_selection[group_id]
                    if selection is not None:
                        rows = (rows[0][selection], rows[1][selection], rows[2][selection])
                transfer_records_by_group[group_id].append(rows)
                keys_by_group[group_id].append(
                    tuple(map_keys(parts, rows[1]) for parts in self._keys[group_id].values())
                )
        return StoreTransferCandidates(
            tuple(
                StoreGroupTransferCandidates(
                    group_id,
                    tuple(transfer_records_by_group[group_id]),
                    merge_keys_by_partition(group_id, keys_by_group[group_id]),
                )
                for group_id in self._groups
            )
        )

    def _checkpoint_rows(
        self,
        command: CheckpointStoreCommand,
    ) -> dict[int, StoreTransferRecords]:
        if not self._state_group_ids:
            raise ValueError("State checkpoint Store requires an align-state cache group")
        source_group_ids = tuple(source.group_id for source in command.sources)
        if len(source_group_ids) != len(set(source_group_ids)):
            raise ValueError("State checkpoint Store contains duplicate cache-group sources")
        invalid_sources = set(source_group_ids).difference(self._state_group_ids)
        if invalid_sources:
            raise ValueError(f"State checkpoint Store contains invalid source groups {sorted(invalid_sources)}")
        boundaries = {source.boundary_token for source in command.sources}
        if len(boundaries) != 1:
            raise ValueError("State checkpoints for one request must share one token boundary")
        boundary = next(iter(boundaries))
        checkpoint_hash = boundary_hash(boundary, command.block_hashes, self._topology.hash_block_size)

        transfer_records_by_group: dict[int, StoreTransferRecords] = {}
        accepted_state_groups = []
        for source in command.sources:
            group = self._groups[source.group_id]
            if source.block_id <= 0:
                continue
            accepted_state_groups.append(group)
            rows = ([group.block_size], [checkpoint_hash], [source.block_id])
            selection = self._store_writer_selection[group.group_id]
            transfer_records_by_group[group.group_id] = (
                rows if selection is None else (rows[0][selection], rows[1][selection], rows[2][selection])
            )

        if accepted_state_groups and any(boundary % group.block_size for group in accepted_state_groups):
            for group_id in self._companion_group_ids:
                group = self._groups[group_id]
                local_block_ids = command.block_ids_by_group[group.group_id]
                logical_count = (boundary + group.block_size - 1) // group.block_size
                block_offset = max(logical_count - len(local_block_ids), 0)
                counts: list[int] = []
                hashes = []
                selected_block_ids: list[int] = []
                for local_index in checkpoint_companion_indices(
                    boundary, command.published_store_end_token, group.block_size, len(local_block_ids)
                ):
                    block_id = local_block_ids[local_index]
                    if block_id < 0:
                        continue
                    block_end = min((local_index + block_offset + 1) * group.block_size, boundary)
                    counts.append(group.block_size)
                    hashes.append(boundary_hash(block_end, command.block_hashes, self._topology.hash_block_size))
                    selected_block_ids.append(block_id)
                selection = self._store_writer_selection[group_id]
                transfer_records_by_group[group_id] = (
                    (counts, hashes, selected_block_ids)
                    if selection is None
                    else (counts[selection], hashes[selection], selected_block_ids[selection])
                )
        return transfer_records_by_group

    def _execute_bulk_load(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        try:
            arguments = self._materialize_bulk_arguments(batch)
        except Exception as error:
            logger.error(
                "Bulk Load materialization failed for requests %s. type=%s, error=%s",
                batch.request_ids,
                type(error).__name__,
                error,
            )
            return _failed_load_completions(batch, None)
        if not arguments.keys:
            return tuple(LoadCompletion(request_id, ()) for request_id in batch.request_ids)
        try:
            result_codes = self._backend.load(arguments.keys, arguments.addresses, arguments.sizes)
        except Exception as error:
            logger.error(
                "Bulk Load failed for requests %s. type=%s, error=%s",
                batch.request_ids,
                type(error).__name__,
                error,
            )
            return _failed_load_completions(batch, None)
        return _bulk_load_completions(batch, result_codes)

    def _execute_bulk_store(
        self, commands: tuple[StoreCommand, ...], source_ready_event: Any
    ) -> tuple[StoreCompletion, ...]:
        selected: KVTransferBatch | None = None
        try:
            candidates = self._build_store_candidates(commands)
            candidate_keys = store_transfer_candidate_keys(candidates)
            accepted, claim_once = self._admitted_store_keys(candidate_keys)
            if accepted is not None and not accepted:
                return _store_command_completions(commands)
            transfer_candidate_mask = (
                None
                if accepted is None
                else select_store_transfer_candidates(candidate_keys, accepted, claim_once=claim_once)
            )
            selected = self._materialize_store_candidates(commands, candidates, transfer_candidate_mask)
            if selected is None:
                return _store_command_completions(commands)
        except Exception as error:
            if selected is None:
                return _store_command_completions(commands, error)
            evidence = tuple(TransferEvidence(source, None, True) for source in _batch_sources(selected, None))
            return _failed_store_completions(selected, evidence, error)
        try:
            arguments = self._materialize_bulk_arguments(selected)
            sources = _batch_sources(selected, None)
            # Address preparation can overlap NPU writes; only the Backend KV read must wait.
            source_ready_event.synchronize()
        except Exception as error:
            evidence = tuple(TransferEvidence(source, None, True) for source in _batch_sources(selected, None))
            return _failed_store_completions(selected, evidence, error)
        if not sources:
            return _empty_store_completions(selected)
        try:
            result_codes = self._backend.store(arguments.keys, arguments.addresses, arguments.sizes)
        except Exception as error:
            evidence = tuple(TransferEvidence(source, None, False) for source in sources)
            return _store_completions(selected, evidence, error)
        evidence = tuple(
            TransferEvidence(source, code, True) for source, code in zip(sources, result_codes, strict=True)
        )
        return _store_completions(selected, evidence)

    def _materialize_bulk_arguments(self, batch: KVTransferBatch) -> BulkBackendArguments:
        return materialize_bulk_arguments(self._local, batch, self._bulk_sizes)
