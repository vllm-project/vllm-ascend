"""Execute the fixed KV Pool dataflow over request-local values."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.logger import logger

from .execution.io import BackendIO, MissingFilter, StoreEvidence
from .execution.resources import KVResources
from .execution.timeline import (
    LoadCompletion,
    LoadTimeline,
    LoadTransfer,
    StoreBatch,
    StoreCompletion,
    StoreTimeline,
    StoreTransfer,
)
from .graph.projection import ConsumerProjection, KVProjection
from .graph.reachability import ChunkAvailability, GroupAvailability, KVReachability
from .graph.topology import KVTopology
from .protocol.lookup import LookupRequest, LookupResult
from .protocol.transfer import LoadCommand, LoadCommandBatch, StoreCommand, StoreCommandBatch


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Terminal Load facts consumed through vLLM's split completion hooks."""

    completed_request_ids: frozenset[str]
    failed_request_ids: frozenset[str]
    failed_block_ids: frozenset[int]


class KVPoolGraph:
    """Expose the fixed Lookup, Load and Store dataflow for one KV Pool participant."""

    def __init__(
        self,
        resources: KVResources,
        topology: KVTopology,
        reachability: KVReachability,
        projection: KVProjection,
        consumer_projection: ConsumerProjection,
        missing_filter: MissingFilter,
        backend_io: BackendIO,
        load_timeline: LoadTimeline,
        store_timeline: StoreTimeline | None,
    ) -> None:
        self._resources = resources
        self._topology = topology
        self._reachability = reachability
        self._projection = projection
        self._consumer_projection = consumer_projection
        self._missing_filter = missing_filter
        self._backend_io = backend_io
        self._load_timeline = load_timeline
        self._store_timeline = store_timeline
        self._load_timeline.attach_operation(self._execute_load)
        if self._store_timeline is not None:
            self._store_timeline.attach_operation(self._execute_store)
        self._pending_store: StoreBatch | None = None
        self._store_error: Exception | None = None
        self._failed_request_ids: set[str] = set()
        self._failed_block_ids: set[int] = set()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            self._resources.register_kv_caches(kv_caches)
            self._projection.compile_memory_mapping()
            self._consumer_projection.compile_memory_mapping()
            if self._store_timeline is not None:
                self._store_timeline.start()
            self._load_timeline.start()
        except BaseException:
            self.close()
            raise

    def lookup(self, request: LookupRequest) -> LookupResult:
        if request.transfer_group_ids != self._reachability.group_ids:
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured groups "
                f"{self._reachability.group_ids}"
            )
        selection = self._reachability.select_for_lookup(request.block_hashes, request.query_range)
        chunk_projections = self._projection.project_chunks(selection)
        availability = []
        for batch in self._projection.project_remote_objects(chunk_projections):
            try:
                observations = self._backend_io.observe_readability(batch)
            except Exception as error:
                logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                return LookupResult(0)
            available_by_chunk = {chunk: True for chunk in batch.chunks}
            for observation in observations:
                available_by_chunk[observation.remote_object.chunk] &= observation.readable
            availability.append(
                GroupAvailability(
                    batch.group_id,
                    tuple(
                        ChunkAvailability(chunk.token_range, chunk.content_hash, available_by_chunk[chunk])
                        for chunk in batch.chunks
                    ),
                )
            )
        return LookupResult(self._reachability.resolve_available_end(selection, availability))

    def load(self, command_batch: LoadCommandBatch) -> None:
        transfers = [self._build_load_transfer(command) for command in command_batch.commands]
        completions = tuple(self._load_timeline.submit(transfers))
        self._record_load_failures(completions)
        if completions and self._failed_request_ids:
            raise RuntimeError(f"Hybrid KV Load failed for requests: {sorted(self._failed_request_ids)}")

    def collect_load_result(self) -> LoadResult:
        completions = tuple(self._load_timeline.collect())
        self._record_load_failures(completions)
        result = LoadResult(
            frozenset(completion.request_id for completion in completions),
            frozenset(self._failed_request_ids),
            frozenset(self._failed_block_ids),
        )
        self._failed_request_ids.clear()
        self._failed_block_ids.clear()
        return result

    def submit_store(self, command_batch: StoreCommandBatch) -> None:
        self._raise_store_error()
        if self._store_timeline is None or not command_batch.commands:
            return
        if self._pending_store is not None:
            raise RuntimeError("Previous Store batch has not reached its fence")
        try:
            source_ready_event = torch.npu.Event()
            source_ready_event.record()
            transfers = [self._build_store_transfer(command, source_ready_event) for command in command_batch.commands]
            self._pending_store = self._store_timeline.submit(transfers)
        except Exception as error:
            self._store_error = error
            raise

    def wait_for_previous_store(self) -> tuple[StoreCompletion, ...]:
        self._raise_store_error()
        if self._store_timeline is None or self._pending_store is None:
            return ()
        try:
            completions = self._store_timeline.wait(self._pending_store)
        except Exception as error:
            self._store_error = error
            raise
        if all(completion.evidence.source_release_confirmed for completion in completions):
            self._pending_store = None
        for completion in completions:
            try:
                self._validate_store_completion(completion)
            except Exception as error:
                self._store_error = error
                raise
        return completions

    def close(self) -> None:
        store_error: BaseException | None = None
        try:
            self.wait_for_previous_store()
        except BaseException as error:
            store_error = error
        if self._store_timeline is not None:
            try:
                self._store_timeline.close()
            except BaseException as error:
                if store_error is None:
                    store_error = error
        try:
            self._load_timeline.close()
        finally:
            if self._pending_store is None:
                self._resources.close()
        if store_error is not None:
            raise store_error

    def _build_load_transfer(self, command: LoadCommand) -> LoadTransfer:
        selection = self._reachability.select_for_load(command.block_hashes, command.load_range)
        chunk_projections = self._projection.project_chunks(selection)
        allocation_batches = self._projection.assign_local_blocks(chunk_projections, command.block_ids_by_group)
        batches = self._projection.bind_representations(allocation_batches)
        traversal = tuple(binding for batch in batches for binding in batch.bindings)
        if traversal:
            offset = self._topology.tp_rank % len(traversal)
            traversal = traversal[offset:] + traversal[:offset]
        return LoadTransfer(command.request_id, traversal)

    def _build_store_transfer(self, command: StoreCommand, source_ready_event) -> StoreTransfer:
        selection = self._reachability.select_for_store(
            command.block_hashes,
            command.store_range,
            command.num_prompt_tokens,
        )
        chunk_projections = self._projection.project_chunks(selection)
        allocation_batches = self._projection.assign_local_blocks(chunk_projections, command.block_ids_by_group)
        owned_allocations = self._projection.select_owned_allocations(allocation_batches)
        batches = self._projection.bind_representations(owned_allocations)
        batches = self._consumer_projection.project(batches)
        return StoreTransfer(command.request_id, batches, source_ready_event)

    def _execute_load(self, transfer: LoadTransfer) -> LoadCompletion:
        return LoadCompletion(transfer.request_id, self._backend_io.load(transfer.traversal))

    def _execute_store(self, transfer: StoreTransfer) -> StoreCompletion:
        try:
            batches = self._missing_filter.select_missing(transfer.batches)
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error))
        if not any(batch.bindings for batch in batches):
            return StoreCompletion(transfer.request_id, StoreEvidence((), True, True))
        try:
            transfer.source_ready_event.synchronize()
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error))

        try:
            evidence = self._backend_io.store(batches)
        except Exception as error:
            evidence = StoreEvidence((), False, False, error)
        return StoreCompletion(transfer.request_id, evidence)

    def _record_load_failures(self, completions: tuple[LoadCompletion, ...]) -> None:
        for completion in completions:
            failed_block_ids = {
                evidence.binding.local_slice.block_id
                for evidence in completion.binding_evidence
                if evidence.result_code != 0
            }
            if not failed_block_ids:
                continue
            if len(self._topology.transfer_group_ids) > 1:
                self._failed_request_ids.add(completion.request_id)
            else:
                self._failed_block_ids.update(failed_block_ids)

    def _raise_store_error(self) -> None:
        if self._store_error is not None:
            raise RuntimeError("KVPoolGraph cannot continue after a previous Store failure") from self._store_error

    @staticmethod
    def _validate_store_completion(completion: StoreCompletion) -> None:
        evidence = completion.evidence
        if evidence.error is not None:
            raise RuntimeError(f"Store failed for request {completion.request_id}") from evidence.error
        failed_codes = [item.result_code for item in evidence.binding_evidence if item.result_code not in (0, None)]
        if failed_codes:
            raise RuntimeError(f"Store failed for request {completion.request_id} with result codes {failed_codes}")
        if not evidence.succeeded:
            raise RuntimeError(f"Store success is unknown for request {completion.request_id}")
        if not evidence.source_release_confirmed:
            raise RuntimeError(f"Store source release is unknown for request {completion.request_id}")
