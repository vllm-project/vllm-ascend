"""Define the compiled KV Pool program over request-local values."""

from __future__ import annotations

from collections.abc import Callable

from vllm.logger import logger

from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import LoadCommand, StoreCommand
from .invocation import (
    BindingEvidence,
    KVPoolStepFrame,
    LoadCompletion,
    LoadTransfer,
    RemoteObjectObservation,
    StoreCompletion,
    StoreEvidence,
    StoreTransfer,
)
from .representation import BindingBatch, KVBinding, KVMemoryGeometry, RemoteObjectBatch
from .spec.schedule import KVPoolSchedule
from .spec.topology import KVPoolTopology
from .stages.chunk import KVChunkProjection
from .stages.memory import BindingProjection, KVBlockProjection
from .stages.reachability import ChunkAvailability, GroupAvailability, KVReachability
from .stages.remote import RemoteObjectProjection
from .stages.store import (
    ConsumerProjection,
    MissingFilter,
    ObjectPresenceObserver,
    StoreOwnershipProjection,
)

ReadabilityObserver = Callable[[RemoteObjectBatch], tuple[RemoteObjectObservation, ...]]
LoadBindings = Callable[[tuple[KVBinding, ...]], tuple[BindingEvidence, ...]]
StoreBindings = Callable[[tuple[BindingBatch, ...]], StoreEvidence]


class KVPoolProgram:
    """Expose the compiled Lookup, Load and Store workflow for one KV Pool participant."""

    def __init__(
        self,
        topology: KVPoolTopology,
        reachability: KVReachability,
        chunk_projection: KVChunkProjection,
        remote_object_projection: RemoteObjectProjection,
        block_projection: KVBlockProjection,
        binding_projection: BindingProjection,
        store_ownership_projection: StoreOwnershipProjection,
        consumer_projection: ConsumerProjection,
        missing_filter: MissingFilter,
        backend_name: str,
        schedule: KVPoolSchedule,
    ) -> None:
        self._topology = topology
        self._reachability = reachability
        self._chunk_projection = chunk_projection
        self._remote_object_projection = remote_object_projection
        self._block_projection = block_projection
        self._binding_projection = binding_projection
        self._store_ownership_projection = store_ownership_projection
        self._consumer_projection = consumer_projection
        self._missing_filter = missing_filter
        self.backend_name = backend_name
        self.schedule = schedule

    @property
    def topology(self) -> KVPoolTopology:
        return self._topology

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None:
        self._binding_projection.bind_memory(memory_geometry)
        self._consumer_projection.bind_memory(memory_geometry)

    def lookup(self, request: LookupRequest, observe_readability: ReadabilityObserver) -> LookupResult:
        if request.transfer_group_ids != self._reachability.group_ids:
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured groups "
                f"{self._reachability.group_ids}"
            )
        selection = self._reachability.select_for_lookup(request.block_hashes, request.query_range)
        _, object_keys = self._chunk_projection.project(selection)
        availability = []
        remote_objects = self._remote_object_projection.project(object_keys)
        for key_batch, object_batch in zip(object_keys, remote_objects, strict=True):
            try:
                observations = observe_readability(object_batch)
            except Exception as error:
                logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                return LookupResult(0)
            available_by_chunk = {item.chunk: True for item in key_batch.keys}
            for observation in observations:
                available_by_chunk[observation.remote_object.chunk] &= observation.readable
            availability.append(
                GroupAvailability(
                    key_batch.group_id,
                    tuple(
                        ChunkAvailability(
                            item.chunk.token_range,
                            item.chunk.content_hash,
                            available_by_chunk[item.chunk],
                        )
                        for item in key_batch.keys
                    ),
                )
            )
        return LookupResult(self._reachability.resolve_available_end(selection, availability))

    def build_load_transfers(self, frame: KVPoolStepFrame) -> list[LoadTransfer]:
        return [self._build_load_transfer(command) for command in frame.step.load.commands]

    def execute_load(self, transfer: LoadTransfer, load_bindings: LoadBindings) -> LoadCompletion:
        return LoadCompletion(transfer.request_id, load_bindings(transfer.traversal))

    def record_load_completion(self, frame: KVPoolStepFrame, completion: LoadCompletion) -> None:
        failed_block_ids = {
            evidence.binding.memory.block_id for evidence in completion.binding_evidence if evidence.result_code != 0
        }
        if not failed_block_ids:
            return
        if len(self._topology.transfer_group_ids) > 1:
            frame.failed_request_ids.add(completion.request_id)
        else:
            frame.failed_block_ids.update(failed_block_ids)

    def build_store_transfers(self, frame: KVPoolStepFrame) -> list[StoreTransfer]:
        return [self._build_store_transfer(command) for command in frame.step.store.commands]

    def filter_store_bindings(
        self,
        transfers: list[StoreTransfer],
        observe_presence: ObjectPresenceObserver,
    ) -> list[StoreTransfer]:
        keys = tuple(
            dict.fromkeys(
                binding.remote_object.key
                for transfer in transfers
                for batch in transfer.batches
                for binding in batch.bindings
            )
        )
        if not keys:
            return transfers
        missing_keys = set(self._missing_filter.select_missing_keys(keys, observe_presence))
        if len(missing_keys) == len(keys):
            return transfers
        return [
            StoreTransfer(
                transfer.request_id,
                tuple(
                    BindingBatch(
                        batch.group_id,
                        tuple(binding for binding in batch.bindings if binding.remote_object.key in missing_keys),
                    )
                    for batch in transfer.batches
                ),
            )
            for transfer in transfers
        ]

    def _build_load_transfer(self, command: LoadCommand) -> LoadTransfer:
        selection = self._reachability.select_for_load(command.block_hashes, command.load_range)
        chunks, object_keys = self._chunk_projection.project(selection)
        block_assignments = self._block_projection.project(chunks, command.block_ids_by_group)
        batches = tuple(
            self._binding_projection.project(assignments, keys)
            for assignments, keys in zip(block_assignments, object_keys, strict=True)
        )
        traversal = tuple(binding for batch in batches for binding in batch.bindings)
        if traversal:
            offset = self._topology.tp_rank % len(traversal)
            traversal = traversal[offset:] + traversal[:offset]
        return LoadTransfer(command.request_id, traversal)

    def _build_store_transfer(self, command: StoreCommand) -> StoreTransfer:
        selection = self._reachability.select_for_store(
            command.block_hashes,
            command.store_range,
            command.num_prompt_tokens,
        )
        chunks, object_keys = self._chunk_projection.project(selection)
        block_assignments = self._block_projection.project(chunks, command.block_ids_by_group)
        owned_assignments = self._store_ownership_projection.project(block_assignments)
        batches = tuple(
            self._binding_projection.project(assignments, keys)
            for assignments, keys in zip(owned_assignments, object_keys, strict=True)
        )
        batches = self._consumer_projection.project(batches)
        return StoreTransfer(command.request_id, batches)

    def execute_store(
        self,
        transfer: StoreTransfer,
        wait_for_source: Callable[[], None],
        store_bindings: StoreBindings,
    ) -> StoreCompletion:
        if not any(batch.bindings for batch in transfer.batches):
            return StoreCompletion(transfer.request_id, StoreEvidence((), True, True))
        try:
            wait_for_source()
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error))

        try:
            evidence = store_bindings(transfer.batches)
        except Exception as error:
            evidence = StoreEvidence((), False, False, error)
        return StoreCompletion(transfer.request_id, evidence)

    @staticmethod
    def validate_store_completion(completion: StoreCompletion) -> None:
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
