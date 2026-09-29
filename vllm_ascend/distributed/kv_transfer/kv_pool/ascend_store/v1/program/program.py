"""Run the compiled KV Pool graph over request-local values.

Node labels describe two independent properties: ``D`` changes the domain representation,
``V`` selects a configured variant, and ``F`` is fixed rather than variant. Compilation fixes
every ``V`` node once; each request selects an existing subgraph without rebuilding the graph.

1. Lookup: Reachability[D,V] -> Chunk[D,F] -> Remote identity[D,F] -> Remote object[D,F]
   -> observation -> Availability[D,F] -> reachable frontier[-,V].
2. Load: Reachability[D,V] -> Chunk[D,F] -> Local Block[D,F] -> Transfer Region[D,V]
   -> Remote object[D,F] -> Binding[D,F] -> traversal[-,F].
3. Range Store: Reachability[D,V] -> Chunk[D,F] -> Local Block[D,F] -> shared Store tail.
4. Checkpoint Store: Checkpoint Chunk[D,F] -> Checkpoint Block[D,F] -> shared Store tail.

The shared Store tail is Ownership[-,V] -> Transfer Region[D,V] -> Partition[-,V]
-> Remote object[D,F] -> Binding[D,F] -> Admission[-,V].
"""

from __future__ import annotations

from collections.abc import Callable

from vllm.logger import logger

from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import CheckpointStoreCommand, LoadCommand, StoreCommand
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
from .representation import (
    BindingBatch,
    KVBinding,
    KVMemoryGeometry,
    RemoteObjectBatch,
    RemoteObjectKeyBatch,
    TransferRegionBatch,
)
from .spec.schedule import KVPoolSchedule
from .spec.topology import KVPoolTopology
from .stages.admission import ObjectPresenceObserver, StoreAdmission
from .stages.block import CheckpointBlockResolution, LocalBlockResolution
from .stages.chunk import CheckpointChunkProjection, SemanticChunkProjection
from .stages.ownership import StoreOwnershipSelection
from .stages.partition import RegionPartition
from .stages.reachability import ChunkAvailability, GroupAvailability, ReachableRegionSelection
from .stages.region import TransferRegionProjection
from .stages.remote import RemoteObjectProjection, project_remote_identities

ReadabilityObserver = Callable[[RemoteObjectBatch], tuple[RemoteObjectObservation, ...]]
LoadBindings = Callable[[tuple[KVBinding, ...]], tuple[BindingEvidence, ...]]
StoreBindings = Callable[[tuple[BindingBatch, ...]], StoreEvidence]


class KVPoolProgram:
    """Expose the compiled Lookup, Load and Store workflow for one KV Pool participant."""

    def __init__(
        self,
        topology: KVPoolTopology,
        reachable_region_selection: ReachableRegionSelection,
        semantic_chunk_projection: SemanticChunkProjection,
        checkpoint_chunk_projection: CheckpointChunkProjection,
        remote_object_projection: RemoteObjectProjection,
        local_block_resolution: LocalBlockResolution,
        checkpoint_block_resolution: CheckpointBlockResolution,
        store_ownership_selection: StoreOwnershipSelection,
        transfer_region_projection: TransferRegionProjection,
        region_partition: RegionPartition,
        store_admission: StoreAdmission,
        backend_name: str,
        schedule: KVPoolSchedule,
    ) -> None:
        self._topology = topology
        self._groups_by_id = {group.group_id: group for group in topology.groups}
        self._reachable_region_selection = reachable_region_selection
        self._semantic_chunk_projection = semantic_chunk_projection
        self._checkpoint_chunk_projection = checkpoint_chunk_projection
        self._remote_object_projection = remote_object_projection
        self._local_block_resolution = local_block_resolution
        self._checkpoint_block_resolution = checkpoint_block_resolution
        self._store_ownership_selection = store_ownership_selection
        self._transfer_region_projection = transfer_region_projection
        self._region_partition = region_partition
        self._store_admission = store_admission
        self.backend_name = backend_name
        self.schedule = schedule

    @property
    def topology(self) -> KVPoolTopology:
        return self._topology

    def bind_memory(self, memory_geometry: KVMemoryGeometry) -> None:
        self._transfer_region_projection.bind_memory(memory_geometry)
        self._region_partition.bind_memory(memory_geometry)

    def lookup(self, request: LookupRequest, observe_readability: ReadabilityObserver) -> LookupResult:
        if request.transfer_group_ids != self._reachable_region_selection.group_ids:
            raise ValueError(
                f"Lookup groups {request.transfer_group_ids} do not match configured groups "
                f"{self._reachable_region_selection.group_ids}"
            )
        selection = self._reachable_region_selection.select_for_lookup(request.block_hashes, request.query_range)
        chunks = self._semantic_chunk_projection.project_lookup(selection)
        object_keys = project_remote_identities(chunks, self._groups_by_id, self._topology.transfer_group_ids)
        availability = []
        remote_objects = self._remote_object_projection.project_lookup(object_keys)
        for key_batch, object_batch in zip(object_keys, remote_objects, strict=True):
            try:
                observations = observe_readability(object_batch)
            except Exception as error:
                logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                return LookupResult(0)
            availability.append(_reduce_remote_availability(key_batch, observations))
        reachable_prefix = self._reachable_region_selection.resolve_available_end(selection, availability)
        return LookupResult(reachable_prefix.end_token, reachable_prefix.tail_key_boundaries)

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

    def admit_store_transfers(
        self, transfers: list[StoreTransfer], observe_presence: ObjectPresenceObserver
    ) -> list[StoreTransfer]:
        return self._store_admission.select(transfers, observe_presence)

    def _build_load_transfer(self, command: LoadCommand) -> LoadTransfer:
        selection = self._reachable_region_selection.select_for_load(command.block_hashes, command.load_range)
        chunks = self._semantic_chunk_projection.project_load(selection, command.tail_key_boundaries)
        object_keys = project_remote_identities(chunks, self._groups_by_id, self._topology.transfer_group_ids)
        block_assignments = self._local_block_resolution.resolve(chunks, command.block_ids_by_group)
        transfer_regions = tuple(self._transfer_region_projection.project(batch) for batch in block_assignments)
        remote_objects = tuple(
            self._remote_object_projection.project_transfer(regions, keys)
            for regions, keys in zip(transfer_regions, object_keys, strict=True)
        )
        batches = tuple(
            _bind_transfer_regions(regions, objects)
            for regions, objects in zip(transfer_regions, remote_objects, strict=True)
        )
        return LoadTransfer(command.request_id, _order_load_traversal(batches, self._topology.tp_rank))

    def _build_store_transfer(self, command: StoreCommand) -> StoreTransfer:
        if isinstance(command, CheckpointStoreCommand):
            chunks = self._checkpoint_chunk_projection.project(command)
            block_assignments = self._checkpoint_block_resolution.resolve(chunks, command)
        else:
            selection = self._reachable_region_selection.select_for_store(
                command.block_hashes,
                command.store_range,
                command.num_prompt_tokens,
            )
            chunks = self._semantic_chunk_projection.project(selection)
            block_assignments = self._local_block_resolution.resolve(chunks, command.block_ids_by_group)

        object_keys = project_remote_identities(chunks, self._groups_by_id, self._topology.transfer_group_ids)
        owned_assignments = tuple(self._store_ownership_selection.select_batch(batch) for batch in block_assignments)
        transfer_regions = tuple(self._transfer_region_projection.project(batch) for batch in owned_assignments)
        transfer_regions = self._region_partition.project(transfer_regions)
        remote_objects = tuple(
            self._remote_object_projection.project_transfer(regions, keys)
            for regions, keys in zip(transfer_regions, object_keys, strict=True)
        )
        batches = tuple(
            _bind_transfer_regions(regions, objects)
            for regions, objects in zip(transfer_regions, remote_objects, strict=True)
        )
        return StoreTransfer(command.request_id, batches, command.store_job_id)

    def execute_store(
        self,
        transfer: StoreTransfer,
        wait_for_source: Callable[[], None],
        store_bindings: StoreBindings,
    ) -> StoreCompletion:
        if not any(batch.bindings for batch in transfer.batches):
            return StoreCompletion(transfer.request_id, StoreEvidence((), True, True), transfer.store_job_id)
        try:
            wait_for_source()
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error), transfer.store_job_id)

        try:
            evidence = store_bindings(transfer.batches)
        except Exception as error:
            evidence = StoreEvidence((), False, False, error)
        return StoreCompletion(transfer.request_id, evidence, transfer.store_job_id)

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


def _reduce_remote_availability(
    object_keys: RemoteObjectKeyBatch,
    observations: tuple[RemoteObjectObservation, ...],
) -> GroupAvailability:
    available_by_chunk = {item.chunk: True for item in object_keys.keys}
    for observation in observations:
        available_by_chunk[observation.remote_object.chunk] &= observation.readable
    return GroupAvailability(
        object_keys.group_id,
        tuple(
            ChunkAvailability(item.chunk.token_range, item.chunk.content_hash, available_by_chunk[item.chunk])
            for item in object_keys.keys
        ),
    )


def _bind_transfer_regions(regions: TransferRegionBatch, remote_objects: RemoteObjectBatch) -> BindingBatch:
    if regions.group_id != remote_objects.group_id:
        raise ValueError(
            f"Transfer region group {regions.group_id} does not match remote object group {remote_objects.group_id}"
        )
    if len(regions.regions) != len(remote_objects.remote_objects):
        raise ValueError("Transfer regions and remote objects must have the same length")
    bindings = []
    for transfer_region, remote_object in zip(regions.regions, remote_objects.remote_objects, strict=True):
        if transfer_region.region.chunk != remote_object.chunk:
            raise ValueError("Transfer region and remote object must belong to the same semantic chunk")
        if transfer_region.coordinate != remote_object.coordinate:
            raise ValueError("Transfer region and remote object must use the same physical coordinate")
        bindings.append(
            KVBinding(
                transfer_region.region,
                remote_object,
                transfer_region.remote_object_size,
                transfer_region.remote_offsets,
                transfer_region.memory,
            )
        )
    return BindingBatch(regions.group_id, tuple(bindings))


def _order_load_traversal(batches: tuple[BindingBatch, ...], rank_offset: int) -> tuple[KVBinding, ...]:
    traversal = tuple(binding for batch in batches for binding in batch.bindings)
    if not traversal:
        return ()
    offset = rank_offset % len(traversal)
    return traversal[offset:] + traversal[:offset]
