"""Compile semantic selection once, then bind physical execution once."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from vllm.logger import logger

from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import CheckpointStoreCommand, LoadCommand, StoreCommand
from .lowering import BoundTransferPlans, bind_transfer_rows, enumerate_transfer_work, lower_transfer_plans
from .spec.schedule import KVPoolSchedule
from .spec.topology import KVPoolTopology
from .stages.admission import StoreAdmission
from .stages.block import CheckpointBlockResolution, LocalBlockResolution
from .stages.chunk import CheckpointChunkProjection, SemanticChunkProjection
from .stages.ownership import StoreOwnershipSelection
from .stages.reachability import ReachableRegionSelection
from .stages.remote import RemoteObjectProjection, project_remote_identities
from .values.evidence import (
    ChunkAvailability,
    GroupAvailability,
    LoadCompletion,
    LoadFailure,
    RemoteObjectObservation,
    StoreCompletion,
    StoreEvidence,
    TransferEvidence,
)
from .values.representation import KVMemoryGeometry, RemoteKVObject, RemoteObjectKeyBatch
from .values.selection import LoadTransfer, StoreTransfer, TransferWork

ReadabilityObserver = Callable[[tuple[RemoteKVObject, ...]], tuple[RemoteObjectObservation, ...]]
LoadWork = Callable[[TransferWork], tuple[TransferEvidence, ...]]
StoreWork = Callable[[TransferWork], StoreEvidence]


class KVPoolProgram:
    """Unbound semantic program compiled without process-local memory addresses."""

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
        self._store_admission = store_admission
        self.backend_name = backend_name
        self.schedule = schedule

    @property
    def topology(self) -> KVPoolTopology:
        return self._topology

    def bind_memory(self, memory_geometry: KVMemoryGeometry, block_capacity: int) -> BoundKVPoolProgram:
        """Return a proof-bearing program; the unbound program remains immutable."""

        plans = lower_transfer_plans(self._topology, self.schedule, memory_geometry, block_capacity)
        return BoundKVPoolProgram(self, plans)

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
                observations = observe_readability(object_batch.remote_objects)
            except Exception as error:
                logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                return LookupResult(0)
            availability.append(_reduce_remote_availability(key_batch, observations))
        reachable_prefix = self._reachable_region_selection.resolve_available_end(selection, availability)
        return LookupResult(reachable_prefix.end_token, reachable_prefix.tail_key_boundaries)


class BoundKVPoolProgram:
    """Executable program whose topology, memory and Backend enumeration are fixed."""

    def __init__(self, semantic: KVPoolProgram, plans: BoundTransferPlans) -> None:
        self._semantic = semantic
        self._plans = plans
        self._load_plans = {plan.group_id: plan for plan in plans.load}
        self._store_plans = {plan.group_id: plan for plan in plans.store}

    @property
    def topology(self) -> KVPoolTopology:
        return self._semantic.topology

    @property
    def schedule(self) -> KVPoolSchedule:
        return self._semantic.schedule

    @property
    def backend_name(self) -> str:
        return self._semantic.backend_name

    @property
    def plans(self) -> BoundTransferPlans:
        return self._plans

    def lookup(self, request: LookupRequest, observe_readability: ReadabilityObserver) -> LookupResult:
        return self._semantic.lookup(request, observe_readability)

    def select_load_transfers(self, commands: tuple[LoadCommand, ...]) -> list[LoadTransfer]:
        return [self._build_load_transfer(command) for command in commands]

    def select_store_transfers(self, commands: tuple[StoreCommand, ...]) -> list[StoreTransfer]:
        return [self._build_store_transfer(command) for command in commands]

    def _build_load_transfer(self, command: LoadCommand) -> LoadTransfer:
        semantic = self._semantic
        selection = semantic._reachable_region_selection.select_for_load(command.block_hashes, command.load_range)
        chunks = semantic._semantic_chunk_projection.project_load(selection, command.tail_key_boundaries)
        assignments = semantic._local_block_resolution.resolve(chunks, command.block_ids_by_group)
        rows = tuple(bind_transfer_rows(self._load_plans[batch.group_id], batch) for batch in assignments)
        return LoadTransfer(command.request_id, rows, enumerate_transfer_work(rows, self.topology.tp_rank))

    def _build_store_transfer(self, command: StoreCommand) -> StoreTransfer:
        semantic = self._semantic
        if isinstance(command, CheckpointStoreCommand):
            chunks = semantic._checkpoint_chunk_projection.project(command)
            assignments = semantic._checkpoint_block_resolution.resolve(chunks, command)
        else:
            selection = semantic._reachable_region_selection.select_for_store(
                command.block_hashes,
                command.store_range,
                command.num_prompt_tokens,
            )
            chunks = semantic._semantic_chunk_projection.project(selection)
            assignments = semantic._local_block_resolution.resolve(chunks, command.block_ids_by_group)
        owned = semantic._store_ownership_selection.select(assignments)
        rows = tuple(bind_transfer_rows(self._store_plans[batch.group_id], batch) for batch in owned)
        return StoreTransfer(command.request_id, rows, enumerate_transfer_work(rows), command.store_job_id)

    def execute_load(self, transfer: LoadTransfer, load_work: LoadWork) -> LoadCompletion:
        evidence = tuple(item for work in transfer.work for item in load_work(work))
        return LoadCompletion(transfer.request_id, evidence)

    def reduce_load_completion(self, completion: LoadCompletion) -> LoadFailure:
        failed_block_ids = frozenset(
            evidence.source.block_id for evidence in completion.transfer_evidence if evidence.result_code != 0
        )
        if not failed_block_ids:
            return LoadFailure()
        if len(self.topology.transfer_group_ids) > 1:
            return LoadFailure(failed_request_ids=frozenset({completion.request_id}))
        return LoadFailure(failed_block_ids=failed_block_ids)

    def store_admission_targets(self, transfers: Sequence[StoreTransfer]) -> tuple[RemoteKVObject, ...]:
        return self._semantic._store_admission.observation_targets(transfers)

    def admit_store_transfers(
        self,
        transfers: list[StoreTransfer],
        observations: tuple[RemoteObjectObservation, ...],
    ) -> list[StoreTransfer]:
        return self._semantic._store_admission.select(transfers, observations)

    def execute_store(
        self,
        transfer: StoreTransfer,
        wait_for_source: Callable[[], None],
        store_work: StoreWork,
    ) -> StoreCompletion:
        if not any(not work.empty for work in transfer.work):
            return StoreCompletion(transfer.request_id, StoreEvidence((), True, True), transfer.store_job_id)
        try:
            wait_for_source()
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error), transfer.store_job_id)

        evidence_parts = []
        for work in transfer.work:
            if work.empty:
                continue
            try:
                evidence_parts.append(store_work(work))
            except Exception as error:
                evidence_parts.append(StoreEvidence((), False, False, error))
                break
        evidence = _merge_store_evidence(evidence_parts)
        return StoreCompletion(transfer.request_id, evidence, transfer.store_job_id)

    @staticmethod
    def validate_store_completion(completion: StoreCompletion) -> None:
        evidence = completion.evidence
        if evidence.error is not None:
            raise RuntimeError(f"Store failed for request {completion.request_id}") from evidence.error
        if not evidence.succeeded:
            raise RuntimeError(
                f"Store success was not confirmed for request {completion.request_id}; "
                f"result codes {[item.result_code for item in evidence.transfer_evidence]}"
            )
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


def _merge_store_evidence(parts: list[StoreEvidence]) -> StoreEvidence:
    if not parts:
        return StoreEvidence((), True, True)
    return StoreEvidence(
        tuple(item for part in parts for item in part.transfer_evidence),
        all(part.succeeded for part in parts),
        all(part.source_release_confirmed for part in parts),
        next((part.error for part in parts if part.error is not None), None),
    )
