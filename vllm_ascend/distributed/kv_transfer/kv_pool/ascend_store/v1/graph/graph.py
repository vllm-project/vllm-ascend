"""Define the fixed KV Pool dataflow over request-local values."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from vllm.logger import logger

from ..execution.io import BindingEvidence, MissingFilter, RemoteObjectObservation, StoreEvidence
from ..execution.timeline import LoadCompletion, LoadTransfer, StoreCompletion, StoreTransfer
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import KVTransferStep, LoadCommand, StoreCommand
from .elements import BindingBatch, KVBinding, RemoteObjectBatch
from .evaluation import KVPoolStepEvaluation, LoadResult
from .projection import ConsumerProjection, KVProjection
from .reachability import ChunkAvailability, GroupAvailability, KVReachability
from .topology import KVTopology

ReadabilityObserver = Callable[[RemoteObjectBatch], tuple[RemoteObjectObservation, ...]]
PresenceObserver = Callable[[list[str]], tuple[int, ...]]
LoadBindings = Callable[[tuple[KVBinding, ...]], tuple[BindingEvidence, ...]]
StoreBindings = Callable[[tuple[BindingBatch, ...]], StoreEvidence]


class KVPoolGraph:
    """Expose the fixed Lookup, Load and Store dataflow for one KV Pool participant."""

    def __init__(
        self,
        topology: KVTopology,
        reachability: KVReachability,
        projection: KVProjection,
        consumer_projection: ConsumerProjection,
        missing_filter: MissingFilter,
    ) -> None:
        self._topology = topology
        self._reachability = reachability
        self._projection = projection
        self._consumer_projection = consumer_projection
        self._missing_filter = missing_filter

    @staticmethod
    def begin_step(step: KVTransferStep) -> KVPoolStepEvaluation:
        return KVPoolStepEvaluation(step)

    def compile_memory_mapping(self) -> None:
        self._projection.compile_memory_mapping()
        self._consumer_projection.compile_memory_mapping()

    def lookup(self, request: LookupRequest, observe_readability: ReadabilityObserver) -> LookupResult:
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
                observations = observe_readability(batch)
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

    def build_load_transfers(self, evaluation: KVPoolStepEvaluation) -> list[LoadTransfer]:
        return [self._build_load_transfer(command) for command in evaluation.step.load.commands]

    def execute_load(self, transfer: LoadTransfer, load_bindings: LoadBindings) -> LoadCompletion:
        return LoadCompletion(transfer.request_id, load_bindings(transfer.traversal))

    def record_load_completion(self, evaluation: KVPoolStepEvaluation, completion: LoadCompletion) -> None:
        failed_block_ids = {
            evidence.binding.local_slice.block_id
            for evidence in completion.binding_evidence
            if evidence.result_code != 0
        }
        if not failed_block_ids:
            return
        if len(self._topology.transfer_group_ids) > 1:
            evaluation.failed_request_ids.add(completion.request_id)
        else:
            evaluation.failed_block_ids.update(failed_block_ids)

    @staticmethod
    def consume_load_result(
        evaluations: Iterable[KVPoolStepEvaluation],
        completed_request_ids: Iterable[str],
    ) -> LoadResult:
        evaluations_by_identity = {id(evaluation): evaluation for evaluation in evaluations}
        failed_request_ids = frozenset(
            request_id
            for evaluation in evaluations_by_identity.values()
            for request_id in evaluation.failed_request_ids
        )
        failed_block_ids = frozenset(
            block_id for evaluation in evaluations_by_identity.values() for block_id in evaluation.failed_block_ids
        )
        result = LoadResult(
            frozenset(completed_request_ids),
            failed_request_ids,
            failed_block_ids,
        )
        for evaluation in evaluations_by_identity.values():
            evaluation.failed_request_ids.clear()
            evaluation.failed_block_ids.clear()
        return result

    def build_store_transfers(
        self,
        evaluation: KVPoolStepEvaluation,
        source_ready_event: Any,
    ) -> list[StoreTransfer]:
        return [self._build_store_transfer(command, source_ready_event) for command in evaluation.step.store.commands]

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

    def execute_store(
        self,
        transfer: StoreTransfer,
        observe_presence: PresenceObserver,
        store_bindings: StoreBindings,
    ) -> StoreCompletion:
        try:
            batches = self._missing_filter.select_missing(transfer.batches, observe_presence)
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error))
        if not any(batch.bindings for batch in batches):
            return StoreCompletion(transfer.request_id, StoreEvidence((), True, True))
        try:
            transfer.source_ready_event.synchronize()
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error))

        try:
            evidence = store_bindings(batches)
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
