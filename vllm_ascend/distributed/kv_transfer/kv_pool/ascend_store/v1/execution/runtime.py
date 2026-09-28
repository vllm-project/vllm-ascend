"""Drive the fixed KV Pool graph with process-owned execution resources."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import logger

from ..graph.evaluation import KVPoolStepEvaluation, LoadResult
from ..graph.graph import KVPoolGraph
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import KVTransferStep
from .io import BackendIO, StoreEvidence
from .resources import KVPoolResources
from .timeline import (
    LoadCompletion,
    LoadTimelineProtocol,
    LoadTransfer,
    StoreCompletion,
    StoreTimelineProtocol,
    StoreTransfer,
)
from .timeline.layerwise import LayerwiseLoadTimelineProtocol, LayerwiseStoreTimelineProtocol


class KVPoolRuntime:
    """Advance KV Pool evaluations using Backend, memory and dispatch resources."""

    def __init__(
        self,
        graph: KVPoolGraph,
        resources: KVPoolResources,
        backend_io: BackendIO,
        load_timeline: LoadTimelineProtocol,
        store_timeline: StoreTimelineProtocol | LayerwiseStoreTimelineProtocol | None,
    ) -> None:
        self._graph = graph
        self._resources = resources
        self._backend_io = backend_io
        self._load_timeline = load_timeline
        self._store_timeline = store_timeline
        self._layer_load_timeline: LayerwiseLoadTimelineProtocol | None = None
        self._layer_store_timeline: LayerwiseStoreTimelineProtocol | None = None
        self._whole_step_store_timeline: StoreTimelineProtocol | None = None
        if isinstance(load_timeline, LayerwiseLoadTimelineProtocol):
            self._layer_load_timeline = load_timeline
        if isinstance(store_timeline, LayerwiseStoreTimelineProtocol):
            self._layer_store_timeline = store_timeline
        else:
            self._whole_step_store_timeline = store_timeline
        self._load_evaluations: dict[str, KVPoolStepEvaluation] = {}
        self._pending_store_evaluation: KVPoolStepEvaluation | None = None
        self._store_error: Exception | None = None
        self._load_timeline.attach_operation(self._execute_load)
        if self._layer_store_timeline is not None:
            self._layer_store_timeline.attach_binding_filter(self._filter_store_bindings)
            self._layer_store_timeline.attach_operation(self._execute_selected_store)
        elif self._whole_step_store_timeline is not None:
            self._whole_step_store_timeline.attach_operation(self._execute_store)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            memory_geometry = self._resources.register_kv_caches(kv_caches)
            self._graph.compile_memory_mapping(memory_geometry)
            if self._store_timeline is not None:
                self._store_timeline.start()
            self._load_timeline.start()
        except BaseException:
            self.close()
            raise

    def lookup(self, request: LookupRequest) -> LookupResult:
        return self._graph.lookup(request, self._backend_io.observe_readability)

    def begin_step(self, step: KVTransferStep) -> KVPoolStepEvaluation:
        self._raise_store_error()
        evaluation = self._graph.begin_step(step)
        if self._store_timeline is not None and step.store.commands:
            try:
                evaluation.store_transfers = self._graph.build_store_transfers(evaluation)
                if self._layer_store_timeline is not None:
                    self._layer_store_timeline.prepare(evaluation.store_transfers)
            except Exception as error:
                self._store_error = error
                raise
        return evaluation

    def start_load(self, evaluation: KVPoolStepEvaluation) -> None:
        transfers = self._graph.build_load_transfers(evaluation)
        if self._load_timeline.collects_completions:
            for transfer in transfers:
                if transfer.request_id in self._load_evaluations:
                    raise RuntimeError(f"Request {transfer.request_id} already has a pending asynchronous Load")
                self._load_evaluations[transfer.request_id] = evaluation
        try:
            completions = tuple(self._load_timeline.submit(transfers))
        except BaseException:
            for transfer in transfers:
                if self._load_evaluations.get(transfer.request_id) is evaluation:
                    del self._load_evaluations[transfer.request_id]
            raise
        for completion in completions:
            self._graph.record_load_completion(evaluation, completion)
        self._raise_load_error(evaluation)

    def collect_load_result(self, evaluation: KVPoolStepEvaluation) -> LoadResult:
        completions = tuple(self._load_timeline.collect())
        evaluations = [evaluation]
        for completion in completions:
            owner = self._load_evaluations.pop(completion.request_id, None)
            if owner is None:
                raise RuntimeError(f"Load completion has no owning KV Pool evaluation: {completion.request_id}")
            self._graph.record_load_completion(owner, completion)
            evaluations.append(owner)
        return self._graph.consume_load_result(evaluations, (completion.request_id for completion in completions))

    def wait_for_layer_load(self, evaluation: KVPoolStepEvaluation, layer_name: str) -> None:
        if self._layer_load_timeline is None:
            return
        completions = tuple(self._layer_load_timeline.wait_for_layer(layer_name))
        for completion in completions:
            self._graph.record_load_completion(evaluation, completion)
        self._raise_load_error(evaluation)

    def save_layer(self, evaluation: KVPoolStepEvaluation, layer_name: str) -> None:
        self._raise_store_error()
        if self._layer_store_timeline is None:
            return
        try:
            source_ready_event = torch.npu.Event()
            source_ready_event.record()
            self._layer_store_timeline.submit_layer(layer_name, source_ready_event)
        except Exception as error:
            self._store_error = error
            raise

    def finish_step(self, evaluation: KVPoolStepEvaluation) -> None:
        self._raise_store_error()
        if self._store_timeline is None or not evaluation.step.store.commands:
            return
        if self._pending_store_evaluation is not None:
            raise RuntimeError("Previous Store evaluation has not reached its fence")
        try:
            if self._layer_store_timeline is not None:
                evaluation.pending_store = self._layer_store_timeline.finalize()
            else:
                assert self._whole_step_store_timeline is not None
                source_ready_event = torch.npu.Event()
                source_ready_event.record()
                evaluation.pending_store = self._whole_step_store_timeline.submit(
                    evaluation.store_transfers, source_ready_event
                )
            self._pending_store_evaluation = evaluation
            if self._layer_store_timeline is not None:
                self.fence_previous_store()
        except Exception as error:
            self._store_error = error
            raise

    def fence_previous_store(self) -> tuple[StoreCompletion, ...]:
        self._raise_store_error()
        evaluation = self._pending_store_evaluation
        if self._store_timeline is None or evaluation is None or evaluation.pending_store is None:
            return ()
        try:
            completions = self._store_timeline.wait(evaluation.pending_store)
        except Exception as error:
            self._store_error = error
            raise
        if all(completion.evidence.source_release_confirmed for completion in completions):
            evaluation.pending_store = None
            self._pending_store_evaluation = None
        for completion in completions:
            try:
                self._graph.validate_store_completion(completion)
            except Exception as error:
                self._store_error = error
                raise
        return completions

    def close(self) -> None:
        close_error: BaseException | None = None
        try:
            self.fence_previous_store()
        except BaseException as error:
            close_error = error
        if self._store_timeline is not None:
            try:
                self._store_timeline.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        try:
            self._load_timeline.close()
        finally:
            if self._pending_store_evaluation is None:
                self._resources.close()
        if close_error is not None:
            raise close_error

    def _execute_load(self, transfer: LoadTransfer) -> LoadCompletion:
        return self._graph.execute_load(transfer, self._backend_io.load)

    def _filter_store_bindings(self, transfers: list[StoreTransfer]) -> list[StoreTransfer]:
        try:
            return self._graph.filter_store_bindings(transfers, self._backend_io.observe_presence)
        except Exception as error:
            logger.error("Layerwise Store existence check failed; treating all keys as missing: %s", error)
            return transfers

    def _execute_store(self, transfer: StoreTransfer, source_ready_event: Any) -> StoreCompletion:
        try:
            filtered_transfers = self._graph.filter_store_bindings([transfer], self._backend_io.observe_presence)
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error))
        return self._execute_selected_store(filtered_transfers[0], source_ready_event)

    def _execute_selected_store(self, transfer: StoreTransfer, source_ready_event: Any) -> StoreCompletion:
        return self._graph.execute_store(transfer, source_ready_event, self._backend_io.store)

    def _raise_store_error(self) -> None:
        if self._store_error is not None:
            raise RuntimeError("KVPoolRuntime cannot continue after a previous Store failure") from self._store_error

    def _raise_load_error(self, evaluation: KVPoolStepEvaluation) -> None:
        if not evaluation.failed_request_ids:
            return
        self._load_timeline.abort()
        raise RuntimeError(f"Hybrid KV Load failed for requests: {sorted(evaluation.failed_request_ids)}")
