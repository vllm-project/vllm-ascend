"""Drive the fixed KV Pool graph with process-owned execution resources."""

from __future__ import annotations

import torch

from ..graph.evaluation import KVPoolStepEvaluation
from ..kv_pool import KVPoolGraph, LoadResult
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import KVTransferStep
from .io import BackendIO
from .resources import KVResources
from .timeline import LoadCompletion, LoadTimeline, LoadTransfer, StoreCompletion, StoreTimeline, StoreTransfer


class KVPoolRuntime:
    """Advance KV Pool evaluations using Backend, memory and dispatch resources."""

    def __init__(
        self,
        graph: KVPoolGraph,
        resources: KVResources,
        backend_io: BackendIO,
        load_timeline: LoadTimeline,
        store_timeline: StoreTimeline | None,
    ) -> None:
        self._graph = graph
        self._resources = resources
        self._backend_io = backend_io
        self._load_timeline = load_timeline
        self._store_timeline = store_timeline
        self._load_evaluations: dict[str, KVPoolStepEvaluation] = {}
        self._pending_store_evaluation: KVPoolStepEvaluation | None = None
        self._store_error: Exception | None = None
        self._load_timeline.attach_operation(self._execute_load)
        if self._store_timeline is not None:
            self._store_timeline.attach_operation(self._execute_store)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            self._resources.register_kv_caches(kv_caches)
            self._graph.compile_memory_mapping()
            if self._store_timeline is not None:
                self._store_timeline.start()
            self._load_timeline.start()
        except BaseException:
            self.close()
            raise

    def lookup(self, request: LookupRequest) -> LookupResult:
        return self._graph.lookup(request, self._backend_io.observe_readability)

    def begin_step(self, step: KVTransferStep) -> KVPoolStepEvaluation:
        return self._graph.begin_step(step)

    def start_load(self, evaluation: KVPoolStepEvaluation) -> None:
        transfers = self._graph.build_load_transfers(evaluation)
        if self._load_timeline.is_deferred:
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
        if completions and evaluation.failed_request_ids:
            raise RuntimeError(f"Hybrid KV Load failed for requests: {sorted(evaluation.failed_request_ids)}")

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

    def finish_step(self, evaluation: KVPoolStepEvaluation) -> None:
        self._raise_store_error()
        if self._store_timeline is None or not evaluation.step.store.commands:
            return
        if self._pending_store_evaluation is not None:
            raise RuntimeError("Previous Store evaluation has not reached its fence")
        try:
            source_ready_event = torch.npu.Event()
            source_ready_event.record()
            transfers = self._graph.build_store_transfers(evaluation, source_ready_event)
            evaluation.pending_store = self._store_timeline.submit(transfers)
            self._pending_store_evaluation = evaluation
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

    def _execute_store(self, transfer: StoreTransfer) -> StoreCompletion:
        return self._graph.execute_store(transfer, self._backend_io.observe_presence, self._backend_io.store)

    def _raise_store_error(self) -> None:
        if self._store_error is not None:
            raise RuntimeError("KVPoolRuntime cannot continue after a previous Store failure") from self._store_error
