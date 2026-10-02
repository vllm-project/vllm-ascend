"""Drive the compiled KV Pool program with process-owned runtime resources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Concatenate, ParamSpec, TypeVar

import torch

from ...attention_fence import reset_attention_compute_start_gate
from ..backend import LayerwiseAccessKind
from ..program.program import BoundKVPoolProgram, KVPoolProgram
from ..program.values.evidence import (
    LoadCompletion,
    StoreCompletion,
    StoreEvidence,
)
from ..program.values.selection import LoadTransfer, StoreTransfer
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import KVTransferStep
from .backend import BackendIO, GVABackendIO, KeyRangeBackendIO
from .resources import KVPoolResources
from .result import LoadResult
from .timeline import StoreBatch
from .timeline.composition import KVPoolTimelineRuntime

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


@dataclass(slots=True)
class _KVPoolStepContext:
    """Adapt one upstream hook batch without owning cross-step executions."""

    step: KVTransferStep
    failed_request_ids: set[str] = field(default_factory=set)
    failed_block_ids: set[int] = field(default_factory=set)
    store_transfers: list[StoreTransfer] = field(default_factory=list)
    store_submitted: bool = False


class KVPoolRuntime:
    """Drive KV Pool program invocations with Backend, memory and dispatch resources."""

    def __init__(
        self,
        program: KVPoolProgram,
        resources: KVPoolResources,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        timeline = KVPoolTimelineRuntime(program.schedule, program.topology)
        backend_io: BackendIO
        layerwise_backend: GVABackendIO | KeyRangeBackendIO | None = None
        if program.schedule.requires_layerwise_backend:
            access_kind = resources.backend_spec.layerwise_access
            if access_kind is None:
                raise ValueError("Layerwise timeline requires a session Backend")
            backend_io_type: type[GVABackendIO] | type[KeyRangeBackendIO] = (
                GVABackendIO if access_kind is LayerwiseAccessKind.GVA else KeyRangeBackendIO
            )
            layerwise_backend = backend_io_type(resources.backend, resources.backend_spec)
            backend_io = layerwise_backend
        else:
            backend_io = BackendIO(resources.backend, resources.backend_spec)
        self._unbound_program = program
        self._program: BoundKVPoolProgram | None = None
        self._resources = resources
        self._backend_io = backend_io
        self._timeline = timeline
        self._active_step: _KVPoolStepContext | None = None
        # One outstanding Load per request; its Timeline retains the transfer, not this step context.
        self._pending_load_request_ids: set[str] = set()
        self._pending_store_batch: StoreBatch | None = None
        self._released_store_job_ids: set[int] = set()
        self._store_error: Exception | None = None
        self._timeline.bind_resources(
            thread_initializer=backend_io.initialize_thread,
            source_ready_event_factory=source_ready_event_factory or (lambda: torch.npu.Event()),
            load_operation=self._execute_load,
            store_operation=self._execute_store,
            store_transfer_operation=self._execute_store_transfer,
            store_admission=self._admit_store_transfers,
            layerwise_backend=layerwise_backend,
            start_gate_factory=start_gate_factory,
        )

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            memory_geometry = self._resources.bind_kv_caches(kv_caches)
            self._program = self._unbound_program.bind_memory(memory_geometry, self._resources.num_blocks)
            self._timeline.start()
        except BaseException:
            self.close()
            raise

    def lookup(self, request: LookupRequest) -> LookupResult:
        return self._bound_program.lookup(request, self._backend_io.observe_objects)

    def begin_step(self, step: KVTransferStep) -> None:
        self._raise_store_error()
        if self._active_step is not None:
            raise RuntimeError("Previous KV Pool step has not ended")
        context = _KVPoolStepContext(step)
        if self._timeline.store_enabled and step.store.commands:
            try:
                context.store_transfers = self._bound_program.select_store_transfers(step.store.commands)
                self._timeline.prepare_store(context.store_transfers)
                if step.store.all_sources_ready:
                    self._submit_store(context)
            except Exception as error:
                self._store_error = error
                raise
        self._active_step = context

    def end_step(self) -> None:
        if self._active_step is None:
            raise RuntimeError("KV Pool step has not begun")
        self._active_step = None

    @staticmethod
    def _with_active_step(
        method: Callable[Concatenate[KVPoolRuntime, _KVPoolStepContext, _Parameters], _Result],
    ) -> Callable[Concatenate[KVPoolRuntime, _Parameters], _Result]:
        """Inject the upstream step context into one runtime operation."""

        @wraps(method)
        def guarded(self: KVPoolRuntime, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            if self._active_step is None:
                raise RuntimeError("KV Pool step has not begun")
            return method(self, self._active_step, *args, **kwargs)

        return guarded

    @_with_active_step
    def start_load(self, context: _KVPoolStepContext) -> None:
        transfers = self._bound_program.select_load_transfers(context.step.load.commands)
        if self._timeline.collects_load_completions:
            for transfer in transfers:
                if transfer.request_id in self._pending_load_request_ids:
                    raise RuntimeError(f"Request {transfer.request_id} already has a pending asynchronous Load")
            self._pending_load_request_ids.update(transfer.request_id for transfer in transfers)
        try:
            completions = self._timeline.submit_load(transfers)
        except BaseException:
            self._pending_load_request_ids.difference_update(transfer.request_id for transfer in transfers)
            raise
        for completion in completions:
            self._record_load_completion(context, completion)
        self._raise_load_error(context)

    @_with_active_step
    def collect_load_result(self, context: _KVPoolStepContext) -> LoadResult:
        completions = self._timeline.collect_load()
        for completion in completions:
            if completion.request_id not in self._pending_load_request_ids:
                raise RuntimeError(f"Load completion has no pending execution: {completion.request_id}")
            self._pending_load_request_ids.remove(completion.request_id)
            self._record_load_completion(context, completion)
        result = LoadResult(
            frozenset(completion.request_id for completion in completions),
            frozenset(context.failed_request_ids),
            frozenset(context.failed_block_ids),
        )
        context.failed_request_ids.clear()
        context.failed_block_ids.clear()
        return result

    @_with_active_step
    def wait_for_layer_load(self, context: _KVPoolStepContext, layer_name: str) -> None:
        completions = self._timeline.wait_for_load_layer(layer_name)
        for completion in completions:
            self._record_load_completion(context, completion)
        self._raise_load_error(context)

    @_with_active_step
    def save_layer(self, _context: _KVPoolStepContext, layer_name: str) -> None:
        self._raise_store_error()
        try:
            self._timeline.submit_store_layer(layer_name)
        except Exception as error:
            self._store_error = error
            raise

    @_with_active_step
    def finish_step(self, context: _KVPoolStepContext) -> None:
        self._raise_store_error()
        if not self._timeline.store_enabled or not context.step.store.commands or context.store_submitted:
            return
        try:
            self._submit_store(context)
            if self._timeline.fences_store_on_finish:
                self.fence_previous_store()
        except Exception as error:
            self._store_error = error
            raise

    def _submit_store(self, context: _KVPoolStepContext) -> None:
        if self._pending_store_batch is not None:
            raise RuntimeError("Previous Store invocation has not reached its fence")
        self._pending_store_batch = self._timeline.finish_store(context.store_transfers)
        context.store_submitted = True

    def fence_previous_store(self) -> tuple[StoreCompletion, ...]:
        self._raise_store_error()
        pending_store = self._pending_store_batch
        if not self._timeline.store_enabled or pending_store is None:
            return ()
        try:
            completions = self._timeline.wait_store(pending_store)
        except Exception as error:
            self._store_error = error
            raise
        for completion in completions:
            if completion.evidence.source_release_confirmed and completion.store_job_id is not None:
                self._released_store_job_ids.add(completion.store_job_id)
        if all(completion.evidence.source_release_confirmed for completion in completions):
            self._pending_store_batch = None
        for completion in completions:
            try:
                self._bound_program.validate_store_completion(completion)
            except Exception as error:
                self._store_error = error
                raise
        return completions

    def take_released_store_job_ids(self) -> set[int]:
        """Drain jobs whose Backend call can no longer read Worker memory."""

        released = self._released_store_job_ids
        self._released_store_job_ids = set()
        return released

    def close(self) -> None:
        close_error: BaseException | None = None
        # An absent batch proves no pending Store only after the Timeline has handed off prepared work.
        store_handoff_complete = False
        try:
            if self._pending_store_batch is None:
                self._pending_store_batch = self._timeline.prepare_store_close()
            store_handoff_complete = True
            self.fence_previous_store()
        except BaseException as error:
            close_error = error
        try:
            self._timeline.close()
        except BaseException as error:
            if close_error is None:
                close_error = error
        finally:
            if store_handoff_complete and self._pending_store_batch is None:
                self._resources.close()
        if close_error is not None:
            raise close_error

    def _execute_load(self, transfer: LoadTransfer) -> LoadCompletion:
        return self._bound_program.execute_load(transfer, self._backend_io.load)

    def _admit_store_transfers(self, transfers: list[StoreTransfer]) -> list[StoreTransfer]:
        return self._select_admitted_store_transfers(transfers)

    def _execute_store(self, transfer: StoreTransfer, source_ready_event: Any) -> StoreCompletion:
        try:
            admitted_transfers = self._select_admitted_store_transfers([transfer])
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error), transfer.store_job_id)
        return self._execute_store_transfer(admitted_transfers[0], source_ready_event)

    def _select_admitted_store_transfers(self, transfers: list[StoreTransfer]) -> list[StoreTransfer]:
        targets = self._bound_program.store_admission_targets(transfers)
        observations = self._backend_io.observe_objects(targets)
        return self._bound_program.admit_store_transfers(transfers, observations)

    def _execute_store_transfer(self, transfer: StoreTransfer, source_ready_event: Any) -> StoreCompletion:
        return self._bound_program.execute_store(transfer, source_ready_event.synchronize, self._backend_io.store)

    def _raise_store_error(self) -> None:
        if self._store_error is not None:
            raise RuntimeError("KVPoolRuntime cannot continue after a previous Store failure") from self._store_error

    def _record_load_completion(self, context: _KVPoolStepContext, completion: LoadCompletion) -> None:
        failure = self._bound_program.reduce_load_completion(completion)
        context.failed_request_ids.update(failure.failed_request_ids)
        context.failed_block_ids.update(failure.failed_block_ids)

    def _raise_load_error(self, context: _KVPoolStepContext) -> None:
        if not context.failed_request_ids:
            return
        self._timeline.abort_load()
        raise RuntimeError(f"Hybrid KV Load failed for requests: {sorted(context.failed_request_ids)}")

    @property
    def _bound_program(self) -> BoundKVPoolProgram:
        if self._program is None:
            raise RuntimeError("KV Pool program is unavailable before cache registration")
        return self._program
