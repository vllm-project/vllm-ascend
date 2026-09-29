"""Drive the compiled KV Pool program with process-owned runtime resources."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import torch
from vllm.logger import logger

from ...attention_fence import reset_attention_compute_start_gate
from ..program.compiler import compile_kv_pool_program
from ..program.invocation import (
    KVPoolStepFrame,
    LoadCompletion,
    LoadTransfer,
    StoreCompletion,
    StoreEvidence,
    StoreTransfer,
)
from ..program.program import KVPoolProgram
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.transfer import KVTransferStep
from .io import BackendIO, LayerwiseBackendIO
from .resources import KVPoolResources
from .result import LoadResult
from .timeline import StoreBatch
from .timeline.composition import KVPoolTimelineRuntime

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


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
        backend_io_type = LayerwiseBackendIO if program.schedule.requires_layerwise_backend else BackendIO
        backend_io = backend_io_type(resources.backend, resources.backend_spec)
        self._program = program
        self._resources = resources
        self._backend_io = backend_io
        self._timeline = timeline
        self._active_frame: KVPoolStepFrame | None = None
        self._load_frames: dict[str, KVPoolStepFrame] = {}
        self._pending_store_batch: StoreBatch | None = None
        self._store_error: Exception | None = None
        self._timeline.bind_resources(
            thread_initializer=backend_io.initialize_thread,
            source_ready_event_factory=source_ready_event_factory or (lambda: torch.npu.Event()),
            load_operation=self._execute_load,
            store_operation=self._execute_store,
            store_transfer_operation=self._execute_store_transfer,
            store_filter=self._filter_store_bindings,
            layerwise_backend=backend_io if isinstance(backend_io, LayerwiseBackendIO) else None,
            start_gate_factory=start_gate_factory,
        )

    @classmethod
    def from_config(
        cls,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> KVPoolRuntime:
        """Compile one program and bind its process-owned execution resources."""

        program = compile_kv_pool_program(vllm_config, kv_cache_config)
        resources = KVPoolResources.bind(
            program.backend_name,
            vllm_config.parallel_config,
            vllm_config.kv_transfer_config.kv_connector_extra_config,
            program.topology.groups,
            kv_cache_config.num_blocks,
        )
        try:
            return cls(program, resources)
        except BaseException:
            resources.close()
            raise

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            memory_geometry = self._resources.bind_kv_caches(kv_caches)
            self._program.bind_memory(memory_geometry)
            self._timeline.start()
        except BaseException:
            self.close()
            raise

    def lookup(self, request: LookupRequest) -> LookupResult:
        return self._program.lookup(request, self._backend_io.observe_readability)

    def begin_step(self, step: KVTransferStep) -> None:
        self._raise_store_error()
        if self._active_frame is not None:
            raise RuntimeError("Previous KV Pool step has not ended")
        frame = KVPoolStepFrame(step)
        if self._timeline.store_enabled and step.store.commands:
            try:
                frame.store_transfers = self._program.build_store_transfers(frame)
                self._timeline.prepare_store(frame.store_transfers)
            except Exception as error:
                self._store_error = error
                raise
        self._active_frame = frame

    def end_step(self) -> None:
        if self._active_frame is None:
            raise RuntimeError("KV Pool step has not begun")
        self._active_frame = None

    @staticmethod
    def _with_active_frame(
        method: Callable[Concatenate[KVPoolRuntime, KVPoolStepFrame, _Parameters], _Result],
    ) -> Callable[Concatenate[KVPoolRuntime, _Parameters], _Result]:
        """Inject the active step frame into one runtime operation."""

        @wraps(method)
        def guarded(self: KVPoolRuntime, *args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            if self._active_frame is None:
                raise RuntimeError("KV Pool step has not begun")
            return method(self, self._active_frame, *args, **kwargs)

        return guarded

    @_with_active_frame
    def start_load(self, frame: KVPoolStepFrame) -> None:
        transfers = self._program.build_load_transfers(frame)
        if self._timeline.collects_load_completions:
            for transfer in transfers:
                if transfer.request_id in self._load_frames:
                    raise RuntimeError(f"Request {transfer.request_id} already has a pending asynchronous Load")
                self._load_frames[transfer.request_id] = frame
        try:
            completions = self._timeline.submit_load(transfers)
        except BaseException:
            for transfer in transfers:
                if self._load_frames.get(transfer.request_id) is frame:
                    del self._load_frames[transfer.request_id]
            raise
        for completion in completions:
            self._program.record_load_completion(frame, completion)
        self._raise_load_error(frame)

    @_with_active_frame
    def collect_load_result(self, frame: KVPoolStepFrame) -> LoadResult:
        completions = self._timeline.collect_load()
        frames = [frame]
        for completion in completions:
            owner = self._load_frames.pop(completion.request_id, None)
            if owner is None:
                raise RuntimeError(f"Load completion has no owning KV Pool frame: {completion.request_id}")
            self._program.record_load_completion(owner, completion)
            frames.append(owner)
        return self._finalize_load_result(frames, (completion.request_id for completion in completions))

    @_with_active_frame
    def wait_for_layer_load(self, frame: KVPoolStepFrame, layer_name: str) -> None:
        completions = self._timeline.wait_for_load_layer(layer_name)
        for completion in completions:
            self._program.record_load_completion(frame, completion)
        self._raise_load_error(frame)

    @_with_active_frame
    def save_layer(self, _frame: KVPoolStepFrame, layer_name: str) -> None:
        self._raise_store_error()
        try:
            self._timeline.submit_store_layer(layer_name)
        except Exception as error:
            self._store_error = error
            raise

    @_with_active_frame
    def finish_step(self, frame: KVPoolStepFrame) -> None:
        self._raise_store_error()
        if not self._timeline.store_enabled or not frame.step.store.commands:
            return
        if self._pending_store_batch is not None:
            raise RuntimeError("Previous Store invocation has not reached its fence")
        try:
            self._pending_store_batch = self._timeline.finish_store(frame.store_transfers)
            if self._timeline.fences_store_on_finish:
                self.fence_previous_store()
        except Exception as error:
            self._store_error = error
            raise

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
        if all(completion.evidence.source_release_confirmed for completion in completions):
            self._pending_store_batch = None
        for completion in completions:
            try:
                self._program.validate_store_completion(completion)
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
        try:
            self._timeline.close()
        except BaseException as error:
            if close_error is None:
                close_error = error
        finally:
            if self._pending_store_batch is None:
                self._resources.close()
        if close_error is not None:
            raise close_error

    def _execute_load(self, transfer: LoadTransfer) -> LoadCompletion:
        return self._program.execute_load(transfer, self._backend_io.load)

    def _filter_store_bindings(self, transfers: list[StoreTransfer]) -> list[StoreTransfer]:
        try:
            return self._program.filter_store_bindings(transfers, self._backend_io.observe_presence)
        except Exception as error:
            logger.error("Layerwise Store existence check failed; treating all keys as missing: %s", error)
            return transfers

    def _execute_store(self, transfer: StoreTransfer, source_ready_event: Any) -> StoreCompletion:
        try:
            filtered_transfers = self._program.filter_store_bindings([transfer], self._backend_io.observe_presence)
        except Exception as error:
            return StoreCompletion(transfer.request_id, StoreEvidence((), False, True, error))
        return self._execute_store_transfer(filtered_transfers[0], source_ready_event)

    def _execute_store_transfer(self, transfer: StoreTransfer, source_ready_event: Any) -> StoreCompletion:
        return self._program.execute_store(transfer, source_ready_event.synchronize, self._backend_io.store)

    def _raise_store_error(self) -> None:
        if self._store_error is not None:
            raise RuntimeError("KVPoolRuntime cannot continue after a previous Store failure") from self._store_error

    def _raise_load_error(self, frame: KVPoolStepFrame) -> None:
        if not frame.failed_request_ids:
            return
        self._timeline.abort_load()
        raise RuntimeError(f"Hybrid KV Load failed for requests: {sorted(frame.failed_request_ids)}")

    @staticmethod
    def _finalize_load_result(
        frames: Iterable[KVPoolStepFrame],
        completed_request_ids: Iterable[str],
    ) -> LoadResult:
        frames_by_identity = {id(frame): frame for frame in frames}
        result = LoadResult(
            frozenset(completed_request_ids),
            frozenset(request_id for frame in frames_by_identity.values() for request_id in frame.failed_request_ids),
            frozenset(block_id for frame in frames_by_identity.values() for block_id in frame.failed_block_ids),
        )
        for frame in frames_by_identity.values():
            frame.failed_request_ids.clear()
            frame.failed_block_ids.clear()
        return result
