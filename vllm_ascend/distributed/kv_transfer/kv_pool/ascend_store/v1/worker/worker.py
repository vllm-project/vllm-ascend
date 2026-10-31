"""Bind KV caches and adapt execution hooks to Lookup, Load and Store services."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger

from ...attention_fence import reset_attention_compute_start_gate
from ...metrics import AscendStoreKVConnectorStats
from ..backend import GVABackend, KeyRangeBackend, LayerwiseAccessKind
from ..backend.naming import WireKeyBuilder
from ..infrastructure.collective import CollectiveGroup
from ..infrastructure.coordinator import Coordinator
from ..infrastructure.executor import BulkExecutor, Executor, LayerwiseExecutor
from ..infrastructure.mapping import BlockHashMapping
from ..infrastructure.resources import Resources
from ..infrastructure.topology import KVPoolTopology
from ..protocol.messages import KVTransferStep, LookupRequest, LookupResult
from ..protocol.results import DrainResult, LoadResult, StoreResult
from .data.context import bind_context
from .service.load import LoadService
from .service.lookup import LookupService
from .service.store import StoreService
from .transfer.bulk import BulkTransfer
from .transfer.layerwise import LayerwiseTransfer


class Worker:
    """Bind once, route framework hooks, and close shared resources after both queues drain."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: Resources,
        *,
        coordinator: Coordinator,
        kv_cache_layout: str = "NHD",
        store_enabled: bool = True,
        source_ready_event_factory: Callable[[], Any] | None = None,
        layerwise: bool = False,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        key_builder: WireKeyBuilder | None = None,
        collective: CollectiveGroup | None = None,
    ) -> None:
        self._resources = resources
        self._closing = False
        self._bound = False
        self._binding_started = False
        self._step: KVTransferStep | None = None
        # torch_npu installs torch.npu at runtime; the CPU torch stubs omit it.
        self._source_ready_event_factory = source_ready_event_factory or (lambda: torch.npu.Event())  # type: ignore[attr-defined]
        self._fatal_error: BaseException | None = None
        load_executor = Executor("AscendStoreWrite", resources.backend.set_device)
        store_executor = Executor("AscendStoreRead", resources.backend.set_device) if store_enabled else None
        try:
            resources.bind_executors(
                tuple(executor for executor in (store_executor, load_executor) if executor is not None)
            )
            access = resources.backend_spec.layerwise_access if layerwise else None
            if layerwise:
                if access is None:
                    raise ValueError("Backend has no Layerwise access")
                if layerwise_prefetch_layers <= 0:
                    raise ValueError("Layerwise prefetch depth must be positive")
                config = topology.config
                if config.tp_partition.tp_mismatch or config.align_state_group_ids or config.has_private_state:
                    raise ValueError("Layerwise requires ordinary KV groups without TP mismatch")
                if (
                    config.pp_partition.target_layer_counts is not None
                    and len(config.pp_partition.target_layer_counts) > 1
                ):
                    raise ValueError("Layerwise does not support Consumer pipeline Store")
            shared_store = (
                access is LayerwiseAccessKind.GVA
                and store_enabled
                and topology.config.cp_partition.object_shard_count > 1
            )
            if shared_store:
                if collective is None:
                    collective = CollectiveGroup(topology.context_global_ranks)
                resources.bind_execution_cleanup(collective.close)
            elif collective is not None:
                raise ValueError("A Store collective requires shared GVA Store objects")
            self._block_hash_mappings = {
                group.group_id: BlockHashMapping(group.block_size, topology.config.hash_block_size)
                for group in topology.transfer_groups
            }
            self._lookup = LookupService(
                topology,
                resources.backend,
                coordinator,
                block_hash_mappings=self._block_hash_mappings,
                key_builder=key_builder,
            )
            backend_sessions = None
            if access is not None:
                backend_sessions = cast(KeyRangeBackend | GVABackend, resources.backend).create_sessions()
                resources.bind_backend_sessions(backend_sessions)
            self._topology = topology
            self._coordinator = coordinator
            self._key_builder = key_builder
            self._kv_cache_layout = kv_cache_layout
            self._access = access
            self._backend_sessions = backend_sessions
            self._collective = collective
            self._executor = (
                LayerwiseExecutor(
                    load_executor,
                    store_executor,
                    prefetch_layers=layerwise_prefetch_layers,
                    start_gate_factory=start_gate_factory,
                )
                if access is not None
                else BulkExecutor(load_executor, store_executor)
            )
        except BaseException:
            try:
                resources.stop_executors()
                resources.close()
            except BaseException:
                logger.exception("Failed to close Worker resources after construction failed")
            raise

    def bind_kv_caches(self, kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]]) -> None:
        """Bind addresses once, then register memory before starting execution queues."""
        if self._closing or self._binding_started:
            raise RuntimeError("KV caches cannot be rebound or bound after close")
        self._binding_started = True
        resources = self._resources
        try:
            resources.prepare_kv_caches(kv_caches)
            caches = resources.kv_caches
            assert caches is not None
            self.context = bind_context(
                self._topology,
                caches,
                self._kv_cache_layout,
                block_hash_mappings=self._block_hash_mappings,
                layerwise_access=self._access,
                key_builder=self._key_builder,
            )
            transfer: BulkTransfer | LayerwiseTransfer
            if self._access is not None:
                assert self._backend_sessions is not None and isinstance(self._executor, LayerwiseExecutor)
                transfer = LayerwiseTransfer(
                    self.context,
                    cast(KeyRangeBackend | GVABackend, resources.backend),
                    self._access,
                    self._backend_sessions,
                    self._executor,
                )
            else:
                assert isinstance(self._executor, BulkExecutor)
                transfer = BulkTransfer(
                    self.context,
                    resources.backend,
                    self._executor,
                    write_async=self._topology.config.load_async,
                )
            self.transfer = transfer
            self.store = StoreService(
                self.context,
                self._coordinator,
                transfer,
                requires_exists=resources.backend_spec.requires_exists_before_put,
                collective=self._collective,
            )
            self.load = LoadService(
                self.context,
                self._coordinator,
                transfer,
                require_complete_load=(
                    len(self._topology.config.group_specs) > 1
                    and (self._access is not None or not self._topology.config.load_async)
                ),
                on_failure=self.store.invalidate_requests,
            )
            resources.register_kv_buffers()
            resources.start_executors()
            self._bound = True
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("Failed to close Worker resources after cache binding failed")
            raise

    def lookup(self, request: LookupRequest) -> LookupResult:
        self._check_available()
        return self._lookup(request)

    def begin_step(self, step: KVTransferStep) -> None:
        self._check_available()
        if self._step is not None:
            raise RuntimeError("Previous KV Pool step has not ended")
        try:
            predecessor = self.transfer.begin_step()
            self._step = step
            if step.store.commands and predecessor is not None:
                self.store.fence()
            self.store.submit(step.store.commands)
            if step.store.all_sources_ready:
                self.transfer.read_all_layers(self._record_source_ready)
        except BaseException as error:
            self.abort_step(error)
            raise

    def start_load(self) -> None:
        self._check_available()
        self.load.submit(self._active_step().load_commands)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._check_available()
        self._active_step()
        layer_id = self.context.layer_ids_by_name.get(layer_name)
        if layer_id is None:
            return
        try:
            self.transfer.wait_for_layer(layer_id, self._record_source_ready)
        except BaseException as error:
            self.abort_step(error)
            raise

    def save_layer(self, layer_name: str) -> None:
        self._check_available()
        self._active_step()
        layer_id = self.context.layer_ids_by_name.get(layer_name)
        if layer_id is not None:
            try:
                self.transfer.read_layer(layer_id, self._record_source_ready)
            except BaseException as error:
                self.abort_step(error)
                raise

    def collect_load_result(self) -> LoadResult:
        self._check_available(allow_closing=True)
        return self.load.collect()

    def finish_step(self) -> None:
        self._active_step()
        self._check_available()
        try:
            if self.transfer.finish_step(self._record_source_ready) is not None:
                self.store.fence()
        except BaseException as error:
            self.abort_step(error)
            raise

    def end_step(self) -> None:
        if self._step is None:
            return
        try:
            self.transfer.end_step(wait=True)
        except BaseException as error:
            self._fatal_error = self._fatal_error or error
            raise
        finally:
            self._step = None

    def abort_step(self, error: BaseException) -> None:
        """Detach framework work, retaining submitted batches and unsafe resources for close."""
        self._fatal_error = self._fatal_error or error
        try:
            if self._bound:
                self.transfer.end_step(wait=False)
        except BaseException:
            logger.exception("Failed to end the transfer step after an earlier operation failure")
        finally:
            self._step = None

    def fence_previous_store(self, released_request_ids: frozenset[str] | None = None) -> tuple[StoreResult, ...]:
        return self.store.fence(released_request_ids)

    def take_released_store_job_ids(self) -> set[int]:
        return self.store.take_released_jobs()

    def take_events(self) -> list[BlockStored]:
        return self.store.take_events()

    def get_stats(self) -> AscendStoreKVConnectorStats | None:
        return self.load.take_stats()

    def reset_cache(self) -> bool:
        """Reset between paused steps without closing queues, registrations or admission."""
        self._check_available()
        if self._step is not None or self._access is not None or self.load.pending:
            return False
        reset = getattr(self._resources.backend, "reset_cache", None)
        if not callable(reset):
            return False
        self.store.fence()
        self.store.take_events()
        return reset()

    def close(self) -> None:
        """Call after synchronous Lookup/RPC callers have quiesced; keep resources on unsafe drain."""
        self._closing = True
        if not self._bound:
            self._resources.stop_executors()
            self._resources.close()
            return
        self.store.stop_accepting()
        self.load.stop_accepting()
        error: BaseException | None = self._fatal_error
        safe_to_release = True
        try:
            self.transfer.end_step(wait=False)
        except BaseException as caught:
            safe_to_release = False
            error = error or caught
        finally:
            self._step = None
        for operation in (self.store, self.load):
            try:
                drained = operation.drain()
            except BaseException as caught:
                drained = DrainResult(False, caught)
            safe_to_release &= drained.safe_to_release
            if error is None:
                error = drained.error
            elif drained.error is not None:
                failure = drained.error
                logger.error(
                    "Operation drain failed after an earlier close failure",
                    exc_info=(type(failure), failure, failure.__traceback__),
                )
        try:
            self._resources.stop_executors()
        except BaseException as caught:
            safe_to_release = False
            if error is None:
                error = caught
            else:
                logger.exception("Failed to stop executors after an earlier close failure")
        if not safe_to_release and error is None:
            error = RuntimeError("Operation drain did not confirm safe memory release")
        if safe_to_release:
            try:
                self._resources.close()
            except BaseException as caught:
                if error is None:
                    error = caught
                else:
                    logger.exception("Failed to close resources after an earlier close failure")
        if error is not None:
            raise error

    def _record_source_ready(self) -> Any:
        try:
            # Admission runs asynchronously; record the producer's stream before this hook returns.
            source_ready = self._source_ready_event_factory()
            source_ready.record()
        except Exception as error:
            self._fatal_error = error
            raise
        return source_ready

    def _check_available(self, *, allow_closing: bool = False) -> None:
        if not self._bound:
            raise RuntimeError("KV caches must be bound before execution")
        if self._fatal_error is not None:
            raise RuntimeError("Worker cannot continue after a fatal hook failure") from self._fatal_error
        self.load.check_available()
        self.store.check_available()
        if self._closing and not allow_closing:
            raise RuntimeError("Worker is closing and cannot accept new work")

    def _active_step(self) -> KVTransferStep:
        if self._step is None:
            raise RuntimeError("KV Pool step has not begun")
        return self._step
