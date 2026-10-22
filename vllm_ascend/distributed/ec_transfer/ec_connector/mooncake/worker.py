# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""NPU worker for vLLM's Mooncake encoder-cache connector."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import (
    _RESERVATION_TTL_SECONDS,
    MooncakeECConfig,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.control import (
    ControlClient,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.producer import (
    ProducerPushManager,
    ProducerPushRecord,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservationManager,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    ensure_mooncake_available,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    _MAX_CANCELLED_TRANSFER_IDS,
    _RESERVATION_REFRESH_SECONDS,
    ECMooncakeWorker,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    build_embedding_namespace,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    EmbeddingStoreOperationError,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    _resolve_bounce_arena_size,
    _TransferFragmentPlan,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    AscendConsumerMemoryPool,
    AscendContiguousAllocator,
    AscendProducerAllocator,
    AscendProducerMemoryPool,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_backend import (
    AscendDeferredStoreBackend,
    PreparedStorePut,
    PreparedStoreSave,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_setup import (
    configure_non_fabric_transport,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.backend import (
        MooncakeEmbeddingStoreBackend,
    )


_TRANSFER_WORKERS = 4
_CONTROL_WORKERS = 8
# Store-borrowed slabs and uncertain transfers retain their EC-side owners
# until process exit, alongside the native Store-side owners.
_PROCESS_LIFETIME_EC_OWNERS: list[tuple[Any, Any]] = []
_SessionItemT = TypeVar("_SessionItemT")


def _resolve_ascend_config(vllm_config: VllmConfig) -> MooncakeECConfig:
    config = MooncakeECConfig.from_vllm_config(vllm_config)
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None

    protocol = config.protocol
    if "mooncake_protocol" not in ec_config.ec_connector_extra_config:
        protocol = "ascend"
    elif protocol != "ascend":
        raise ValueError("Ascend ECMooncakeConnector requires mooncake_protocol='ascend'")

    buffer_device = config.buffer_device
    if buffer_device == "cuda":
        buffer_device = "npu"
    device_type, separator, device_index = buffer_device.partition(":")
    is_valid_npu_device = device_type == "npu" and (
        not separator or (device_index.isascii() and device_index.isdigit())
    )
    if not is_valid_npu_device:
        raise ValueError("Ascend ECMooncakeWorker requires ec_buffer_device='npu'")
    return replace(
        config,
        protocol=protocol,
        buffer_device=buffer_device,
    )


@dataclass(frozen=True)
class _AcquiredTransferWave:
    fragments: list[_TransferFragmentPlan]
    store_puts: list[PreparedStorePut] = field(default_factory=list)
    store_futures: list[Future[None]] = field(default_factory=list, compare=False)


class AscendECMooncakeWorker(ECMooncakeWorker):
    """Reuse upstream orchestration with Ascend-specific primitives."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None

        config = _resolve_ascend_config(vllm_config)

        self._store_config = (
            (
                build_embedding_namespace(vllm_config),
                config.store_max_pending_items,
                config.store_max_pending_bytes,
                config.store_read_buffer_bytes,
            )
            if config.cross_encoder_cache
            else None
        )
        self._output_store: MooncakeEmbeddingStoreBackend | None = None
        self._deferred_store: AscendDeferredStoreBackend | None = None

        resolved_bounce_arena_size = _resolve_bounce_arena_size(vllm_config)
        self._bounce_arena_size = resolved_bounce_arena_size if ec_config.is_ec_producer else 0

        if os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") != "1":
            configure_non_fabric_transport()
        ensure_mooncake_available()
        self.is_producer = config.is_producer
        self.is_consumer = config.is_consumer
        self._buffer_device = config.buffer_device
        self._control_host = config.control_host
        self._control_port = config.control_port
        transfer = AscendMooncakeTransfer(get_ip(), torch.npu.current_device())
        consumer_memory = AscendConsumerMemoryPool(
            config.pool_size,
            transfer,
            allocator=AscendContiguousAllocator(config.pool_size),
        )
        self._transfer = transfer
        self._consumer_memory = consumer_memory
        self._reservations = ConsumerReservationManager(
            consumer_memory,
            _RESERVATION_TTL_SECONDS,
            _MAX_CANCELLED_TRANSFER_IDS,
        )
        self._consumer_rank_resolved = False
        self._is_receiving_rank = True
        self._tp_rank = 0
        self._tp_size = 1
        self._control_server = None
        self._producer_memory = AscendProducerMemoryPool(
            config.pool_size,
            transfer,
            allocator=AscendProducerAllocator(
                staging_capacity=config.pool_size,
                bounce_capacity=self._bounce_arena_size,
            ),
        )
        self._control_client = ControlClient(config.control_timeout_ms)
        self._shutdown_drain_timeout_s = config.control_timeout_ms / 1000
        self._io_executor = ThreadPoolExecutor(
            max_workers=_TRANSFER_WORKERS,
            thread_name_prefix="ec-mooncake-transfer",
        )
        self._control_executor = ThreadPoolExecutor(
            max_workers=_CONTROL_WORKERS,
            thread_name_prefix="ec-mooncake-control",
        )
        self._shard_pool = None
        self._shard_pool_lock = threading.Lock()
        self._push_ready = threading.Event()
        self._producer_pushes = ProducerPushManager(self._push_ready.set)
        self._dispatch_stop = threading.Event()
        self._dispatcher = None
        self._failed_saves: set[str] = set()
        self._collecting_sources = False
        self._completed_loads: set[str] = set()
        self._failed_loads: set[str] = set()
        self._shutdown = False
        self._use_store_independent_te = False
        self._retain_producer_slab = True
        self._control_thread = threading.local()
        if self.is_producer:
            self._dispatcher = threading.Thread(
                target=self._dispatch_pushes,
                name="ec-mooncake-ready",
                daemon=True,
            )
            self._dispatcher.start()

    def start_services(self) -> None:
        if not getattr(self, "_native_owners_retained", False):
            _PROCESS_LIFETIME_EC_OWNERS.append((self._transfer, (self._producer_memory, self._consumer_memory)))
            self._native_owners_retained = True
        if not self.is_producer or self._store_config is None or self._output_store is not None:
            if self.is_producer:
                self._producer_memory.ensure_prepared(torch.device(self._buffer_device))
            super().start_services()
            return

        from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_setup import (
            create_ascend_mooncake_embedding_store_client,
            ensure_mooncake_store_imported,
            use_store_independent_te,
        )

        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        namespace, max_items, max_bytes, read_bytes = self._store_config
        ensure_mooncake_store_imported()
        use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
        use_independent_store = use_store_independent_te()
        self._use_store_independent_te = use_independent_store
        use_shared_roce_engine = not use_fabric_mem and not use_independent_store
        if use_fabric_mem or use_independent_store:
            # Independent topology is deliberately staged: initialize the EC
            # engine first under the top-level resource config, before the
            # Store factory enables the process-global RoCE selectors for its
            # separately owned engine.
            producer_memory.ensure_prepared(torch.device(self._buffer_device))
        try:
            store_client = create_ascend_mooncake_embedding_store_client(
                producer_memory,
                cast(AscendMooncakeTransfer, self._transfer),
                read_buffer_bytes=read_bytes,
            )
            try:
                if use_shared_roce_engine:
                    # The validated RoCE topology attaches Store to the global
                    # engine before registering the shared NPU staging slab.
                    producer_memory.ensure_prepared(torch.device(self._buffer_device))
                staging = producer_memory.tensor
                if staging is None:
                    raise RuntimeError("Producer staging slab is unavailable")
                store_client.bind_staging_buffer(staging)
                self._retain_producer_slab = True
                backend_type = AscendDeferredStoreBackend
                output_store = backend_type(
                    store_client,
                    namespace,
                    max_pending_items=max_items,
                    max_pending_bytes=max_bytes,
                )
            except BaseException:
                store_client.close()
                raise

            self._output_store = output_store
            self._deferred_store = output_store
            try:
                super().start_services()
            except BaseException:
                self._output_store = None
                self._deferred_store = None
                output_store.shutdown()
                raise
        except BaseException:
            if not getattr(self, "_retain_producer_slab", False):
                producer_memory.close()
            raise

    def close(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is not None:
            self._dispatch_stop.set()
            self._push_ready.set()
            dispatcher.join()
        if self._control_server is not None:
            self._reservations.begin_shutdown()
        self._flush_pending_pushes()
        for record in self._producer_pushes.cancel_requests(None):
            self._producer_pushes.submit_cancel(
                record,
                self._io_executor,
                self._cancel_orphaned_reservation,
            )
        self._control_executor.shutdown(wait=True)
        self._producer_pushes.submit_batches(
            self._io_executor,
            self._push_batch,
            wait=True,
        )
        self._io_executor.shutdown(wait=True)
        if self._shard_pool is not None:
            self._shard_pool.shutdown(wait=True)
        self._control_client.close()
        if self._output_store is not None:
            # No producer task can choose or submit a Store path after this
            # point. Drain Store before releasing its arena or shared engine.
            self._output_store.shutdown()
            self._output_store = None
            self._deferred_store = None
        drained = True
        if self._control_server is not None:
            drained = self._reservations.wait_for_writers(self._shutdown_drain_timeout_s)
            self._control_server.close()
        if drained:
            _PROCESS_LIFETIME_EC_OWNERS.append((self._transfer, self._consumer_memory))
        else:
            _PROCESS_LIFETIME_EC_OWNERS.append((self._transfer, self._consumer_memory))
            logger.error("Timed out waiting for Mooncake EC writers; keeping the consumer receive pool registered")
        if getattr(self, "_retain_producer_slab", False):
            _PROCESS_LIFETIME_EC_OWNERS.append((self._transfer, self._producer_memory))
        else:
            self._producer_memory.close()
            self._transfer.close()

    def _reserve_push_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not getattr(self._control_thread, "device_set", False):
            pool = self._consumer_memory.tensor
            assert pool is not None
            torch.npu.set_device(pool.device)
            self._control_thread.device_set = True
        return super()._reserve_push_destination(payload)

    def _retain_transfer_owners(self, owners: list[Any]) -> None:
        """Keep uncertain native users alive and wake blocked allocators."""
        self._producer_memory.poison()
        self._retain_producer_slab = True
        _PROCESS_LIFETIME_EC_OWNERS.append((self._transfer, (self._producer_memory, *owners)))
        deferred_store = getattr(self, "_deferred_store", None)
        if deferred_store is not None:
            deferred_store.store_client.retain_fallback_owners(owners)

    def _run_session_writes(
        self,
        by_session: dict[str, list[_SessionItemT]],
        session_records: dict[str, dict[str, ProducerPushRecord]],
        write: Callable[[str, list[_SessionItemT]], None],
    ) -> None:
        sessions = list(by_session.items())

        def track_write(index: int, future: Future[None]) -> None:
            session = sessions[index][0]
            self._producer_pushes.track_shard_futures(
                list(session_records[session].values()),
                [future],
            )

        self._run_fanout(
            [partial(write, *session) for session in sessions],
            track_write,
        )

    def _write_transfer_wave(
        self,
        pushes: list[ProducerPushRecord],
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]],
        acquired: _AcquiredTransferWave,
    ) -> bool:
        """Submit Store work and wait only for EC writes, retaining PUT futures."""
        source_index = {push.spec.transfer_id: index for index, push in enumerate(pushes)}
        fragments_by_source: list[list[_TransferFragmentPlan]] = [[] for _ in pushes]
        for fragment in acquired.fragments:
            fragments_by_source[fragment.source_index].append(fragment)

        by_session: dict[
            str,
            list[tuple[_TransferFragmentPlan, int]],
        ] = {}
        session_records: dict[str, dict[str, ProducerPushRecord]] = {}
        for push, shard in ready:
            index = source_index.get(push.spec.transfer_id)
            if index is None:
                continue

            session = str(shard["dst_session"])
            destination = int(shard["dst_ptr"])
            for fragment in fragments_by_source[index]:
                by_session.setdefault(session, []).append((fragment, destination))
            session_records.setdefault(session, {})[push.spec.transfer_id] = push

        def write(
            session: str,
            items: list[tuple[_TransferFragmentPlan, int]],
        ) -> None:
            self._transfer.write(
                session,
                [fragment.source_address for fragment, _ in items],
                [destination + fragment.destination_offset for fragment, destination in items],
                [fragment.nbytes for fragment, _ in items],
            )

        deferred_store = getattr(self, "_deferred_store", None)
        assert deferred_store is not None or not acquired.store_puts
        error: BaseException | None = None
        for store_put in acquired.store_puts:
            assert deferred_store is not None
            try:
                acquired.store_futures.append(deferred_store.submit_prepared(store_put))
            except BaseException as exc:
                error = exc
                break

        if error is None:
            try:
                self._run_session_writes(by_session, session_records, write)
            except BaseException as exc:
                error = exc
        if error is not None:
            owners: list[Any] = []
            for store_put in acquired.store_puts:
                owners.extend(store_put.owners)
                owners.append(store_put.ticket.tensor)
            if deferred_store is not None and acquired.store_puts:
                deferred_store.store_client.retain_fallback_owners(owners)
            return False
        return True

    def _release_after_store(self, acquired: _AcquiredTransferWave, staged: Any = None) -> None:
        """After EC completes, hand memory ownership to the PUT callbacks.

        No callback acquires staging space. The publisher can release
        a completed wave even while an EC thread waits for the next allocation.
        Executor shutdown joins publisher callbacks before Store/native teardown.
        """

        def release() -> None:
            if staged is not None:
                self._producer_memory.release(staged)

        if not acquired.store_futures:
            release()
            return

        lock = threading.Lock()
        futures = tuple(acquired.store_futures)
        # Futures retain their callbacks. Move them out of the acquired wave
        # so completed saves cannot form a cycle retaining original outputs.
        acquired.store_futures.clear()
        remaining = len(futures)
        uncertain = False

        def completed(future: Future[None]) -> None:
            nonlocal remaining, uncertain
            error = None
            try:
                error = future.exception()  # The callback only sees completed PUTs.
            except BaseException as exc:
                error = exc
            if isinstance(error, (EmbeddingStoreOperationError, OSError)):
                # Confirmed failure: the save failed, but its memory is reusable.
                error = None
            with lock:
                first_error = error is not None and not uncertain
                uncertain |= error is not None
                remaining -= 1
                reusable = remaining == 0 and not uncertain
            if first_error:
                self._retain_transfer_owners([staged, acquired])
                logger.error("Store completion is unconfirmed; retaining EC/Store wave: %s", error)
            if reusable:
                try:
                    release()
                except BaseException:
                    self._retain_transfer_owners([staged, acquired])
                    logger.exception("Failed to reclaim completed EC/Store wave; retaining owners")

        for future in futures:
            future.add_done_callback(completed)

    def _resolve_writable_pushes(
        self,
        pushes: list[ProducerPushRecord],
        started_at: float,
    ) -> tuple[
        list[tuple[ProducerPushRecord, dict[str, Any]]],
        list[ProducerPushRecord],
    ]:
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]] = []
        ordered: dict[str, ProducerPushRecord] = {}
        for push in pushes:
            self._validate_push_source(push)
            reservations = self._producer_pushes.resolve_reservations(push)
            stale = [
                index
                for index, shard in enumerate(reservations)
                if not shard.get("ready", False)
                and not shard.get("cancelled", False)
                and time.monotonic() - float(shard.get("_received_at", started_at)) >= _RESERVATION_REFRESH_SECONDS
            ]
            if stale:
                reservations = self._refresh_remote_reservations(
                    push.spec,
                    reservations,
                    push,
                )
                self._producer_pushes.replace_reservations(push, reservations)
            self._producer_pushes.begin_writing(push)
            source = push.source_tensor
            assert source is not None
            for shard in reservations:
                if shard.get("cached", False) or shard.get("cancelled", False) or not shard.get("write", True):
                    continue
                if int(shard["nbytes"]) != source.nbytes:
                    raise RuntimeError(f"Reserved EC size does not match tensor for mm_hash={push.spec.mm_hash}")
                ready.append((push, shard))
                ordered.setdefault(push.spec.transfer_id, push)
        return ready, list(ordered.values())

    def _acquire_staged_wave(
        self,
        stage_pushes: list[ProducerPushRecord],
        staged: Any,
        store_saves: dict[str, PreparedStoreSave],
    ) -> _AcquiredTransferWave:
        """Describe whole staged EC payloads and their coallocated Store headers."""
        fragments = [
            _TransferFragmentPlan(
                source_index=index,
                source_address=tensor.data_ptr(),
                destination_offset=0,
                nbytes=tensor.nbytes,
            )
            for index, tensor in enumerate(staged.tensors)
        ]
        staged_by_hash = {}
        for push, tensor, header in zip(stage_pushes, staged.tensors, staged.headers, strict=True):
            staged_by_hash.setdefault(push.spec.mm_hash, (tensor, header))
        store_puts = []
        for identifier, ticket in store_saves.items():
            payload, header = staged_by_hash[identifier]
            assert header is not None
            store_puts.append(
                PreparedStorePut(
                    ticket=ticket,
                    addresses=[header.data_ptr(), payload.data_ptr()],
                    sizes=[header.nbytes, payload.nbytes],
                    owners=[staged],
                )
            )
        return _AcquiredTransferWave(fragments, store_puts)

    def _push_staged_sources(
        self,
        pushes: list[ProducerPushRecord],
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]],
        staged: Any,
        store_saves: dict[str, PreparedStoreSave],
    ) -> None:
        """Notify EC independently; retain staged bytes through Store completion."""
        acquired = None
        submitted = False
        reusable = False
        try:
            acquired = self._acquire_staged_wave(pushes, staged, store_saves)
            submitted = True
            if not self._write_transfer_wave(pushes, ready, acquired):
                raise RuntimeError("EC/Store staging completion is unconfirmed")
            self._release_after_store(acquired, staged)
            reusable = True
        finally:
            if not submitted:
                self._producer_memory.release(staged)
            elif not reusable:
                self._retain_transfer_owners([staged, acquired])

    def _push_store_batch(
        self,
        pushes: list[ProducerPushRecord],
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]],
        ordered_pushes: list[ProducerPushRecord],
    ) -> None:
        """Stage whole payloads and headers atomically; publish asynchronously."""
        deferred_store = cast(AscendDeferredStoreBackend, self._deferred_store)
        independent = getattr(self, "_use_store_independent_te", False)
        prepared: dict[str, PreparedStoreSave] = {}
        try:
            for identifier in dict.fromkeys(push.spec.mm_hash for push in pushes):
                ticket = deferred_store.prepare_save(identifier)
                if ticket is not None:
                    prepared[identifier] = ticket

            sources = list(ordered_pushes)

            # Preserve each topology's existing source layout: independent
            # mode stages distinct transfers; shared mode adds Store-only hashes.
            def source_id(push: ProducerPushRecord) -> str:
                return push.spec.transfer_id if independent else push.spec.mm_hash

            staged_ids = {source_id(push) for push in sources}
            for push in pushes:
                if push.spec.mm_hash in prepared and source_id(push) not in staged_ids:
                    sources.append(push)
                    staged_ids.add(source_id(push))
            if not sources:
                return
            tensors = [cast(torch.Tensor, push.source_tensor) for push in sources]
            producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
            headers = [
                deferred_store.store_client.prepare_put_header(tensor) if push.spec.mm_hash in prepared else None
                for push, tensor in zip(sources, tensors, strict=True)
            ]
            staged = producer_memory.stage_with_headers(tensors, headers)
            if staged is None:
                raise RuntimeError("encoder batch including Store headers exceeds the fixed staging pool")
            self._push_staged_sources(sources, ready, staged, prepared)
        finally:
            for ticket in prepared.values():
                deferred_store.abort_prepared(ticket)

    def _push_batch(self, pushes: list[ProducerPushRecord]) -> None:
        started_at = time.monotonic()
        deferred_store = getattr(self, "_deferred_store", None)
        failure: Exception | None = None
        try:
            ready, ordered_pushes = self._resolve_writable_pushes(
                pushes,
                started_at,
            )

            if deferred_store is not None:
                self._push_store_batch(pushes, ready, ordered_pushes)
            elif ready:
                tensors = [cast(torch.Tensor, push.source_tensor) for push in ordered_pushes]
                staged = self._producer_memory.stage_wait(tensors)
                if staged is None:
                    raise RuntimeError("encoder batch exceeds the fixed staging pool")
                self._push_staged_sources(ordered_pushes, ready, staged, {})

            self._producer_pushes.begin_notifying(pushes)
            self._notify_completions(ready)
            self._producer_pushes.complete(pushes)
        except Exception as exc:
            failure = exc
            logger.exception(
                "EC Mooncake push batch failed for mm_hashes=%s",
                [push.spec.mm_hash for push in pushes],
            )
            self._producer_pushes.settle_all(pushes)
            self._abandon_pushes(pushes)
        finally:
            if failure is not None:
                self._producer_pushes.fail(pushes, failure)

    def _bind_push_source(self, tensor: torch.Tensor, mm_hash: str) -> None:
        """Bind an NPU source using an NPU readiness event."""
        if tensor.device.type != "npu":
            raise ValueError("Ascend ECMooncake source tensor must be on NPU")
        ready_event = torch.npu.Event()
        ready_event.record(torch.npu.current_stream(tensor.device))
        self._producer_pushes.bind_source(mm_hash, tensor, ready_event)

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs: Any,
    ) -> None:
        if self._deferred_store is None:
            super().save_caches(encoder_cache, mm_hash, **kwargs)
            return
        if not self.is_producer:
            return
        tensor = encoder_cache[mm_hash]
        self._bind_push_source(tensor, mm_hash)
        self._deferred_store.save_output(mm_hash, tensor)
