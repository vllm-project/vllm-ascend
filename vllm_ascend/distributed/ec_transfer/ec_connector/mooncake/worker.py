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
    _MOONCAKE_TENSOR_HEADER,
    EmbeddingStoreOperationError,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    _BOUNCE_ARENA_CONFIG_KEY,
    _BounceLease,
    _flatten_transfer_wave,
    _plan_transfer_waves,
    _resolve_bounce_arena_size,
    _TransferFragmentPlan,
    _TransferWavePlan,
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
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import (
    configure_non_fabric_roce,
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
# A slab registered in both independent native engines cannot lose either
# owner before process exit.  Keep the EC-side objects alive just as
# store_client keeps the Store-side objects alive.
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
    registration_addresses: list[int]
    bounce_lease: _BounceLease | None
    store_puts: list[PreparedStorePut] = field(default_factory=list)


class _RegistrationReleaseError(RuntimeError):
    pass


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
            configure_non_fabric_roce()
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
        self._failed_saves = set()
        self._collecting_sources = False
        self._completed_loads = set()
        self._failed_loads = set()
        self._shutdown = False
        self._use_store_independent_te = False
        self._retain_producer_slab = False
        self._control_thread = threading.local()
        if self.is_producer:
            self._dispatcher = threading.Thread(
                target=self._dispatch_pushes,
                name="ec-mooncake-ready",
                daemon=True,
            )
            self._dispatcher.start()

    def start_services(self) -> None:
        if not self.is_producer or self._store_config is None or self._output_store is not None:
            super().start_services()
            return

        from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.backend import (
            MooncakeEmbeddingStoreBackend,
        )

        from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import (
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
        if use_fabric_mem:
            producer_memory.ensure_prepared(torch.device(self._buffer_device))
        try:
            store_client = create_ascend_mooncake_embedding_store_client(
                producer_memory.bounce_arena,
                cast(AscendMooncakeTransfer, self._transfer),
                read_buffer_bytes=read_bytes,
            )
            try:
                if use_shared_roce_engine:
                    # The validated RoCE topology attaches Store to the global
                    # engine before registering the shared NPU staging slab.
                    producer_memory.ensure_prepared(
                        torch.device(self._buffer_device)
                    )
                elif use_independent_store:
                    producer_memory.ensure_prepared(
                        torch.device(self._buffer_device)
                    )
                    staging = producer_memory.tensor
                    if staging is None:
                        raise RuntimeError("Producer staging slab is unavailable")
                    store_client.register_staging_buffer(staging)
                    self._retain_producer_slab = True
                backend_type = (
                    AscendDeferredStoreBackend
                    if not use_fabric_mem
                    else MooncakeEmbeddingStoreBackend
                )
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
            self._deferred_store = (
                output_store
                if isinstance(output_store, AscendDeferredStoreBackend)
                else None
            )
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
            drained = self._reservations.wait_for_writers(
                self._shutdown_drain_timeout_s
            )
            self._control_server.close()
        if drained:
            self._consumer_memory.close()
        else:
            logger.error(
                "Timed out waiting for Mooncake EC writers; keeping the "
                "consumer receive pool registered"
            )
        if getattr(self, "_retain_producer_slab", False):
            _PROCESS_LIFETIME_EC_OWNERS.append(
                (self._transfer, self._producer_memory)
            )
        else:
            self._producer_memory.close()
            self._transfer.close()

    def _reserve_push_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not getattr(self._control_thread, "device_set", False):
            torch.npu.set_device(torch.device(self._buffer_device))
            self._control_thread.device_set = True
        return super()._reserve_push_destination(payload)

    def _acquire_transfer_wave(
        self,
        wave: _TransferWavePlan,
        store_saves: list[PreparedStoreSave | None] | None = None,
    ) -> _AcquiredTransferWave:
        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        transfer = cast(AscendMooncakeTransfer, self._transfer)
        bounce_arena = producer_memory.bounce_arena
        lease = bounce_arena.acquire(wave.bounce_nbytes)
        registration_addresses: list[int] = []

        try:
            bounce_address: int | None = None
            if lease is not None:
                copies: list[tuple[torch.Tensor, int, int]] = []
                for index, wave_source in enumerate(wave.sources):
                    source = wave_source.source
                    ticket = store_saves[index] if store_saves is not None else None
                    if ticket is not None:
                        assert wave_source.store_header_offset is not None
                        store_client = cast(
                            AscendDeferredStoreBackend,
                            self._deferred_store,
                        ).store_client
                        header = store_client.prepare_put_header(ticket.tensor)
                        copies.append(
                            (
                                header,
                                wave_source.store_header_offset,
                                _MOONCAKE_TENSOR_HEADER.size,
                            )
                        )
                    if source.prefix_nbytes == 0:
                        continue
                    assert wave_source.bounce_offset is not None
                    copies.append(
                        (
                            source.owner,
                            wave_source.bounce_offset,
                            source.prefix_nbytes,
                        )
                    )

                bounce_address = bounce_arena.copy(lease, copies)

            fragments = _flatten_transfer_wave(wave, bounce_address)
            registration_addresses = transfer.acquire_registration_ranges(wave.registration_ranges)
            store_puts: list[PreparedStorePut] = []
            if store_saves is not None:
                assert bounce_address is not None or not any(store_saves)
                for ticket, wave_source in zip(store_saves, wave.sources, strict=True):
                    if ticket is None:
                        continue
                    source = wave_source.source
                    assert wave_source.store_header_offset is not None
                    assert bounce_address is not None
                    addresses = [
                        bounce_address + wave_source.store_header_offset,
                    ]
                    sizes = [_MOONCAKE_TENSOR_HEADER.size]
                    if source.prefix_nbytes:
                        assert wave_source.bounce_offset is not None
                        addresses.append(bounce_address + wave_source.bounce_offset)
                        sizes.append(source.prefix_nbytes)
                    if source.direct_address is not None:
                        addresses.append(source.direct_address)
                        sizes.append(source.direct_nbytes)
                    store_puts.append(
                        PreparedStorePut(
                            ticket=ticket,
                            addresses=addresses,
                            sizes=sizes,
                            owners=[bounce_arena, lease, ticket.tensor],
                        )
                    )
            return _AcquiredTransferWave(
                fragments=fragments,
                registration_addresses=registration_addresses,
                bounce_lease=lease,
                store_puts=store_puts,
            )
        except Exception as error:
            try:
                if registration_addresses:
                    released = transfer.release_registration_ranges(
                        registration_addresses
                    )
                    if not released:
                        raise _RegistrationReleaseError(
                            "EC direct registration rollback could not be confirmed"
                        ) from error
            except _RegistrationReleaseError:
                raise
            except BaseException as cleanup_error:
                raise _RegistrationReleaseError(
                    "EC direct registration rollback raised an exception"
                ) from cleanup_error
            finally:
                bounce_arena.release(lease)
            raise

    def _release_transfer_wave(self, acquired: _AcquiredTransferWave) -> None:
        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        transfer = cast(AscendMooncakeTransfer, self._transfer)

        try:
            released = transfer.release_registration_ranges(
                acquired.registration_addresses
            )
        except BaseException as error:
            raise _RegistrationReleaseError(
                "EC direct registration release raised an exception"
            ) from error
        if not released:
            raise _RegistrationReleaseError(
                "EC direct registration release could not be confirmed"
            )
        producer_memory.bounce_arena.release(acquired.bounce_lease)

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
        source_index = {
            push.spec.transfer_id: index
            for index, push in enumerate(pushes)
        }
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
                [
                    destination + fragment.destination_offset
                    for fragment, destination in items
                ],
                [fragment.nbytes for fragment, _ in items],
            )

        store_futures: list[Future[None]] = []
        deferred_store = getattr(self, "_deferred_store", None)
        assert deferred_store is not None or not acquired.store_puts
        error: BaseException | None = None
        for store_put in acquired.store_puts:
            assert deferred_store is not None
            try:
                store_futures.append(
                    deferred_store.submit_prepared(store_put)
                )
            except BaseException as exc:
                error = exc
                break

        if error is None:
            try:
                self._run_session_writes(by_session, session_records, write)
            except BaseException as exc:
                error = exc
        for future in store_futures:
            try:
                future.result()
            except (EmbeddingStoreOperationError, OSError):
                # These outcomes confirm that no native operation remains in flight.
                continue
            except BaseException as exc:
                if error is None:
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
                and time.monotonic()
                - float(shard.get("_received_at", started_at))
                >= _RESERVATION_REFRESH_SECONDS
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
                if (
                    shard.get("cached", False)
                    or shard.get("cancelled", False)
                    or not shard.get("write", True)
                ):
                    continue
                if int(shard["nbytes"]) != source.nbytes:
                    raise RuntimeError(
                        "Reserved EC size does not match tensor for "
                        f"mm_hash={push.spec.mm_hash}"
                    )
                ready.append((push, shard))
                ordered.setdefault(push.spec.transfer_id, push)
        return ready, list(ordered.values())

    def _submit_original_store_outputs(
        self,
        pushes: list[ProducerPushRecord],
    ) -> None:
        """Publish immutable outputs through the Store's normal PUT path."""
        deferred_store = getattr(self, "_deferred_store", None)
        if deferred_store is None:
            return
        for mm_hash in dict.fromkeys(push.spec.mm_hash for push in pushes):
            deferred_store.submit_original(mm_hash)

    def _acquire_staged_wave(
        self,
        stage_pushes: list[ProducerPushRecord],
        staged: Any,
        store_saves: dict[str, PreparedStoreSave],
    ) -> _AcquiredTransferWave:
        """Describe staged EC payloads and pack any Store headers."""
        fragments = [
            _TransferFragmentPlan(
                source_index=index,
                source_address=tensor.data_ptr(),
                destination_offset=0,
                nbytes=tensor.nbytes,
            )
            for index, tensor in enumerate(staged.tensors)
        ]
        if not store_saves:
            return _AcquiredTransferWave(fragments, [], None, [])

        deferred_store = cast(AscendDeferredStoreBackend, self._deferred_store)
        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        bounce_arena = producer_memory.bounce_arena
        header_size = _MOONCAKE_TENSOR_HEADER.size
        lease = bounce_arena.acquire(header_size * len(store_saves))
        if lease is None:
            raise RuntimeError(
                "Store-independent staging requires a nonzero bounce arena"
            )

        try:
            copies: list[tuple[torch.Tensor, int, int]] = []
            header_offsets: dict[str, int] = {}
            for ticket in store_saves.values():
                offset = len(header_offsets) * header_size
                header_offsets[ticket.identifier] = offset
                copies.append(
                    (
                        deferred_store.store_client.prepare_put_header(
                            ticket.tensor
                        ),
                        offset,
                        header_size,
                    )
                )
            bounce_address = bounce_arena.copy(lease, copies)

            staged_by_hash: dict[str, torch.Tensor] = {}
            for push, tensor in zip(
                stage_pushes,
                staged.tensors,
                strict=True,
            ):
                staged_by_hash.setdefault(push.spec.mm_hash, tensor)
            store_puts = []
            for identifier, ticket in store_saves.items():
                payload = staged_by_hash[identifier]
                store_puts.append(
                    PreparedStorePut(
                        ticket=ticket,
                        addresses=[
                            bounce_address + header_offsets[identifier],
                            payload.data_ptr(),
                        ],
                        sizes=[header_size, payload.nbytes],
                        owners=[bounce_arena, lease, payload, ticket.tensor],
                    )
                )
            return _AcquiredTransferWave(fragments, [], lease, store_puts)
        except BaseException:
            bounce_arena.release(lease)
            raise

    def _push_store_independent_staging(
        self,
        pushes: list[ProducerPushRecord],
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]],
        ordered_pushes: list[ProducerPushRecord],
    ) -> tuple[Any, _AcquiredTransferWave, list[PreparedStoreSave]]:
        """Stage the union of EC sources and accepted independent Store PUTs."""
        deferred_store = cast(AscendDeferredStoreBackend, self._deferred_store)
        prepared: dict[str, PreparedStoreSave] = {}
        for push in pushes:
            identifier = push.spec.mm_hash
            if identifier in prepared:
                continue
            ticket = deferred_store.prepare_fallback(identifier)
            if ticket is not None:
                prepared[identifier] = ticket

        stage_pushes = list(ordered_pushes)
        staged_ids = {push.spec.transfer_id for push in stage_pushes}
        for push in pushes:
            if (
                push.spec.mm_hash in prepared
                and push.spec.transfer_id not in staged_ids
            ):
                stage_pushes.append(push)
                staged_ids.add(push.spec.transfer_id)

        if not stage_pushes:
            return (
                None,
                _AcquiredTransferWave([], [], None, []),
                list(prepared.values()),
            )

        tensors = [cast(torch.Tensor, push.source_tensor) for push in stage_pushes]
        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        staged = producer_memory.stage_wait(tensors)
        if staged is None:
            for ticket in prepared.values():
                deferred_store.abort_prepared(ticket)
            raise RuntimeError(
                "one independent-Store encoder batch is larger than the "
                "producer staging pool"
            )
        try:
            acquired = self._acquire_staged_wave(
                stage_pushes,
                staged,
                prepared,
            )
        except BaseException:
            producer_memory.release(staged)
            for ticket in prepared.values():
                deferred_store.abort_prepared(ticket)
            raise
        reusable = self._write_transfer_wave(
            stage_pushes,
            ready,
            acquired,
        )
        if not reusable:
            producer_memory.poison()
            for ticket in prepared.values():
                deferred_store.abort_prepared(ticket)
            raise RuntimeError(
                "joint EC/independent-Store staging completion is unconfirmed"
            )
        return staged, acquired, list(prepared.values())

    def _write_fallback_batch(
        self,
        ordered_pushes: list[ProducerPushRecord],
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]],
        tensors: list[torch.Tensor],
        store_saves: list[PreparedStoreSave | None],
    ) -> None:
        """Run joint waves that share prefix copies and suffix registrations."""
        deferred_store = getattr(self, "_deferred_store", None)
        if deferred_store is None:
            waves = _plan_transfer_waves(tensors, self._bounce_arena_size)
        else:
            waves = _plan_transfer_waves(
                tensors,
                self._bounce_arena_size,
                [save is not None for save in store_saves],
                _MOONCAKE_TENSOR_HEADER.size,
            )

        cursor = 0
        for wave in waves:
            next_cursor = cursor + len(wave.sources)
            wave_pushes = ordered_pushes[cursor:next_cursor]
            wave_saves = store_saves[cursor:next_cursor]
            try:
                acquired = (
                    self._acquire_transfer_wave(wave, wave_saves)
                    if deferred_store is not None
                    else self._acquire_transfer_wave(wave)
                )
            except BaseException:
                if deferred_store is not None:
                    for ticket in wave_saves:
                        if ticket is not None:
                            deferred_store.abort_prepared(ticket)
                raise

            reusable = False
            try:
                reusable = (
                    self._write_transfer_wave(
                        wave_pushes,
                        ready,
                        acquired,
                    )
                    is not False
                )
            finally:
                if reusable:
                    self._release_transfer_wave(acquired)
            if not reusable:
                raise RuntimeError(
                    "joint EC/Store fallback completion is unconfirmed"
                )
            cursor = next_cursor
        assert cursor == len(ordered_pushes)

    def _push_batch(self, pushes: list[ProducerPushRecord]) -> None:
        started_at = time.monotonic()
        deferred_store = getattr(self, "_deferred_store", None)
        fallback_store_saves: list[PreparedStoreSave | None] = []
        failure: Exception | None = None
        try:
            ready, ordered_pushes = self._resolve_writable_pushes(
                pushes,
                started_at,
            )

            if getattr(self, "_use_store_independent_te", False):
                staged, acquired, prepared = (
                    self._push_store_independent_staging(
                        pushes,
                        ready,
                        ordered_pushes,
                    )
                )
                fallback_store_saves = prepared
                if staged is not None:
                    producer_memory = cast(
                        AscendProducerMemoryPool,
                        self._producer_memory,
                    )
                    self._release_transfer_wave(acquired)
                    producer_memory.release(staged)

                self._producer_pushes.begin_notifying(pushes)
                self._notify_completions(ready)
                self._producer_pushes.complete(pushes)
                return

            if ready:
                tensors = [
                    cast(torch.Tensor, push.source_tensor)
                    for push in ordered_pushes
                ]
                staged = self._producer_memory.stage(tensors)
                if staged is not None:
                    self._submit_original_store_outputs(pushes)
                    try:
                        acquired = self._acquire_staged_wave(
                            ordered_pushes,
                            staged,
                            {},
                        )
                        if not self._write_transfer_wave(
                            ordered_pushes,
                            ready,
                            acquired,
                        ):
                            raise RuntimeError(
                                "EC staging completion is unconfirmed"
                            )
                    finally:
                        self._producer_memory.release(staged)
                else:
                    if self._bounce_arena_size == 0:
                        raise RuntimeError(
                            "Ascend Mooncake producer staging pool cannot fit "
                            "the transfer batch and direct/bounce fallback is "
                            f"disabled because {_BOUNCE_ARENA_CONFIG_KEY}=0"
                        )
                    store_saves = (
                        [
                            deferred_store.prepare_fallback(push.spec.mm_hash)
                            for push in ordered_pushes
                        ]
                        if deferred_store is not None
                        else [None] * len(ordered_pushes)
                    )
                    fallback_store_saves = store_saves
                    ordered_ids = {
                        push.spec.transfer_id for push in ordered_pushes
                    }
                    self._submit_original_store_outputs(
                        [
                            push
                            for push in pushes
                            if push.spec.transfer_id not in ordered_ids
                        ]
                    )
                    self._write_fallback_batch(
                        ordered_pushes,
                        ready,
                        tensors,
                        store_saves,
                    )

            else:
                self._submit_original_store_outputs(pushes)

            self._producer_pushes.begin_notifying(pushes)
            self._notify_completions(ready)
            self._producer_pushes.complete(pushes)
        except Exception as exc:
            failure = exc
            if deferred_store is not None:
                for ticket in fallback_store_saves:
                    if ticket is not None:
                        deferred_store.abort_prepared(ticket)
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
