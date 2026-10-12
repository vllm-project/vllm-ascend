# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""NPU worker for vLLM's Mooncake encoder-cache connector."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

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
from vllm.logger import init_logger
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
logger = init_logger(__name__)


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

        resolved_bounce_arena_size = _resolve_bounce_arena_size(vllm_config)
        self._bounce_arena_size = resolved_bounce_arena_size if ec_config.is_ec_producer else 0

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
        )

        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        namespace, max_items, max_bytes, read_bytes = self._store_config
        ensure_mooncake_store_imported()
        producer_memory.ensure_prepared(torch.device(self._buffer_device))
        try:
            store_client = create_ascend_mooncake_embedding_store_client(
                producer_memory.bounce_arena,
                cast(AscendMooncakeTransfer, self._transfer),
                read_buffer_bytes=read_bytes,
            )
            try:
                output_store = MooncakeEmbeddingStoreBackend(
                    store_client,
                    namespace,
                    max_pending_items=max_items,
                    max_pending_bytes=max_bytes,
                )
            except BaseException:
                store_client.close()
                raise

            self._output_store = output_store
            try:
                super().start_services()
            except BaseException:
                self._output_store = None
                output_store.shutdown()
                raise
        except BaseException:
            producer_memory.close()
            raise

    def close(self) -> None:
        if self._shutdown:
            return
        if self._output_store is not None:
            self._output_store.shutdown()
            self._output_store = None
        super().close()

    def _reserve_push_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not getattr(self._control_thread, "device_set", False):
            torch.npu.set_device(torch.device(self._buffer_device))
            self._control_thread.device_set = True
        return super()._reserve_push_destination(payload)

    def _acquire_transfer_wave(
        self,
        wave: _TransferWavePlan,
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
                for wave_source in wave.sources:
                    source = wave_source.source
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
            return _AcquiredTransferWave(
                fragments=fragments,
                registration_addresses=registration_addresses,
                bounce_lease=lease,
            )
        except Exception:
            if registration_addresses:
                transfer.release_registration_ranges(registration_addresses)
            bounce_arena.release(lease)
            raise

    def _release_transfer_wave(self, acquired: _AcquiredTransferWave) -> None:
        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        transfer = cast(AscendMooncakeTransfer, self._transfer)

        try:
            transfer.release_registration_ranges(acquired.registration_addresses)
        finally:
            producer_memory.bounce_arena.release(acquired.bounce_lease)

    def _write_transfer_wave(
        self,
        pushes: list[ProducerPushRecord],
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]],
        acquired: _AcquiredTransferWave,
    ) -> None:
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

        sessions = list(by_session.items())

        def track_write(index: int, future: Future[None]) -> None:
            session = sessions[index][0]
            self._producer_pushes.track_shard_futures(
                list(session_records[session].values()),
                [future],
            )

        writes = [partial(write, *session) for session in sessions]
        self._run_fanout(writes, track_write)

    def _push_batch(self, pushes: list[ProducerPushRecord]) -> None:
        started_at = time.monotonic()
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]] = []
        written_pushes: dict[str, ProducerPushRecord] = {}
        failure: Exception | None = None
        try:
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
                    reservations = self._refresh_remote_reservations(push.spec, reservations, push)
                    self._producer_pushes.replace_reservations(push, reservations)
                self._producer_pushes.begin_writing(push)
                writable = [
                    shard
                    for shard in reservations
                    if not shard.get("cached", False) and not shard.get("cancelled", False) and shard.get("write", True)
                ]
                source = push.source_tensor
                assert source is not None
                for shard in writable:
                    if int(shard["nbytes"]) != source.nbytes:
                        raise RuntimeError(f"Reserved EC size does not match tensor for mm_hash={push.spec.mm_hash}")
                    ready.append((push, shard))
                    written_pushes.setdefault(push.spec.transfer_id, push)

            if ready:
                ordered_pushes = list(written_pushes.values())
                tensors = [
                    cast(torch.Tensor, push.source_tensor) for push in ordered_pushes if push.source_tensor is not None
                ]
                staged = self._producer_memory.stage(tensors)
                if staged is not None:
                    lengths = [tensor.nbytes for tensor in tensors]
                    addresses = [tensor.data_ptr() for tensor in staged.tensors]
                    try:
                        source_index = {push.spec.transfer_id: index for index, push in enumerate(ordered_pushes)}
                        by_session: dict[str, list[tuple[int, int]]] = {}
                        session_records: dict[str, dict[str, ProducerPushRecord]] = {}
                        for push, shard in ready:
                            session = str(shard["dst_session"])
                            by_session.setdefault(session, []).append(
                                (
                                    source_index[push.spec.transfer_id],
                                    int(shard["dst_ptr"]),
                                )
                            )
                            session_records.setdefault(session, {})[push.spec.transfer_id] = push

                        def write(
                            session: str,
                            items: list[tuple[int, int]],
                        ) -> None:
                            self._transfer.write(
                                session,
                                [addresses[index] for index, _ in items],
                                [dst for _, dst in items],
                                [lengths[index] for index, _ in items],
                            )

                        sessions = list(by_session.items())

                        def track_write(
                            index: int,
                            future: Future[None],
                        ) -> None:
                            session = sessions[index][0]
                            self._producer_pushes.track_shard_futures(
                                list(session_records[session].values()),
                                [future],
                            )

                        writes: list[Callable[[], None]] = [partial(write, *session) for session in sessions]
                        self._run_fanout(writes, track_write)
                    finally:
                        self._producer_memory.release(staged)
                else:
                    if self._bounce_arena_size == 0:
                        raise RuntimeError(
                            "Ascend Mooncake producer staging pool cannot fit "
                            "the transfer batch and direct/bounce fallback is "
                            f"disabled because {_BOUNCE_ARENA_CONFIG_KEY}=0"
                        )
                    waves = _plan_transfer_waves(
                        tensors,
                        self._bounce_arena_size,
                    )
                    cursor = 0
                    for wave in waves:
                        next_cursor = cursor + len(wave.sources)
                        wave_pushes = ordered_pushes[cursor:next_cursor]
                        acquired = self._acquire_transfer_wave(wave)
                        try:
                            self._write_transfer_wave(
                                wave_pushes,
                                ready,
                                acquired,
                            )
                        finally:
                            self._release_transfer_wave(acquired)
                        cursor = next_cursor
                    assert cursor == len(ordered_pushes)

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
