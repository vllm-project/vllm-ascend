# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""NPU worker for vLLM's Mooncake encoder-cache connector."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import (
    MooncakeECConfig,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    ProducerMemoryPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.producer import (
    ProducerPushRecord,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    _RESERVATION_REFRESH_SECONDS,
    ECMooncakeWorker,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    AscendConsumerMemoryPool,
    AscendContiguousAllocator,
    AscendProducerAllocator,
    AscendProducerMemoryPool,
    _BounceLease,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
    _plan_transfer_waves,
    _TransferWavePlan,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_DEFAULT_BOUNCE_LIMIT = 128
_BOUNCE_ARENA_CONFIG_KEY = "ascend_mooncake_bounce_arena_size"
logger = init_logger(__name__)


def _resolve_bounce_arena_size(vllm_config: VllmConfig) -> int:
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None

    extra_config = ec_config.ec_connector_extra_config

    if _BOUNCE_ARENA_CONFIG_KEY not in extra_config:
        quantum_count = min(
            _DEFAULT_BOUNCE_LIMIT,
            vllm_config.scheduler_config.max_num_seqs,
        )
        return ASCEND_DIRECT_MEMORY_ALIGNMENT * quantum_count

    value = extra_config[_BOUNCE_ARENA_CONFIG_KEY]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{_BOUNCE_ARENA_CONFIG_KEY} must be an integer")
    if value < ASCEND_DIRECT_MEMORY_ALIGNMENT:
        raise ValueError(
            f"{_BOUNCE_ARENA_CONFIG_KEY} must be at least "
            f"{ASCEND_DIRECT_MEMORY_ALIGNMENT} bytes"
        )

    return round_up(value, ASCEND_DIRECT_MEMORY_ALIGNMENT)


@dataclass(frozen=True)
class _TransferFragmentPlan:
    """One physical Mooncake write fragment of a logical source."""

    source_index: int
    source_address: int
    destination_offset: int
    nbytes: int


@dataclass(frozen=True)
class _AcquiredTransferWave:
    fragments: list[_TransferFragmentPlan]
    registration_addresses: list[int]
    bounce_lease: _BounceLease | None


def _flatten_transfer_wave(
    wave: _TransferWavePlan,
    bounce_address: int | None,
) -> list[_TransferFragmentPlan]:
    fragments: list[_TransferFragmentPlan] = []

    for source_index, wave_source in enumerate(wave.sources):
        source = wave_source.source

        if source.prefix_nbytes > 0:
            assert bounce_address is not None
            assert wave_source.bounce_offset is not None

            fragments.append(
                _TransferFragmentPlan(
                    source_index=source_index,
                    source_address=bounce_address + wave_source.bounce_offset,
                    destination_offset=0,
                    nbytes=source.prefix_nbytes,
                )
            )

        if source.direct_nbytes > 0:
            assert source.direct_address is not None

            fragments.append(
                _TransferFragmentPlan(
                    source_index=source_index,
                    source_address=source.direct_address,
                    destination_offset=source.prefix_nbytes,
                    nbytes=source.direct_nbytes,
                )
            )

    return fragments


class AscendECMooncakeWorker(ECMooncakeWorker):
    """Reuse upstream orchestration with Ascend-specific primitives."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None

        resolved_bounce_arena_size = _resolve_bounce_arena_size(vllm_config)

        self._bounce_arena_size = (
            resolved_bounce_arena_size if ec_config.is_ec_producer else 0
        )

        super().__init__(vllm_config)

    def _make_config(self, vllm_config: VllmConfig) -> MooncakeECConfig:
        config = super()._make_config(vllm_config)

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

    def _make_transfer(self, hostname: str, protocol: str) -> AscendMooncakeTransfer:
        if protocol != "ascend":
            raise ValueError("Ascend ECMooncakeConnector requires mooncake_protocol='ascend'")
        device_index = torch.npu.current_device()
        return AscendMooncakeTransfer(hostname, device_index)

    def _make_consumer_memory(self, capacity: int, transfer: MooncakeTransfer) -> ConsumerMemoryPool:
        return AscendConsumerMemoryPool(
            capacity,
            transfer,
            allocator=AscendContiguousAllocator(capacity),
        )

    def _make_producer_memory(self, capacity: int, transfer: MooncakeTransfer) -> ProducerMemoryPool:
        return AscendProducerMemoryPool(
            capacity,
            transfer,
            allocator=AscendProducerAllocator(
                staging_capacity=capacity,
                bounce_capacity=self._bounce_arena_size,
            ),
        )

    def _acquire_transfer_wave(
        self,
        wave: _TransferWavePlan,
    ) -> _AcquiredTransferWave:
        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        transfer = cast(AscendMooncakeTransfer, self._transfer)
        lease = producer_memory.acquire_bounce(wave.bounce_nbytes)
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

                bounce_address = producer_memory.copy_to_bounce(lease, copies)

            fragments = _flatten_transfer_wave(wave, bounce_address)
            registration_addresses = transfer.acquire_registration_ranges(
                wave.registration_ranges
            )
            return _AcquiredTransferWave(
                fragments=fragments,
                registration_addresses=registration_addresses,
                bounce_lease=lease,
            )
        except Exception:
            if registration_addresses:
                transfer.release_registration_ranges(registration_addresses)
            producer_memory.release_bounce(lease)
            raise

    def _release_transfer_wave(self, acquired: _AcquiredTransferWave) -> None:
        producer_memory = cast(AscendProducerMemoryPool, self._producer_memory)
        transfer = cast(AscendMooncakeTransfer, self._transfer)

        try:
            transfer.release_registration_ranges(
                acquired.registration_addresses
            )
        finally:
            producer_memory.release_bounce(acquired.bounce_lease)

    def _write_transfer_wave(
        self,
        pushes: list[ProducerPushRecord],
        ready: list[tuple[ProducerPushRecord, dict[str, Any]]],
        acquired: _AcquiredTransferWave,
    ) -> None:
        source_index = {
            push.spec.transfer_id: index
            for index, push in enumerate(pushes)
        }
        fragments_by_source: list[list[_TransferFragmentPlan]] = [
            [] for _ in pushes
        ]
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
                by_session.setdefault(session, []).append(
                    (fragment, destination)
                )
            session_records.setdefault(session, {})[
                push.spec.transfer_id
            ] = push

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
                    and time.monotonic()
                    - float(shard.get("_received_at", started_at))
                    >= _RESERVATION_REFRESH_SECONDS
                ]
                if stale:
                    reservations = self._refresh_remote_reservations(
                        push.spec, reservations, push
                    )
                    self._producer_pushes.replace_reservations(
                        push, reservations
                    )
                self._producer_pushes.begin_writing(push)
                writable = [
                    shard
                    for shard in reservations
                    if not shard.get("cached", False)
                    and not shard.get("cancelled", False)
                    and shard.get("write", True)
                ]
                source = push.source_tensor
                assert source is not None
                for shard in writable:
                    if int(shard["nbytes"]) != source.nbytes:
                        raise RuntimeError(
                            "Reserved EC size does not match tensor for "
                            f"mm_hash={push.spec.mm_hash}"
                        )
                    ready.append((push, shard))
                    written_pushes.setdefault(push.spec.transfer_id, push)

            if ready:
                ordered_pushes = list(written_pushes.values())
                tensors = [
                    cast(torch.Tensor, push.source_tensor)
                    for push in ordered_pushes
                    if push.source_tensor is not None
                ]
                staged = self._producer_memory.stage(tensors)
                if staged is not None:
                    lengths = [tensor.nbytes for tensor in tensors]
                    addresses = [tensor.data_ptr() for tensor in staged.tensors]
                    try:
                        source_index = {
                            push.spec.transfer_id: index
                            for index, push in enumerate(ordered_pushes)
                        }
                        by_session: dict[str, list[tuple[int, int]]] = {}
                        session_records: dict[
                            str, dict[str, ProducerPushRecord]
                        ] = {}
                        for push, shard in ready:
                            session = str(shard["dst_session"])
                            by_session.setdefault(session, []).append(
                                (
                                    source_index[push.spec.transfer_id],
                                    int(shard["dst_ptr"]),
                                )
                            )
                            session_records.setdefault(session, {})[
                                push.spec.transfer_id
                            ] = push

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

                        writes: list[Callable[[], None]] = [
                            partial(write, *session) for session in sessions
                        ]
                        self._run_fanout(writes, track_write)
                    finally:
                        self._producer_memory.release(staged)
                else:
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

    def _record_source_ready_event(self, tensor: torch.Tensor) -> torch.Event | None:
        if tensor.device.type != "npu":
            return super()._record_source_ready_event(tensor)
        ready_event = torch.npu.Event()
        ready_event.record(torch.npu.current_stream(tensor.device))
        return ready_event
