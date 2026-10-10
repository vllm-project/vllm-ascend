# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Shared Ascend source planning and bounce-arena lease coordination.

Ascend direct registration requires a 2 MiB-aligned start address. A source
tensor may begin before the first aligned address contained in its storage,
so registering the tensor in place could cover memory that the storage does
not own. The fallback path therefore splits such a source into an
unregistrable prefix and an aligned suffix. It copies the prefix into a
pre-registered bounce arena and registers the suffix directly.

Requests use the staging pool first. If a batch does not fit, transfer
waves lease space from the shared bounce arena for their prefixes and use
direct registration for their suffixes. Each fragment retains its
destination offset, so Mooncake reconstructs the original tensor byte order
in the consumer pool. A wave releases its bounce lease only after all
writes using those bytes have finished.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ContiguousAllocator,
)
from vllm.utils.math_utils import round_down, round_up

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ASCEND_DIRECT_MEMORY_ALIGNMENT = 2 * 1024 * 1024  # 2 MiB
_BOUNCE_ARENA_CONFIG_KEY = "ascend_mooncake_bounce_arena_size"
_DEFAULT_BOUNCE_LIMIT = 128


@dataclass(frozen=True)
class _BounceLease:
    offset: int
    nbytes: int
    allocated_nbytes: int


class _BounceLeaseManager:
    """Manage complete wave leases over one existing bounce arena."""

    def __init__(self, capacity: int, alignment: int = 256) -> None:
        self.capacity = capacity
        self.alignment = alignment
        self._regions = ContiguousAllocator(capacity, alignment)
        self._condition = threading.Condition()
        self._waiters: deque[object] = deque()
        self._active: dict[int, int] = {}

    def acquire(self, nbytes: int) -> _BounceLease | None:
        if nbytes == 0:
            return None

        allocated_nbytes = round_up(nbytes, self.alignment)
        if allocated_nbytes > self.capacity:
            raise ValueError(
                f"bounce lease requires {allocated_nbytes} bytes but arena capacity is {self.capacity} bytes"
            )

        waiter = object()

        with self._condition:
            self._waiters.append(waiter)
            queued = True

            try:
                while True:
                    if self._waiters[0] is waiter:
                        region = self._regions.allocate(nbytes)

                        if region is not None:
                            offset, size = region
                            self._waiters.popleft()
                            queued = False
                            self._active[offset] = size
                            self._condition.notify_all()

                            return _BounceLease(
                                offset=offset,
                                nbytes=nbytes,
                                allocated_nbytes=size,
                            )
                    self._condition.wait()
            finally:
                if queued:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()

    def release(self, lease: _BounceLease | None) -> None:
        if lease is None:
            return

        with self._condition:
            allocated_nbytes = self._active.get(lease.offset)

            if allocated_nbytes != lease.allocated_nbytes:
                raise ValueError("bounce lease is not active")

            del self._active[lease.offset]
            self._regions.free(
                lease.offset,
                lease.allocated_nbytes,
            )
            self._condition.notify_all()


class _BounceTensorProvider(Protocol):
    @property
    def bounce_tensor(self) -> torch.Tensor | None: ...


class AscendBounceArena:
    """Share one physical bounce view, lease manager, and thread-local copy-stream state."""

    def __init__(
        self,
        provider: _BounceTensorProvider,
        capacity: int,
        thread_local: threading.local,
    ) -> None:
        self._provider = provider
        self._leases = _BounceLeaseManager(capacity)
        self._local = thread_local

    @property
    def tensor(self) -> torch.Tensor | None:
        return self._provider.bounce_tensor

    def acquire(self, nbytes: int) -> _BounceLease | None:
        return self._leases.acquire(nbytes)

    def release(self, lease: _BounceLease | None) -> None:
        self._leases.release(lease)

    def copy(
        self,
        lease: _BounceLease,
        copies: list[tuple[torch.Tensor, int, int]],
    ) -> int:
        """Pack prefixes into a lease and return its base address.

        Each copy contains source, offset within the lease, and byte count.
        """
        bounce = self.tensor
        if bounce is None:
            raise RuntimeError("Mooncake bounce arena is not prepared")
        assert copies

        stream = getattr(self._local, "stream", None)
        if stream is None:
            stream = torch.npu.Stream(device=bounce.device)
            self._local.stream = stream

        # The same source-readiness event also guards this fallback copy.
        with torch.npu.stream(stream):
            for source, bounce_offset, nbytes in copies:
                assert bounce_offset >= 0
                assert 0 < nbytes <= source.nbytes
                assert bounce_offset + nbytes <= lease.nbytes

                source_prefix = source.view(torch.uint8).view(-1).narrow(0, 0, nbytes)
                destination = bounce.narrow(
                    0,
                    lease.offset + bounce_offset,
                    nbytes,
                )
                destination.copy_(source_prefix, non_blocking=True)

        stream.synchronize()
        return bounce.data_ptr() + lease.offset


@dataclass(frozen=True)
class _SourceFragmentPlan:
    owner: torch.Tensor
    prefix_nbytes: int
    direct_address: int | None
    direct_nbytes: int
    registration_address: int | None
    registration_nbytes: int


@dataclass(frozen=True)
class _RegistrationRangePlan:
    address: int
    nbytes: int
    owners: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _WaveSourcePlan:
    source: _SourceFragmentPlan
    bounce_offset: int | None


@dataclass(frozen=True)
class _TransferWavePlan:
    sources: tuple[_WaveSourcePlan, ...]
    registration_ranges: tuple[_RegistrationRangePlan, ...]
    bounce_nbytes: int


@dataclass(frozen=True)
class _TransferFragmentPlan:
    """One physical Mooncake write fragment of a logical source."""

    source_index: int
    source_address: int
    destination_offset: int
    nbytes: int


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
    if value == 0:
        return 0
    if value < ASCEND_DIRECT_MEMORY_ALIGNMENT:
        raise ValueError(
            f"{_BOUNCE_ARENA_CONFIG_KEY} must be 0 to disable "
            f"fallback or at least {ASCEND_DIRECT_MEMORY_ALIGNMENT} bytes"
        )

    return round_up(value, ASCEND_DIRECT_MEMORY_ALIGNMENT)


def _plan_source(tensor: torch.Tensor) -> _SourceFragmentPlan:
    storage = tensor.untyped_storage()
    storage_start = storage.data_ptr()
    source_start = tensor.data_ptr()
    source_end = source_start + tensor.nbytes

    down = round_down(
        source_start,
        ASCEND_DIRECT_MEMORY_ALIGNMENT,
    )

    if down >= storage_start:
        return _SourceFragmentPlan(
            owner=tensor,
            prefix_nbytes=0,
            direct_address=source_start,
            direct_nbytes=tensor.nbytes,
            registration_address=down,
            registration_nbytes=source_end - down,
        )
    else:
        split = min(
            round_up(source_start, ASCEND_DIRECT_MEMORY_ALIGNMENT),
            source_end,
        )
        prefix_nbytes = split - source_start
        if split == source_end:
            return _SourceFragmentPlan(
                owner=tensor,
                prefix_nbytes=prefix_nbytes,
                direct_address=None,
                direct_nbytes=0,
                registration_address=None,
                registration_nbytes=0,
            )
        else:
            return _SourceFragmentPlan(
                owner=tensor,
                prefix_nbytes=prefix_nbytes,
                direct_address=split,
                direct_nbytes=source_end - split,
                registration_address=split,
                registration_nbytes=source_end - split,
            )


def _plan_registration_ranges(
    sources: list[_SourceFragmentPlan],
) -> list[_RegistrationRangePlan]:
    by_storage: dict[
        int,
        list[tuple[int, int, torch.Tensor]],
    ] = {}
    for source in sources:
        address = source.registration_address
        if address is None:
            assert source.registration_nbytes == 0
            continue

        assert source.registration_nbytes > 0
        storage_start = source.owner.untyped_storage().data_ptr()
        end = address + source.registration_nbytes

        by_storage.setdefault(storage_start, []).append((address, end, source.owner))

    merged: list[_RegistrationRangePlan] = []
    for ranges in by_storage.values():
        ranges.sort(key=lambda item: item[0])

        current_start, current_end, first_owner = ranges[0]
        current_owners = [first_owner]

        for start, end, owner in ranges[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                current_owners.append(owner)
                continue

            merged.append(
                _RegistrationRangePlan(
                    address=current_start,
                    nbytes=current_end - current_start,
                    owners=tuple(current_owners),
                )
            )
            current_start = start
            current_end = end
            current_owners = [owner]

        merged.append(
            _RegistrationRangePlan(
                address=current_start,
                nbytes=current_end - current_start,
                owners=tuple(current_owners),
            )
        )

    return merged


def _make_transfer_wave(
    sources: list[_WaveSourcePlan],
    bounce_nbytes: int,
) -> _TransferWavePlan:
    return _TransferWavePlan(
        sources=tuple(sources),
        registration_ranges=tuple(_plan_registration_ranges([item.source for item in sources])),
        bounce_nbytes=bounce_nbytes,
    )


def _plan_transfer_waves(
    tensors: list[torch.Tensor],
    bounce_capacity: int,
) -> list[_TransferWavePlan]:
    waves: list[_TransferWavePlan] = []
    wave_sources: list[_WaveSourcePlan] = []
    wave_bounce_nbytes = 0

    for tensor in tensors:
        source = _plan_source(tensor)
        if wave_sources and (wave_bounce_nbytes + source.prefix_nbytes > bounce_capacity):
            waves.append(_make_transfer_wave(wave_sources, wave_bounce_nbytes))
            wave_sources = []
            wave_bounce_nbytes = 0

        wave_sources.append(
            _WaveSourcePlan(
                source=source,
                bounce_offset=(wave_bounce_nbytes if source.prefix_nbytes else None),
            )
        )
        wave_bounce_nbytes += source.prefix_nbytes

    if wave_sources:
        waves.append(_make_transfer_wave(wave_sources, wave_bounce_nbytes))
    return waves


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
