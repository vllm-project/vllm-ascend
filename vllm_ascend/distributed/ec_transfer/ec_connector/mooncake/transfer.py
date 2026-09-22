# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend Direct initialization for an ECMooncake TransferEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.logger import logger
from vllm.utils.math_utils import round_down, round_up

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine


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


@dataclass
class _DirectRegistration:
    nbytes: int
    owners: tuple[torch.Tensor, ...]
    users: int = 1


@dataclass(frozen=True)
class _WaveSourcePlan:
    source: _SourceFragmentPlan
    bounce_offset: int | None


@dataclass(frozen=True)
class _TransferWavePlan:
    sources: tuple[_WaveSourcePlan, ...]
    registration_ranges: tuple[_RegistrationRangePlan, ...]
    bounce_nbytes: int


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


class AscendMooncakeTransfer(MooncakeTransfer):
    """Use the process-wide Ascend engine with EC registration ownership."""

    def __init__(self, hostname: str, device_index: int) -> None:
        super().__init__(hostname, "ascend")
        self._device_index = device_index
        self._direct_registrations: dict[int, _DirectRegistration] = {}

    def _ensure_engine(self) -> TransferEngine:
        engine = self._engine
        if engine is not None:
            return engine

        with self._engine_lock:
            engine = self._engine
            if engine is None:
                torch.npu.set_device(self._device_index)
                engine = global_te.get_transfer_engine(
                    self._hostname,
                    device_name=None,
                )
                self._engine = engine
        return engine

    def acquire_registration_ranges(
        self,
        ranges: tuple[_RegistrationRangePlan, ...],
    ) -> list[int]:
        engine = self._ensure_engine()
        acquired: list[int] = []

        with self._registration_lock:
            try:
                for item in ranges:
                    entry = self._direct_registrations.get(item.address)

                    if entry is not None:
                        if entry.nbytes != item.nbytes:
                            raise RuntimeError("Mooncake direct registration range changed size")
                        entry.users += 1
                        acquired.append(item.address)
                        continue

                    status = engine.batch_register_memory(
                        [item.address],
                        [item.nbytes],
                    )
                    if status != 0:
                        raise RuntimeError(
                            f"Mooncake direct registration failed for address {item.address} with status {status}"
                        )

                    self._direct_registrations[item.address] = _DirectRegistration(
                        nbytes=item.nbytes,
                        owners=item.owners,
                    )
                    acquired.append(item.address)

            except Exception:
                for address in reversed(acquired):
                    entry = self._direct_registrations[address]
                    entry.users -= 1

                    if entry.users != 0:
                        continue

                    status = engine.unregister_memory(address)
                    if status == 0:
                        del self._direct_registrations[address]

                raise

        return acquired

    def release_registration_ranges(self, addresses: list[int]) -> bool:
        if not addresses:
            return True

        engine = self._ensure_engine()
        released = True

        with self._registration_lock:
            for address in addresses:
                entry = self._direct_registrations.get(address)
                if entry is None:
                    continue

                entry.users -= 1
                if entry.users > 0:
                    continue

                status = engine.unregister_memory(address)
                if status != 0:
                    released = False
                    continue
                del self._direct_registrations[address]
        return released

    def close(self) -> None:
        if self._closed:
            return

        super().close()
        engine = self._engine
        if engine is None:
            return

        with self._registration_lock:
            for address in list(self._direct_registrations):
                status = engine.unregister_memory(address)
                if status != 0:
                    logger.error(
                        "Mooncake direct registration cleanup failed for address %d with status %d",
                        address,
                        status,
                    )
                    continue
                del self._direct_registrations[address]

    def acquire_sources(self, tensors: list[torch.Tensor]) -> list[int]:
        """Register whole aligned storages only when they cover every source.

        The active fallback path plans direct ranges and bounced prefixes
        separately; this inherited API cannot represent a bounced prefix.
        """
        regions: dict[int, torch.UntypedStorage] = {}

        for tensor in tensors:
            storage = tensor.untyped_storage()
            storage_start = storage.data_ptr()
            offset = (-storage_start) % ASCEND_DIRECT_MEMORY_ALIGNMENT
            if storage.nbytes() <= offset:
                raise ValueError("Mooncake source storage has no 2 MiB-aligned bytes to register")
            aligned_start = storage_start + offset
            source_start = tensor.data_ptr()
            if source_start < aligned_start:
                raise ValueError("Mooncake source starts before the aligned registration region")
            regions[storage_start] = storage

        aligned_regions: list[torch.Tensor] = []

        for storage_start, storage in regions.items():
            offset = (-storage_start) % ASCEND_DIRECT_MEMORY_ALIGNMENT
            nbytes = storage.nbytes() - offset
            region = torch.empty(
                0,
                dtype=torch.uint8,
                device=storage.device,
            ).set_(storage, offset, (nbytes,), (1,))
            aligned_regions.append(region)

        return super().acquire_sources(aligned_regions)
