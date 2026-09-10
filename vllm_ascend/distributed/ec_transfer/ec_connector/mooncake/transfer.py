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
from vllm.utils.math_utils import round_down, round_up

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
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
    """Bind the NPU before initializing an Ascend Direct transport."""

    def __init__(self, hostname: str, device_index: int) -> None:
        super().__init__(hostname, "ascend")
        self._device_index = device_index

    def _initialize_engine(self, engine: TransferEngine) -> int:
        torch.npu.set_device(self._device_index)
        return super()._initialize_engine(engine)

    def acquire_sources(self, tensors: list[torch.Tensor]) -> list[int]:
        regions: dict[int, torch.UntypedStorage] = {}

        for tensor in tensors:
            storage = tensor.untyped_storage()
            storage_start = storage.data_ptr()
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
