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

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    _RegistrationRangePlan,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine


@dataclass
class _DirectRegistration:
    nbytes: int
    owners: tuple[torch.Tensor, ...]
    users: int = 1


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
