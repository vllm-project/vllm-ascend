# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend Direct initialization for an ECMooncake TransferEngine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    _REGISTERED_MEMORY_ALIGNMENT,
)

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine


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
            offset = (-storage_start) % _REGISTERED_MEMORY_ALIGNMENT
            nbytes = storage.nbytes() - offset
            region = torch.empty(
                0,
                dtype=torch.uint8,
                device=storage.device,
            ).set_(storage, offset, (nbytes,), (1,))
            aligned_regions.append(region)

        return super().acquire_sources(aligned_regions)
