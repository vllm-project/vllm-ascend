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

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine  # type: ignore[import-not-found]


class AscendMooncakeTransfer(MooncakeTransfer):
    """Use only fixed registered slabs with the process-wide Ascend engine."""

    _engine: TransferEngine | None

    def __init__(self, hostname: str, device_index: int) -> None:
        super().__init__(hostname, "ascend")
        self._device_index = device_index
        self._retained_owners: list[object] = []

    def _ensure_engine(self) -> TransferEngine:
        with self._engine_lock:
            if self._engine is None:
                torch.npu.set_device(self._device_index)
                self._engine = global_te.get_transfer_engine(self._hostname, device_name=None)
            return self._engine

    def acquire_sources(self, tensors: list[torch.Tensor]) -> list[int]:
        raise RuntimeError("Ascend Mooncake requires sources in the fixed staging pool")

    def acquire_registration_ranges(self, ranges: object) -> list[int]:
        raise RuntimeError("Dynamic Ascend Mooncake registration is disabled")

    def retain_owners(self, owners: list[object]) -> None:
        self._retained_owners.extend(owners)

    def register_memory(self, tensor: torch.Tensor) -> int:
        # Even an unconfirmed partial registration must keep its allocation.
        self.retain_owners([tensor])
        return super().register_memory(tensor)

    def unregister_memory(self, tensor: torch.Tensor) -> bool:
        # The shared engine has no public global channel-retirement barrier.
        self.retain_owners([tensor])
        return False

    def close(self) -> None:
        self._closed = True
