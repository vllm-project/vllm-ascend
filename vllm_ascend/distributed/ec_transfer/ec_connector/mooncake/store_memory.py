# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Bind one fixed pool to the selected Store engine and validate I/O ranges."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    EmbeddingStoreOperationError,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import AscendMooncakeTransfer


class AscendStoreTopology(Enum):
    SHARED = auto()
    INDEPENDENT = auto()
    FABRIC_MEM = auto()


class AscendStoreMemory:
    """Own Store registrations while borrowing the physical producer slab."""

    def __init__(self, store: Any, transfer: AscendMooncakeTransfer, topology: AscendStoreTopology) -> None:
        self.store = store
        self.transfer = transfer
        self.topology = topology
        self.staging_owner: torch.Tensor | None = None

    def bind_staging(self, tensor: torch.Tensor) -> None:
        """Borrow the EC slab; only an independent engine registers it again."""
        if self.staging_owner is tensor:
            return
        if self.staging_owner is not None:
            raise RuntimeError("A different producer staging slab is already bound")
        if self.topology is AscendStoreTopology.INDEPENDENT:
            status = self.store.register_buffer(tensor.data_ptr(), tensor.nbytes)
            if status != 0:
                raise EmbeddingStoreOperationError(f"Failed to register producer staging slab with Store: {status}")
        self.staging_owner = tensor

    def validate_buffers(self, addresses: list[int], sizes: list[int]) -> None:
        if self.staging_owner is None:
            raise EmbeddingStoreOperationError("Store staging pool is not bound")
        start = self.staging_owner.data_ptr()
        end = start + self.staging_owner.nbytes
        if len(addresses) != len(sizes) or any(
            size <= 0 or address < start or address + size > end for address, size in zip(addresses, sizes, strict=True)
        ):
            raise EmbeddingStoreOperationError("Store I/O must use the fixed staging pool")
