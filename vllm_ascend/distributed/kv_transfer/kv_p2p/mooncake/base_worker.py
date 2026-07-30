# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Worker-side interface for Mooncake KV transfer connectors.

This module intentionally contains no transfer implementation. Memory
registration, Mooncake engine lifecycle, rank mapping, and D2D execution belong
in concrete worker implementations.
"""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)

from .metadata import MooncakeConnectorMetadata

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class MooncakeBaseConnectorWorker:
    """Common worker-side contract for Mooncake connectors."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.kv_cache_config = kv_cache_config
        self.xfer_handshake_metadata: (
            KVConnectorHandshakeMetadata | None
        ) = None

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Register model KV cache tensors for D2D transfer."""
        raise NotImplementedError

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return requests with completed receive and send operations."""
        raise NotImplementedError

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return local block IDs whose KV load failed."""
        raise NotImplementedError

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return and reset transfer statistics for the current interval."""
        raise NotImplementedError

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        """Start D2D KV loading described by scheduler metadata."""
        raise NotImplementedError


__all__ = ["MooncakeBaseConnectorWorker"]
