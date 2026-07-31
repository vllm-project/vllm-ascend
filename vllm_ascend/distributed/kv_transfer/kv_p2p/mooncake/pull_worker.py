# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Worker-side implementation entry point for Mooncake pull transfers."""

from typing import TYPE_CHECKING

from vllm.config import VllmConfig

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class MooncakePullConnectorWorker(MooncakeBaseConnectorWorker):
    """Worker-side framework for Mooncake pull transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        """Start the pull operations described by scheduler metadata."""
        raise NotImplementedError

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return requests with completed receive and send operations."""
        raise NotImplementedError

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return local block IDs whose pull operations failed."""
        raise NotImplementedError


__all__ = ["MooncakePullConnectorWorker"]
