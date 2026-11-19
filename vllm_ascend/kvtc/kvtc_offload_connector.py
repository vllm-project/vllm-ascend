"""KVTC scheduler and Ascend worker adapter for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    DEFAULT_CPU_CAPACITY_BYTES,
    SimpleCPUOffloadConnector,
)
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

from .manager import KVTCOffloadScheduler
from .worker import KVTCOffloadWorker

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class AscendKVTCOffloadConnector(SimpleCPUOffloadConnector):
    """Reuse base connector plumbing, then replace its role-specific handler."""

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str:
        return "NHD"

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        if extra.get("kv_offload_backend", "cpu") != "cpu":
            raise ValueError("AscendKVTCOffloadConnector requires the CPU offload backend")
        if extra.get("num_cpu_blocks") is not None:
            raise ValueError("KVTC derives its raw and compressed pool sizes from the CPU byte budget")
        super().__init__(vllm_config, role, kv_cache_config)
        if self.worker_handler is None and self.scheduler_manager is None:
            return
        if role == KVConnectorRole.WORKER:
            cpu_capacity = self.worker_handler.cpu_capacity_bytes
            self.worker_handler = KVTCOffloadWorker(vllm_config, kv_cache_config, cpu_capacity)
            logger.info(
                "[KVTC] worker CPU capacity per rank=%.2f GiB",
                cpu_capacity / (1024**3),
            )
        elif role == KVConnectorRole.SCHEDULER:
            extra = self._kv_transfer_config.kv_connector_extra_config or {}
            capacity = int(extra.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES))
            per_rank = capacity // vllm_config.parallel_config.world_size
            if "cpu_bytes_to_use_per_rank" in extra:
                per_rank = int(extra["cpu_bytes_to_use_per_rank"])
            scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
            self.scheduler_manager = KVTCOffloadScheduler(
                vllm_config,
                kv_cache_config,
                per_rank,
                scheduler_block_size=scheduler_block_size,
                hash_block_size=hash_block_size,
                lazy_offload=bool(extra.get("lazy_offload", False)),
            )
            logger.info("[KVTC] scheduler initialized")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.worker_handler is not None:
            self.worker_handler.register_kv_caches(kv_caches)
