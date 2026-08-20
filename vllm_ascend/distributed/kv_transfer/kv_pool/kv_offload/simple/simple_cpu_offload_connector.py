"""Ascend NPU adaptation of vLLM's simple CPU offload connector.

The scheduler-side implementation is platform agnostic and reused directly
from vLLM. Only the worker-side handler is replaced with the NPU-specific
implementation in this package.
"""

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.logger import logger
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.simple.worker import (
    SimpleCPUOffloadNPUWorker,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class AscendSimpleCPUOffloadConnector(SimpleCPUOffloadConnector):
    """Reuse vLLM's connector while replacing its CUDA worker on NPU."""

    # vLLM imports are skipped by the project mypy command. Mirror the
    # upstream declaration without changing its runtime initialization.
    worker_handler: SimpleCPUOffloadWorker | None

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        # The upstream worker and its DMA backend only initialize Python state
        # here. Device resources are allocated later by register_kv_caches(),
        # so replacing the worker after super().__init__ has no CUDA side
        # effects and lets us continue to reuse upstream configuration parsing.
        super().__init__(vllm_config, role, kv_cache_config)

        # When prefix caching is disabled, upstream intentionally leaves the
        # worker unset and the connector behaves as a no-op.
        worker_handler = self.worker_handler
        if role == KVConnectorRole.WORKER and worker_handler is not None:
            cpu_capacity = worker_handler.cpu_capacity_bytes
            self.worker_handler = SimpleCPUOffloadNPUWorker(vllm_config, kv_cache_config, cpu_capacity)
            logger.info(
                "AscendSimpleCPUOffloadConnector: using NPU worker (per_rank=%.2f GB)",
                cpu_capacity / (1024**3),
            )
