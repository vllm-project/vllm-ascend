# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Ascend NPU adaptation of vLLM's ``SimpleCPUOffloadConnector``.

The scheduler-side ``SimpleCPUOffloadScheduler`` is platform-agnostic
and reused as-is from upstream vLLM. The Ascend variant only swaps the
worker-side handler with an NPU-native implementation that uses
``aclrtMemcpyBatchAsync`` and ``torch.npu`` streams/events.
"""

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (  # noqa: E501
    SimpleCPUOffloadConnector,
)
from vllm.logger import init_logger

from vllm_ascend.simple_kv_offload.worker import SimpleCPUOffloadNPUWorker

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class AscendSimpleCPUOffloadConnector(SimpleCPUOffloadConnector):
    """NPU-flavored ``SimpleCPUOffloadConnector``.

    Inherits the full scheduler/worker plumbing from upstream and only
    replaces the CUDA worker handler with the NPU one. All other public
    APIs (``register_kv_caches``, ``bind_connector_metadata``,
    ``get_finished``, ``handle_preemptions``, every scheduler-side
    method, etc.) are inherited verbatim — they all route through
    ``self.worker_handler`` / ``self.scheduler_manager``.

    Why post-init swap (instead of skipping ``super().__init__``):
    ``SimpleCPUOffloadWorker.__init__`` and ``DmaCopyBackend.__init__``
    only assign ``None``/empty-field defaults — no CUDA resource is
    allocated until ``register_kv_caches`` runs. So letting the parent
    construct a transient CUDA worker and then replacing it costs
    nothing and keeps us free of duplicated configuration parsing.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)

        # If prefix caching is disabled, the parent leaves both handlers
        # as None and the connector is a no-op — nothing to swap.
        if role == KVConnectorRole.WORKER and self.worker_handler is not None:
            cpu_capacity = self.worker_handler.cpu_capacity_bytes
            self.worker_handler = SimpleCPUOffloadNPUWorker(
                vllm_config, kv_cache_config, cpu_capacity
            )
            logger.info(
                "AscendSimpleCPUOffloadConnector: swapped CUDA worker for "
                "NPU worker (per_rank=%.2f GB)",
                cpu_capacity / (1024**3),
            )
