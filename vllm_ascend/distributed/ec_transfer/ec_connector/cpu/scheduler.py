# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler delegate for Ascend CPU encoder-cache offloading."""

from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
    ECCPUScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    EmbeddingCache,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.step_tracker import (
    StepTracker,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    create_ascend_ec_shared_region,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AscendECCPUScheduler(ECCPUScheduler):
    """Use the Ascend-compatible shared region with upstream scheduling."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer

        self._region = create_ascend_ec_shared_region(vllm_config)
        # Block allocator + LRU eviction policy for the shared region.
        self._cache = EmbeddingCache(self._region.num_blocks)

        max_batches = vllm_config.max_concurrent_batches
        # Delays mark_ready until the GPU→mmap DMA is guaranteed complete.
        self._ready_tracker = StepTracker(max_batches)
        # Delays unpin until the mmap→GPU DMA is guaranteed complete.
        self._unpin_tracker = StepTracker(max_batches)

        # mm_hash → block IDs allocated this step for GPU→mmap saves.
        self._pending_saves: dict[str, list[int]] = {}
        # mm_hash → block IDs to load from mmap→GPU this step.
        self._pending_loads: dict[str, list[int]] = {}
