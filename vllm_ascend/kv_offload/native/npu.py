# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend adaptation of vLLM's native CPU offloading spec."""

from __future__ import annotations

from collections.abc import MutableMapping

from typing_extensions import override
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec as _CPUOffloadingSpec

from vllm_ascend.kv_offload.native.cpu_npu import NPUOffloadingWorker


def _set_cpu_bytes_from_legacy_num_blocks(
    config: OffloadingConfig,
    alignment: int,
) -> None:
    """Translate the legacy Ascend block count to vLLM's byte capacity."""
    extra_config = config.extra_config
    if extra_config.get("cpu_bytes_to_use") is not None:
        return

    num_cpu_blocks = extra_config.get("num_cpu_blocks")
    if num_cpu_blocks is None:
        return
    num_cpu_blocks = int(num_cpu_blocks)
    if num_cpu_blocks <= 0:
        raise ValueError("num_cpu_blocks must be greater than 0")
    if not isinstance(extra_config, MutableMapping):
        raise TypeError("kv_connector_extra_config must be mutable when using the legacy num_cpu_blocks option")

    world_size = config.parallel.world_size
    worker_kv_bytes_per_block = config.worker_kv_bytes_per_block
    if worker_kv_bytes_per_block <= 0 or world_size <= 0:
        # The scheduler can construct the spec before worker cache sizing is
        # available. Match vLLM's zero-capacity initialization behavior.
        extra_config["cpu_bytes_to_use"] = 1
        return

    kv_bytes_per_chunk = worker_kv_bytes_per_block * world_size * config.cache.blocks_per_chunk
    aligned_kv_bytes_per_chunk = round_up(
        kv_bytes_per_chunk,
        alignment,
    )
    extra_config["cpu_bytes_to_use"] = num_cpu_blocks * aligned_kv_bytes_per_chunk


class NPUOffloadingSpec(_CPUOffloadingSpec):
    """Use vLLM's CPU manager with an Ascend-specific transfer worker."""

    def __init__(self, config: OffloadingConfig):
        _set_cpu_bytes_from_legacy_num_blocks(
            config,
            self.BLOCK_SIZE_ALIGNMENT,
        )
        super().__init__(config)
        self._npu_worker: NPUOffloadingWorker | None = None

    @override
    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        return NPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
        )

    @override
    def get_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> OffloadingWorker:
        if self._npu_worker is None:
            self._npu_worker = self.create_worker(kv_caches)
        return self._npu_worker


# Compatibility alias for configurations that load this module directly.
CPUOffloadingSpec = NPUOffloadingSpec
