# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ascend specs for vLLM v0.27 native KV-cache offloading."""

from __future__ import annotations

import torch
from typing_extensions import override
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec as _CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import (
    TieringOffloadingSpec as _TieringOffloadingSpec,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import (
    NPUOffloadingWorker,
)


class _NPUWorkerMixin:
    """Replace only the accelerator worker and upstream platform gate."""

    _worker: NPUOffloadingWorker | None

    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        raise NotImplementedError

    def get_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> OffloadingWorker:
        if self._worker is None:
            self._worker = self.create_worker(kv_caches)
        return self._worker


class NPUOffloadingSpec(_NPUWorkerMixin, _CPUOffloadingSpec):
    """Single CPU-tier offloading with pinned host tensors and NPU DMA."""

    @override
    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        # Keep the single-tier path on PyTorch's pinned allocator. Unlike CUDA,
        # Ascend has no public cudaHostRegister-equivalent for an arbitrary
        # mmap buffer, and the pinned tensor path has the best proven H2D/D2H
        # performance. Consequently replicated_layout remains safely disabled
        # for this spec by the upstream _uses_shared_region() gate.
        return NPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
        )


class NPUTieringOffloadingSpec(_NPUWorkerMixin, _TieringOffloadingSpec):
    """CPU primary tier plus vLLM secondary tiers on Ascend NPU."""

    def _uses_shared_region(self) -> bool:
        # Unlike the CPU-only NPU spec, tiering always connects every worker
        # and the scheduler-side primary tier through SharedOffloadRegion.
        # Advertising that fact lets upstream safely enable its single-copy
        # layout for configurations it has certified as byte-replicated
        # (currently pure MLA under the supported TP topology).
        return True

    @override
    def create_worker(
        self,
        kv_caches: CanonicalKVCaches,
    ) -> NPUOffloadingWorker:
        world_size = self.config.parallel.world_size
        if self.replicated_layout:
            rank = 0
        else:
            # engine_id identifies the DP replica; fold the process-local
            # physical device index into that replica's mmap slot range.
            rank = int(torch.npu.current_device()) % world_size

        worker_mmap = SharedOffloadRegion(
            engine_id=self._engine_id,
            num_blocks=self.num_blocks,
            rank=rank,
            kv_bytes_per_block=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )
        return NPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
            mmap_region=worker_mmap,
        )
