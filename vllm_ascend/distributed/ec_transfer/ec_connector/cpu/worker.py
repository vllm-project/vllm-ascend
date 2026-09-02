# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""NPU worker for vLLM's mmap-backed CPU encoder-cache connector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import (
    ECCPUTransferDirection,
    ECCPUWorker,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker.descriptor_buffers import DescriptorBuffers
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_DIRECTION_H2D = 0
_DIRECTION_D2H = 1


def _supports_eccpu_offload() -> bool:
    return bool(torch.ops._C_ascend.supports_eccpu_offload())


def _register_pinned_host_mmap(blocks: torch.Tensor) -> None:
    torch.ops._C_ascend.register_pinned_host_mmap(blocks)


def _unregister_pinned_host_mmap(blocks: torch.Tensor) -> None:
    torch.ops._C_ascend.unregister_pinned_host_mmap(blocks)


def _swap_blocks_batch(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    direction: int,
) -> None:
    torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, direction)


class AscendECCPUWorker(ECCPUWorker):
    """Use pinned shared mmap and CANN batched DMA for encoder caches.

    Shared-region lifecycle, scheduling and cache bookkeeping stay upstream;
    only host registration, NPU event creation and CANN DMA are replaced.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self._mmap_pinned = False
        try:
            super().__init__(vllm_config)
        except Exception:
            try:
                self._shutdown_transfer_backend()
            finally:
                region = getattr(self, "_region", None)
                if region is not None:
                    region.cleanup()
            raise

    def _pin_shared_region(self) -> None:
        if not _supports_eccpu_offload():
            raise RuntimeError(
                "Ascend ECCPUConnector requires both "
                "aclrtMemcpyBatchAsync and "
                "aclrtHostRegisterV2(ACL_HOST_REG_PINNED). Rebuild "
                "vllm-ascend against a CANN stack that provides both APIs."
            )
        _register_pinned_host_mmap(self._region.blocks)
        self._mmap_pinned = True

    def _acquire_event(self) -> torch.npu.Event:
        if self._event_pool:
            return self._event_pool.pop()
        return torch.npu.Event(enable_timing=True)

    def _submit_transfer(
        self,
        bufs: DescriptorBuffers,
        count: int,
        direction: ECCPUTransferDirection,
    ) -> None:
        cann_direction = _DIRECTION_H2D if direction == ECCPUTransferDirection.HOST_TO_DEVICE else _DIRECTION_D2H
        try:
            _swap_blocks_batch(
                bufs.src_ptrs[:count],
                bufs.dst_ptrs[:count],
                bufs.sizes[:count],
                cann_direction,
            )
        except Exception:
            current_platform.current_stream().synchronize()
            self._buf_pool.release(bufs)
            raise

    def flush_saves(self) -> None:
        try:
            super().flush_saves()
        except Exception:
            self._save_bufs = None
            self._save_stream = None
            self._save_count = 0
            self._save_bytes = 0
            self._save_mm_hashes: list[str] = []
            raise

    def _shutdown_transfer_backend(self) -> None:
        if not self._mmap_pinned:
            return
        torch.npu.synchronize()
        _unregister_pinned_host_mmap(self._region.blocks)
        self._mmap_pinned = False
