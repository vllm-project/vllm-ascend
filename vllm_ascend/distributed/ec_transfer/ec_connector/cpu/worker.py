# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""NPU worker for vLLM's mmap-backed CPU encoder-cache connector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    create_ec_shared_region,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker
from vllm.distributed.ec_transfer.ec_connector.cpu.worker.descriptor_buffers import (
    DescriptorBufferPool,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
)
from vllm.platforms import current_platform

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

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

    This class intentionally depends on a small set of upstream private fields
    (``_region``, ``_dtype``, ``_buf_pool``, ``_save_bufs`` and
    ``_save_count``). Scheduler state, mmap layout and descriptor pooling stay
    upstream-owned while only device-specific transfer behavior is replaced.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        # Do not call ECCPUWorker.__init__: it optionally invokes the CUDA-only
        # ECSharedRegion.pin_memory(). The fields below deliberately mirror the
        # small upstream initialization seam while replacing pinning/streams.
        self._region = create_ec_shared_region(vllm_config)
        self._dtype = vllm_config.model_config.dtype
        self._is_save_rank = get_tensor_model_parallel_rank() == 0 and get_pcp_group().rank_in_group == 0
        self._buf_pool = DescriptorBufferPool()
        self._save_bufs = None
        self._save_count = 0
        self._mmap_pinned = False
        try:
            device_type = get_ascend_device_type()
            if device_type not in (AscendDeviceType.A2, AscendDeviceType.A3):
                raise RuntimeError(f"Ascend ECCPUConnector currently supports only A2/A3, got {device_type.name}")

            if not _supports_eccpu_offload():
                raise RuntimeError(
                    "Ascend ECCPUConnector requires both "
                    "aclrtMemcpyBatchAsync and "
                    "aclrtHostRegisterV2(ACL_HOST_REG_PINNED). Rebuild "
                    "vllm-ascend against a supported A2/A3 CANN stack."
                )

            self._load_stream = current_platform.Stream()
            _register_pinned_host_mmap(self._region.blocks)
            self._mmap_pinned = True
        except Exception:
            self._region.cleanup()
            raise

    def _validate_block_ids(self, mm_hash: str, block_ids: list[int]) -> None:
        if not block_ids:
            raise RuntimeError(f"EC metadata has no blocks for mm_hash={mm_hash}")
        if len(set(block_ids)) != len(block_ids):
            raise RuntimeError(f"EC metadata contains duplicate blocks for mm_hash={mm_hash}: {block_ids}")
        invalid = [
            block_id
            for block_id in block_ids
            if not isinstance(block_id, int) or block_id < 0 or block_id >= self._region.num_blocks
        ]
        if invalid:
            raise RuntimeError(
                f"EC metadata contains out-of-range blocks for mm_hash={mm_hash}: "
                f"{invalid}; region has {self._region.num_blocks} blocks"
            )

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        """Build D2H descriptors for submission on the current stream."""
        if not self._is_save_rank:
            return
        block_ids = connector_metadata.saves.get(mm_hash)
        if block_ids is None:
            return
        self._validate_block_ids(mm_hash, block_ids)

        src = encoder_cache[mm_hash]
        if not src.is_contiguous():
            raise RuntimeError(
                f"Non-contiguous EC encoder cache is not supported by the batched D2H copy path: {mm_hash}"
            )

        total_bytes = src.numel() * src.element_size()
        block_size = self._region.block_size_bytes
        assert block_size % src.element_size() == 0, (
            f"EC block size {block_size} is not divisible by source element size {src.element_size()}"
        )
        required_blocks = (total_bytes + block_size - 1) // block_size
        assert len(block_ids) == required_blocks, (
            f"EC allocated block count mismatch for mm_hash={mm_hash}: need {required_blocks}, got {len(block_ids)}"
        )
        allocated_bytes = len(block_ids) * block_size
        assert total_bytes <= allocated_bytes, (
            f"EC: encoder output exceeds allocated blocks for mm_hash={mm_hash}: "
            f"{total_bytes} bytes but only {allocated_bytes} allocated"
        )

        if self._save_bufs is None:
            total = sum(len(v) for v in connector_metadata.saves.values())
            self._save_bufs = self._buf_pool.acquire(total)

        assert self._save_count + len(block_ids) <= self._save_bufs.src_ptrs.numel()
        src_ptrs, dst_ptrs, sizes = self._save_bufs
        src_base = src.data_ptr()
        dst_base = self._region.blocks.data_ptr()
        idx = self._save_count

        for block_offset, block_idx in enumerate(block_ids):
            start = block_offset * block_size
            src_ptrs[idx] = src_base + start
            dst_ptrs[idx] = dst_base + block_idx * block_size
            sizes[idx] = min(block_size, total_bytes - start)
            idx += 1

        self._save_count = idx

    def flush_saves(self) -> None:
        if self._save_count == 0:
            return

        bufs = self._save_bufs
        assert bufs is not None
        src_ptrs, dst_ptrs, sizes = bufs
        n = self._save_count

        try:
            _swap_blocks_batch(src_ptrs[:n], dst_ptrs[:n], sizes[:n], _DIRECTION_D2H)
        finally:
            self._buf_pool.release(bufs)
            self._save_bufs = None
            self._save_count = 0

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        """Load all missing hashes from pinned mmap with one H2D batch."""
        load_items = {
            mm_hash: block_ids
            for mm_hash, block_ids in connector_metadata.loads.items()
            if mm_hash not in encoder_cache
        }
        if not load_items:
            return
        for mm_hash, block_ids in load_items.items():
            self._validate_block_ids(mm_hash, block_ids)

        block_size = self._region.block_size_bytes
        element_size = torch.empty((), dtype=self._dtype).element_size()
        assert block_size % element_size == 0, (
            f"EC block size {block_size} is not divisible by dtype element size {element_size}"
        )
        elements_per_block = block_size // element_size
        total_blocks = sum(len(block_ids) for block_ids in load_items.values())
        src_base = self._region.blocks.data_ptr()

        with current_platform.stream(self._load_stream):
            # Preserve the source dtype so each block matches the mmap layout.
            dst_buf = torch.empty(
                (total_blocks, elements_per_block),
                dtype=self._dtype,
                device=current_platform.device_type,
            )
            dst_base = dst_buf.data_ptr()
            bufs = self._buf_pool.acquire(total_blocks)
            src_ptrs = bufs.src_ptrs[:total_blocks]
            dst_ptrs = bufs.dst_ptrs[:total_blocks]
            sizes = bufs.sizes[:total_blocks]
            sizes[:] = block_size

            op_idx = 0
            for block_ids in load_items.values():
                for block_idx in block_ids:
                    src_ptrs[op_idx] = src_base + block_idx * block_size
                    dst_ptrs[op_idx] = dst_base + op_idx * block_size
                    op_idx += 1

            try:
                _swap_blocks_batch(src_ptrs, dst_ptrs, sizes, _DIRECTION_H2D)
            finally:
                self._buf_pool.release(bufs)

            offset = 0
            for mm_hash, block_ids in load_items.items():
                count = len(block_ids)
                encoder_cache[mm_hash] = dst_buf[offset : offset + count]
                offset += count

        current_platform.current_stream().wait_stream(self._load_stream)

    def shutdown(self) -> None:
        torch.npu.synchronize()

        if self._mmap_pinned:
            _unregister_pinned_host_mmap(self._region.blocks)
            self._mmap_pinned = False

        self._save_bufs = None
        self._save_count = 0
        self._region.cleanup()
