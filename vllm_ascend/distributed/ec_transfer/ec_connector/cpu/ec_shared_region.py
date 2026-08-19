# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight mmap-backed shared memory region for encoder cache (EC) data.

Modeled after SharedOffloadRegion (vllm/v1/kv_offload/cpu/) but simplified
for EC: flat shared layout, no multi-tensor cursor, no block_size_factor.
"""

import errno
import mmap
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    _get_encoder_cache_hidden_dim,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    ECSharedRegion,
    _wait_for_file_size,
    logger,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _fallback_populate_write(
    mmap_obj: mmap.mmap,
    offset: int,
    length: int,
) -> None:
    # Touch one byte per page via a read-modify-write so existing bytes are
    # preserved — a peer worker may have already written EC data into this
    # shared mmap by the time we run on a kernel without MADV_POPULATE_WRITE.
    arr = np.frombuffer(mmap_obj, dtype=np.uint8)
    arr[offset : offset + length : mmap.PAGESIZE] |= 0


class AscendECSharedRegion(ECSharedRegion):
    """Flat mmap-backed memory region shared across TP workers for
    encoder cache blocks.

    Layout: (num_blocks, block_size_bytes) — contiguous, no per-worker
    interleaving. All workers map the same file and see identical data.

    File path: /dev/shm/vllm_ec_{engine_id}.mmap

    This class owns only the shared memory substrate (mmap lifecycle, the
    `blocks` view, CUDA host registration). Block allocation and eviction
    are tracked by `EmbeddingCache` in the scheduler process.
    """

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        block_size_bytes: int,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size_bytes = block_size_bytes

        total_size_bytes = num_blocks * block_size_bytes
        # Path in /dev/shm (tmpfs); unique per engine instance.
        self._mmap_path = f"/dev/shm/vllm_ec_{engine_id}.mmap"
        # True for the process that created the file (responsible for unlink).
        self._is_creator = False
        # True after successful cudaHostRegister (cleanup must unregister).
        self._is_pinned = False

        # File descriptor for the shared memory backing file.
        try:
            self._fd: int | None = os.open(
                self._mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            os.ftruncate(self._fd, total_size_bytes)
            self._is_creator = True
            logger.info(
                "Created EC mmap file %s (%.2f MiB)",
                self._mmap_path,
                total_size_bytes / (1 << 20),
            )
        except FileExistsError:
            self._fd = os.open(self._mmap_path, os.O_RDWR)
            try:
                _wait_for_file_size(self._fd, total_size_bytes)
            except Exception:
                os.close(self._fd)
                self._fd = None
                raise
            logger.info("Opened existing EC mmap file %s", self._mmap_path)

        # MAP_SHARED mmap over _fd; all processes see the same pages.
        self._mmap_obj: mmap.mmap | None = mmap.mmap(
            self._fd,
            total_size_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

        if self._is_creator:
            _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
            try:
                self._mmap_obj.madvise(_MADV_POPULATE_WRITE, 0, total_size_bytes)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                logger.warning(
                    "MADV_POPULATE_WRITE is not supported; falling back to "
                    "per-page writes for EC mmap pre-population. Startup may "
                    "be slower."
                )
                _fallback_populate_write(self._mmap_obj, 0, total_size_bytes)

        # (num_blocks, block_size_bytes) int8 tensor over the mmap buffer.
        self.blocks: torch.Tensor = torch.frombuffer(
            memoryview(self._mmap_obj), dtype=torch.int8
        ).view(num_blocks, block_size_bytes)
        # Cached for cudaHostRegister/Unregister and pointer math.
        self._blocks_ptr: int = self.blocks.data_ptr()
        self._blocks_nbytes: int = self.blocks.nbytes


def create_ascend_ec_shared_region(
    vllm_config: "VllmConfig",
) -> AscendECSharedRegion:
    """Build the EC mmap region from `vllm_config`.

    Both `AscendECCPUScheduler` and `AscendECCPUWorker` call this to get the
    same shared region (same engine_id, same block_size_bytes).
    """
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None, "ec_transfer_config required to build region"

    dp_rank = vllm_config.parallel_config.data_parallel_rank
    engine_id = f"{vllm_config.instance_id}_dp{dp_rank}"

    dtype = vllm_config.model_config.dtype
    hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
    element_size = torch.empty(0, dtype=dtype).element_size()
    block_size_bytes = hidden_dim * element_size

    cpu_bytes = ec_config.ec_connector_extra_config.get("ec_cpu_bytes")
    if not cpu_bytes:
        raise ValueError("ec_cpu_bytes must be specified in ec_connector_extra_config")
    num_blocks = int(cpu_bytes) // block_size_bytes

    return AscendECSharedRegion(
        engine_id=engine_id,
        num_blocks=num_blocks,
        block_size_bytes=block_size_bytes,
    )
