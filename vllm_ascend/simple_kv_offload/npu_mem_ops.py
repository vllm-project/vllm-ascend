# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Low-level NPU memory helpers: batched DMA transfers.

Mirrors :mod:`vllm.v1.simple_kv_offload.cuda_mem_ops` but uses the
Ascend ``aclrtMemcpyBatchAsync`` path exposed via
``torch.ops._C_ascend.swap_blocks_batch`` (see
``csrc/torch_binding.cpp``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Direction codes shared with csrc/torch_binding.cpp::swap_blocks_batch.
DIRECTION_H2D = 0
DIRECTION_D2H = 1


class BatchMemcpyParams(NamedTuple):
    """Pre-computed per-tensor descriptors for batched block copy."""

    src_bases: np.ndarray  # [num_sub_tensors] int64 — data_ptr per tensor
    dst_bases: np.ndarray  # [num_sub_tensors] int64
    bpb: np.ndarray  # [num_sub_tensors] int64 — bytes per block
    num_sub_tensors: int
    direction: int  # DIRECTION_H2D or DIRECTION_D2H


def _ordered_tensors(caches: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Return values in insertion order (kept as a function for clarity)."""
    return list(caches.values())


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    direction: int,
) -> BatchMemcpyParams:
    """Build cached pointer/stride descriptors for all sub-tensors.

    Both ``src_caches`` and ``dst_caches`` must have identical keys and a
    matching ``[num_blocks, block_bytes]`` layout (already prepared by
    :class:`SimpleCPUOffloadNPUWorker.register_kv_caches`).
    """
    assert list(src_caches.keys()) == list(dst_caches.keys()), (
        "src/dst cache key order must match"
    )
    src_tensors = _ordered_tensors(src_caches)
    dst_tensors = _ordered_tensors(dst_caches)

    src_bases: list[int] = []
    dst_bases: list[int] = []
    bpb: list[int] = []
    for s, d in zip(src_tensors, dst_tensors):
        s_bpb = s.stride(0) * s.element_size()
        d_bpb = d.stride(0) * d.element_size()
        assert s_bpb == d_bpb, (
            f"per-block bytes mismatch src={s_bpb} dst={d_bpb}"
        )
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    return BatchMemcpyParams(
        src_bases=np.array(src_bases, dtype=np.int64),
        dst_bases=np.array(dst_bases, dtype=np.int64),
        bpb=np.array(bpb, dtype=np.int64),
        num_sub_tensors=len(src_tensors),
        direction=direction,
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
) -> None:
    """Issue a batched async DMA on the *current* NPU stream.

    The caller is expected to be inside a ``torch.npu.stream(...)``
    context so the issued copies bind to the dedicated transfer stream.
    """
    n = len(src_block_ids)
    if n == 0:
        return
    assert n == len(dst_block_ids), "src/dst block counts must match"

    src_ids = np.asarray(src_block_ids, dtype=np.int64)
    dst_ids = np.asarray(dst_block_ids, dtype=np.int64)

    # Layout: (num_sub_tensors, n) flattened — contract of swap_blocks_batch.
    bpb_col = params.bpb[:, None]
    src_all = (params.src_bases[:, None] + src_ids[None, :] * bpb_col).ravel()
    dst_all = (params.dst_bases[:, None] + dst_ids[None, :] * bpb_col).ravel()
    sz_all = np.broadcast_to(
        bpb_col, (params.num_sub_tensors, n)
    ).ravel().copy()

    batch_src = torch.from_numpy(src_all)
    batch_dst = torch.from_numpy(dst_all)
    batch_sizes = torch.from_numpy(sz_all)

    torch.ops._C_ascend.swap_blocks_batch(
        batch_src, batch_dst, batch_sizes, params.direction
    )
