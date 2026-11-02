# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-cache addressing and stable index compaction; no Attention registration."""

import torch

_LATENT_DIM = 512
_ROPE_DIM = 64
_ROW_WIDTH = _LATENT_DIM + _ROPE_DIM
_MAX_TOPK = 32768
_MAX_FUSED_COMPACT_TOPK = 2048
_INT64_MAX = (1 << 63) - 1
_INT32_MAX = (1 << 31) - 1


def packed_cache_storage(cache: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """Return a bounded contiguous alias starting at this layer's first row.

    The alias includes inter-page gaps without copying them. Its pointer keeps
    the layer's storage offset; explicit page strides decide which rows the
    kernel may read. Both ordinary strided views and expanded attention views
    use the same addressing contract.
    """
    if cache.dtype != torch.bfloat16 or cache.ndim != 3 or cache.shape[-1] != _ROW_WIDTH:
        raise ValueError("Packed HiSparse KV must be BF16 [pages, block_size, 576]")
    pages, block_size, _ = cache.shape
    if pages <= 0 or block_size <= 0 or block_size > 4095:
        raise ValueError("Packed HiSparse KV requires nonempty pages with block_size <= 4095")
    block_stride = cache.stride(0)
    if cache.stride(1) != _ROW_WIDTH or cache.stride(2) != 1 or block_stride < block_size * _ROW_WIDTH:
        raise ValueError("Packed HiSparse KV requires contiguous rows and nonoverlapping pages")
    capacity = pages * block_size
    span = (pages - 1) * block_stride + block_size * _ROW_WIDTH
    offset = cache.storage_offset()
    if span > _INT64_MAX or capacity > _INT32_MAX or offset < 0:
        raise ValueError("Packed HiSparse KV exceeds int64 addressing or int32 physical-row capacity")
    if (offset + span) * cache.element_size() > cache.untyped_storage().nbytes():
        raise ValueError("Packed HiSparse KV view exceeds its backing storage")
    raw = torch.as_strided(cache, (span,), (1,), storage_offset=offset)
    return raw, block_size, block_stride, capacity


def compact_physical_indices(
    indices: torch.Tensor, capacity: int, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep valid indices in their original order, including duplicates.

    Bounds are checked on device. Out-of-range entries are excluded like -1,
    so no malformed physical index can address another allocation. The NPU
    prefix-scan path preserves order and duplicates without sorting. Other
    devices and larger selections retain the FP32 position-key implementation.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32 or indices.stride(1) != 1:
        raise ValueError("HiSparse physical indices must be int32 [queries, top_k] with contiguous columns")
    topk = indices.shape[1]
    if topk <= 0 or topk > _MAX_TOPK or topk % 128:
        raise ValueError("HiSparse direct attention requires top_k in [128, 32768], divisible by 128")
    if positions.shape != (1, topk) or positions.dtype != torch.float32 or positions.device != indices.device:
        raise ValueError("HiSparse position keys must be FP32 [1, top_k] on the indices device")
    use_fused = False
    if indices.device.type == "npu" and topk <= _MAX_FUSED_COMPACT_TOPK:
        from vllm_ascend.device import HardwareCapability, get_current_hardware_profile

        use_fused = get_current_hardware_profile().supports(HardwareCapability.HISPARSE_FUSED_INDEX_OPS)
    if use_fused:
        from vllm.triton_utils import triton

        from vllm_ascend.ops.triton.hisparse import compact_physical_indices_kernel

        rows = indices.shape[0]
        compacted = torch.empty((rows, topk), dtype=torch.int32, device=indices.device)
        counts = torch.empty((rows,), dtype=torch.int32, device=indices.device)
        if rows:
            compact_physical_indices_kernel[(rows,)](
                indices,
                compacted,
                counts,
                INPUT_STRIDE=indices.stride(0),
                TOP_K=topk,
                CAPACITY=capacity,
                BLOCK=triton.next_power_of_2(topk),
                multibuffer=False,
            )
        return compacted, counts
    valid = (indices >= 0) & (indices.to(torch.int64) < capacity)
    keys = positions + (~valid).to(torch.float32) * topk
    order = keys.sort(dim=-1).indices
    compacted = indices.masked_fill(~valid, -1).gather(1, order)
    counts = valid.sum(dim=-1, dtype=torch.int32)
    return compacted, counts
