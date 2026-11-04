# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Native C8 payload/layout bridge; quantization and Attention stay in existing ops."""

import math

import torch

_LATENT = 512
_ROPE_BYTES = 128
_SCALE_BYTES = 16
C8_ROW_BYTES = _LATENT + _ROPE_BYTES + _SCALE_BYTES
_INT32_MAX = 2**31 - 1


def pack_c8_components(nope: torch.Tensor, rope: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack existing SFA quantizer outputs using byte copies only."""
    if nope.dtype != torch.float8_e4m3fn:
        raise ValueError("HiSparse C8 requires FP8 E4M3 NoPE")
    components = (nope, rope, scales)
    if any(t.device != nope.device or not t.is_contiguous() for t in components):
        raise ValueError("HiSparse C8 components must be contiguous on one device")
    rows = nope.numel() // _LATENT
    if rows == 0 or nope.numel() != rows * _LATENT:
        raise ValueError("HiSparse C8 requires complete 512-element NoPE rows")
    if rope.numel() * rope.element_size() != rows * _ROPE_BYTES:
        raise ValueError("HiSparse C8 requires 64 BF16 RoPE elements per row")
    if scales.numel() * scales.element_size() != rows * _SCALE_BYTES:
        raise ValueError("HiSparse C8 requires four FP32 scale values per row")
    return torch.cat(
        [
            t.view(torch.uint8).reshape(rows, width)
            for t, width in zip(components, (_LATENT, _ROPE_BYTES, _SCALE_BYTES))
        ],
        dim=-1,
    )


def c8_storage_view(cache: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    """Alias all addressable pages, retaining HMA gaps and the layer offset.

    The allocator must align the physical block stride to a whole packed row.
    The returned TND view shares storage; no payload is gathered or dequantized.
    """
    if cache.ndim != 3 or cache.dtype != torch.float8_e4m3fn or cache.shape[-1] != C8_ROW_BYTES:
        raise ValueError("HiSparse C8 cache must be FP8 [pages, block_size, 656]")
    pages, block, width = cache.shape
    stride = cache.stride(0)
    if min(pages, block) <= 0 or cache.stride(1) != width or cache.stride(2) != 1:
        raise ValueError("HiSparse C8 requires nonempty pages with dense token rows")
    if stride < block * width or stride % width:
        raise ValueError("HiSparse C8 physical block stride must cover whole packed rows")
    logical_capacity = pages * block
    span = (pages - 1) * stride + block * width
    addressable_rows = span // width
    if max(logical_capacity, addressable_rows) > _INT32_MAX:
        raise ValueError("HiSparse C8 physical row IDs must fit int32")
    offset = cache.storage_offset()
    if (offset + span) * cache.element_size() > cache.untyped_storage().nbytes():
        raise ValueError("HiSparse C8 view exceeds its backing allocation")
    view = torch.as_strided(cache, (addressable_rows, 1, width), (width, width, 1), storage_offset=offset)
    return view, block, stride // width, logical_capacity


def map_c8_physical_indices(indices: torch.Tensor, block: int, stride_rows: int, capacity: int) -> torch.Tensor:
    """Map virtual page rows to native contiguous-row addresses on device."""
    if indices.ndim != 2 or indices.dtype != torch.int32 or block <= 0 or stride_rows < block:
        raise ValueError("HiSparse C8 requires int32 [queries, top_k] indices and valid page geometry")
    if capacity <= 0 or capacity % block or (capacity // block - 1) * stride_rows + block > _INT32_MAX:
        raise ValueError("HiSparse C8 physical row capacity is invalid")
    valid = (indices >= 0) & (indices < capacity)
    safe = indices.clamp_min(0).to(torch.int64)
    mapped = safe.div(block, rounding_mode="floor") * stride_rows + safe.remainder(block)
    return torch.where(valid, mapped, -1).to(torch.int32)


def compact_c8_indices(indices: torch.Tensor) -> torch.Tensor:
    """Keep valid IDs in their original order, including every duplicate."""
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("C8 compaction requires int32 [queries, top_k] indices")
    width = indices.shape[1]
    if width > 2**23:
        raise ValueError("C8 compaction keys must remain exact FP32 integers")
    positions = torch.arange(width, dtype=torch.float32, device=indices.device)
    keys = positions + (indices < 0).to(torch.float32) * width
    order = torch.argsort(keys, dim=-1)
    return torch.gather(indices, -1, order)


def c8_physical_attention(query, query_rope, cache, indices, scale, query_ends, key_length):
    """Adapt native packed rows to the existing quantized SFA operation.

    Keep compaction stable, including duplicates. The two current lengths are
    device writes captured on the consumer stream; no Host read or stream/event
    policy is introduced here.
    """
    view, block, stride_rows, capacity = c8_storage_view(cache)
    rows = query.shape[0]
    if query.ndim != 3 or query.dtype != torch.bfloat16 or query.shape[-1] != _LATENT or rows <= 0:
        raise ValueError("HiSparse C8 query must be nonempty BF16 [queries, heads, 512]")
    if query_rope.shape != (*query.shape[:-1], 64) or query_rope.dtype != query.dtype:
        raise ValueError("HiSparse C8 requires matching BF16 query RoPE")
    if indices.ndim != 2 or indices.shape[0] != rows or indices.shape[1] <= 0 or indices.shape[1] % 128:
        raise ValueError("HiSparse C8 top-k width must be positive and divisible by 128")
    if query.device.type != "npu" or any(
        tensor.device != query.device for tensor in (query_rope, cache, indices, query_ends, key_length)
    ):
        raise ValueError("HiSparse C8 inputs must share one NPU")
    if any(tensor.shape != (1,) or tensor.dtype != torch.int32 for tensor in (query_ends, key_length)):
        raise ValueError("HiSparse C8 lengths must be int32 scalar tensors")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("HiSparse C8 attention scale must be finite and positive")
    mapped = compact_c8_indices(map_c8_physical_indices(indices, block, stride_rows, capacity))
    packed_query = torch.cat((query, query_rope), dim=-1)
    query_ends.fill_(rows)
    key_length.fill_(view.shape[0])
    result = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
        query=packed_query,
        key=view,
        value=view,
        sparse_indices=mapped.unsqueeze(1),
        scale_value=scale,
        sparse_block_size=1,
        actual_seq_lengths_query=query_ends,
        actual_seq_lengths_kv=key_length,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=0,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=64,
        return_softmax_lse=True,
    )
    if isinstance(result, tuple):
        result = result[0]
    return torch.where((mapped >= 0).any(dim=-1)[:, None, None], result, 0)
