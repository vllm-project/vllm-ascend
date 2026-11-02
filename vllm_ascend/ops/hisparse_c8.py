# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Candidate byte-preserving FP8 HiSparse storage and physical SFA adapter.

This staged module is not installed in the running BF16 experiment. Packed
Ascend rows contain FP8 NoPE, BF16 RoPE bytes, then FP32 scale bytes. The common
HiSparse runtime must treat these rows as opaque payloads, without converting
the RoPE or scale bytes through floating-point arithmetic.
"""

import math

import torch

_LATENT = 512
_ROPE_BYTES = 128
_SCALE_BYTES = 16
_ROW_BYTES = _LATENT + _ROPE_BYTES + _SCALE_BYTES
_INT32_MAX = 2**31 - 1
C8_PAYLOAD_FORMAT = "ascend_sfa_e4m3fn_nope512_rope64_scale4_tile128_v1"


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
    if cache.ndim != 3 or cache.dtype != torch.float8_e4m3fn or cache.shape[-1] != _ROW_BYTES:
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


def c8_physical_attention(
    query, query_rope, cache, indices, scale, query_ends, key_length, *, _prepared_indices=False, _mapping_scope=None
):
    """Use custom quantized SFA with its LSE-enabled valid-length path.

    Logical causality must already be enforced by the indexer and resolver.
    sparse_mode=0 avoids applying a second causal mask to physical addresses.
    The pinned custom kernel clips both AIC and AIV task lengths to the valid
    prefix when LSE is enabled. Device compaction preserves all selected rows
    and duplicates. This candidate still requires NPU and model qualification.
    """
    view, block, stride_rows, capacity = c8_storage_view(cache)
    rows = query.shape[0]
    if query.ndim != 3 or query.dtype != torch.bfloat16 or query.shape[-1] != _LATENT or rows <= 0:
        raise ValueError("HiSparse C8 query must be nonempty BF16 [queries, heads, 512]")
    if query_rope.shape != (*query.shape[:-1], 64) or query_rope.dtype != query.dtype:
        raise ValueError("HiSparse C8 requires matching BF16 query RoPE")
    if indices.ndim != 2 or indices.shape[0] != rows or indices.shape[1] <= 0 or indices.shape[1] % 128:
        raise ValueError("HiSparse C8 top-k must have one row per query and a width divisible by 128")
    if query.device.type != "npu" or any(
        t.device != query.device for t in (query_rope, cache, indices, query_ends, key_length)
    ):
        raise ValueError("HiSparse C8 inputs must share one NPU")
    if query_ends.dtype != torch.int32 or query_ends.shape != (1,):
        raise ValueError("HiSparse C8 requires one cumulative query end equal to the current query count")
    if key_length.dtype != torch.int32 or key_length.shape != (1,):
        raise ValueError("HiSparse C8 requires one key length equal to the aliased physical row capacity")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("HiSparse C8 attention scale must be finite and positive")
    from vllm_ascend.ops.hisparse_c8_mapping import map_compact_c8_indices

    packed_query = None
    if _prepared_indices:
        if indices.dtype != torch.int32 or stride_rows != block or not indices.is_contiguous():
            raise ValueError("Prepared C8 indices require contiguous int32 rows and an identity staging layout")
        mapped = indices
    else:
        # The existing group scope certifies one current captured producer.
        # Followers reuse only its mapped indices; each packs its own query.
        fused = None
        if (
            _mapping_scope is not None
            and rows in (8, 16, 32)
            and _mapping_scope.context() is not None
            and torch.npu.is_current_stream_capturing()
        ):
            binding = _mapping_scope.binding(indices, block, stride_rows, capacity)
            if binding is not None and _mapping_scope.result is None:
                from vllm_ascend.ops.triton.hisparse_c8_map_pack import map_pack

                fused = map_pack(
                    indices,
                    block,
                    stride_rows,
                    _mapping_scope.physical_bound,
                    query,
                    query_rope,
                    query_ends,
                    key_length,
                    view.shape[0],
                )
                _mapping_scope.key = binding
                _mapping_scope.result = fused[0]
        if fused is None:
            mapped = map_compact_c8_indices(indices, block, stride_rows, capacity, _scope=_mapping_scope)
        else:
            mapped, packed_query = fused
    from vllm_ascend.ops.triton.hisparse_c8_fused import pack_c8_query_lengths

    if packed_query is None:
        packed_query = pack_c8_query_lengths(query, query_rope, query_ends, key_length, view.shape[0])
    output = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
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
    if isinstance(output, tuple):
        output = output[0]
    # The pinned LSE-enabled C8 kernel clears all outputs before
    # skipping empty valid prefixes. Preserve that raw output.
    return output
