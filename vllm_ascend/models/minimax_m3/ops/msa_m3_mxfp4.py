# SPDX-License-Identifier: Apache-2.0
"""MXFP4 query/key packing for MiniMax M3 index scoring on Ascend 950."""

import torch
import torch_npu

from vllm_ascend.ops.triton.msa.msa_m3_mxfp4_cache import scatter_mxfp4_cache

_MXFP4_GROUP_SIZE = 32
_MXFP4_ELEMENTS_PER_BYTE = 2


def pack_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the logical last dimension into packed bytes and E8M0 scales."""
    width = tensor.shape[-1]
    if width not in (64, 128):
        raise ValueError("MSAIndexScore MXFP4 supports logical head dimensions 64 and 128")
    if tensor.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("MiniMax M3 MXFP4 index quantization requires BF16 or FP16 inputs")
    if tensor.numel() == 0:
        return (
            torch.empty(
                (*tensor.shape[:-1], width // _MXFP4_ELEMENTS_PER_BYTE),
                dtype=torch.uint8,
                device=tensor.device,
            ),
            torch.empty(
                (*tensor.shape[:-1], width // _MXFP4_GROUP_SIZE),
                dtype=torch.uint8,
                device=tensor.device,
            ),
        )
    matrix = tensor.reshape(-1, width).contiguous()
    # TorchNPU expects its extended dtype enum for packed FP4 allocation.
    packed, scales = torch_npu.npu_dynamic_mx_quant(
        matrix,
        dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=_MXFP4_GROUP_SIZE,
        scale_alg=0,
    )
    prefix = tensor.shape[:-1]
    packed = packed.view(torch.uint8).reshape(*prefix, width // _MXFP4_ELEMENTS_PER_BYTE)
    # DynamicMxQuant returns scales with its last two dimensions interleaved
    # in pairs. Quantizing the last axis makes flattening those pairs lossless.
    scales = scales.view(torch.uint8).reshape(*prefix, width // _MXFP4_GROUP_SIZE)
    return packed, scales


def insert_mxfp4_index_cache(
    cache: tuple[torch.Tensor, torch.Tensor],
    index_key: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Quantize only newly inserted keys, retaining cached prefixes unchanged."""
    key_cache, scale_cache = cache
    packed, scales = pack_mxfp4(index_key)
    scatter_mxfp4_cache(packed, scales, slots, key_cache, scale_cache)


def pack_mxfp4_query_and_insert_cache(
    cache: tuple[torch.Tensor, torch.Tensor],
    index_qk: torch.Tensor,
    num_index_heads: int,
    slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous [all Q rows | all K rows] in one native launch.

    Padding rows are quantized but never inserted. Grouping remains along
    the last dimension, so combining rows does not change their scales.
    """
    if index_qk.ndim != 2 or num_index_heads < 1 or not index_qk.is_contiguous():
        raise ValueError("Combined MXFP4 input must be a contiguous Q/K row matrix")
    tokens, remainder = divmod(index_qk.shape[0], num_index_heads + 1)
    if remainder or slots.ndim != 1 or slots.numel() > tokens:
        raise ValueError("Combined MXFP4 rows and index slots have incompatible sizes")
    actual_tokens = slots.numel()
    if actual_tokens == 0:
        packed, scales = pack_mxfp4(index_qk[:0])
        return packed.reshape(0, num_index_heads, packed.shape[-1]), scales.reshape(
            0, num_index_heads, scales.shape[-1]
        )
    packed, scales = pack_mxfp4(index_qk)
    query_rows = tokens * num_index_heads
    scatter_mxfp4_cache(
        packed[query_rows : query_rows + actual_tokens],
        scales[query_rows : query_rows + actual_tokens],
        slots,
        *cache,
    )
    return (
        packed[:query_rows].reshape(tokens, num_index_heads, packed.shape[-1])[:actual_tokens],
        scales[:query_rows].reshape(tokens, num_index_heads, scales.shape[-1])[:actual_tokens],
    )


def mxfp4_index_score(
    query: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    cache: tuple[torch.Tensor, torch.Tensor],
    start_loc: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Dispatch actual FP4 operands and E8M0 scales to the experimental operator."""
    # Load the isolated extension only in a worker that has enabled A4C4.
    from cann_ops_transformer_a4c4.ops import msa_index_score

    if isinstance(query, tuple):
        packed, query_scale = query
        if packed.dtype != torch.uint8 or query_scale.dtype != torch.uint8:
            raise TypeError("Prequantized MXFP4 queries require packed bytes and scale bytes")
        if packed.shape[-1] not in (32, 64) or query_scale.shape != (
            *packed.shape[:-1],
            packed.shape[-1] // 16,
        ):
            raise ValueError("Prequantized MXFP4 query and scale shapes differ")
        logical_width = packed.shape[-1] * _MXFP4_ELEMENTS_PER_BYTE
    else:
        packed, query_scale = pack_mxfp4(query)
        logical_width = query.shape[-1]
    key_cache, scale_cache = cache
    key_scale = scale_cache.reshape(*key_cache.shape[:-1], logical_width // _MXFP4_GROUP_SIZE)
    return msa_index_score(
        packed.view(torch.float4_e2m1fn_x2),
        key_cache.view(torch.float4_e2m1fn_x2).unsqueeze(2),
        start_loc,
        query_scale=query_scale.view(torch.float8_e8m0fnu),
        key_scale=key_scale.unsqueeze(2).view(torch.float8_e8m0fnu),
        **kwargs,
    )
