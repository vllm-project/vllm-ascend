# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 SWA RMSNorm, partial RoPE and paged-cache store.

The projection GEMM remains separate. Triton normalizes each WKV row,
applies interleaved partial RoPE, and writes directly to the builder-prepared
physical SWA cache coordinate. No intermediate KV tensor is materialized.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_vectorcore_num, init_device_properties_triton

HEAD_DIM = 512
ROPE_DIM = 64


@triton.jit(do_not_specialize=["num_tokens"])
def _dsv41_rmsnorm_rope_cache_persistent_kernel(
    projected_kv,
    norm_weight,
    cos,
    sin,
    cache,
    slot_mapping,
    num_tokens,
    KV_TOKEN_STRIDE: tl.constexpr,
    COS_TOKEN_STRIDE: tl.constexpr,
    SIN_TOKEN_STRIDE: tl.constexpr,
    SLOT_TOKEN_STRIDE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    nope_dim: tl.constexpr = HEAD_DIM - ROPE_DIM
    dims = tl.arange(0, HEAD_DIM)
    rotary_dims = tl.arange(0, ROPE_DIM)
    weight = tl.load(norm_weight + dims).to(tl.float32)
    for row_index in range(tl.program_id(0), num_tokens, tl.num_programs(0)):
        token = row_index.to(tl.int64)
        slot_row = slot_mapping + token * SLOT_TOKEN_STRIDE
        block = tl.load(slot_row).to(tl.int64)
        offset = tl.load(slot_row + 1).to(tl.int64)
        valid = (block >= 0) & (offset >= 0)
        cache_row = cache + tl.where(valid, block * CACHE_BLOCK_STRIDE + offset * CACHE_TOKEN_STRIDE, 0)

        value = tl.load(projected_kv + token * KV_TOKEN_STRIDE + dims).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / HEAD_DIM + EPS)
        normalized = (value * inverse_rms * weight).to(tl.bfloat16)
        rotary = extract_slice(normalized, (nope_dim,), (ROPE_DIM,), (1,)).to(tl.float32)
        cos_value = tl.load(cos + token * COS_TOKEN_STRIDE + rotary_dims).to(tl.float32)
        sin_value = tl.load(sin + token * SIN_TOKEN_STRIDE + rotary_dims).to(tl.float32)
        partner = tl.gather(rotary, rotary_dims ^ 1, 0)
        rotated = rotary * cos_value + partner * sin_value * tl.where(rotary_dims % 2 == 0, -1.0, 1.0)

        tl.store(cache_row + dims, normalized, valid & (dims < nope_dim))
        tl.store(cache_row + nope_dim + rotary_dims, rotated, valid)


def dsv41_rmsnorm_rope_cache(
    projected_kv: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    eps: float,
    nope_dim: int,
) -> bool:
    """Write normalized and rotated KV in place, or return False for unsupported inputs.

    Fusion supports 448 NoPE and 64 RoPE dimensions. The physical
    ``[block, offset]`` coordinates must be in bounds and unique when valid.
    A negative coordinate masks the entire store. The cache uses its actual
    block and row strides. A valid empty batch succeeds without launching.
    """
    if (
        nope_dim != HEAD_DIM - ROPE_DIM
        or not isinstance(cache, torch.Tensor)
        or cache.device.type != "npu"
        or cache.dtype != torch.bfloat16
        or cache.ndim != 4
        or cache.shape[-2:] != (1, HEAD_DIM)
        or cache.stride(-1) != 1
    ):
        return False

    device = cache.device
    if (
        not isinstance(norm_weight, torch.Tensor)
        or norm_weight.shape != (HEAD_DIM,)
        or norm_weight.device != device
        or not norm_weight.is_contiguous()
        or projected_kv.ndim != 2
        or projected_kv.dtype != torch.bfloat16
        or projected_kv.device != device
        or projected_kv.shape[1] != HEAD_DIM
        or projected_kv.stride(1) != 1
    ):
        return False

    num_tokens = projected_kv.shape[0]
    for rotary in (cos, sin):
        if (
            rotary.ndim < 2
            or rotary.shape[0] < num_tokens
            or rotary.shape[-1] != ROPE_DIM
            or rotary.stride(-1) != 1
            or rotary.device != device
        ):
            return False
    if (
        slot_mapping.ndim != 2
        or slot_mapping.shape[0] < num_tokens
        or slot_mapping.shape[1] != 2
        or slot_mapping.device != device
        or slot_mapping.dtype not in (torch.int32, torch.int64)
        or slot_mapping.stride(1) != 1
    ):
        return False

    if num_tokens:
        init_device_properties_triton()
        programs = min(num_tokens, get_vectorcore_num())
        _dsv41_rmsnorm_rope_cache_persistent_kernel[(programs,)](
            projected_kv,
            norm_weight,
            cos,
            sin,
            cache.squeeze(-2),
            slot_mapping,
            num_tokens,
            projected_kv.stride(0),
            cos.stride(0),
            sin.stride(0),
            slot_mapping.stride(0),
            cache.stride(0),
            cache.stride(1),
            HEAD_DIM,
            ROPE_DIM,
            eps,
            enable_fp_fusion=False,
        )
    return True
