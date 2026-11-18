# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 SWA RMSNorm, partial RoPE and paged-cache store.

The projection GEMM remains separate.  One Triton program consumes one WKV
row, normalizes it, applies the model's interleaved partial RoPE, and writes
the result directly to the builder-prepared physical SWA cache coordinate.
No normalized/rotated KV tensor is materialized.
"""

from dataclasses import dataclass
from weakref import ReferenceType, ref

import torch
from vllm.triton_utils import tl, triton

HEAD_DIM = 512


@triton.jit
def _dsv41_rmsnorm_rope_cache_kernel(
    projected_kv,
    norm_weight,
    cos,
    sin,
    cache,
    slot_mapping,
    KV_TOKEN_STRIDE: tl.constexpr,
    COS_TOKEN_STRIDE: tl.constexpr,
    COS_DIM_STRIDE: tl.constexpr,
    SIN_TOKEN_STRIDE: tl.constexpr,
    SIN_DIM_STRIDE: tl.constexpr,
    SLOT_TOKEN_STRIDE: tl.constexpr,
    SLOT_DIM_STRIDE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    CACHE_DIM_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, HEAD_DIM)
    row = projected_kv + token * KV_TOKEN_STRIDE

    value = tl.load(row + dims).to(tl.float32)
    weight = tl.load(norm_weight + dims).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / HEAD_DIM + EPS)

    # The standalone RMSNorm publishes BF16 before RoPE consumes it.  Keep
    # that rounding boundary instead of fusing the two FP32 expressions.
    normalized = (value * inverse_rms * weight).to(projected_kv.dtype.element_ty).to(tl.float32)

    rotary_index = dims - NOPE_DIM
    rotary_mask = dims >= NOPE_DIM
    even = (rotary_index % 2) == 0
    pair_dims = tl.where(rotary_mask, dims + tl.where(even, 1, -1), 0)
    pair_value = tl.load(row + pair_dims, mask=rotary_mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(norm_weight + pair_dims, mask=rotary_mask, other=0.0).to(tl.float32)
    pair_normalized = (pair_value * inverse_rms * pair_weight).to(projected_kv.dtype.element_ty).to(tl.float32)

    cos_value = tl.load(
        cos + token * COS_TOKEN_STRIDE + rotary_index * COS_DIM_STRIDE,
        mask=rotary_mask,
        other=1.0,
    ).to(tl.float32)
    sin_value = tl.load(
        sin + token * SIN_TOKEN_STRIDE + rotary_index * SIN_DIM_STRIDE,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    sin_sign = tl.where(even, -1.0, 1.0)
    rotated = normalized * cos_value + pair_normalized * sin_value * sin_sign
    output = tl.where(rotary_mask, rotated, normalized)

    slot_row = slot_mapping + token * SLOT_TOKEN_STRIDE
    block = tl.load(slot_row).to(tl.int64)
    offset = tl.load(slot_row + SLOT_DIM_STRIDE).to(tl.int64)
    valid = (block >= 0) & (offset >= 0)
    cache_offsets = block * CACHE_BLOCK_STRIDE + offset * CACHE_TOKEN_STRIDE + dims * CACHE_DIM_STRIDE
    tl.store(cache + cache_offsets, output, mask=valid)


@dataclass(frozen=True, slots=True)
class DSV41SWACacheLayout:
    """Validated layout for one bound SWA cache and RMSNorm weight."""

    cache_ref: ReferenceType[torch.Tensor]
    weight_ref: ReferenceType[torch.Tensor]
    device: torch.device
    nope_dim: int
    cache_strides: tuple[int, int, int]

    @classmethod
    def prepare(
        cls, cache: torch.Tensor, weight: torch.Tensor, head_dim: int, nope_dim: int
    ) -> "DSV41SWACacheLayout | None":
        if (
            head_dim != HEAD_DIM
            or not 0 <= nope_dim < head_dim
            or (head_dim - nope_dim) % 2
            or not isinstance(cache, torch.Tensor)
            or cache.device.type != "npu"
            or cache.dtype != torch.bfloat16
            or cache.ndim != 4
            or cache.shape[-2:] != (1, head_dim)
            or cache.stride(-1) != 1
            or not isinstance(weight, torch.Tensor)
            or weight.shape != (head_dim,)
            or weight.device != cache.device
            or not weight.is_contiguous()
        ):
            return None
        return cls(ref(cache), ref(weight), cache.device, nope_dim, (cache.stride(0), cache.stride(1), 1))

    def try_store(
        self,
        projected_kv: torch.Tensor,
        norm_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        eps: float,
    ) -> bool:
        """Validate the batch and submit a write for the bound cache and weight."""
        if (
            projected_kv.ndim != 2
            or projected_kv.dtype != torch.bfloat16
            or projected_kv.device != self.device
            or projected_kv.shape[1] != HEAD_DIM
            or projected_kv.stride(1) != 1
        ):
            return False
        num_tokens = projected_kv.shape[0]
        rope_dim = HEAD_DIM - self.nope_dim
        for rotary in (cos, sin):
            if (
                rotary.ndim < 2
                or rotary.shape[0] < num_tokens
                or rotary.shape[-1] != rope_dim
                or rotary.stride(-1) != 1
                or rotary.device != self.device
            ):
                return False
        if (
            slot_mapping.ndim != 2
            or slot_mapping.shape[0] < num_tokens
            or slot_mapping.shape[1] != 2
            or slot_mapping.device != self.device
            or slot_mapping.dtype not in (torch.int32, torch.int64)
            or slot_mapping.stride(1) != 1
        ):
            return False
        if num_tokens:
            _dsv41_rmsnorm_rope_cache_kernel[(num_tokens,)](
                projected_kv,
                norm_weight,
                cos,
                sin,
                cache.squeeze(-2),
                slot_mapping,
                projected_kv.stride(0),
                cos.stride(0),
                1,
                sin.stride(0),
                1,
                slot_mapping.stride(0),
                1,
                *self.cache_strides,
                HEAD_DIM,
                self.nope_dim,
                eps,
                enable_fp_fusion=False,
            )
        return True


def dsv41_rmsnorm_rope_cache(
    projected_kv: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    eps: float,
    nope_dim: int,
) -> None:
    """Normalize/rotate projected KV and write it directly to the SWA cache.

    ``slot_mapping`` contains physical ``[block, offset]`` coordinates.  A
    negative coordinate suppresses the whole row, including graph-padding
    rows.  Cache offsets use the tensor's real strides, so a cache plane may
    remain a view into a layer-outermost allocation.
    """
    layout = DSV41SWACacheLayout.prepare(cache, norm_weight, HEAD_DIM, nope_dim)
    if layout is None or not layout.try_store(projected_kv, norm_weight, cos, sin, cache, slot_mapping, eps):
        raise ValueError("Unsupported tensors for the DeepSeek V4.1 fused SWA cache write")
