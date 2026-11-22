# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_vectorcore_num, init_device_properties_triton

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM


@dataclass(frozen=True)
class LongKVStore:
    """Decode epilogue inputs for a standard BF16 paged Long KV cache."""

    weight: torch.Tensor
    eps: float
    cache: torch.Tensor
    slots: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

    def kernel_args(self):
        pointers = (self.cache, self.slots, self.cos, self.sin)
        strides = (
            self.cache.stride(0),
            self.cache.stride(1),
            self.slots.stride(0),
            self.cos.stride(0),
            self.sin.stride(0),
        )
        return pointers, strides


@triton.jit
def _normalize_and_store_long_kv(
    value,
    weight,
    row,
    store_ptrs,
    STRIDES: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    # Round the pooled value before RMSNorm and normalize to BF16 before RoPE.
    # The RMSNorm and interleaved rotation follow the SWA writer in Ascend #21183.
    value = value.to(tl.bfloat16).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / HEAD_DIM + EPS)
    normalized = (value * inverse_rms * weight).to(tl.bfloat16)

    cache, slots, cos, sin = store_ptrs
    block_stride, token_stride, slot_stride, cos_stride, sin_stride = STRIDES
    row = row.to(tl.int64)
    slot_row = slots + row * slot_stride
    block = tl.load(slot_row).to(tl.int64)
    offset = tl.load(slot_row + 1).to(tl.int64)
    valid = (block >= 0) & (offset >= 0)
    cache_row = cache + tl.where(valid, block * block_stride + offset * token_stride, 0)

    nope_dim: tl.constexpr = HEAD_DIM - ROPE_DIM
    rotary_dims = tl.arange(0, ROPE_DIM)
    rotary = extract_slice(normalized, (nope_dim,), (ROPE_DIM,), (1,)).to(tl.float32)
    cos_value = tl.load(cos + row * cos_stride + rotary_dims).to(tl.float32)
    sin_value = tl.load(sin + row * sin_stride + rotary_dims).to(tl.float32)
    partner = tl.gather(rotary, rotary_dims ^ 1, 0)
    rotated = rotary * cos_value + partner * sin_value * tl.where(rotary_dims % 2 == 0, -1.0, 1.0)

    dims = tl.arange(0, HEAD_DIM)
    tl.store(cache_row + dims, normalized, valid & (dims < nope_dim))
    tl.store(cache_row + nope_dim + rotary_dims, rotated, valid)
    return normalized


@triton.jit(do_not_specialize=["num_tokens"])
def _normalize_and_store_c1_kernel(
    projected,
    weight_ptr,
    latent,
    store_ptrs,
    num_tokens,
    STRIDES: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    dims = tl.arange(0, HEAD_DIM)
    weight = tl.load(weight_ptr + dims).to(tl.float32)
    for row in range(tl.program_id(0), num_tokens, tl.num_programs(0)):
        row_offset = row.to(tl.int64) * HEAD_DIM + dims
        value = tl.load(projected + row_offset)
        normalized = _normalize_and_store_long_kv(value, weight, row, store_ptrs, STRIDES, EPS, HEAD_DIM, ROPE_DIM)
        tl.store(latent + row_offset, normalized)


def normalize_and_store_c1(projected: torch.Tensor, cache_store: LongKVStore) -> torch.Tensor:
    """Return unrotated BF16 latent and publish rotated rows to Long KV."""
    latent = torch.empty_like(projected)
    num_tokens = projected.shape[0]
    if num_tokens:
        init_device_properties_triton()
        pointers, strides = cache_store.kernel_args()
        _normalize_and_store_c1_kernel[(min(num_tokens, get_vectorcore_num()),)](
            projected,
            cache_store.weight,
            latent,
            pointers,
            num_tokens,
            strides,
            cache_store.eps,
            HEAD_DIM,
            ROPE_DIM,
            enable_fp_fusion=False,
        )
    return latent
