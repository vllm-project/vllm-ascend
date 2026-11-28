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
        return dict(
            long_cache=self.cache,
            long_slots=self.slots,
            long_cos=self.cos,
            long_sin=self.sin,
            CACHE_PAGE_STRIDE=self.cache.stride(0),
            CACHE_ROW_STRIDE=self.cache.stride(1),
            SLOT_ROW_STRIDE=self.slots.stride(0),
            COS_ROW_STRIDE=self.cos.stride(0),
            SIN_ROW_STRIDE=self.sin.stride(0),
        )


@triton.jit
def _normalize_and_store_long_kv(
    value,
    weight,
    row,
    long_cache,
    long_slots,
    long_cos,
    long_sin,
    CACHE_PAGE_STRIDE: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    SLOT_ROW_STRIDE: tl.constexpr,
    COS_ROW_STRIDE: tl.constexpr,
    SIN_ROW_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    # Round the pooled value before RMSNorm and normalize to BF16 before RoPE.
    # The RMSNorm and interleaved rotation follow the SWA writer in Ascend #21183.
    value = value.to(tl.bfloat16).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(value * value, axis=0) / HEAD_DIM + EPS)
    normalized = (value * inverse_rms * weight).to(tl.bfloat16)
    row = row.to(tl.int64)
    slot = long_slots + row * SLOT_ROW_STRIDE
    block = tl.load(slot).to(tl.int64)
    offset = tl.load(slot + 1).to(tl.int64)
    valid = (block >= 0) & (offset >= 0)
    dst = long_cache + tl.where(valid, block * CACHE_PAGE_STRIDE + offset * CACHE_ROW_STRIDE, 0)
    nope_dim: tl.constexpr = HEAD_DIM - ROPE_DIM
    d = tl.arange(0, ROPE_DIM)
    rotary = extract_slice(normalized, (nope_dim,), (ROPE_DIM,), (1,)).to(tl.float32)
    c = tl.load(long_cos + row * COS_ROW_STRIDE + d).to(tl.float32)
    s = tl.load(long_sin + row * SIN_ROW_STRIDE + d).to(tl.float32)
    partner = tl.gather(rotary, d ^ 1, 0)
    rotated = rotary * c + partner * s * tl.where(d % 2 == 0, -1.0, 1.0)
    dims = tl.arange(0, HEAD_DIM)
    tl.store(dst + dims, normalized, valid & (dims < nope_dim))
    tl.store(dst + nope_dim + d, rotated, valid)
    return normalized


@triton.jit(do_not_specialize=["num_tokens"])
def _normalize_and_store_c1_kernel(
    projected,
    weight_ptr,
    latent,
    num_tokens,
    long_cache,
    long_slots,
    long_cos,
    long_sin,
    CACHE_PAGE_STRIDE: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    SLOT_ROW_STRIDE: tl.constexpr,
    COS_ROW_STRIDE: tl.constexpr,
    SIN_ROW_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    dims = tl.arange(0, HEAD_DIM)
    weight = tl.load(weight_ptr + dims).to(tl.float32)
    for row in range(tl.program_id(0), num_tokens, tl.num_programs(0)):
        offset = row.to(tl.int64) * HEAD_DIM + dims
        value = tl.load(projected + offset)
        result = _normalize_and_store_long_kv(
            value,
            weight,
            row,
            long_cache,
            long_slots,
            long_cos,
            long_sin,
            CACHE_PAGE_STRIDE,
            CACHE_ROW_STRIDE,
            SLOT_ROW_STRIDE,
            COS_ROW_STRIDE,
            SIN_ROW_STRIDE,
            EPS,
            HEAD_DIM,
            ROPE_DIM,
        )
        tl.store(latent + offset, result)


def normalize_and_store_c1(projected: torch.Tensor, cache_store: LongKVStore) -> torch.Tensor:
    """Return unrotated BF16 latent and publish rotated rows to Long KV."""
    latent = torch.empty_like(projected)
    num_tokens = projected.shape[0]
    if num_tokens:
        init_device_properties_triton()
        _normalize_and_store_c1_kernel[(min(num_tokens, get_vectorcore_num()),)](
            projected,
            cache_store.weight,
            latent,
            num_tokens,
            EPS=cache_store.eps,
            HEAD_DIM=HEAD_DIM,
            ROPE_DIM=ROPE_DIM,
            enable_fp_fusion=False,
            **cache_store.kernel_args(),
        )
    return latent
