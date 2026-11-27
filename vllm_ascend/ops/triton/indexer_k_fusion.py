# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 RMSNorm/RoPE/INT8 publication for ordinary paged caches.

Only k_cache and scale_cache are modified, at valid unique [block, slot]
coordinates. Inputs must be finite and RoPE tables must repeat each angle
twice for interleaved rotation. Packed/folded caches and biased normalization
are handled by the production caller's existing path. See docs/indexer_k_fusion.md.
Hardware acceptance remains required; this module alone is not validation.
"""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

INT8_QUANT_MAX = 127.0  # Symmetric INT8 dynamic quantization uses [-127, 127].


@triton.jit
def _round_even(value):
    lower = tl.floor(value)
    fraction = value - lower
    lower_is_odd = (lower.to(tl.int32) & 1) != 0
    return lower + ((fraction > 0.5) | ((fraction == 0.5) & lower_is_odd)).to(tl.float32)


# Specialize on vector dimensions, weight/RoPE column strides and the fixed
# quantization algorithm. Row/slot addressing and cache bounds stay dynamic so
# changing row layouts or cache allocations does not add scalar-value variants.
# Epsilon is a runtime float; its value needs no integer specialization controls.
@triton.jit(
    do_not_specialize=[
        "input_row_stride",
        "cos_row_stride",
        "sin_row_stride",
        "slots_row_stride",
        "slots_col_stride",
        "key_block_stride",
        "key_row_stride",
        "scale_block_stride",
        "scale_row_stride",
        "num_blocks",
        "page_size",
    ]
)
def _indexer_k_norm_rope_int8_store_kernel(
    input_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    slots_ptr,
    key_cache_ptr,
    scale_cache_ptr,
    input_row_stride,
    cos_row_stride,
    sin_row_stride,
    slots_row_stride,
    slots_col_stride,
    key_block_stride,
    key_row_stride,
    scale_block_stride,
    scale_row_stride,
    num_blocks,
    page_size,
    rms_eps,
    WEIGHT_STRIDE: tl.constexpr,
    COS_COL_STRIDE: tl.constexpr,
    SIN_COL_STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    QUANT_MAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.load(slots_ptr + row * slots_row_stride).to(tl.int64)
    slot = tl.load(slots_ptr + row * slots_row_stride + slots_col_stride).to(tl.int64)
    valid = (block >= 0) & (block < num_blocks) & (slot >= 0) & (slot < page_size)
    if valid:
        pairs = tl.arange(0, HEAD_DIM // 2)
        even_offsets = 2 * pairs
        odd_offsets = even_offsets + 1
        even_values = tl.load(input_ptr + row * input_row_stride + even_offsets).to(tl.float32)
        odd_values = tl.load(input_ptr + row * input_row_stride + odd_offsets).to(tl.float32)
        mean_square = (tl.sum(even_values * even_values, 0) + tl.sum(odd_values * odd_values, 0)) / HEAD_DIM
        inv_rms = tl.rsqrt(mean_square + rms_eps)
        even_weight = tl.load(weight_ptr + even_offsets * WEIGHT_STRIDE).to(tl.float32)
        odd_weight = tl.load(weight_ptr + odd_offsets * WEIGHT_STRIDE).to(tl.float32)
        # The unfused RMSNorm produces BF16 before RoPE consumes its output.
        even_values = (even_values * inv_rms * even_weight).to(tl.bfloat16).to(tl.float32)
        odd_values = (odd_values * inv_rms * odd_weight).to(tl.bfloat16).to(tl.float32)

        rotary_offsets = even_offsets - (HEAD_DIM - ROTARY_DIM)
        rotary_mask = rotary_offsets >= 0
        safe_offsets = tl.maximum(rotary_offsets, 0)
        cos_values = tl.load(
            cos_ptr + row * cos_row_stride + safe_offsets * COS_COL_STRIDE,
            mask=rotary_mask,
            other=1.0,
        ).to(tl.float32)
        sin_values = tl.load(
            sin_ptr + row * sin_row_stride + safe_offsets * SIN_COL_STRIDE,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        rotated_even = (even_values * cos_values - odd_values * sin_values).to(tl.bfloat16).to(tl.float32)
        rotated_odd = (odd_values * cos_values + even_values * sin_values).to(tl.bfloat16).to(tl.float32)
        abs_max = tl.maximum(tl.max(tl.abs(rotated_even), 0), tl.max(tl.abs(rotated_odd), 0))
        scale = abs_max / QUANT_MAX
        # Quantize with FP32 scale; only the published scale is converted to FP16.
        # Zero-row semantics must be checked against the deployed native operator.
        divisor = tl.where(abs_max > 0.0, scale, 1.0)
        quant_even = tl.minimum(tl.maximum(_round_even(rotated_even / divisor), -QUANT_MAX), QUANT_MAX)
        quant_odd = tl.minimum(tl.maximum(_round_even(rotated_odd / divisor), -QUANT_MAX), QUANT_MAX)
        key_base = key_cache_ptr + block * key_block_stride + slot * key_row_stride
        tl.store(key_base + even_offsets, quant_even.to(tl.int8))
        tl.store(key_base + odd_offsets, quant_odd.to(tl.int8))
        tl.store(
            scale_cache_ptr + block * scale_block_stride + slot * scale_row_stride,
            scale.to(tl.float16),
        )


class IndexerKFusionKernel(AscendTritonJitKernel["IndexerKFusionKernel.CompileKey"]):
    kernel = _indexer_k_norm_rope_int8_store_kernel

    @dataclass(frozen=True)
    class CompileKey:
        head_dim: int
        rotary_dim: int
        weight_dtype: torch.dtype
        slots_dtype: torch.dtype
        weight_stride: int
        cos_col_stride: int
        sin_col_stride: int

    def dispatch(
        self,
        *,
        head_dim: int,
        rotary_dim: int,
        weight_dtype: torch.dtype,
        slots_dtype: torch.dtype,
        weight_stride: int,
        cos_col_stride: int,
        sin_col_stride: int,
    ) -> "IndexerKFusionKernel.CompileKey":
        return self.CompileKey(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            weight_dtype=weight_dtype,
            slots_dtype=slots_dtype,
            weight_stride=weight_stride,
            cos_col_stride=cos_col_stride,
            sin_col_stride=sin_col_stride,
        )

    def get_warmup_keys(
        self,
        *,
        head_dim: int,
        rotary_dim: int,
        vector_strides: tuple[int, int, int] = (1, 1, 1),
    ) -> list["IndexerKFusionKernel.CompileKey"]:
        # Production weights and source-selected RoPE rows have unit column
        # strides. Other supported views can register their own vector strides.
        # Row strides/bounds are do_not_specialize; batch size only sets grid.
        return [
            self.dispatch(
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                weight_dtype=weight_dtype,
                slots_dtype=slots_dtype,
                weight_stride=vector_strides[0],
                cos_col_stride=vector_strides[1],
                sin_col_stride=vector_strides[2],
            )
            for weight_dtype in (torch.bfloat16, torch.float32)
            for slots_dtype in (torch.int32, torch.int64)
        ]

    def warmup_inputs(self, key: "IndexerKFusionKernel.CompileKey") -> dict[str, Any]:
        return dict(
            input=TritonWarmupTensor(torch.bfloat16, shape=(1, key.head_dim)),
            weight=TritonWarmupTensor(
                key.weight_dtype,
                shape=(key.head_dim,),
                strides=(key.weight_stride,),
            ),
            cos=TritonWarmupTensor(
                torch.float32,
                shape=(1, key.rotary_dim),
                strides=(key.rotary_dim * key.cos_col_stride, key.cos_col_stride),
            ),
            sin=TritonWarmupTensor(
                torch.float32,
                shape=(1, key.rotary_dim),
                strides=(key.rotary_dim * key.sin_col_stride, key.sin_col_stride),
            ),
            slots=TritonWarmupTensor(key.slots_dtype, shape=(1, 2)),
            key_cache=TritonWarmupTensor(torch.int8, shape=(1, 1, 1, key.head_dim)),
            scale_cache=TritonWarmupTensor(torch.float16, shape=(1, 1, 1, 1)),
            rms_eps=1e-6,
        )

    @kernel_launcher
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slots: torch.Tensor,
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
        *,
        rms_eps: float,
    ) -> LaunchSpec:
        return (input.shape[0],), dict(
            input_row_stride=input.stride(0),
            cos_row_stride=cos.stride(0),
            sin_row_stride=sin.stride(0),
            slots_row_stride=slots.stride(0),
            slots_col_stride=slots.stride(1),
            key_block_stride=key_cache.stride(0),
            key_row_stride=key_cache.stride(1),
            scale_block_stride=scale_cache.stride(0),
            scale_row_stride=scale_cache.stride(1),
            num_blocks=key_cache.shape[0],
            page_size=key_cache.shape[1],
            WEIGHT_STRIDE=weight.stride(0),
            COS_COL_STRIDE=cos.stride(1),
            SIN_COL_STRIDE=sin.stride(1),
            HEAD_DIM=input.shape[1],
            ROTARY_DIM=cos.shape[1],
            QUANT_MAX=INT8_QUANT_MAX,
            enable_fp_fusion=False,
        )


# Shared by model registration and runtime; only the warmup framework owns
# transient compilation state. No request tensors are retained here.
_INDEXER_K_FUSION_KERNEL = IndexerKFusionKernel()


def indexer_k_norm_rope_int8_store(
    k_pre: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    scale_cache: torch.Tensor,
) -> None:
    """Publish pre-normalization wk output into caller-owned K/scale caches.

    Shapes: k_pre [T,D], weight [D], slots [T,2], cos/sin [T,R] or
    [T,1,1,R], k_cache [B,P,1,D], scale_cache [B,P,1,1]. D is an even
    power of two; 0 < R <= D and R is even. Inputs use one NPU. k_pre is
    BF16, weight BF16/FP32, cos/sin FP32, slots INT32/INT64, and caches
    INT8/FP16. Cache rows may have physical padding. Valid slots must be
    unique; out-of-range slots are skipped. Inputs must not alias outputs.

    The caller guarantees finite values, paired RoPE columns, unique slots
    and stream ordering. Checking tensor contents here would synchronize
    the hot path. Empty batches are a no-op after metadata validation.
    No tensor-content host reads, implicit contiguous copies or allocation.
    """
    if k_pre.ndim != 2 or k_pre.dtype != torch.bfloat16:
        raise ValueError("k_pre must be [T,D] BF16")
    num_tokens, head_dim = k_pre.shape
    if head_dim < 2 or head_dim & (head_dim - 1) or k_pre.stride(1) != 1:
        raise ValueError("D must be an even power of two with stride(1)=1")
    if weight.shape != (head_dim,) or weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("weight must be BF16/FP32 [D]")
    if not 0 < eps < float("inf"):
        raise ValueError("Require finite positive model RMSNorm epsilon")
    if slots.shape != (num_tokens, 2) or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slots must be INT32/INT64 [T,2]")
    if cos.ndim == 4 and cos.shape[1:3] == (1, 1):
        cos = cos[:, 0, 0, :]
    if sin.ndim == 4 and sin.shape[1:3] == (1, 1):
        sin = sin[:, 0, 0, :]
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != num_tokens:
        raise ValueError("cos/sin must be [T,R] or [T,1,1,R]")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("Require source-selected FP32 RoPE tables")
    rotary_dim = cos.shape[1]
    if not (0 < rotary_dim <= head_dim and rotary_dim % 2 == 0):
        raise ValueError("R must be positive, even, and <= D")
    if k_cache.ndim != 4 or k_cache.shape[2:] != (1, head_dim):
        raise ValueError("k_cache must be [blocks,page,1,D]")
    num_blocks, page_size = k_cache.shape[:2]
    if scale_cache.shape != (num_blocks, page_size, 1, 1):
        raise ValueError("scale_cache must be [blocks,page,1,1]")
    if k_cache.dtype != torch.int8 or scale_cache.dtype != torch.float16:
        raise ValueError("Require INT8 K and FP16 dequant scales")
    if (
        k_cache.stride(3) != 1
        or k_cache.stride(1) < head_dim
        or k_cache.stride(0) < page_size * k_cache.stride(1)
        or scale_cache.stride(1) < 1
        or scale_cache.stride(0) < page_size * scale_cache.stride(1)
    ):
        raise ValueError("Require internally contiguous, nonoverlapping cache rows")
    tensors = (weight, cos, sin, slots, k_cache, scale_cache)
    if k_pre.device.type != "npu" or any(tensor.device != k_pre.device for tensor in tensors):
        raise ValueError("All tensors must share one NPU")
    if num_tokens == 0:
        return
    _INDEXER_K_FUSION_KERNEL(
        input=k_pre,
        weight=weight,
        cos=cos,
        sin=sin,
        slots=slots,
        key_cache=k_cache,
        scale_cache=scale_cache,
        rms_eps=float(eps),
    )
