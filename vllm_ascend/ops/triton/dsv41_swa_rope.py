# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AIV-budgeted persistent Triton RoPE + paged scatter kernel for DeepSeek V4.1 SWA."""

from __future__ import annotations

import torch
import torch_npu
from vllm.triton_utils import tl, triton


def swa_aux_vector_core_budget() -> int:
    """Return the AIV vector core budget for the SWA auxiliary overlap stream.

    Caps background SWA vector work to 20% of AIV cores (min 4; 8/40 on
    Ascend 910C/A3) so the main stream retains 80% of AIV cores and shared
    L2/HBM bandwidth during high-concurrency decode.
    """
    vec_cores = 0
    try:
        from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

        vec_cores = int(get_vectorcore_num())
    except Exception:
        vec_cores = 0
    if vec_cores <= 0:
        try:
            get_limit = getattr(torch_npu.npu, "get_device_limit", None)
            if callable(get_limit):
                limit = get_limit(torch.npu.current_device())
                if isinstance(limit, dict):
                    vec_cores = int(limit.get("vector_core_num", 0))
        except Exception:
            vec_cores = 0
    if vec_cores <= 0:
        vec_cores = 40
    return max(4, vec_cores // 5)


@triton.jit
def _store_swa_with_rope_kernel(
    kv_ptr,
    cos_ptr,
    sin_ptr,
    cache_ptr,
    slots_ptr,
    num_tokens,
    KV_ROW: tl.constexpr,
    COS_ROW: tl.constexpr,
    SIN_ROW: tl.constexpr,
    SLOT_ROW: tl.constexpr,
    SLOT_COL: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    HEAD: tl.constexpr,
    ROTARY: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    ROTARY_TILE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    stride = tl.num_programs(0).to(tl.int64)
    n_tok = num_tokens.to(tl.int64)
    plain: tl.constexpr = HEAD - ROTARY
    dim = tl.arange(0, HEAD_TILE)
    r = tl.arange(0, ROTARY_TILE)

    row = pid
    while row < n_tok:
        page = tl.load(slots_ptr + row * SLOT_ROW).to(tl.int64)
        offset = tl.load(slots_ptr + row * SLOT_ROW + SLOT_COL).to(tl.int64)
        if (page >= 0) & (offset >= 0):
            dst = cache_ptr + page * PAGE_STRIDE + offset * ROW_STRIDE
            # Copy unrotated prefix unchanged (strict 0-ULP).
            plain_vals = tl.load(kv_ptr + row * KV_ROW + dim, mask=dim < plain)
            tl.store(dst + dim, plain_vals, mask=dim < plain)

            # FP32 interleaved partial RoPE matching inplace_partial_rotary_mul (strict 0-ULP).
            x = tl.load(kv_ptr + row * KV_ROW + plain + r, mask=r < ROTARY, other=0).to(tl.float32)
            c = tl.load(cos_ptr + row * COS_ROW + r, mask=r < ROTARY, other=0).to(tl.float32)
            s = tl.load(sin_ptr + row * SIN_ROW + r, mask=r < ROTARY, other=0).to(tl.float32)
            even, odd = tl.split(tl.reshape(x * s, (ROTARY_TILE // 2, 2)))
            rotated = x * c + tl.reshape(tl.join(-odd, even), (ROTARY_TILE,))
            tl.store(dst + plain + r, rotated.to(tl.bfloat16), mask=r < ROTARY)
        row += stride


def can_store_swa_with_rope(
    kv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    head_dim: int,
    nope_head_dim: int,
) -> bool:
    """Return True iff the BF16 SWA tensors satisfy the fused RoPE+scatter contract."""
    rotary = head_dim - nope_head_dim
    if kv.ndim != 2 or cos.ndim != 4 or sin.shape != cos.shape:
        return False
    tokens, width = kv.shape
    return (
        tokens > 0
        and width == head_dim
        and 0 < rotary <= width
        and (rotary % 2 == 0)
        and kv.device.type == "npu"
        and kv.dtype == cache.dtype == torch.bfloat16
        and cos.dtype == sin.dtype == torch.float32
        and cos.shape[0] >= tokens
        and cos.shape[1:] == (1, 1, rotary)
        and cache.ndim == 4
        and cache.shape[-2:] == (1, width)
        and slots.ndim == 2
        and slots.shape[0] >= tokens
        and slots.shape[1] == 2
        and slots.dtype in (torch.int32, torch.int64)
        and all(t.stride(-1) == 1 for t in (kv, cos, sin, cache))
    )


def store_swa_with_rope(
    kv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    head_dim: int,
    nope_head_dim: int,
) -> None:
    """Apply FP32 interleaved partial RoPE and scatter BF16 SWA rows into paged cache.

    Launches a persistent Triton grid capped to ``swa_aux_vector_core_budget()``
    programs because Triton kernels on Ascend ignore ``set_stream_limit``.
    """
    num_tokens = int(kv.shape[0])
    if num_tokens <= 0:
        return
    rotary = head_dim - nope_head_dim
    num_programs = min(num_tokens, swa_aux_vector_core_budget())
    _store_swa_with_rope_kernel[(num_programs,)](
        kv,
        cos,
        sin,
        cache,
        slots,
        num_tokens,
        KV_ROW=kv.stride(0),
        COS_ROW=cos.stride(0),
        SIN_ROW=sin.stride(0),
        SLOT_ROW=slots.stride(0),
        SLOT_COL=slots.stride(1),
        PAGE_STRIDE=cache.stride(0),
        ROW_STRIDE=cache.stride(1),
        HEAD=head_dim,
        ROTARY=rotary,
        HEAD_TILE=triton.next_power_of_2(head_dim),
        ROTARY_TILE=triton.next_power_of_2(rotary),
    )
