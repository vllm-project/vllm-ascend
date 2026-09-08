# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""K3 Phase-1 ``block_attn_res_prepare`` CANNDSL implementation.

整网接口沿有效历史 block 轴做多 Query 注意力残差聚合：

    qe_s     = effective_queries[s]                     整网预先融合 q ⊙ g
    logit_sn = (qe_s · V_n) / sqrt(mean_d V_n^2 + eps)  RMSNorm 后点积（rms 折进打分,不物化 K）
    max      = max_n(logit_n)                            每个 token 的稳定 softmax 最大值
    sum      = Σ_n exp(logit_n - max)                   每个 token 的稳定 softmax 指数和
    numerator = Σ_n exp(logit_n - max) · V_n             未归一化加权和

纯 AIV（Vector）核，D 进 lane（段循环）。Prefill 按 token 分核，每个 token 把固定
容量 [max_blocks,D] 搬入 UB 并复用给全部 slots；decode 的小 token shape 按
token×slot 分核，用少量 V 重读换取多核并行。valid_blocks 从设备标量 Tensor
读取，eps 是编译期 Python float；无效 block lane 不参与
max/sum/numerator。核内 fp32 计算，numerator/max/sum 均固定写回 fp32。
三个输出直接作为 ``block_attn_res_update`` 的 Phase-1 输入。
"""

from dataclasses import dataclass
from functools import lru_cache

import cannbotdsl
import torch

from cannbotdsl import dtypes
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.control_flow import range as dsl_range
from cannbotdsl.core.frontend.compiler import compile_function
from cannbotdsl.buffer import Buffer
from cannbotdsl.dtypes import (
    bfloat16 as BFloat16,
    float16 as Float16,
    float32 as Float32,
)
from cannbotdsl.integer import Int64
from cannbotdsl.tensor import local_slice, tile_view, mem_copy
from cannbotdsl.typing.types import MemLoc, Tensor
from cannbotdsl.vf import vf
from cannbotdsl.raw_reg import (
    UnpackMode,
    full_mask,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup_lane0,
    vdup_scalar,
    vexp_sub,
    vload,
    vload_brc,
    vload_unpack,
    vmem_bar,
    vmerge,
    vmax,
    vmul,
    vmuls,
    vreduce_max,
    vreduce_sum,
    vsqrt,
    vstore,
    vstore_first,
)

VL = 64
DEFAULT_BLOCK_NUM = 64
MAX_D = 8192
UB_BYTES = 240 * 1024  # Arch35 UB 256KB,留 ~16KB 给对齐/保留
# K3 decode 网络 shape 为 T=1/2。仅对这两类 shape 使用 token×slot
# 并行，避免改变 T>=4 的 prefill 路径及其 V 复用特性。
DECODE_SLOT_PARALLEL_MAX_TOKENS = 2

_TORCH_TO_DSL = {torch.bfloat16: BFloat16, torch.float16: Float16, torch.float32: Float32}
_COMPILED_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


@dataclass(frozen=True, slots=True)
class PrepareStaticShape:
    """Static dimensions and dtype shared by launch planning and compilation."""

    max_blocks: int
    tokens: int
    slots: int
    d: int
    input_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class PrepareLaunchPlan:
    """Named compile/launch configuration cached for one static network shape."""

    shape: PrepareStaticShape
    mode: str
    d_tile: int
    block_num: int
    eps: float


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _require(condition: bool, message: str) -> None:
    """Raise a stable input-contract error."""
    if not condition:
        raise ValueError(message)


def _use_decode_slot_parallel(
    tokens: int,
    slots: int,
    required_ub: int,
) -> bool:
    """Return whether a hold-capable shape should parallelize token×slot.

    The stream kernel already uses depth-2 D tiling and is left unchanged.  The
    decode path deliberately targets only K3's T=1/2 shapes so prefill keeps
    loading each token's V once and reusing it across all query slots.
    """
    return (
        0 < int(tokens) <= DECODE_SLOT_PARALLEL_MAX_TOKENS
        and int(slots) > 1
        and int(required_ub) <= UB_BYTES
    )


def _stream_decode_query_group(tokens: int, slots: int) -> int:
    """Balance decode core parallelism against V reuse per core.

    Keep roughly 24 independent token×query-group work items. Small slot sets
    use group=1 for maximum parallelism; larger sets reuse one V tile across a
    small group instead of serializing every query on a single token core.
    """
    work = max(1, int(tokens) * int(slots))
    return max(1, _ceil_div(work, 24))


def _stream_prefill_query_group(tokens: int, slots: int) -> int:
    """Use query groups only when token-only prefill underfills AIV cores."""
    tokens = int(tokens)
    slots = int(slots)
    if tokens <= 0 or tokens >= DEFAULT_BLOCK_NUM or slots <= 1:
        return max(1, slots)
    groups_per_token = _ceil_div(DEFAULT_BLOCK_NUM, tokens)
    return max(1, slots // groups_per_token)


def _decode_online_query_group(tokens: int, slots: int, d: int) -> int:
    """Choose a query group whose Q/numerator state fits with one full V."""
    wanted = _stream_decode_query_group(tokens, slots)
    w = _ceil_div(int(d), VL) * VL
    for group in range(int(wanted), 0, -1):
        required = (
            (2 * group + 1) * w * 4
            + (2 * group + 3) * VL * 4
        )
        if required <= UB_BYTES:
            return group
    return 0


def _launch_block_num(mode: str, tokens: int, slots: int, d: int = 0) -> int:
    """Launch only as many blocks as the selected path can use."""
    if mode == "decode":
        work_items = int(tokens) * int(slots)
    elif mode == "stream_decode":
        group = _stream_decode_query_group(tokens, slots)
        work_items = int(tokens) * _ceil_div(int(slots), group)
    elif mode == "decode_online":
        group = _decode_online_query_group(tokens, slots, d)
        work_items = int(tokens) * _ceil_div(int(slots), max(1, group))
    elif mode == "stream_prefill_group":
        group = _stream_prefill_query_group(tokens, slots)
        work_items = int(tokens) * _ceil_div(int(slots), group)
    else:
        work_items = int(tokens)
    return max(1, min(DEFAULT_BLOCK_NUM, work_items))


# These VF callbacks are expanded by ``compile_function``. Their Buffer/scalar
# operands must remain flattened because the DSL preprocessor does not accept a
# Python dataclass as a kernel-region ABI value.
# pylint: disable=too-many-arguments,too-many-positional-arguments
# --- 打分 body：seg 4 路 unroll(4 部分累加器打破依赖链)。logit_n = (qe·V_n)/sqrt(mean V_n^2 + eps) --
def _score_body_16bit(v_ub, qe_ub, logits_ub, nplus1, num_seg, w, avg, eps, num_col):
    ng = (num_col // VL) // 4          # 4 路 unroll 只覆盖完整 VL 段;尾段（含非对齐 D 的偏段）走掩码
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w
            d0 = vdup_scalar(0.0, Float32, mask=full)
            d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full)
            d3 = vdup_scalar(0.0, Float32, mask=full)
            s0 = vdup_scalar(0.0, Float32, mask=full)
            s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full)
            s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL)
                qb = it * (4 * VL)
                x0 = vcast(vload_unpack(v_ub, vb, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x1 = vcast(vload_unpack(v_ub, vb + VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x2 = vcast(vload_unpack(v_ub, vb + 2 * VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x3 = vcast(vload_unpack(v_ub, vb + 3 * VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                q0 = vload(qe_ub, qb)
                q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL)
                q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full)
                d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full)
                d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full)
                s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full)
                s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            dot = vadd(vadd(d0, d1, mask=full), vadd(d2, d3, mask=full), mask=full)
            ssq = vadd(vadd(s0, s1, mask=full), vadd(s2, s3, mask=full), mask=full)
            for seg in range(Int64(4 * ng), Int64(num_seg)):
                mask, _ = update_mask(Int64(num_col) - seg * VL, elem_bits=32)   # 尾段 partial(非对齐 D)/满(对齐)
                off = base_n + seg * VL
                xf = vcast(vload_unpack(v_ub, off, mode=UnpackMode.B16_TO_B32), Float32, mask=mask)
                qf = vload(qe_ub, seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=mask), mask=full)
                ssq = vadd(ssq, vmul(xf, xf, mask=mask), mask=full)
            denom = vsqrt(vadds(vmuls(vreduce_sum(ssq, mask=full), avg, mask=full), eps, mask=full), mask=full)
            vstore_first(logits_ub, n, vdiv(vreduce_sum(dot, mask=full), denom, mask=full))


def _score_body_fp32(v_ub, qe_ub, logits_ub, nplus1, num_seg, w, avg, eps, num_col):
    ng = (num_col // VL) // 4          # 4 路 unroll 只覆盖完整 VL 段;尾段（含非对齐 D 的偏段）走掩码
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w
            d0 = vdup_scalar(0.0, Float32, mask=full)
            d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full)
            d3 = vdup_scalar(0.0, Float32, mask=full)
            s0 = vdup_scalar(0.0, Float32, mask=full)
            s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full)
            s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL)
                qb = it * (4 * VL)
                x0 = vload(v_ub, vb)
                x1 = vload(v_ub, vb + VL)
                x2 = vload(v_ub, vb + 2 * VL)
                x3 = vload(v_ub, vb + 3 * VL)
                q0 = vload(qe_ub, qb)
                q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL)
                q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full)
                d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full)
                d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full)
                s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full)
                s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            dot = vadd(vadd(d0, d1, mask=full), vadd(d2, d3, mask=full), mask=full)
            ssq = vadd(vadd(s0, s1, mask=full), vadd(s2, s3, mask=full), mask=full)
            for seg in range(Int64(4 * ng), Int64(num_seg)):
                mask, _ = update_mask(Int64(num_col) - seg * VL, elem_bits=32)   # 尾段 partial(非对齐 D)/满(对齐)
                off = base_n + seg * VL
                xf = vload(v_ub, off)
                qf = vload(qe_ub, seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=mask), mask=full)
                ssq = vadd(ssq, vmul(xf, xf, mask=mask), mask=full)
            denom = vsqrt(vadds(vmuls(vreduce_sum(ssq, mask=full), avg, mask=full), eps, mask=full), mask=full)
            vstore_first(logits_ub, n, vdiv(vreduce_sum(dot, mask=full), denom, mask=full))


def _hold_inv_rms_body_fp32(v_ub, inv_rms_ub, nplus1, num_seg,
                            w, avg, eps, num_col):
    """Compute query-independent inverse RMS once for every resident block."""
    ng = (num_col // VL) // 4
    with vf(mode="raw"):
        full = full_mask(32)
        one = vdup_scalar(1.0, Float32, mask=full)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w
            s0 = vdup_scalar(0.0, Float32, mask=full)
            s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full)
            s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                base = base_n + it * (4 * VL)
                x0 = vload(v_ub, base)
                x1 = vload(v_ub, base + VL)
                x2 = vload(v_ub, base + 2 * VL)
                x3 = vload(v_ub, base + 3 * VL)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full)
                s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full)
                s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            ssq = vadd(
                vadd(s0, s1, mask=full),
                vadd(s2, s3, mask=full),
                mask=full,
            )
            for seg in range(Int64(4 * ng), Int64(num_seg)):
                mask, _ = update_mask(
                    Int64(num_col) - seg * VL, elem_bits=32,
                )
                value = vload(v_ub, base_n + seg * VL)
                ssq = vadd(
                    ssq, vmul(value, value, mask=mask), mask=full,
                )
            denom = vsqrt(
                vadds(
                    vmuls(vreduce_sum(ssq, mask=full), avg, mask=full),
                    eps,
                    mask=full,
                ),
                mask=full,
            )
            vstore_first(
                inv_rms_ub, n, vdiv(one, denom, mask=full),
            )


def _score_hold_dot_body_fp32(v_ub, qe_ub, logits_ub, inv_rms_ub,
                              nplus1, num_seg, w, num_col):
    """Compute only q dot V, then reuse the precomputed inverse RMS."""
    ng = (num_col // VL) // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w
            d0 = vdup_scalar(0.0, Float32, mask=full)
            d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full)
            d3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL)
                qb = it * (4 * VL)
                x0 = vload(v_ub, vb)
                x1 = vload(v_ub, vb + VL)
                x2 = vload(v_ub, vb + 2 * VL)
                x3 = vload(v_ub, vb + 3 * VL)
                q0 = vload(qe_ub, qb)
                q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL)
                q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full)
                d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full)
                d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
            dot = vadd(
                vadd(d0, d1, mask=full),
                vadd(d2, d3, mask=full),
                mask=full,
            )
            for seg in range(Int64(4 * ng), Int64(num_seg)):
                mask, _ = update_mask(
                    Int64(num_col) - seg * VL, elem_bits=32,
                )
                value = vload(v_ub, base_n + seg * VL)
                query = vload(qe_ub, seg * VL)
                dot = vadd(
                    dot, vmul(value, query, mask=mask), mask=full,
                )
            score = vmul(
                vreduce_sum(dot, mask=full),
                vload_brc(inv_rms_ub, n),
                mask=full,
            )
            vstore_first(logits_ub, n, score)


def _online_init_body_fp32(v_ub, numerator_ub, max_state_ub,
                           sum_state_ub, logits_ub, query_lane,
                           num_seg, w, num_col):
    """Initialize online softmax state from candidate zero."""
    with vf(mode="raw"):
        full = full_mask(32)
        state_off = query_lane * VL
        numerator_off = query_lane * w
        score = vload_brc(logits_ub, 0)
        one = vdup_scalar(1.0, Float32, mask=full)
        vstore(max_state_ub, state_off, score, full)
        vstore(sum_state_ub, state_off, one, full)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(num_col) - off, elem_bits=32)
            vstore(
                numerator_ub,
                numerator_off + off,
                vload(v_ub, off),
                mask,
            )


def _online_merge_body_fp32(v_ub, numerator_ub, max_state_ub,
                            sum_state_ub, logits_ub, query_lane,
                            num_seg, w, num_col):
    """Merge one candidate into stable (numerator,max,sum) state."""
    with vf(mode="raw"):
        full = full_mask(32)
        state_off = query_lane * VL
        numerator_off = query_lane * w
        score = vload_brc(logits_ub, 0)
        old_max = vload(max_state_ub, state_off)
        old_sum = vload(sum_state_ub, state_off)
        new_max = vmax(old_max, score, mask=full)
        old_scale = vexp_sub(old_max, new_max, mask=full)
        new_scale = vexp_sub(score, new_max, mask=full)
        new_sum = vadd(
            vmul(old_sum, old_scale, mask=full),
            new_scale,
            mask=full,
        )
        vstore(max_state_ub, state_off, new_max, full)
        vstore(sum_state_ub, state_off, new_sum, full)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(num_col) - off, elem_bits=32)
            old_numerator = vload(numerator_ub, numerator_off + off)
            candidate = vload(v_ub, off)
            merged = vadd(
                vmul(old_numerator, old_scale, mask=mask),
                vmul(candidate, new_scale, mask=mask),
                mask=mask,
            )
            vstore(numerator_ub, numerator_off + off, merged, mask)


def _online_zero_body_fp32(numerator_ub, max_state_ub, sum_state_ub,
                           query_lane, num_seg, w, num_col):
    """Materialize the empty-attention identity for one query lane."""
    with vf(mode="raw"):
        full = full_mask(32)
        state_off = query_lane * VL
        numerator_off = query_lane * w
        zero = vdup_scalar(0.0, Float32, mask=full)
        one = vdup_scalar(1.0, Float32, mask=full)
        neg_flt_max = vdup_scalar(
            -3.4028234663852886e38, Float32, mask=full,
        )
        vstore(max_state_ub, state_off, neg_flt_max, full)
        vstore(sum_state_ub, state_off, one, full)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            mask, _ = update_mask(Int64(num_col) - off, elem_bits=32)
            vstore(numerator_ub, numerator_off + off, zero, mask)


# --- 聚合 body：候选 n 4 路 unroll(4 部分累加器)。h[seg] = Σ_n w_n · V_n[seg] --------------------------------
def _agg_body_16bit(v_ub, logits_ub, h_ub, nplus1, num_seg, w,
                    logits_off):
    ng = nplus1 // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            a0 = vdup_scalar(0.0, Float32, mask=full)
            a1 = vdup_scalar(0.0, Float32, mask=full)
            a2 = vdup_scalar(0.0, Float32, mask=full)
            a3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                nb = it * 4
                w0 = vload_brc(logits_ub, logits_off + nb)
                w1 = vload_brc(logits_ub, logits_off + nb + 1)
                w2 = vload_brc(logits_ub, logits_off + nb + 2)
                w3 = vload_brc(logits_ub, logits_off + nb + 3)
                x0 = vcast(vload_unpack(v_ub, nb * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x1 = vcast(vload_unpack(v_ub, (nb + 1) * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x2 = vcast(vload_unpack(v_ub, (nb + 2) * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x3 = vcast(vload_unpack(v_ub, (nb + 3) * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                a0 = vadd(a0, vmul(x0, w0, mask=full), mask=full)
                a1 = vadd(a1, vmul(x1, w1, mask=full), mask=full)
                a2 = vadd(a2, vmul(x2, w2, mask=full), mask=full)
                a3 = vadd(a3, vmul(x3, w3, mask=full), mask=full)
            acc = vadd(vadd(a0, a1, mask=full), vadd(a2, a3, mask=full), mask=full)
            for n in range(Int64(4 * ng), Int64(nplus1)):
                wn = vload_brc(logits_ub, logits_off + n)
                xf = vcast(vload_unpack(v_ub, n * w + off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                acc = vadd(acc, vmul(xf, wn, mask=full), mask=full)
            vstore(h_ub, off, acc, full)


def _agg_body_fp32(v_ub, logits_ub, h_ub, nplus1, num_seg, w,
                   logits_off):
    ng = nplus1 // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for seg in range(Int64(0), Int64(num_seg)):
            off = seg * VL
            a0 = vdup_scalar(0.0, Float32, mask=full)
            a1 = vdup_scalar(0.0, Float32, mask=full)
            a2 = vdup_scalar(0.0, Float32, mask=full)
            a3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                nb = it * 4
                w0 = vload_brc(logits_ub, logits_off + nb)
                w1 = vload_brc(logits_ub, logits_off + nb + 1)
                w2 = vload_brc(logits_ub, logits_off + nb + 2)
                w3 = vload_brc(logits_ub, logits_off + nb + 3)
                x0 = vload(v_ub, nb * w + off)
                x1 = vload(v_ub, (nb + 1) * w + off)
                x2 = vload(v_ub, (nb + 2) * w + off)
                x3 = vload(v_ub, (nb + 3) * w + off)
                a0 = vadd(a0, vmul(x0, w0, mask=full), mask=full)
                a1 = vadd(a1, vmul(x1, w1, mask=full), mask=full)
                a2 = vadd(a2, vmul(x2, w2, mask=full), mask=full)
                a3 = vadd(a3, vmul(x3, w3, mask=full), mask=full)
            acc = vadd(vadd(a0, a1, mask=full), vadd(a2, a3, mask=full), mask=full)
            for n in range(Int64(4 * ng), Int64(nplus1)):
                wn = vload_brc(logits_ub, logits_off + n)
                xf = vload(v_ub, n * w + off)
                acc = vadd(acc, vmul(xf, wn, mask=full), mask=full)
            vstore(h_ub, off, acc, full)


# --- 流式打分 body：只算「当前 Dt 块」每候选的部分 dot/ssq,写 pdot/pssq 的 lane n（跨块累加见 kernel）----
#   大 D 装不下 hold-all 时用:分 Dt 块流式过 D,dot/ssq 是标量(每候选)可跨块累加。qe 常驻[D],读偏移 qe_off。
#   Dt 为 VL 整数倍(host 保证),块内无非对齐尾;不除 rms(要全 D 的 ssq),留到 finalize。
def _score_stream_body_16bit(v_dt, qe_ub, pdot, pssq, nplus1, num_seg_dt, w_dt, qe_off):
    ng = num_seg_dt // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w_dt
            d0 = vdup_scalar(0.0, Float32, mask=full)
            d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full)
            d3 = vdup_scalar(0.0, Float32, mask=full)
            s0 = vdup_scalar(0.0, Float32, mask=full)
            s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full)
            s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL)
                qb = qe_off + it * (4 * VL)
                x0 = vcast(vload_unpack(v_dt, vb, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x1 = vcast(vload_unpack(v_dt, vb + VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x2 = vcast(vload_unpack(v_dt, vb + 2 * VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                x3 = vcast(vload_unpack(v_dt, vb + 3 * VL, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                q0 = vload(qe_ub, qb)
                q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL)
                q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full)
                d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full)
                d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full)
                s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full)
                s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            dot = vadd(vadd(d0, d1, mask=full), vadd(d2, d3, mask=full), mask=full)
            ssq = vadd(vadd(s0, s1, mask=full), vadd(s2, s3, mask=full), mask=full)
            for seg in range(Int64(4 * ng), Int64(num_seg_dt)):
                off = base_n + seg * VL
                xf = vcast(vload_unpack(v_dt, off, mode=UnpackMode.B16_TO_B32), Float32, mask=full)
                qf = vload(qe_ub, qe_off + seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=full), mask=full)
                ssq = vadd(ssq, vmul(xf, xf, mask=full), mask=full)
            vstore_first(pdot, n, vreduce_sum(dot, mask=full))
            vstore_first(pssq, n, vreduce_sum(ssq, mask=full))


def _score_stream_body_fp32(v_dt, qe_ub, pdot, pssq, nplus1, num_seg_dt, w_dt, qe_off):
    ng = num_seg_dt // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w_dt
            d0 = vdup_scalar(0.0, Float32, mask=full)
            d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full)
            d3 = vdup_scalar(0.0, Float32, mask=full)
            s0 = vdup_scalar(0.0, Float32, mask=full)
            s1 = vdup_scalar(0.0, Float32, mask=full)
            s2 = vdup_scalar(0.0, Float32, mask=full)
            s3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL)
                qb = qe_off + it * (4 * VL)
                x0 = vload(v_dt, vb)
                x1 = vload(v_dt, vb + VL)
                x2 = vload(v_dt, vb + 2 * VL)
                x3 = vload(v_dt, vb + 3 * VL)
                q0 = vload(qe_ub, qb)
                q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL)
                q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full)
                d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full)
                d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
                s0 = vadd(s0, vmul(x0, x0, mask=full), mask=full)
                s1 = vadd(s1, vmul(x1, x1, mask=full), mask=full)
                s2 = vadd(s2, vmul(x2, x2, mask=full), mask=full)
                s3 = vadd(s3, vmul(x3, x3, mask=full), mask=full)
            dot = vadd(vadd(d0, d1, mask=full), vadd(d2, d3, mask=full), mask=full)
            ssq = vadd(vadd(s0, s1, mask=full), vadd(s2, s3, mask=full), mask=full)
            for seg in range(Int64(4 * ng), Int64(num_seg_dt)):
                off = base_n + seg * VL
                xf = vload(v_dt, off)
                qf = vload(qe_ub, qe_off + seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=full), mask=full)
                ssq = vadd(ssq, vmul(xf, xf, mask=full), mask=full)
            vstore_first(pdot, n, vreduce_sum(dot, mask=full))
            vstore_first(pssq, n, vreduce_sum(ssq, mask=full))


def _score_stream_dot_body_fp32(v_dt, qe_ub, pdot, nplus1,
                                num_seg_dt, w_dt):
    """Score one query tile while reusing the query-independent RMS sum."""
    ng = num_seg_dt // 4
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w_dt
            d0 = vdup_scalar(0.0, Float32, mask=full)
            d1 = vdup_scalar(0.0, Float32, mask=full)
            d2 = vdup_scalar(0.0, Float32, mask=full)
            d3 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                vb = base_n + it * (4 * VL)
                qb = it * (4 * VL)
                x0 = vload(v_dt, vb)
                x1 = vload(v_dt, vb + VL)
                x2 = vload(v_dt, vb + 2 * VL)
                x3 = vload(v_dt, vb + 3 * VL)
                q0 = vload(qe_ub, qb)
                q1 = vload(qe_ub, qb + VL)
                q2 = vload(qe_ub, qb + 2 * VL)
                q3 = vload(qe_ub, qb + 3 * VL)
                d0 = vadd(d0, vmul(x0, q0, mask=full), mask=full)
                d1 = vadd(d1, vmul(x1, q1, mask=full), mask=full)
                d2 = vadd(d2, vmul(x2, q2, mask=full), mask=full)
                d3 = vadd(d3, vmul(x3, q3, mask=full), mask=full)
            dot = vadd(
                vadd(d0, d1, mask=full),
                vadd(d2, d3, mask=full),
                mask=full,
            )
            for seg in range(Int64(4 * ng), Int64(num_seg_dt)):
                off = base_n + seg * VL
                xf = vload(v_dt, off)
                qf = vload(qe_ub, seg * VL)
                dot = vadd(dot, vmul(xf, qf, mask=full), mask=full)
            vstore_first(pdot, n, vreduce_sum(dot, mask=full))


def _score_stream_dot2_body_fp32(v_dt, qe_pair_ub, pdot,
                                 nplus1, num_seg_dt, w_dt):
    """Score two query tiles while loading each V segment only once."""
    ng = num_seg_dt // 2
    with vf(mode="raw"):
        full = full_mask(32)
        for n in range(Int64(0), Int64(nplus1)):
            base_n = n * w_dt
            d00 = vdup_scalar(0.0, Float32, mask=full)
            d01 = vdup_scalar(0.0, Float32, mask=full)
            d10 = vdup_scalar(0.0, Float32, mask=full)
            d11 = vdup_scalar(0.0, Float32, mask=full)
            for it in range(Int64(0), Int64(ng)):
                off = it * (2 * VL)
                x0 = vload(v_dt, base_n + off)
                x1 = vload(v_dt, base_n + off + VL)
                q00 = vload(qe_pair_ub, off)
                q01 = vload(qe_pair_ub, off + VL)
                q10 = vload(qe_pair_ub, w_dt + off)
                q11 = vload(qe_pair_ub, w_dt + off + VL)
                d00 = vadd(d00, vmul(x0, q00, mask=full), mask=full)
                d01 = vadd(d01, vmul(x1, q01, mask=full), mask=full)
                d10 = vadd(d10, vmul(x0, q10, mask=full), mask=full)
                d11 = vadd(d11, vmul(x1, q11, mask=full), mask=full)
            dot0 = vadd(d00, d01, mask=full)
            dot1 = vadd(d10, d11, mask=full)
            for seg in range(Int64(2 * ng), Int64(num_seg_dt)):
                off = seg * VL
                value = vload(v_dt, base_n + off)
                query0 = vload(qe_pair_ub, off)
                query1 = vload(qe_pair_ub, w_dt + off)
                dot0 = vadd(
                    dot0, vmul(value, query0, mask=full), mask=full,
                )
                dot1 = vadd(
                    dot1, vmul(value, query1, mask=full), mask=full,
                )
            vstore_first(pdot, n, vreduce_sum(dot0, mask=full))
            vstore_first(pdot, VL + n, vreduce_sum(dot1, mask=full))


_score_body_16bit = compile_function(
    _score_body_16bit, enable_preprocessor=True,
).function
_score_body_fp32 = compile_function(
    _score_body_fp32, enable_preprocessor=True,
).function
_hold_inv_rms_body_fp32 = compile_function(
    _hold_inv_rms_body_fp32, enable_preprocessor=True,
).function
_score_hold_dot_body_fp32 = compile_function(
    _score_hold_dot_body_fp32, enable_preprocessor=True,
).function
_online_init_body_fp32 = compile_function(
    _online_init_body_fp32, enable_preprocessor=True,
).function
_online_merge_body_fp32 = compile_function(
    _online_merge_body_fp32, enable_preprocessor=True,
).function
_online_zero_body_fp32 = compile_function(
    _online_zero_body_fp32, enable_preprocessor=True,
).function
_score_stream_body_16bit = compile_function(
    _score_stream_body_16bit, enable_preprocessor=True,
).function
_score_stream_body_fp32 = compile_function(
    _score_stream_body_fp32, enable_preprocessor=True,
).function
_score_stream_dot_body_fp32 = compile_function(
    _score_stream_dot_body_fp32, enable_preprocessor=True,
).function
_score_stream_dot2_body_fp32 = compile_function(
    _score_stream_dot2_body_fp32, enable_preprocessor=True,
).function
_agg_body_16bit = compile_function(
    _agg_body_16bit, enable_preprocessor=True,
).function
_agg_body_fp32 = compile_function(
    _agg_body_fp32, enable_preprocessor=True,
).function
# pylint: enable=too-many-arguments,too-many-positional-arguments


# The vector object is instantiated inside a DSL kernel. Constructor and VF
# method operands therefore stay as compile-time primitives/Buffer handles.
# pylint: disable=too-many-arguments,too-many-positional-arguments
class BlockAttnResPrepareVector:
    """Whole-network multi-query AIV buffers and per-token primitives."""

    def __init__(self, nplus1: int, d: int, eps, dtype=BFloat16,
                 dt: int = 0, stream_queries: int = 1,
                 reuse_hold_rms: bool = False):
        self.nplus1 = int(nplus1)
        self.d = int(d)
        self.eps = eps
        self.avg = 1.0 / self.d
        self.is16 = dtype in (BFloat16, Float16)
        self.num_seg = _ceil_div(self.d, VL)
        w = self.num_seg * VL
        self.w = w
        self.dt = int(dt)                 # 流式 Dt 分块宽（>0 时走大 D 流式:候选不全驻留,分块过 D）
        self.stream_queries = int(stream_queries)
        self.reuse_hold_rms = bool(reuse_hold_rms)
        state = dtype if self.is16 else Float32
        # Hold keeps one full query; stream keeps one D tile and reuses the
        # current V tile across every query slot.
        q_width = self.dt if self.dt > 0 else w
        self.effective_q_ub = Channel(
            MemLoc.UB,
            shape=(1, q_width),
            dtype=Float32,
            depth=1,
        )
        if self.dt > 0:
            # 大 D 流式:只驻留一个 Dt 块的 N+1 候选（(N+1)·Dt,与 D 解耦）。dot/ssq 标量跨块累加。
            self.num_dt = self.d // self.dt          # 块数（host 保证 dt|D 且 dt 为 VL 整数倍）
            self.num_seg_dt = self.dt // VL
            # 两槽 FIFO：MTE2 预取下一 D 块时，Vector 消费当前块。
            self.v_dt = Channel(MemLoc.UB, shape=(1, self.nplus1 * self.dt), dtype=state, depth=2)
            self.effective_q_pair_ub = Channel(
                MemLoc.UB,
                shape=(1, 2 * self.dt),
                dtype=Float32,
                depth=1,
            )
            self.dot_acc_ub = Buffer(
                MemLoc.UB, (1, self.stream_queries * VL), Float32,
            )
            self.ssq_acc_ub = Buffer(MemLoc.UB, (1, VL), Float32)
            self.pdot_ub = Buffer(MemLoc.UB, (1, 2 * VL), Float32)
            self.pssq_ub = Buffer(MemLoc.UB, (1, VL), Float32)
            self.h_dt_ub = Channel(MemLoc.UB, shape=(1, self.dt), dtype=Float32, depth=2)
        else:
            # 一个 token 的全部历史 block 常驻 UB，供所有 query slot 复用。
            self.v_ub = Channel(
                MemLoc.UB, shape=(1, self.nplus1 * w), dtype=state, depth=1,
            )
        logits_width = self.stream_queries * VL if self.dt > 0 else VL
        self.logits_ub = Buffer(
            MemLoc.UB, (1, logits_width), Float32,
        )
        if self.reuse_hold_rms:
            self.inv_rms_ub = Buffer(MemLoc.UB, (1, VL), Float32)
        # softmax 原本就会计算 m/s；用独立 Channel 直接写回，不重算也不与 logits 复用冲突。
        self.stats_ub = Channel(
            MemLoc.UB, shape=(1, VL), dtype=Float32, depth=1,
        )
        if self.dt == 0:
            self.h_ub = Channel(
                MemLoc.UB, shape=(1, w), dtype=Float32, depth=1,
            )

    def load_effective_query(self, gm_effective_queries, slot_idx):
        """Load one fp32 prefused query [D] into its UB Channel."""
        q_slot = self.effective_q_ub.acquire()
        mem_copy(
            local_slice(q_slot, (1, self.d), offset=0),
            tile_view(gm_effective_queries, (1, self.d), (slot_idx, Int64(0))),
        )
        self.effective_q_ub.commit(q_slot)

    def load_effective_query_dt(self, gm_effective_queries, slot_idx,
                                db_idx):
        """Load one fp32 query D tile for the stream kernel."""
        q_slot = self.effective_q_ub.acquire()
        mem_copy(
            local_slice(q_slot, (1, self.dt), offset=0),
            tile_view(
                gm_effective_queries,
                (1, self.dt),
                (slot_idx, db_idx),
            ),
        )
        self.effective_q_ub.commit(q_slot)

    def load_effective_query_pair_dt(self, gm_effective_queries, slot_idx,
                                     db_idx):
        """Load two adjacent fp32 query D tiles in one Channel epoch."""
        q_slot = self.effective_q_pair_ub.acquire()
        # This is a host-side helper method, so keep the fixed two-lane loop
        # as a Python range instead of constructing DSL Int64 bounds.
        for pair_lane in range(2):
            mem_copy(
                tile_view(
                    q_slot,
                    (1, self.dt),
                    (Int64(0), pair_lane),
                ),
                tile_view(
                    gm_effective_queries,
                    (1, self.dt),
                    (slot_idx + pair_lane, db_idx),
                ),
            )
        self.effective_q_pair_ub.commit(q_slot)

    # --- GM -> UB：载入当前 token 的 N+1 个候选到指定槽（vf 外）------------------------------------
    @jit
    def load_row(self, gm_v2d, row, valid_blocks):
        d, w = self.d, self.w
        slot = self.v_ub.acquire()
        # The physical buffer has max_blocks rows, but later score/aggregate
        # only read [0, valid_blocks).  Do not spend GM bandwidth on inactive
        # history slots, especially in early AttnRes blocks.
        for n in dsl_range(Int64(0), valid_blocks, Int64(1)):
            # 唯一公开布局为 [*M,N+1,D]，展平 GM 行号为 row*(N+1)+n。
            gm_row = row * Int64(self.nplus1) + Int64(n)
            src = tile_view(gm_v2d, (1, d), (gm_row, Int64(0)))
            # ``local_slice`` requires a compile-time byte offset.  Use a
            # dynamic tile coordinate for the block row, then trim the padded
            # UB width ``w`` back to the logical ``d`` columns.
            dst_block = tile_view(slot, (1, w), (Int64(0), n))
            mem_copy(tile_view(dst_block, (1, d), (Int64(0), Int64(0))), src)
        self.v_ub.commit(slot)

    def prepare_hold_rms(self, v_ub, valid_blocks):
        """Materialize one inverse RMS scalar per block before the slot loop."""
        _hold_inv_rms_body_fp32(
            v_ub,
            self.inv_rms_ub,
            valid_blocks,
            self.num_seg,
            self.w,
            self.avg,
            self.eps,
            self.d,
        )
        with vf(mode="raw"):
            vmem_bar(mode="vst_vld")

    # --- Stage 1：使用预融合 effective query 打分 -----------------------------------------------
    def score_with_query(self, v_ub, effective_q_ub, valid_blocks):
        if self.reuse_hold_rms:
            _score_hold_dot_body_fp32(
                v_ub,
                effective_q_ub,
                self.logits_ub,
                self.inv_rms_ub,
                valid_blocks,
                self.num_seg,
                self.w,
                self.d,
            )
        elif self.is16:
            _score_body_16bit(
                v_ub, effective_q_ub, self.logits_ub, valid_blocks,
                self.num_seg, self.w, self.avg, self.eps, self.d,
            )
        else:
            _score_body_fp32(
                v_ub, effective_q_ub, self.logits_ub, valid_blocks,
                self.num_seg, self.w, self.avg, self.eps, self.d,
            )
        # score body 写 logits_ub，后续 VF 读取前显式建立 VST→VLD 顺序。
        with vf(mode="raw"):
            vmem_bar(mode="vst_vld")

    # --- Stage 2：softmax over N+1（masked reduce/exp；无效 lane 精确置 0）--------------------------
    def softmax(self, valid_blocks, query_lane=0):
        stats_slot = self.stats_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            mask_n, _ = update_mask(Int64(valid_blocks), elem_bits=32)
            logits_off = query_lane * VL
            v = vload(self.logits_ub, logits_off)
            # 用第 0 个有效 logit 填充无效 lane；重复有效元素不改变 max，因此归约可用 full mask。
            # exp 的无效 lane 再动态填成 m-80，避开极端输入，最后精确 merge 为 0。
            lane0 = vload_brc(self.logits_ub, logits_off)
            max_input = vmerge(mask_n, v, lane0)
            m = vdup_lane0(vreduce_max(max_input, mask=full), mask=full)
            exp_floor = vadds(m, -80.0, mask=full)
            exp_input = vmerge(mask_n, v, exp_floor)
            e_all = vexp_sub(exp_input, m, mask=full)
            zero = vdup_scalar(0.0, Float32, mask=full)
            e = vmerge(mask_n, e_all, zero)
            s = vdup_lane0(vreduce_sum(e, mask=full), mask=full)
            vstore_first(stats_slot, 0, m)
            # MTE 小块搬运的 UB 源地址需 32B 对齐；s 放 lane 8（fp32 偏移 32B）。
            vstore_first(stats_slot, 8, s)
            # 保留未归一化的 stable exp，Stage 3 聚合得到
            # o_tilde = Σ exp(logit-max) * V，供 online-softmax merge 直接使用。
            vstore(self.logits_ub, logits_off, e, full)
            vmem_bar(mode="vst_vld")
        self.stats_ub.commit(stats_slot)

    # --- Stage 3：聚合（运行时 seg×候选 循环 body）-------------------------------------------------
    def aggregate(self, v_ub, valid_blocks):
        h_slot = self.h_ub.acquire()
        if self.is16:
            _agg_body_16bit(
                v_ub, self.logits_ub, h_slot, valid_blocks,
                self.num_seg, self.w, Int64(0),
            )
        else:
            _agg_body_fp32(
                v_ub, self.logits_ub, h_slot, valid_blocks,
                self.num_seg, self.w, Int64(0),
            )
        self.h_ub.commit(h_slot)

    def single_valid_result(self, v_ub):
        """N=1 identity: numerator=V, max=logit, exp_sum=1."""
        h_slot = self.h_ub.acquire()
        stats_slot = self.stats_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            for seg in range(self.num_seg):
                off = seg * VL
                if self.is16:
                    value = vcast(
                        vload_unpack(
                            v_ub, off, mode=UnpackMode.B16_TO_B32,
                        ),
                        Float32,
                        mask=full,
                    )
                else:
                    value = vload(v_ub, off)
                vstore(h_slot, off, value, full)
            vstore_first(
                stats_slot, 0, vload_brc(self.logits_ub, 0),
            )
            one = vdup_scalar(1.0, Float32, mask=full)
            vstore_first(stats_slot, 8, one)
            vmem_bar(mode="vst_vld")
        self.h_ub.commit(h_slot)
        self.stats_ub.commit(stats_slot)

    def single_valid_stats(self, query_lane):
        """Store N=1 max/sum without running exp/reduce softmax."""
        stats_slot = self.stats_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            logits_off = query_lane * VL
            vstore_first(
                stats_slot,
                0,
                vload_brc(self.logits_ub, logits_off),
            )
            one = vdup_scalar(1.0, Float32, mask=full)
            vstore_first(stats_slot, 8, one)
            vmem_bar(mode="vst_vld")
        self.stats_ub.commit(stats_slot)

    def zero_result(self):
        """Materialize the empty-attention identity: h=0, max=-FLT_MAX, sum=0."""
        h_slot = self.h_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, Float32, mask=full)
            for seg in range(self.num_seg):
                off = seg * VL
                vstore(h_slot, off, zero, full)
        self.h_ub.commit(h_slot)
        self.zero_stats()

    def zero_stats(self):
        stats_slot = self.stats_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, Float32, mask=full)
            one = vdup_scalar(1.0, Float32, mask=full)
            neg_flt_max = vdup_scalar(-3.4028234663852886e38, Float32, mask=full)
            vstore_first(stats_slot, 0, neg_flt_max)
            vstore_first(stats_slot, 8, one)
            vmem_bar(mode="vst_vld")
        self.stats_ub.commit(stats_slot)

    def zero_stream_block(self, gm_h, row, db_idx):
        h_slot = self.h_dt_ub.acquire()
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, Float32, mask=full)
            for seg in range(self.num_seg_dt):
                off = seg * VL
                vstore(h_slot, off, zero, full)
        self.h_dt_ub.commit(h_slot)
        h_cur = self.h_dt_ub.wait()
        mem_copy(
            tile_view(gm_h, (1, self.dt), (row, Int64(db_idx))),
            local_slice(h_cur, (1, self.dt), offset=0),
        )
        self.h_dt_ub.release(h_cur)

    def store_row(self, gm_h, row):
        d = self.d
        h_cur = self.h_ub.wait()
        mem_copy(tile_view(gm_h, (1, d), (row, Int64(0))), local_slice(h_cur, (1, d), offset=0))
        self.h_ub.release(h_cur)

    def store_stats(self, gm_max, gm_sum, row):
        """Write per-token softmax stats: max(logits), sum(exp(logits-max))."""
        stats_cur = self.stats_ub.wait()
        mem_copy(tile_view(gm_max, (1, 1), (row, Int64(0))),
                 local_slice(stats_cur, (1, 1), offset=0))
        mem_copy(tile_view(gm_sum, (1, 1), (row, Int64(0))),
                 local_slice(stats_cur, (1, 1), offset=32))
        self.stats_ub.release(stats_cur)

    # === 大 D 流式（候选不全驻留,分 Dt 块过 D;V 读两遍：score 一遍、agg 一遍）====================
    # 载入 token row 的 N+1 候选在 D 块 db_idx 的那段 [db_idx·Dt : +Dt] → v_dt（候选步长 Dt）。
    @jit
    def load_dt(self, gm_v2d, row, db_idx, valid_blocks):
        # tile_view 第 2 维坐标是「瓦片单位」(瓦片宽=dt)→ 传块索引 db_idx,元素列自动 = db_idx·dt。
        slot = self.v_dt.acquire()
        for n in dsl_range(Int64(0), valid_blocks, Int64(1)):
            gm_row = row * Int64(self.nplus1) + Int64(n)
            src = tile_view(gm_v2d, (1, self.dt), (gm_row, Int64(db_idx)))
            mem_copy(
                tile_view(slot, (1, self.dt), (Int64(0), n)),
                src,
            )
        self.v_dt.commit(slot)

    def score_stream_init(self):
        with vf(mode="raw"):
            full = full_mask(32)
            z = vdup_scalar(0.0, Float32, mask=full)
            for query_lane in range(self.stream_queries):
                vstore(self.dot_acc_ub, query_lane * VL, z, full)
            vstore(self.ssq_acc_ub, 0, z, full)
            vmem_bar(mode="vst_vld")

    # 打分:算本块每候选 partial dot/ssq(lane n)→ 累加到常驻 dot/ssq_acc。qe_off 编译期(块循环 Python 展开)。
    def score_stream_block(self, v_dt, effective_q_ub, valid_blocks,
                           query_lane):
        if self.is16:
            _score_stream_body_16bit(
                v_dt, effective_q_ub, self.pdot_ub, self.pssq_ub,
                valid_blocks, self.num_seg_dt, self.dt, Int64(0),
            )
        else:
            _score_stream_body_fp32(
                v_dt, effective_q_ub, self.pdot_ub, self.pssq_ub,
                valid_blocks, self.num_seg_dt, self.dt, Int64(0),
            )
        with vf(mode="raw"):
            full = full_mask(32)
            vmem_bar(mode="vst_vld")
            dot_off = query_lane * VL
            dot_value = vadd(
                vload(self.dot_acc_ub, dot_off),
                vload(self.pdot_ub, 0),
                mask=full,
            )
            vstore(self.dot_acc_ub, dot_off, dot_value, full)
            vstore(self.ssq_acc_ub, 0, vadd(vload(self.ssq_acc_ub, 0), vload(self.pssq_ub, 0), mask=full), full)
            vmem_bar(mode="vst_vld")

    def score_stream_dot_block(self, v_dt, effective_q_ub, valid_blocks,
                               query_lane):
        if self.is16:
            # The public interface is fp32. Keep the old 16-bit calculation
            # available for internal compatibility, but discard its repeated
            # query-independent ssq result.
            _score_stream_body_16bit(
                v_dt, effective_q_ub, self.pdot_ub, self.pssq_ub,
                valid_blocks, self.num_seg_dt, self.dt, Int64(0),
            )
        else:
            _score_stream_dot_body_fp32(
                v_dt, effective_q_ub, self.pdot_ub,
                valid_blocks, self.num_seg_dt, self.dt,
            )
        with vf(mode="raw"):
            full = full_mask(32)
            vmem_bar(mode="vst_vld")
            dot_off = query_lane * VL
            vstore(
                self.dot_acc_ub,
                dot_off,
                vadd(
                    vload(self.dot_acc_ub, dot_off),
                    vload(self.pdot_ub, 0),
                    mask=full,
                ),
                full,
            )
            vmem_bar(mode="vst_vld")

    def score_stream_dot2_block(self, v_dt, query_pair_ub,
                                valid_blocks, query_lane):
        """Accumulate two fp32 query dots from one set of V register loads."""
        _score_stream_dot2_body_fp32(
            v_dt,
            query_pair_ub,
            self.pdot_ub,
            valid_blocks,
            self.num_seg_dt,
            self.dt,
        )
        with vf(mode="raw"):
            full = full_mask(32)
            vmem_bar(mode="vst_vld")
            dot0_off = query_lane * VL
            dot1_off = (query_lane + Int64(1)) * VL
            vstore(
                self.dot_acc_ub,
                dot0_off,
                vadd(
                    vload(self.dot_acc_ub, dot0_off),
                    vload(self.pdot_ub, 0),
                    mask=full,
                ),
                full,
            )
            vstore(
                self.dot_acc_ub,
                dot1_off,
                vadd(
                    vload(self.dot_acc_ub, dot1_off),
                    vload(self.pdot_ub, VL),
                    mask=full,
                ),
                full,
            )
            vmem_bar(mode="vst_vld")

    # 全块累加完:lane n = 候选 n 的完整 dot/ssq → logit_n = dot/sqrt(ssq·avg+eps)。
    def score_stream_finalize(self, query_lane):
        with vf(mode="raw"):
            full = full_mask(32)
            logits_off = query_lane * VL
            dot = vload(self.dot_acc_ub, logits_off)
            ssq = vload(self.ssq_acc_ub, 0)
            denom = vsqrt(vadds(vmuls(ssq, self.avg, mask=full), self.eps, mask=full), mask=full)
            vstore(self.logits_ub, logits_off, vdiv(dot, denom, mask=full), full)
            vmem_bar(mode="vst_vld")

    # 聚合:本块 v_dt 复用 agg body 算 h_dt[Dt],写回 gm_h[row, db_idx·Dt : +Dt]。
    def agg_stream_block(self, v_dt, gm_h, row, db_idx, valid_blocks,
                         query_lane):
        h_slot = self.h_dt_ub.acquire()
        if self.is16:
            _agg_body_16bit(
                v_dt, self.logits_ub, h_slot, valid_blocks,
                self.num_seg_dt, self.dt, query_lane * VL,
            )
        else:
            _agg_body_fp32(
                v_dt, self.logits_ub, h_slot, valid_blocks,
                self.num_seg_dt, self.dt, query_lane * VL,
            )
        self.h_dt_ub.commit(h_slot)
        # 输出块 db_idx → gm_h[row, db_idx·dt : +dt]。列坐标同样是瓦片单位,传 db_idx。
        h_cur = self.h_dt_ub.wait()
        mem_copy(tile_view(gm_h, (1, self.dt), (row, Int64(db_idx))),
                 local_slice(h_cur, (1, self.dt), offset=0))
        self.h_dt_ub.release(h_cur)

    def single_valid_stream_block(self, v_dt, gm_h, row, db_idx):
        """N=1 identity numerator: copy candidate 0 to one output tile."""
        if self.is16:
            h_slot = self.h_dt_ub.acquire()
            with vf(mode="raw"):
                full = full_mask(32)
                for seg in range(self.num_seg_dt):
                    off = seg * VL
                    value = vcast(
                        vload_unpack(
                            v_dt, off, mode=UnpackMode.B16_TO_B32,
                        ),
                        Float32,
                        mask=full,
                    )
                    vstore(h_slot, off, value, full)
            self.h_dt_ub.commit(h_slot)
            h_cur = self.h_dt_ub.wait()
            mem_copy(
                tile_view(gm_h, (1, self.dt), (row, db_idx)),
                local_slice(h_cur, (1, self.dt), offset=0),
            )
            self.h_dt_ub.release(h_cur)
        else:
            mem_copy(
                tile_view(gm_h, (1, self.dt), (row, db_idx)),
                local_slice(v_dt, (1, self.dt), offset=0),
            )
# pylint: enable=too-many-arguments,too-many-positional-arguments


# Public kernel entry operands must mirror the flattened Tensor ABI generated
# by CANNBotDSL; Python containers cannot cross this boundary.
# pylint: disable=too-many-arguments,too-many-positional-arguments
@kernel
class BlockAttnResPrepareMultiQueryKernel:
    """Whole-network single-launch multi-query kernel.

    Each core owns token rows.  A token's [max_blocks,D] values are loaded once and
    retained in UB while every effective query slot computes its own score,
    softmax statistics and unnormalized numerator. ``valid_blocks`` remains a
    device scalar, while ``eps`` is a compile-time Python float.
    """

    def __init__(self, max_blocks: int, num_row: int, num_query: int, d: int,
                 eps: float, dtype=BFloat16):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.eps = float(eps)
        self.dtype = dtype

    def __call__(self, gm_v2d: Tensor, gm_effective_queries: Tensor,
                 gm_valid_blocks: Tensor,
                 gm_h: Tensor, gm_max: Tensor, gm_sum: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()
        valid_blocks = gm_valid_blocks[(Int64(0),)]
        # K3 guarantees 0 <= valid_blocks <= max_blocks. Clamp in-kernel as a
        # final OOB guard without copying the scalar back to the host.
        if valid_blocks < Int64(0):
            valid_blocks = Int64(0)
        if valid_blocks > Int64(self.max_blocks):
            valid_blocks = Int64(self.max_blocks)
        vector = BlockAttnResPrepareVector(
            self.max_blocks,
            self.d,
            self.eps,
            dtype=self.dtype,
            reuse_hold_rms=True,
        )
        num_row = Int64(self.num_row)
        logical = block_num
        if logical > num_row:
            logical = num_row
        rows_per_core = (num_row + logical - Int64(1)) // logical
        row_start = block_idx * rows_per_core
        row_end = row_start + rows_per_core
        if row_end > num_row:
            row_end = num_row
        if block_idx < logical:
            for row in range(row_start, row_end):
                if valid_blocks > Int64(0):
                    vector.load_row(gm_v2d, row, valid_blocks)
                    v_cur = vector.v_ub.wait()
                    vector.prepare_hold_rms(v_cur, valid_blocks)
                    for slot_idx in range(Int64(0), Int64(self.num_query)):
                        vector.load_effective_query(gm_effective_queries, slot_idx)
                        q_cur = vector.effective_q_ub.wait()
                        vector.score_with_query(
                            v_cur, q_cur, valid_blocks=valid_blocks,
                        )
                        vector.effective_q_ub.release(q_cur)
                        if valid_blocks == Int64(1):
                            vector.single_valid_result(v_cur)
                        else:
                            vector.softmax(valid_blocks)
                            vector.aggregate(
                                v_cur, valid_blocks=valid_blocks,
                            )
                        out_row = slot_idx * num_row + row
                        vector.store_row(gm_h, out_row)
                        vector.store_stats(gm_max, gm_sum, out_row)
                    vector.v_ub.release(v_cur)
                else:
                    for slot_idx in range(Int64(0), Int64(self.num_query)):
                        vector.zero_result()
                        out_row = slot_idx * num_row + row
                        vector.store_row(gm_h, out_row)
                        vector.store_stats(gm_max, gm_sum, out_row)


@kernel
class BlockAttnResPrepareMultiQueryKernelDecode:
    """Small-token kernel: distribute flattened ``[slot, token]`` rows.

    The regular hold kernel owns a token row and serially reuses its resident V
    across every query.  That is efficient for prefill, but T=1/2 activates only
    one or two vector cores.  Decode instead exposes ``slots * tokens`` work
    items.  Each work item reloads one token's small ``[max_blocks,D]`` buffer,
    while up to 64 cores compute different query slots concurrently.
    """

    def __init__(self, max_blocks: int, num_row: int, num_query: int, d: int,
                 eps: float, dtype=BFloat16):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.eps = float(eps)
        self.dtype = dtype

    def __call__(self, gm_v2d: Tensor, gm_effective_queries: Tensor,
                 gm_valid_blocks: Tensor,
                 gm_h: Tensor, gm_max: Tensor, gm_sum: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()
        valid_blocks = gm_valid_blocks[(Int64(0),)]
        if valid_blocks < Int64(0):
            valid_blocks = Int64(0)
        if valid_blocks > Int64(self.max_blocks):
            valid_blocks = Int64(self.max_blocks)

        vector = BlockAttnResPrepareVector(
            self.max_blocks, self.d, self.eps, dtype=self.dtype,
        )
        num_row = Int64(self.num_row)
        num_work = Int64(self.num_row * self.num_query)
        logical = block_num
        if logical > num_work:
            logical = num_work
        work_per_core = (num_work + logical - Int64(1)) // logical
        work_start = block_idx * work_per_core
        work_end = work_start + work_per_core
        if work_end > num_work:
            work_end = num_work

        if block_idx < logical:
            for out_row in range(work_start, work_end):
                # Output is slot-major: out_row = slot_idx * num_row + row.
                slot_idx = out_row // num_row
                row = out_row - slot_idx * num_row
                if valid_blocks > Int64(0):
                    vector.load_row(gm_v2d, row, valid_blocks)
                    v_cur = vector.v_ub.wait()
                    vector.load_effective_query(
                        gm_effective_queries, slot_idx,
                    )
                    q_cur = vector.effective_q_ub.wait()
                    vector.score_with_query(
                        v_cur, q_cur, valid_blocks=valid_blocks,
                    )
                    vector.effective_q_ub.release(q_cur)
                    if valid_blocks == Int64(1):
                        vector.single_valid_result(v_cur)
                    else:
                        vector.softmax(valid_blocks)
                        vector.aggregate(
                            v_cur, valid_blocks=valid_blocks,
                        )
                    vector.v_ub.release(v_cur)
                    vector.store_row(gm_h, out_row)
                    vector.store_stats(gm_max, gm_sum, out_row)
                else:
                    vector.zero_result()
                    vector.store_row(gm_h, out_row)
                    vector.store_stats(gm_max, gm_sum, out_row)


class BlockAttnResPrepareDecodeOnlineVector:
    """Decode-local full-D state for candidate-major online softmax."""

    def __init__(self, d: int, eps, query_group: int):
        self.d = int(d)
        self.eps = eps
        self.query_group = int(query_group)
        self.num_seg = _ceil_div(self.d, VL)
        self.w = self.num_seg * VL
        self.avg = 1.0 / self.d

        self.query_ub = Channel(
            MemLoc.UB,
            shape=(1, self.query_group * self.w),
            dtype=Float32,
            depth=1,
        )
        self.v_ub = Channel(
            MemLoc.UB, shape=(1, self.w), dtype=Float32, depth=1,
        )
        self.numerator_ub = Channel(
            MemLoc.UB,
            shape=(1, self.query_group * self.w),
            dtype=Float32,
            depth=1,
        )
        self.inv_rms_ub = Buffer(MemLoc.UB, (1, VL), Float32)
        self.logits_ub = Buffer(MemLoc.UB, (1, VL), Float32)
        self.max_state_ub = Buffer(
            MemLoc.UB, (1, self.query_group * VL), Float32,
        )
        self.sum_state_ub = Buffer(
            MemLoc.UB, (1, self.query_group * VL), Float32,
        )
        self.stats_ub = Channel(
            MemLoc.UB, shape=(1, VL), dtype=Float32, depth=1,
        )

    @jit
    def load_query_group(self, gm_queries, group_start, active_queries):
        query_slot = self.query_ub.acquire()
        for query_lane in dsl_range(
            Int64(0), active_queries, Int64(1),
        ):
            dst = tile_view(
                query_slot,
                (1, self.w),
                (Int64(0), query_lane),
            )
            src = tile_view(
                gm_queries,
                (1, self.d),
                (group_start + query_lane, Int64(0)),
            )
            mem_copy(
                tile_view(dst, (1, self.d), (Int64(0), Int64(0))),
                src,
            )
        self.query_ub.commit(query_slot)

    @jit
    def load_candidate(self, gm_v2d, row, candidate, max_blocks):
        v_slot = self.v_ub.acquire()
        gm_row = row * Int64(max_blocks) + candidate
        mem_copy(
            local_slice(v_slot, (1, self.d), offset=0),
            tile_view(
                gm_v2d, (1, self.d), (gm_row, Int64(0)),
            ),
        )
        self.v_ub.commit(v_slot)

    def prepare_candidate(self, v_cur):
        _hold_inv_rms_body_fp32(
            v_cur,
            self.inv_rms_ub,
            Int64(1),
            self.num_seg,
            self.w,
            self.avg,
            self.eps,
            self.d,
        )
        with vf(mode="raw"):
            vmem_bar(mode="vst_vld")

    def score_query(self, v_cur, query_cur, query_lane):
        query_view = tile_view(
            query_cur,
            (1, self.w),
            (Int64(0), query_lane),
        )
        _score_hold_dot_body_fp32(
            v_cur,
            query_view,
            self.logits_ub,
            self.inv_rms_ub,
            Int64(1),
            self.num_seg,
            self.w,
            self.d,
        )
        with vf(mode="raw"):
            vmem_bar(mode="vst_vld")

    def init_query(self, v_cur, numerator_cur, query_lane):
        _online_init_body_fp32(
            v_cur,
            numerator_cur,
            self.max_state_ub,
            self.sum_state_ub,
            self.logits_ub,
            query_lane,
            self.num_seg,
            self.w,
            self.d,
        )

    def merge_query(self, v_cur, numerator_cur, query_lane):
        _online_merge_body_fp32(
            v_cur,
            numerator_cur,
            self.max_state_ub,
            self.sum_state_ub,
            self.logits_ub,
            query_lane,
            self.num_seg,
            self.w,
            self.d,
        )

    def zero_query(self, numerator_cur, query_lane):
        _online_zero_body_fp32(
            numerator_cur,
            self.max_state_ub,
            self.sum_state_ub,
            query_lane,
            self.num_seg,
            self.w,
            self.d,
        )

    @staticmethod
    def finish_candidate():
        with vf(mode="raw"):
            vmem_bar(mode="vst_vld")

    def store_stats(self, gm_max, gm_sum, out_row, query_lane):
        stats_slot = self.stats_ub.acquire()
        state_off = query_lane * VL
        with vf(mode="raw"):
            vstore_first(
                stats_slot, 0, vload(self.max_state_ub, state_off),
            )
            vstore_first(
                stats_slot, 8, vload(self.sum_state_ub, state_off),
            )
            vmem_bar(mode="vst_vld")
        self.stats_ub.commit(stats_slot)
        stats_cur = self.stats_ub.wait()
        mem_copy(
            tile_view(gm_max, (1, 1), (out_row, Int64(0))),
            local_slice(stats_cur, (1, 1), offset=0),
        )
        mem_copy(
            tile_view(gm_sum, (1, 1), (out_row, Int64(0))),
            local_slice(stats_cur, (1, 1), offset=32),
        )
        self.stats_ub.release(stats_cur)

    @jit
    def store_group(self, gm_h, gm_max, gm_sum, group_start,
                    active_queries, row, num_row):
        numerator_cur = self.numerator_ub.wait()
        for query_lane in dsl_range(
            Int64(0), active_queries, Int64(1),
        ):
            out_row = (group_start + query_lane) * num_row + row
            numerator_view = tile_view(
                numerator_cur,
                (1, self.w),
                (Int64(0), query_lane),
            )
            mem_copy(
                tile_view(gm_h, (1, self.d), (out_row, Int64(0))),
                tile_view(
                    numerator_view,
                    (1, self.d),
                    (Int64(0), Int64(0)),
                ),
            )
            self.store_stats(
                gm_max, gm_sum, out_row, query_lane,
            )
        self.numerator_ub.release(numerator_cur)


@kernel
class BlockAttnResPrepareMultiQueryKernelDecodeOnline:
    """Decode path that reads every candidate V once and merges online."""

    def __init__(self, max_blocks: int, num_row: int, num_query: int,
                 d: int, query_group: int, eps: float):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.query_group = int(query_group)
        self.eps = float(eps)

    def __call__(self, gm_v2d: Tensor, gm_effective_queries: Tensor,
                 gm_valid_blocks: Tensor,
                 gm_h: Tensor, gm_max: Tensor, gm_sum: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()
        valid_blocks = gm_valid_blocks[(Int64(0),)]
        if valid_blocks < Int64(0):
            valid_blocks = Int64(0)
        if valid_blocks > Int64(self.max_blocks):
            valid_blocks = Int64(self.max_blocks)

        vector = BlockAttnResPrepareDecodeOnlineVector(
            self.d, self.eps, self.query_group,
        )
        num_row = Int64(self.num_row)
        num_group = _ceil_div(self.num_query, self.query_group)
        num_work = Int64(self.num_row * num_group)
        logical = block_num
        if logical > num_work:
            logical = num_work
        work_per_core = (num_work + logical - Int64(1)) // logical
        work_start = block_idx * work_per_core
        work_end = work_start + work_per_core
        if work_end > num_work:
            work_end = num_work

        if block_idx < logical:
            for work_idx in range(work_start, work_end):
                group_idx = work_idx // num_row
                row = work_idx - group_idx * num_row
                group_start = group_idx * Int64(self.query_group)
                active_queries = Int64(self.query_group)
                remaining = Int64(self.num_query) - group_start
                if active_queries > remaining:
                    active_queries = remaining

                numerator_cur = vector.numerator_ub.acquire()
                if valid_blocks > Int64(0):
                    vector.load_query_group(
                        gm_effective_queries,
                        group_start,
                        active_queries,
                    )
                    query_cur = vector.query_ub.wait()

                    vector.load_candidate(
                        gm_v2d, row, Int64(0), self.max_blocks,
                    )
                    v_cur = vector.v_ub.wait()
                    vector.prepare_candidate(v_cur)
                    for query_lane in dsl_range(
                        Int64(0), active_queries, Int64(1),
                    ):
                        vector.score_query(v_cur, query_cur, query_lane)
                        vector.init_query(
                            v_cur, numerator_cur, query_lane,
                        )
                    vector.finish_candidate()
                    vector.v_ub.release(v_cur)

                    for candidate in dsl_range(
                        Int64(1), valid_blocks, Int64(1),
                    ):
                        vector.load_candidate(
                            gm_v2d, row, candidate, self.max_blocks,
                        )
                        v_cur = vector.v_ub.wait()
                        vector.prepare_candidate(v_cur)
                        for query_lane in dsl_range(
                            Int64(0), active_queries, Int64(1),
                        ):
                            vector.score_query(
                                v_cur, query_cur, query_lane,
                            )
                            vector.merge_query(
                                v_cur, numerator_cur, query_lane,
                            )
                        vector.finish_candidate()
                        vector.v_ub.release(v_cur)
                    vector.query_ub.release(query_cur)
                else:
                    for query_lane in dsl_range(
                        Int64(0), active_queries, Int64(1),
                    ):
                        vector.zero_query(numerator_cur, query_lane)
                    vector.finish_candidate()

                vector.numerator_ub.commit(numerator_cur)
                vector.store_group(
                    gm_h,
                    gm_max,
                    gm_sum,
                    group_start,
                    active_queries,
                    row,
                    num_row,
                )


@kernel
class BlockAttnResPrepareMultiQueryKernelStream:
    """D-tiled whole-network fallback when [max_blocks,D] cannot fit in UB."""

    def __init__(self, max_blocks: int, num_row: int, num_query: int, d: int,
                 dt: int, eps: float, dtype=BFloat16):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.dt = int(dt)
        self.eps = float(eps)
        self.dtype = dtype

    def __call__(self, gm_v2d: Tensor, gm_effective_queries: Tensor,
                 gm_valid_blocks: Tensor,
                 gm_h: Tensor, gm_max: Tensor, gm_sum: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()
        valid_blocks = gm_valid_blocks[(Int64(0),)]
        if valid_blocks < Int64(0):
            valid_blocks = Int64(0)
        if valid_blocks > Int64(self.max_blocks):
            valid_blocks = Int64(self.max_blocks)
        vector = BlockAttnResPrepareVector(
            self.max_blocks,
            self.d,
            self.eps,
            dtype=self.dtype,
            dt=self.dt,
            stream_queries=self.num_query,
        )
        num_row = Int64(self.num_row)
        num_dt = self.d // self.dt
        logical = block_num
        if logical > num_row:
            logical = num_row
        rows_per_core = (num_row + logical - Int64(1)) // logical
        row_start = block_idx * rows_per_core
        row_end = row_start + rows_per_core
        if row_end > num_row:
            row_end = num_row
        if block_idx < logical:
            for row in range(row_start, row_end):
                if valid_blocks > Int64(0):
                    # Score pass: each V tile is loaded once, then reused by
                    # every query. RMS ssq is query-independent and is only
                    # accumulated while processing slot 0.
                    vector.score_stream_init()
                    vector.load_dt(
                        gm_v2d, row, Int64(0), valid_blocks,
                    )
                    for db in range(Int64(0), Int64(num_dt)):
                        nxt = db + Int64(1)
                        if nxt < Int64(num_dt):
                            vector.load_dt(
                                gm_v2d, row, nxt, valid_blocks,
                            )
                        v_cur = vector.v_dt.wait()
                        vector.load_effective_query_dt(
                            gm_effective_queries, Int64(0), db,
                        )
                        q_cur = vector.effective_q_ub.wait()
                        vector.score_stream_block(
                            v_cur, q_cur, valid_blocks, Int64(0),
                        )
                        vector.effective_q_ub.release(q_cur)
                        pair_end = self.num_query - (
                            1 if self.num_query % 2 == 0 else 0
                        )
                        for slot_idx in range(
                            Int64(1), Int64(pair_end), Int64(2),
                        ):
                            vector.load_effective_query_pair_dt(
                                gm_effective_queries, slot_idx, db,
                            )
                            q_pair_cur = vector.effective_q_pair_ub.wait()
                            vector.score_stream_dot2_block(
                                v_cur,
                                q_pair_cur,
                                valid_blocks,
                                slot_idx,
                            )
                            vector.effective_q_pair_ub.release(q_pair_cur)
                        if self.num_query > 1 and self.num_query % 2 == 0:
                            last_slot = Int64(self.num_query - 1)
                            vector.load_effective_query_dt(
                                gm_effective_queries, last_slot, db,
                            )
                            q_cur = vector.effective_q_ub.wait()
                            vector.score_stream_dot_block(
                                v_cur, q_cur, valid_blocks, last_slot,
                            )
                            vector.effective_q_ub.release(q_cur)
                        if valid_blocks == Int64(1):
                            for slot_idx in range(
                                Int64(0), Int64(self.num_query),
                            ):
                                out_row = slot_idx * num_row + row
                                vector.single_valid_stream_block(
                                    v_cur, gm_h, out_row, db,
                                )
                        vector.v_dt.release(v_cur)

                    for slot_idx in range(
                        Int64(0), Int64(self.num_query),
                    ):
                        vector.score_stream_finalize(slot_idx)
                        if valid_blocks == Int64(1):
                            vector.single_valid_stats(slot_idx)
                        else:
                            vector.softmax(valid_blocks, slot_idx)
                        out_row = slot_idx * num_row + row
                        vector.store_stats(gm_max, gm_sum, out_row)

                    # N=1 writes its identity numerator during the score pass.
                    # Other depths still need a second V pass after softmax.
                    if valid_blocks > Int64(1):
                        vector.load_dt(
                            gm_v2d, row, Int64(0), valid_blocks,
                        )
                        for db in range(Int64(0), Int64(num_dt)):
                            nxt = db + Int64(1)
                            if nxt < Int64(num_dt):
                                vector.load_dt(
                                    gm_v2d, row, nxt, valid_blocks,
                                )
                            v_cur = vector.v_dt.wait()
                            for slot_idx in range(
                                Int64(0), Int64(self.num_query),
                            ):
                                out_row = slot_idx * num_row + row
                                vector.agg_stream_block(
                                    v_cur,
                                    gm_h,
                                    out_row,
                                    db,
                                    valid_blocks,
                                    slot_idx,
                                )
                            vector.v_dt.release(v_cur)
                else:
                    for slot_idx in range(Int64(0), Int64(self.num_query)):
                        out_row = slot_idx * num_row + row
                        for db in range(Int64(0), Int64(num_dt)):
                            vector.zero_stream_block(gm_h, out_row, db)
                        vector.zero_stats()
                        vector.store_stats(gm_max, gm_sum, out_row)


@kernel
class BlockAttnResPrepareMultiQueryKernelStreamDecode:
    """Decode stream kernel with parallel query groups and intra-group V reuse."""

    def __init__(self, max_blocks: int, num_row: int, num_query: int, d: int,
                 dt: int, query_group: int, eps: float, dtype=BFloat16):
        self.max_blocks = int(max_blocks)
        self.num_row = int(num_row)
        self.num_query = int(num_query)
        self.d = int(d)
        self.dt = int(dt)
        self.query_group = int(query_group)
        self.num_groups = _ceil_div(self.num_query, self.query_group)
        self.eps = float(eps)
        self.dtype = dtype

    def __call__(self, gm_v2d: Tensor, gm_effective_queries: Tensor,
                 gm_valid_blocks: Tensor,
                 gm_h: Tensor, gm_max: Tensor, gm_sum: Tensor):
        block_idx = get_block_idx()
        block_num = get_block_num()
        valid_blocks = gm_valid_blocks[(Int64(0),)]
        if valid_blocks < Int64(0):
            valid_blocks = Int64(0)
        if valid_blocks > Int64(self.max_blocks):
            valid_blocks = Int64(self.max_blocks)

        vector = BlockAttnResPrepareVector(
            self.max_blocks,
            self.d,
            self.eps,
            dtype=self.dtype,
            dt=self.dt,
            stream_queries=self.query_group,
        )
        num_row = Int64(self.num_row)
        num_dt = self.d // self.dt
        num_work = Int64(self.num_row * self.num_groups)
        logical = block_num
        if logical > num_work:
            logical = num_work
        work_per_core = (num_work + logical - Int64(1)) // logical
        work_start = block_idx * work_per_core
        work_end = work_start + work_per_core
        if work_end > num_work:
            work_end = num_work

        if block_idx < logical:
            for work_idx in range(work_start, work_end):
                group_idx = work_idx // num_row
                row = work_idx - group_idx * num_row
                group_start = group_idx * Int64(self.query_group)
                active_queries = Int64(self.num_query) - group_start
                if active_queries > Int64(self.query_group):
                    active_queries = Int64(self.query_group)

                if valid_blocks > Int64(0):
                    vector.score_stream_init()
                    vector.load_dt(
                        gm_v2d, row, Int64(0), valid_blocks,
                    )
                    for db in range(Int64(0), Int64(num_dt)):
                        nxt = db + Int64(1)
                        if nxt < Int64(num_dt):
                            vector.load_dt(
                                gm_v2d, row, nxt, valid_blocks,
                            )
                        v_cur = vector.v_dt.wait()
                        vector.load_effective_query_dt(
                            gm_effective_queries, group_start, db,
                        )
                        q_cur = vector.effective_q_ub.wait()
                        vector.score_stream_block(
                            v_cur, q_cur, valid_blocks, Int64(0),
                        )
                        vector.effective_q_ub.release(q_cur)
                        for query_lane in range(
                            Int64(1),
                            Int64(self.query_group),
                            Int64(2),
                        ):
                            if query_lane < active_queries:
                                slot_idx = group_start + query_lane
                                if query_lane + Int64(1) < active_queries:
                                    vector.load_effective_query_pair_dt(
                                        gm_effective_queries,
                                        slot_idx,
                                        db,
                                    )
                                    q_pair_cur = (
                                        vector.effective_q_pair_ub.wait()
                                    )
                                    vector.score_stream_dot2_block(
                                        v_cur,
                                        q_pair_cur,
                                        valid_blocks,
                                        query_lane,
                                    )
                                    vector.effective_q_pair_ub.release(
                                        q_pair_cur,
                                    )
                                else:
                                    vector.load_effective_query_dt(
                                        gm_effective_queries, slot_idx, db,
                                    )
                                    q_cur = vector.effective_q_ub.wait()
                                    vector.score_stream_dot_block(
                                        v_cur,
                                        q_cur,
                                        valid_blocks,
                                        query_lane,
                                    )
                                    vector.effective_q_ub.release(q_cur)
                        if valid_blocks == Int64(1):
                            for query_lane in range(
                                Int64(0), active_queries,
                            ):
                                slot_idx = group_start + query_lane
                                out_row = slot_idx * num_row + row
                                vector.single_valid_stream_block(
                                    v_cur, gm_h, out_row, db,
                                )
                        vector.v_dt.release(v_cur)

                    for query_lane in range(
                        Int64(0), active_queries,
                    ):
                        slot_idx = group_start + query_lane
                        vector.score_stream_finalize(query_lane)
                        if valid_blocks == Int64(1):
                            vector.single_valid_stats(query_lane)
                        else:
                            vector.softmax(valid_blocks, query_lane)
                        out_row = slot_idx * num_row + row
                        vector.store_stats(gm_max, gm_sum, out_row)

                    if valid_blocks > Int64(1):
                        vector.load_dt(
                            gm_v2d, row, Int64(0), valid_blocks,
                        )
                        for db in range(Int64(0), Int64(num_dt)):
                            nxt = db + Int64(1)
                            if nxt < Int64(num_dt):
                                vector.load_dt(
                                    gm_v2d, row, nxt, valid_blocks,
                                )
                            v_cur = vector.v_dt.wait()
                            for query_lane in range(
                                Int64(0), active_queries,
                            ):
                                slot_idx = group_start + query_lane
                                out_row = slot_idx * num_row + row
                                vector.agg_stream_block(
                                    v_cur,
                                    gm_h,
                                    out_row,
                                    db,
                                    valid_blocks,
                                    query_lane,
                                )
                            vector.v_dt.release(v_cur)
                else:
                    for query_lane in range(
                        Int64(0), active_queries,
                    ):
                        slot_idx = group_start + query_lane
                        out_row = slot_idx * num_row + row
                        for db in range(Int64(0), Int64(num_dt)):
                            vector.zero_stream_block(
                                gm_h, out_row, db,
                            )
                        vector.zero_stats()
                        vector.store_stats(gm_max, gm_sum, out_row)
# pylint: enable=too-many-arguments,too-many-positional-arguments


class BlockAttnResPrepare:
    """Host launcher for the whole-network multi-query path."""

    def __init__(self, plan: PrepareLaunchPlan):
        shape = plan.shape
        self.max_blocks = int(shape.max_blocks)
        self.num_row = int(shape.tokens)
        self.num_query = int(shape.slots)
        self.d = int(shape.d)
        self.eps = float(plan.eps)
        self.dtype = _TORCH_TO_DSL[shape.input_dtype]
        self.block_num = int(plan.block_num)
        if plan.mode == "decode_online":
            query_group = _decode_online_query_group(
                self.num_row, self.num_query, self.d,
            )
            self._kernel_cls = (
                BlockAttnResPrepareMultiQueryKernelDecodeOnline
            )
            self._kernel_args = (
                self.max_blocks,
                self.num_row,
                self.num_query,
                self.d,
                query_group,
                self.eps,
            )
        elif plan.mode == "stream":
            self._kernel_cls = BlockAttnResPrepareMultiQueryKernelStream
            self._kernel_args = (
                self.max_blocks, self.num_row, self.num_query, self.d,
                int(plan.d_tile), self.eps, self.dtype,
            )
        elif plan.mode == "stream_decode":
            query_group = _stream_decode_query_group(
                self.num_row, self.num_query,
            )
            self._kernel_cls = (
                BlockAttnResPrepareMultiQueryKernelStreamDecode
            )
            self._kernel_args = (
                self.max_blocks,
                self.num_row,
                self.num_query,
                self.d,
                int(plan.d_tile),
                query_group,
                self.eps,
                self.dtype,
            )
        elif plan.mode == "stream_prefill_group":
            query_group = _stream_prefill_query_group(
                self.num_row, self.num_query,
            )
            self._kernel_cls = (
                BlockAttnResPrepareMultiQueryKernelStreamDecode
            )
            self._kernel_args = (
                self.max_blocks,
                self.num_row,
                self.num_query,
                self.d,
                int(plan.d_tile),
                query_group,
                self.eps,
                self.dtype,
            )
        elif plan.mode == "decode":
            self._kernel_cls = BlockAttnResPrepareMultiQueryKernelDecode
            self._kernel_args = (
                self.max_blocks, self.num_row, self.num_query, self.d,
                self.eps, self.dtype,
            )
        else:
            self._kernel_cls = BlockAttnResPrepareMultiQueryKernel
            self._kernel_args = (
                self.max_blocks, self.num_row, self.num_query, self.d,
                self.eps, self.dtype,
            )

    # The JIT entry signature is the compiled kernel's flattened Tensor ABI.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @jit
    def run(self, gm_v2d, gm_effective_queries, gm_valid_blocks,
            gm_h, gm_max, gm_sum):
        op = self._kernel_cls(*self._kernel_args)
        op[self.block_num](
            gm_v2d, gm_effective_queries, gm_valid_blocks,
            gm_h, gm_max, gm_sum,
        )

    # pylint: enable=too-many-arguments,too-many-positional-arguments


def _compiled_prepare_kernel(plan: PrepareLaunchPlan):
    """Compile once per static kernel configuration and reuse the callable."""
    shape = plan.shape
    key = (
        shape,
        str(plan.mode),
        int(plan.d_tile),
        int(plan.block_num),
        float(plan.eps),
    )
    compiled = _COMPILED_KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled

    op = BlockAttnResPrepare(plan)
    fake = cannbotdsl.TensorSpec
    compiled = op.run.compile(
        fake(
            (shape.tokens * shape.max_blocks, shape.d),
            _TORCH_TO_DSL[shape.input_dtype],
        ),
        fake((shape.slots, shape.d), dtypes.float32),
        fake((1,), dtypes.int64),
        fake((shape.slots * shape.tokens, shape.d), dtypes.float32),
        fake((shape.slots * shape.tokens, 1), dtypes.float32),
        fake((shape.slots * shape.tokens, 1), dtypes.float32),
    )
    _COMPILED_KERNEL_CACHE[key] = compiled
    return compiled


@lru_cache(maxsize=None)
def _prepare_launch_plan(
    shape: PrepareStaticShape,
    eps: float,
):
    """Cache static UB sizing, path selection, and compiled callable."""
    max_blocks = shape.max_blocks
    tokens = shape.tokens
    slots = shape.slots
    d = shape.d
    ebytes = 2 if shape.input_dtype in (torch.bfloat16, torch.float16) else 4
    w_pad = _ceil_div(d, VL) * VL
    required_ub = (
        max_blocks * w_pad * ebytes
        + w_pad * 4
        + w_pad * 4
        + 3 * VL * 4
    )
    mode = "hold"
    dt = 0
    if required_ub > UB_BYTES:
        use_stream_decode = 0 < tokens <= 2 and slots > 1
        prefill_query_group = _stream_prefill_query_group(tokens, slots)
        use_stream_prefill_group = (
            DEFAULT_BLOCK_NUM // 2 <= tokens < DEFAULT_BLOCK_NUM
            and prefill_query_group >= 2
        )
        online_group = (
            _decode_online_query_group(tokens, slots, d)
            if use_stream_decode else 0
        )
        if online_group > 0:
            mode = "decode_online"
        else:
            _require(
                d % VL == 0,
                f"整网多 query D 分块暂要求 D 是 {VL} 的整数倍；D={d}",
            )
            if use_stream_decode:
                stream_queries = _stream_decode_query_group(tokens, slots)
            elif use_stream_prefill_group:
                stream_queries = _stream_prefill_query_group(tokens, slots)
            else:
                stream_queries = slots
            fixed_stream_bytes = (2 * stream_queries + 7) * VL * 4
            stream_bytes_per_d = 2 * max_blocks * ebytes + 2 * 4 + 3 * 4
            dt_max = (
                (UB_BYTES - fixed_stream_bytes)
                // stream_bytes_per_d
                // VL
            ) * VL
            candidate = min(dt_max, d)
            while candidate >= VL:
                if d % candidate == 0:
                    dt = candidate
                    break
                candidate -= VL
            if dt < VL:
                raise RuntimeError(
                    "整网多 query D 分块仍超 UB: "
                    f"max_blocks={max_blocks}, D={d}"
                )
            if use_stream_decode:
                mode = "stream_decode"
            elif use_stream_prefill_group:
                mode = "stream_prefill_group"
            else:
                mode = "stream"
    elif _use_decode_slot_parallel(tokens, slots, required_ub):
        mode = "decode"

    block_num = _launch_block_num(mode, tokens, slots, d)
    plan = PrepareLaunchPlan(
        shape=shape,
        mode=mode,
        d_tile=dt,
        block_num=block_num,
        eps=float(eps),
    )
    return _compiled_prepare_kernel(plan)


def _block_attn_res_prepare_eager(
    v: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K3 Phase 1 block attention residual.

    Args:
        v: [tokens,max_blocks,D] fp32 resident block buffer.
        effective_queries: [slots,D] fp32, pre-fused q * RMSNorm gain.
        valid_blocks: scalar int64 tensor on the same device. Only
            v[:, :valid_blocks, :] participates in attention.
        eps: optional positive Python float; defaults to 1e-6 when omitted.

    Returns:
        numerator/max/sum are all fp32. numerator has shape
        [slots,tokens,D], while max/sum have shape [slots,tokens].
        numerator is Σ exp(logit-max)*V and is intentionally not divided by sum;
        all three outputs can be passed directly to the update operator.
    """
    _require(v.dtype == torch.float32, f"v 必须为 fp32,实际 {v.dtype}")
    _require(v.dim() == 3, "v 必须为 [tokens,max_blocks,D]")
    _require(
        effective_queries.dtype == torch.float32,
        f"effective_queries 必须为 fp32,实际 {effective_queries.dtype}",
    )
    _require(
        effective_queries.dim() == 2,
        "effective_queries 必须为 [slots,D]",
    )
    _require(
        isinstance(valid_blocks, torch.Tensor),
        "valid_blocks 必须为标量 int64 Tensor",
    )
    _require(
        valid_blocks.dtype == torch.int64 and valid_blocks.numel() == 1,
        "valid_blocks 必须为标量 int64 Tensor",
    )
    _require(
        v.is_contiguous()
        and effective_queries.is_contiguous()
        and valid_blocks.is_contiguous(),
        "v / effective_queries / valid_blocks 必须 contiguous",
    )
    _require(
        v.device == effective_queries.device == valid_blocks.device,
        "v / effective_queries / valid_blocks 必须同 device",
    )

    tokens, max_blocks, d = (int(size) for size in v.shape)
    slots = int(effective_queries.shape[0])
    _require(
        int(effective_queries.shape[1]) == d,
        f"effective_queries hidden 必须为 D={d},实际 {effective_queries.shape[1]}",
    )
    _require(d >= 1, f"D={d} 非法")
    _require(d <= MAX_D, f"D={d} 超过当前上限 {MAX_D}")
    _require(
        1 <= max_blocks <= VL,
        f"max_blocks={max_blocks} 当前实现须在 [1,{VL}]",
    )

    _require(isinstance(eps, (float, int)), "eps 必须为 Python float")
    eps_value = float(eps)
    _require(eps_value > 0.0, "eps 必须大于 0")

    out_shape = (slots, tokens, d)
    stats_shape = (slots, tokens)
    out = torch.empty(out_shape, dtype=torch.float32, device=v.device)
    max_out = torch.empty(stats_shape, dtype=torch.float32, device=v.device)
    sum_out = torch.empty(stats_shape, dtype=torch.float32, device=v.device)

    # Empty dimensions are valid host-level shapes and need no kernel launch.
    if tokens == 0 or slots == 0:
        return out, max_out, sum_out

    static_shape = PrepareStaticShape(
        max_blocks=max_blocks,
        tokens=tokens,
        slots=slots,
        d=d,
        input_dtype=v.dtype,
    )
    compiled = _prepare_launch_plan(static_shape, eps_value)
    compiled(
        v.reshape(tokens * max_blocks, d),
        effective_queries,
        valid_blocks.reshape(1),
        out.reshape(slots * tokens, d),
        max_out.reshape(slots * tokens, 1),
        sum_out.reshape(slots * tokens, 1),
    )
    return out, max_out, sum_out


# Publish the functional whole-network entry as one opaque dispatcher op. The
# PrivateUse1 implementation passes provider-owned tensors directly to the
# CANNBotDSL 0.3 Host Runtime after graph capture has finished.
_PREPARE_TORCH_LIBRARY = torch.library.Library("cannbot_attn_res", "FRAGMENT")
_PREPARE_TORCH_LIBRARY.define(
    "block_attn_res_prepare(Tensor v, Tensor effective_queries, "
    "Tensor valid_blocks, float eps=1e-6) -> (Tensor, Tensor, Tensor)"
)


@torch.library.impl(_PREPARE_TORCH_LIBRARY, "block_attn_res_prepare", "Meta")
def _block_attn_res_prepare_meta(
    v: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del valid_blocks, eps
    tokens, _, d = v.shape
    slots = effective_queries.shape[0]
    numerator = v.new_empty((slots, tokens, d), dtype=torch.float32)
    max_out = v.new_empty((slots, tokens), dtype=torch.float32)
    sum_out = v.new_empty((slots, tokens), dtype=torch.float32)
    return numerator, max_out, sum_out


@torch.library.impl(
    _PREPARE_TORCH_LIBRARY, "block_attn_res_prepare", "PrivateUse1",
)
def _block_attn_res_prepare_npu(
    v: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _block_attn_res_prepare_eager(
        v, effective_queries, valid_blocks, eps=eps,
    )


def block_attn_res_prepare(
    v: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Graph-safe public entry for K3 Phase-1 block attention residual.

    The four-input functional form is emitted as one opaque Torch operator, so
    Dynamo does not trace the CANNBotDSL provider call.
    """
    _require(isinstance(eps, (float, int)), "eps 必须为 Python float")
    eps_value = float(eps)
    _require(eps_value > 0.0, "eps 必须大于 0")
    if v.device.type != "npu":
        return _block_attn_res_prepare_eager(
            v,
            effective_queries,
            valid_blocks,
            eps=eps_value,
        )

    return torch.ops.cannbot_attn_res.block_attn_res_prepare.default(
        v, effective_queries, valid_blocks, eps_value,
    )
