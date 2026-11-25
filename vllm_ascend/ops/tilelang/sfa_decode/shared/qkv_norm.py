# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from .whole_rms_math import normalize_qkv


def qkv_norm(batch, pages, *, split=False):
    dtype = T.bfloat16
    norm = make_norm_pack(1, pages * 128, 1e-5)

    @T.prim_func
    def main(
        QKV: T.Tensor((batch, 6144 if split else 2048), T.float32 if split else dtype),
        GQ: T.Tensor((1536,), dtype),
        GK: T.Tensor((512,), dtype),
        Slot: T.Tensor((batch, 1), T.int32),
        CacheRows: T.Tensor((pages * 128, 132), T.int32),
        Norm: T.Tensor((batch, 1536), dtype),
    ):
        with T.Kernel(batch) as row:
            src = T.alloc_shared((1, 2048), dtype)
            dst = T.alloc_shared((1, 1536), dtype)
            if split:
                part = T.alloc_shared((3 * 2048,), T.float32)
                T.copy(QKV[row, :], part)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(2048):
                        pair = part[i] + part[i + 2048]
                        src[0, i] = T.cast(pair + part[i + 4096], T.bfloat16)
            else:
                T.copy(QKV[row : row + 1, :], src)
            norm(src, GQ, GK, Slot, CacheRows, row, 0, dst)
            T.copy(dst, Norm[row : row + 1, :])

    return main


def make_norm_pack(qhalf, cache_rows, eps):
    q_width, kv_width = 1536, 512
    block, words, scales, stride = 128, 128, 4, 132
    dtype = T.bfloat16

    @T.macro
    def norm_pack(QKV, QGamma, KVGamma, Slot, Cache, bx, sid, QNorm):
        q_gamma = T.alloc_shared((q_width,), dtype)
        kv_gamma = T.alloc_shared((kv_width,), dtype)
        packed = T.alloc_shared((1, stride), T.int32)
        packed_codes = T.view(packed, (stride * 4,), dtype=T.float8_e4m3fn)
        packed_scales = T.view(packed, (stride,), dtype=T.float32)
        norm_kv = T.alloc_shared((1, kv_width), dtype)
        T.copy(QGamma, q_gamma)
        T.copy(KVGamma, kv_gamma)
        for r in T.serial(qhalf):
            normalize_qkv(QKV, kv_gamma, r, norm_kv, eps, kv_width, q_width)
            normalize_qkv(QKV, q_gamma, r, QNorm, eps, q_width, 0)
            with T.SimdVF():
                mask = S.pset(32)
                scalar = S.pset(32, "PAT_VL1")
                fp8_max = S.vdup(T.float32(448.0), "float32", mask)
                for g in range(scales):
                    start = g * block
                    left = S.vcvt(S.vld(norm_kv[0, start], "UNPK_B16"), "float32", mask, part=0)
                    right = S.vcvt(S.vld(norm_kv[0, start + 64], "UNPK_B16"), "float32", mask, part=0)
                    peak = S.vdupv(S.vcmax(S.vmax(S.vabs(left, mask), S.vabs(right, mask), mask), mask), mask)
                    # Native writes the rounded FP32 quotient to the cache. A rounded reciprocal changes its bits.
                    scale = S.vmaxs(S.vdiv(peak, fp8_max, mask, precision="ftz_true"), T.float32(1.0e-12), mask)
                    S.vsts(packed_scales[words + g], scale, scalar, "ONEPT_B32", extent=1)
                    left_codes = S.vcvt(S.vdiv(left, scale, mask, precision="ftz_true"), "float8_e4m3fn", mask)
                    right_codes = S.vcvt(S.vdiv(right, scale, mask, precision="ftz_true"), "float8_e4m3fn", mask)
                    S.vsts(packed_codes[start], left_codes, mask, "PK4_B32")
                    S.vsts(packed_codes[start + 64], right_codes, mask, "PK4_B32")

            slot = Slot[bx, sid * qhalf + r]
            if slot >= 0 and slot < cache_rows:
                T.copy(packed, Cache[slot : slot + 1, :])

    return norm_pack
