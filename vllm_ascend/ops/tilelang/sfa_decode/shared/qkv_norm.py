# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: disable-error-code="valid-type"

import tilelang.ascend.language as T  # type: ignore[import-not-found]
from tilelang.ascend.language import simd as S  # type: ignore[import-not-found]

from .whole_rms_math import _rms_norm


def qkv_norm(batch, pages, *, split=False):
    dtype = T.bfloat16
    q_width, kv_width = 1536, 512
    block, words, scales, stride = 128, 128, 4, 132

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
            q_gamma = T.alloc_shared((q_width,), dtype)
            kv_gamma = T.alloc_shared((kv_width,), dtype)
            packed = T.alloc_shared((1, stride), T.int32)
            packed_codes = T.view(packed, (stride * 4,), dtype=T.float8_e4m3fn)
            packed_scales = T.view(packed, (stride,), dtype=T.float32)
            norm_kv = T.alloc_shared((1, kv_width), dtype)
            T.copy(GQ, q_gamma)
            T.copy(GK, kv_gamma)
            # Scope the reductions to one row so their temporary buffers can share UB.
            for r in T.serial(src.shape[0]):
                _rms_norm(src, kv_gamma, r, norm_kv, kv_width, q_width)
                _rms_norm(src, q_gamma, r, dst, q_width, 0)
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

                slot = Slot[row, r]
                if slot >= 0 and slot < pages * 128:
                    T.copy(packed, CacheRows[slot : slot + 1, :])
            T.copy(dst, Norm[row : row + 1, :])

    return main
