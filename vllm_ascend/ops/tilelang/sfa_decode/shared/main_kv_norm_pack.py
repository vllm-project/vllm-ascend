# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from .whole_kv_rms512 import make_kv_norm
from .whole_rms_math import make_query_norm


def make_norm_pack(qhalf, cache_rows, eps, mode):
    query_norm = make_query_norm(mode.startswith("perf"), mode.endswith("nr"))
    kv_norm = make_kv_norm(2)
    q_width, kv_width = 1536, 512
    block, words, scales, stride = 128, 128, 4, 132
    dtype = T.bfloat16

    @T.macro
    def norm_pack(QKV, QGamma, KVGamma, Slot, Cache, bx, sid, QNorm):
        scratch_stats = T.alloc_shared((8,), T.float32)
        q_gamma = T.alloc_shared((q_width,), dtype)
        kv_gamma = T.alloc_shared((kv_width,), dtype)
        packed = T.alloc_shared((1, stride), T.int32)
        packed_codes = T.view(packed, (stride * 4,), dtype=T.float8_e4m3fn)
        packed_scales = T.view(packed, (stride,), dtype=T.float32)
        norm_kv = T.alloc_shared((1, kv_width), dtype)
        T.copy(QGamma, q_gamma)
        T.copy(KVGamma, kv_gamma)
        for r in T.serial(qhalf):
            kv_norm(QKV, kv_gamma, r, norm_kv, eps, scratch_stats)
            query_norm(QKV, q_gamma, r, QNorm, eps, scratch_stats)
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
