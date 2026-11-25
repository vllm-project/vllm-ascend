# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from ..shared.simd_mxrow import encode_mx16


def mx_quant(batch, width):
    bk = 256
    bf16, fp8 = T.bfloat16, T.float8_e4m3fn

    @T.prim_func
    def main(
        Value: T.Tensor((batch, width), bf16),
        C3: T.Tensor((batch, width), fp8),
        S3: T.Tensor((width // 64, batch), T.int16),
    ):
        with T.Kernel((batch // 16) * (width // bk)) as task:
            gr = task // (width // bk)
            k = task % (width // bk)
            inp = T.alloc_shared((16, bk), bf16)
            code = T.alloc_shared((16, bk), fp8)
            sf = T.alloc_shared((bk // 64, 16), T.int16)
            T.copy(Value[gr * 16 : (gr + 1) * 16, k * bk : (k + 1) * bk], inp)
            encode_mx16(inp, code, sf, bk)
            T.copy(code, C3[gr * 16 : (gr + 1) * 16, k * bk : (k + 1) * bk])
            T.copy(sf, S3[k * (bk // 64) : (k + 1) * (bk // 64), gr * 16 : (gr + 1) * 16])

    return main
