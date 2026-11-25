# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tilelang.ascend.language as T

from .register_quant import encode_block


def wuv_program(batch, bm, bn, *, input_major=False):
    bk, heads = 128, 64
    assert batch % bm == 0 and bn in (128, 256)

    weight_shape = (64, 512, 256) if input_major else (64, 256, 512)

    @T.prim_func
    def main(
        Latent3: T.Tensor((batch, 32768), T.bfloat16),
        WV: T.Tensor(weight_shape, T.bfloat16),
        C3: T.Tensor((batch, 16384), T.float8_e4m3fn),
        S3: T.Tensor((256, batch), T.int16),
    ):
        with T.MixedKernel((batch // bm) * heads * (256 // bn), sids=2) as (task, sid):
            gr = task // (heads * (256 // bn))
            head = (task // (256 // bn)) % heads
            col = task % (256 // bn)
            x = T.alloc_l1((bm, bk), T.bfloat16)
            w = T.alloc_l1((bn, bk), T.bfloat16)
            a0 = T.alloc_l0a((bm, bk), T.bfloat16)
            b0 = T.alloc_l0b((bn, bk), T.bfloat16)
            acc = T.alloc_l0c((bm, bn), T.float32)
            vals = T.alloc_shared(((bm // 2), bn), T.float32)
            code = T.alloc_shared(((bm // 2), bn), T.float8_e4m3fn)
            sf = T.alloc_shared((bn // 64, (bm // 2)), T.int16)
            for k in T.Pipelined(512 // bk, num_stages=1):
                T.copy(Latent3[gr * bm : (gr + 1) * bm, head * 512 + k * bk : head * 512 + (k + 1) * bk], x)
                if input_major:
                    T.copy(WV[head, k * bk : (k + 1) * bk, col * bn : (col + 1) * bn], w, transpose=True)
                else:
                    T.copy(WV[head, col * bn : (col + 1) * bn, k * bk : (k + 1) * bk], w)
                T.copy(x, a0)
                T.copy(w, b0)
                T.gemm(a0, b0, acc, transpose_B=True, clear_accum=k == 0)
            T.dual_copy(acc, vals)
            encode_block(vals, code, sf, (bm // 2), bn, 5)
            r0 = gr * bm + sid * (bm // 2)
            off = head * 256 + col * bn
            T.copy(code, C3[r0 : r0 + (bm // 2), off : off + bn])
            T.copy(sf, S3[off // 64 : off // 64 + bn // 64, r0 : r0 + (bm // 2)])

    return main
