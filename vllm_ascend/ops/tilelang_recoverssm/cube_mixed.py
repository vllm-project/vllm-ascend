# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused four-token BF16 projection, Gram and vector recurrence for Ascend 950."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language.kernel import get_thread_binding

from .verify import prepare


@lru_cache(maxsize=32)
def compile_fused(cfg, tokens, blocks):
    desc, rows, hk, hv, pages, L, _, _, norm, scale, sp_beta, threshold = cfg
    al, ga, gb, bias, qs, ks, vs, ss0, rs, off, ids, os = desc
    ss = ss0[1]
    ratio = hv // hk
    height = ratio * 128

    @T.prim_func
    def fused(
        A: T.Tensor((al[0],), "float32"),
        G: T.Tensor((ga[0],), "bfloat16"),
        B: T.Tensor((gb[0],), "bfloat16"),
        Bias: T.Tensor((bias[0],), "float32"),
        Q: T.StridedTensor((tokens, hk, 128), qs[1], "bfloat16"),
        K: T.StridedTensor((tokens, hk, 128), ks[1], "bfloat16"),
        V: T.Tensor((vs[0],), "bfloat16"),
        S: T.Tensor((((pages - 1) * ss[0] + (hv - 1) * ss[1]) // 128 + 128, 128), "bfloat16"),
        R: T.Tensor((rs[0],), "float32"),
        Off: T.Tensor((off[0],), off[2]),
        PageIds: T.Tensor((ids[0],), ids[2]),
        Out: T.Tensor((os[0],), "bfloat16"),
    ):
        with T.MixedKernel(blocks, sids=2) as (block, sid):
            for wave in T.serial(rows * hk // blocks):
                task = wave * blocks + block
                page = T.min(T.max(T.cast(PageIds[(task // hk) * ids[1][0]], "int32"), 0), pages - 1)
                base = page * (ss[0] // 128) + (task % hk) * ratio * 128
                lhs = T.alloc_l1((height, 128), "bfloat16")
                rhs = T.alloc_l1((16, 128), "bfloat16")
                small = T.alloc_l0c((16, 16), "float32")
                large = T.alloc_l0c((16, height), "float32")
                yub = T.alloc_shared((8, height), "float32")
                gub = T.alloc_shared((8, 16), "float32")
                row = task // hk
                head = task % hk
                # Captured buffers and SP may contain padded rows. Read device offsets
                # even at the full capture size. Extra rows fetched by DMA are masked below.
                begin = T.cast(Off[row * off[1][0]], "int64")
                end = T.cast(Off[(row + 1) * off[1][0]], "int64")
                if begin == row * 4 and row * 4 + 4 <= tokens:
                    for copy in T.unroll(2):
                        T.copy(K[row * 4 : row * 4 + 4, head, 0:128], rhs[copy * 8 : copy * 8 + 4, 0:128])
                        T.copy(Q[row * 4 : row * 4 + 4, head, 0:128], rhs[copy * 8 + 4 : copy * 8 + 8, 0:128])
                elif begin + 4 <= tokens:
                    for copy in T.unroll(2):
                        T.copy(K[begin : begin + 4, head, 0:128], rhs[copy * 8 : copy * 8 + 4, 0:128])
                        T.copy(Q[begin : begin + 4, head, 0:128], rhs[copy * 8 + 4 : copy * 8 + 8, 0:128])
                else:
                    for copy in T.unroll(2):
                        for pos in T.unroll(4):
                            valid_row = T.if_then_else(begin + pos < end, begin + pos, 0)
                            T.copy(K[valid_row, head, 0:128], rhs[copy * 8 + pos, 0:128])
                            T.copy(Q[valid_row, head, 0:128], rhs[copy * 8 + 4 + pos, 0:128])
                T.copy(S[base : base + height, 0:128], lhs)
                T.gemm(rhs, lhs, large, transpose_B=True, clear_accum=True)
                T.dual_copy(large, yub)
                T.gemm(rhs, rhs, small, transpose_B=True, clear_accum=True)
                T.dual_copy(small, gub)

                with T.SimtVF(threads=256):
                    tid = get_thread_binding(0)
                    if tid < ratio * 64:
                        h = head * ratio + tid // 64
                        v = sid * 64 + tid % 64
                        begin = T.cast(Off[row * off[1][0]], "int64")
                        end = T.cast(Off[(row + 1) * off[1][0]], "int64")
                        page = T.cast(PageIds[row * ids[1][0]], "int64")
                        weights = T.alloc_local((L,), "float32")
                        for z0 in T.unroll(L):
                            weights[z0] = 0.0
                        kn = T.alloc_local((L,), "float32")
                        initial = T.alloc_var("float32", init=1.0)
                        for t in T.unroll(L):
                            if begin + t < end:
                                token = begin + t
                                if page > 0 and page < pages:
                                    x = T.cast(G[token * ga[1][0] + h * ga[1][1]], "float32") + Bias[h * bias[1][0]]
                                    z = x * sp_beta
                                    sp = T.if_then_else(
                                        z > threshold,
                                        x,
                                        (T.max(z, 0.0) + T.log1p(T.exp(-T.abs(z)))) / sp_beta,
                                    )
                                    d = T.exp(-T.exp(A[h * al[1][0]]) * sp)
                                    beta = 1.0 / (1.0 + T.exp(-T.cast(B[token * gb[1][0] + h * gb[1][1]], "float32")))
                                    initial = initial * d
                                    kn[t] = T.rsqrt(gub[t, t] + 1e-6) if norm else 1.0
                                    qn = T.rsqrt(gub[L + t, L + t] + 1e-6) if norm else 1.0
                                    prediction = T.alloc_var(
                                        "float32",
                                        init=initial * yub[t, (h % ratio) * 128 + v],
                                    )
                                    output = T.alloc_var(
                                        "float32",
                                        init=initial * yub[L + t, (h % ratio) * 128 + v],
                                    )
                                    for j in T.unroll(t):
                                        weights[j] = weights[j] * d
                                        prediction = prediction + weights[j] * gub[j, t]
                                        output = output + weights[j] * gub[j, L + t]
                                    correction = (
                                        T.cast(V[token * vs[1][0] + h * vs[1][1] + v * vs[1][2]], "float32")
                                        - prediction * kn[t]
                                    ) * beta
                                    weights[t] = correction * kn[t]
                                    output = (output + weights[t] * gub[t, L + t]) * qn * scale
                                    record = page * rs[1][0] + h * rs[1][1] + t * rs[1][2]
                                    R[record + v * rs[1][3]] = correction
                                    R[record + (128 + v) * rs[1][3]] = (
                                        T.cast(
                                            K[token, head, v],
                                            "float32",
                                        )
                                        * kn[t]
                                    )
                                    if v == 0:
                                        R[record + 256 * rs[1][3]] = d
                                    Out[token * os[1][0] + h * os[1][1] + v * os[1][2]] = T.cast(output, "bfloat16")
                                else:
                                    Out[token * os[1][0] + h * os[1][1] + v * os[1][2]] = T.cast(0.0, "bfloat16")

    return tilelang.compile(fused, target="ascend")


def cube_verify(A, a, b, bias, q, k, v, **kwargs):
    state = kwargs["checkpoint_state"]
    # The mixed kernel gives both vector subcores a copy of the four Q/K rows.
    # Each consumes 64 value rows per head and immediately emits output and replay.
    if (
        kwargs["spec_query_len"] != 4
        or state.dtype != torch.bfloat16
        or q.ndim != 4
        or k.ndim != 4
        or not q.shape[2]
        or state.ndim != 4
        or state.shape[1] // q.shape[2] > 4
        or state.stride()[-2:] != (128, 1)
        or state.stride(0) % 128
        or state.stride(1) != 16384
        or state.storage_offset() % 16
        or any(t.stride(-1) != 1 or t.storage_offset() % 16 or t.stride(1) % 16 or t.stride(2) % 16 for t in (q, k))
    ):
        from .cube_verify import cube_verify as separate_projection

        return separate_projection(A, a, b, bias, q, k, v, **kwargs)
    out, cfg, buf = prepare(A.float(), a, b, bias.float(), q, k, v, shaped_qk=True, **kwargs)
    if not out.numel():
        return out
    tasks = kwargs["state_indices"].numel() * q.shape[2]
    blocks = 32 if tasks % 32 == 0 else tasks
    compile_fused(cfg, q.shape[1], blocks)(
        *buf[:4], buf[4], buf[5], buf[6], buf[7].view(-1, 128), buf[8], buf[9], buf[10], buf[11]
    )
    return out
