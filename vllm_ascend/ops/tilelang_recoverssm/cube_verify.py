# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short-window BF16 GDN verification via exact raw-key Gram and checkpoint projections."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language.kernel import get_thread_binding

from .verify import prepare


@lru_cache(maxsize=32)
def compile_projection(rows, hv, pages, ss, ids, iddtype):
    @T.prim_func
    def projection(
        S: T.Tensor((((pages - 1) * ss[0] + (hv - 1) * ss[1]) // 128 + 128, 128), "bfloat16"),
        P: T.Tensor((rows * hv * 16, 128), "bfloat16"),
        PageIds: T.Tensor((1 + (rows - 1) * ids,), iddtype),
        Y: T.Tensor((rows * hv * 128, 16), "float32"),
        Gram: T.Tensor((rows * hv * 16, 16), "float32"),
    ):
        with T.Kernel(rows * hv) as task:
            page = T.min(T.max(T.cast(PageIds[(task // hv) * ids], "int32"), 0), pages - 1)
            lhs = T.alloc_l1((128, 128), "bfloat16")
            rhs = T.alloc_l1((16, 128), "bfloat16")
            small = T.alloc_l0c((16, 16), "float32")
            large = T.alloc_l0c((128, 16), "float32")
            T.copy(P[task * 16 : task * 16 + 16, 0:128], rhs)
            T.copy(
                S[
                    page * (ss[0] // 128) + (task % hv) * (ss[1] // 128) : page * (ss[0] // 128)
                    + (task % hv) * (ss[1] // 128)
                    + 128,
                    0:128,
                ],
                lhs,
            )
            T.gemm(lhs, rhs, large, transpose_B=True, clear_accum=True)
            T.copy(large, Y[task * 128, 0])
            T.gemm(rhs, rhs, small, transpose_B=True, clear_accum=True)
            T.copy(small, Gram[task * 16, 0])

    return tilelang.compile(projection, target="ascend")


@lru_cache(maxsize=32)
def compile_pack(cfg):
    desc, n, hk, hv, pages, L, *_ = cfg
    qs, ks, ofs = desc[4], desc[5], desc[9]

    @T.prim_func
    def pack(
        Q: T.Tensor((qs[0],), "bfloat16"),
        K: T.Tensor((ks[0],), "bfloat16"),
        Off: T.Tensor((ofs[0],), ofs[2]),
        P: T.Tensor((n, hv, 16, 128), "bfloat16"),
    ):
        with T.Kernel(n * hv) as task:  # noqa: SIM117 - preserve nested TileLang IR scopes
            with T.SimtVF(threads=512):
                tid = get_thread_binding(0)
                r = task // hv
                h = task % hv
                kh = h // (hv // hk)
                begin = T.cast(Off[r * ofs[1][0]], "int64")
                end = T.cast(Off[(r + 1) * ofs[1][0]], "int64")
                for wave in T.unroll(4):
                    i = wave * 4 + tid // 128
                    col = tid % 128
                    if i < L and begin + i < end:
                        P[r, h, i, col] = K[(begin + i) * ks[1][0] + kh * ks[1][1] + col * ks[1][2]]
                    elif i >= L and i < 2 * L and begin + i - L < end:
                        P[r, h, i, col] = Q[(begin + i - L) * qs[1][0] + kh * qs[1][1] + col * qs[1][2]]
                    else:
                        P[r, h, i, col] = T.cast(0.0, "bfloat16")

    return tilelang.compile(pack, target="ascend")


@lru_cache(maxsize=32)
def compile_finish(cfg):
    desc, n, hk, hv, pages, L, _, _, norm, scale, sp_beta, threshold = cfg
    al, ga, gb, bias, _, _, vs, _, rs, off, ids, os = desc

    @T.prim_func
    def finish(
        A: T.Tensor((al[0],), al[2]),
        G: T.Tensor((ga[0],), "bfloat16"),
        B: T.Tensor((gb[0],), "bfloat16"),
        Bias: T.Tensor((bias[0],), bias[2]),
        V: T.Tensor((vs[0],), "bfloat16"),
        R: T.Tensor((rs[0],), "float32"),
        Offsets: T.Tensor((off[0],), off[2]),
        PageIds: T.Tensor((ids[0],), ids[2]),
        P: T.Tensor((n * hv, 16, 128), "bfloat16"),
        Y: T.Tensor((n * hv, 128, 16), "float32"),
        Gram: T.Tensor((n * hv, 16, 16), "float32"),
        Out: T.Tensor((os[0],), "bfloat16"),
    ):
        with T.Kernel(n * hv) as task:  # noqa: SIM117 - preserve nested TileLang IR scopes
            with T.SimtVF(threads=128):
                row = task // hv
                h = task % hv
                v = get_thread_binding(0)
                page = T.cast(PageIds[row * ids[1][0]], "int64")
                begin = T.cast(Offsets[row * off[1][0]], "int64")
                end = T.cast(Offsets[(row + 1) * off[1][0]], "int64")
                weights = T.alloc_local((L,), "float32")
                kn = T.alloc_local((L,), "float32")
                initial = T.alloc_var("float32", init=1.0)
                for t in T.unroll(L):
                    if begin + t < end:
                        token = begin + t
                        if page > 0 and page < pages:
                            x = T.cast(G[token * ga[1][0] + h * ga[1][1]], "float32") + T.cast(
                                Bias[h * bias[1][0]], "float32"
                            )
                            z = x * sp_beta
                            sp = T.if_then_else(z > threshold, x, (T.max(z, 0.0) + T.log1p(T.exp(-T.abs(z)))) / sp_beta)
                            d = T.exp(-T.exp(T.cast(A[h * al[1][0]], "float32")) * sp)
                            beta = 1.0 / (1.0 + T.exp(-T.cast(B[token * gb[1][0] + h * gb[1][1]], "float32")))
                            initial = initial * d
                            kn[t] = T.rsqrt(Gram[task, t, t] + 1e-6) if norm else 1.0
                            qn = T.rsqrt(Gram[task, L + t, L + t] + 1e-6) if norm else 1.0
                            prediction = T.alloc_var("float32", init=initial * Y[task, v, t])
                            output = T.alloc_var("float32", init=initial * Y[task, v, L + t])
                            for j in T.unroll(t):
                                weights[j] = weights[j] * d
                                prediction = prediction + weights[j] * Gram[task, j, t]
                                output = output + weights[j] * Gram[task, j, L + t]
                            correction = (
                                T.cast(V[token * vs[1][0] + h * vs[1][1] + v * vs[1][2]], "float32")
                                - prediction * kn[t]
                            ) * beta
                            weights[t] = correction * kn[t]
                            output = (output + weights[t] * Gram[task, t, L + t]) * qn * scale
                            record = page * rs[1][0] + h * rs[1][1] + t * rs[1][2]
                            R[record + v * rs[1][3]] = correction
                            R[record + (128 + v) * rs[1][3]] = T.cast(P[task, t, v], "float32") * kn[t]
                            if v == 0:
                                R[record + 256 * rs[1][3]] = d
                            Out[token * os[1][0] + h * os[1][1] + v * os[1][2]] = T.cast(output, "bfloat16")
                        else:
                            Out[token * os[1][0] + h * os[1][1] + v * os[1][2]] = T.cast(0.0, "bfloat16")

    return tilelang.compile(finish, target="ascend")


def cube_verify(A, a, b, bias, q, k, v, **kwargs):
    state = kwargs["checkpoint_state"]
    # Cube DMA needs aligned, contiguous 128-element rows. Keep the SIMT path
    # for FP32 states, unusual cache views and longer speculative windows.
    if (
        kwargs["spec_query_len"] > 4
        or state.dtype != torch.bfloat16
        or state.stride()[-2:] != (128, 1)
        or state.stride(0) % 128
        or state.stride(1) % 128
        or state.storage_offset() % 16
    ):
        from .verify import gdn_recoverssm_verify

        return gdn_recoverssm_verify(A, a, b, bias, q, k, v, **kwargs)
    out, cfg, buf = prepare(A, a, b, bias, q, k, v, **kwargs)
    if not out.numel():
        return out
    desc, n, hk, hv, pages, L, *_ = cfg
    P = torch.empty((n, hv, 16, 128), device=q.device, dtype=torch.bfloat16)
    Y = torch.empty((n * hv, 128, 16), device=q.device, dtype=torch.float32)
    gram = torch.empty((n * hv, 16, 16), device=q.device, dtype=torch.float32)
    compile_pack(cfg)(buf[4], buf[5], buf[9], P)
    compile_projection(n, hv, pages, tuple(state.stride()), desc[10][1][0], desc[10][2])(
        buf[7].view(-1, 128), P.view(-1, 128), buf[10], Y.view(-1, 16), gram.view(-1, 16)
    )
    compile_finish(cfg)(*buf[:4], buf[6], buf[8], buf[9], buf[10], P.view(-1, 16, 128), Y, gram, buf[11])
    return out
