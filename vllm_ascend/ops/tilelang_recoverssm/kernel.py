# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 950 SIMT implementation of speculative GDN Verify."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language.kernel import get_thread_binding


def make_kernel(layouts, rows, hk, hv, pages, window, tile_v, block_cap, normalize, scale, sp_beta, threshold):
    # The launcher passes flat, offset-preserving views. Pitches are in elements.
    sizes, strides, dtypes = zip(*layouts)
    al, ga, gb, bias, qs, ks, vs, ss, rs, starts, indices, os = strides
    lanes = 32
    value_tiles = 128 // tile_v

    tasks = rows * hv * value_tiles
    blocks = min(block_cap, tasks) if block_cap else tasks

    @T.macro
    def warp_sum(x):
        # Five butterfly stages broadcast the sum to every lane of one value row.
        for j in T.unroll(5):
            x[0] = x[0] + T.call_extern("float32", "__shfl_xor", x[0], 1 << j, 32)

    @T.prim_func
    def verify(
        A: T.Tensor((sizes[0],), dtypes[0]),
        G: T.Tensor((sizes[1],), dtypes[1]),
        B: T.Tensor((sizes[2],), dtypes[2]),
        Bias: T.Tensor((sizes[3],), dtypes[3]),
        Q: T.Tensor((sizes[4],), dtypes[4]),
        K: T.Tensor((sizes[5],), dtypes[5]),
        V: T.Tensor((sizes[6],), dtypes[6]),
        S: T.Tensor((sizes[7],), dtypes[7]),
        R: T.Tensor((sizes[8],), dtypes[8]),
        Off: T.Tensor((sizes[9],), dtypes[9]),
        Id: T.Tensor((sizes[10],), dtypes[10]),
        Out: T.Tensor((sizes[11],), dtypes[11]),
    ):
        with T.Kernel(blocks) as core:  # noqa: SIM117 - preserve nested TileLang IR scopes
            with T.SimtVF(threads=lanes * tile_v):
                for wave in T.serial((tasks - 1 - core) // blocks + 1):
                    block = wave * blocks + core
                    tid = get_thread_binding(0)
                    lane = tid % lanes
                    h = block // value_tiles % hv
                    row = block // (value_tiles * hv)
                    vr = block % value_tiles * tile_v + tid // lanes
                    kh = h // (hv // hk)
                    page = T.cast(Id[row * indices[0]], "int64")
                    begin = T.cast(Off[row * starts[0]], "int64")
                    end = T.cast(Off[(row + 1) * starts[0]], "int64")
                    state = T.alloc_local((4,), "float32")
                    key = T.alloc_local((4,), "float32")
                    query = T.alloc_local((4,), "float32")
                    kn = T.alloc_local((1,), "float32")
                    qn = T.alloc_local((1,), "float32")
                    dot = T.alloc_local((1,), "float32")
                    if page > 0 and page < pages and begin < end:
                        for j in T.unroll(4):
                            state[j] = T.cast(
                                S[page * ss[0] + h * ss[1] + vr * ss[2] + (lane + j * 32) * ss[3]], "float32"
                            )
                        for pos in T.serial(end - begin):
                            t = begin + pos
                            kn[0] = 0.0
                            qn[0] = 0.0
                            for j in T.unroll(4):
                                col = lane + j * 32
                                key[j] = T.cast(K[t * ks[0] + kh * ks[1] + col * ks[2]], "float32")
                                query[j] = T.cast(Q[t * qs[0] + kh * qs[1] + col * qs[2]], "float32")
                                if normalize:
                                    kn[0] += key[j] * key[j]
                                    qn[0] += query[j] * query[j]
                            if normalize:
                                warp_sum(kn)
                                warp_sum(qn)
                                kn[0] = T.rsqrt(kn[0] + 1e-6)
                                qn[0] = T.rsqrt(qn[0] + 1e-6)
                            x = T.cast(G[t * ga[0] + h * ga[1]], "float32") + T.cast(Bias[h * bias[0]], "float32")
                            z = x * sp_beta
                            # max(z,0) + log(1+exp(-abs(z))) is overflow safe.
                            sp = T.if_then_else(z > threshold, x, (T.max(z, 0.0) + T.log1p(T.exp(-T.abs(z)))) / sp_beta)
                            decay = T.exp(-T.exp(T.cast(A[h * al[0]], "float32")) * sp)
                            gate = T.cast(B[t * gb[0] + h * gb[1]], "float32")
                            beta = 1.0 / (1.0 + T.exp(-gate))
                            dot[0] = 0.0
                            for j in T.unroll(4):
                                if normalize:
                                    key[j] *= kn[0]
                                    query[j] *= qn[0]
                                state[j] *= decay
                                dot[0] += state[j] * key[j]
                            warp_sum(dot)
                            correction = (T.cast(V[t * vs[0] + h * vs[1] + vr * vs[2]], "float32") - dot[0]) * beta
                            record = page * rs[0] + h * rs[1] + pos * rs[2]
                            if lane == 0:
                                R[record + vr * rs[3]] = correction
                            dot[0] = 0.0
                            for j in T.unroll(4):
                                state[j] += correction * key[j]
                                dot[0] += state[j] * query[j]
                                if vr == 0:
                                    R[record + (128 + lane + j * 32) * rs[3]] = key[j]
                            warp_sum(dot)
                            if vr == 0 and lane == 0:
                                R[record + 256 * rs[3]] = decay
                            if lane == 0:
                                Out[t * os[0] + h * os[1] + vr * os[2]] = T.cast(dot[0] * scale, "bfloat16")
                    else:
                        if lane == 0:
                            for pos in T.serial(end - begin):
                                Out[(begin + pos) * os[0] + h * os[1] + vr * os[2]] = T.cast(0.0, "bfloat16")

    return verify


@lru_cache(maxsize=64)
def compiled(*config):
    return tilelang.compile(make_kernel(*config), target="ascend")
