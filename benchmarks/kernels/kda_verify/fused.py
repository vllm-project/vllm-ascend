# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head-parallel CV fusion between the two large native dense projections.

Native speculative convolution cache is updated in place. SSM checkpoint is
read-only; correction/KG records match the original Replay SSM contract.
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F
from small_batch import small_batch_program


def program(c, unroll_taps=False):
    taps = T.Unroll if unroll_taps else T.Serial
    B, L, H, D = T.dynamic("B"), c.tokens, c.heads, c.dim
    state_rows = 16 if L <= 8 else 32
    stages = 6 if c.batch == 1 else 4
    PITCH = 3 * H * D + H + 2 * D

    @T.prim_func
    def main(
        P: T.Tensor((B * L, PITCH), "bfloat16"),
        WF: T.Tensor((H // 2, 2 * D, D), "bfloat16"),
        WG: T.Tensor((H // 2, 2 * D, D), "bfloat16"),
        CW: T.Tensor((H, 3, D, 4), "bfloat16"),
        CS: T.Tensor((B + 1, L + 2, 3 * H * D), "bfloat16"),
        State: T.Tensor(((B + 1) * H * D, D), "float32"),
        Accepted: T.Tensor((B,), "int32"),
        A: T.Tensor((H,), "float32"),
        DT: T.Tensor((H, D), "float32"),
        NW: T.Tensor((D,), "bfloat16"),
        Y: T.Tensor((B * L, H * D), "bfloat16"),
        Correction: T.Tensor((B + 1, H, L, D), "float32"),
        KG: T.Tensor((B + 1, H, L, 2 * D), "bfloat16"),
    ):
        with T.MixedKernel(32, sids=2) as (bx, sid):
            al1 = T.alloc_l1((16, D), "bfloat16")
            bl1 = T.alloc_l1((16, D), "bfloat16")
            wl1 = T.alloc_l1((2 * D, D), "bfloat16")
            vl1 = T.alloc_l1((2 * D, D), "bfloat16")
            ga = T.alloc_l0c((16, 2 * D), "float32")
            gb = T.alloc_l0c((16, 2 * D), "float32")
            gf = T.alloc_shared((16, D), "float32")
            gg = T.alloc_shared((16, D), "float32")
            g = T.alloc_shared((16, 2 * D), "bfloat16")
            raw = T.alloc_shared((3, L, D), "bfloat16")
            hist = T.alloc_shared((3, 3, D), "bfloat16")
            cache = T.alloc_shared((3, L + 2, D), "bfloat16")
            cw = T.alloc_shared((3, D, 4), "bfloat16")
            conv = T.alloc_shared((3, L, D), "bfloat16")
            cf = T.alloc_shared((3, L, D), "float32")
            gatef = T.alloc_shared((L, 2 * D), "float32")
            betaf = T.alloc_shared((L,), "float32")
            nwf = T.alloc_shared((D,), "float32")
            coref = T.alloc_shared((L, D), "float32")
            yf = T.alloc_shared((L, D), "float32")
            beta = T.alloc_shared((L, 1), "bfloat16")
            bs = T.alloc_shared((L,), "float32")
            ac = T.alloc_shared((1,), "int32")
            alog = T.alloc_shared((1,), "float32")
            dt = T.alloc_shared((D,), "float32")
            nw = T.alloc_shared((D,), "bfloat16")
            knq = T.alloc_shared((2, L, D), "float32")
            decay = T.alloc_shared((L, D), "float32")
            val = T.alloc_shared((L, D), "float32")
            prefix = T.alloc_shared((L, D), "float32")
            qk = T.alloc_shared((state_rows, D), "float32")
            qkl1 = T.alloc_l1((2 * state_rows, D), "float32")
            sl1 = T.alloc_l1((2 * D, D), "float32")
            cp = T.alloc_l0c((2 * state_rows, 2 * D), "float32")
            proj = T.alloc_shared((state_rows, 2 * D), "float32")
            ck = T.alloc_shared((L, L), "float32")
            cq = T.alloc_shared((L, L), "float32")
            T.set_hf32_mode(None)
            corr = T.alloc_shared((L, D), "float32")
            kg = T.alloc_shared((L, 2 * D), "bfloat16")
            of = T.alloc_shared((L, D), "float32")
            core = T.alloc_shared((L, D), "bfloat16")
            y = T.alloc_shared((L, D), "bfloat16")
            T.copy(NW, nw)
            for it in T.Pipelined(
                T.ceildiv(B * (H // 2) - bx, 32), num_stages=stages, annotations={"enable_offset": True}
            ):
                job = it * 32 + bx
                b = job // (H // 2)
                hp = job % (H // 2)
                h = hp * 2 + sid
                T.copy(Accepted[b : b + 1], ac)
                T.copy(P[b * L : (b + 1) * L, 3 * H * D + H : 3 * H * D + H + D], al1[:L, :])
                T.copy(P[b * L : (b + 1) * L, 3 * H * D + H + D : PITCH], bl1[:L, :])
                T.copy(WF[hp, :, :], wl1)
                T.copy(WG[hp, :, :], vl1)
                T.gemm(al1, wl1, ga, transpose_B=True, clear_accum=True)
                T.gemm(bl1, vl1, gb, transpose_B=True, clear_accum=True)
                T.dual_copy(ga, gf)
                T.dual_copy(gb, gg)
                with T.SimtVF(threads=128):
                    for t, j in T.Parallel(16, D):
                        g[t, j] = gf[t, j]
                        g[t, D + j] = gg[t, j]
                T.copy(CW[h, :, :, :], cw)
                T.copy(A[h : h + 1], alog)
                T.copy(DT[h, :], dt)
                T.copy(P[b * L : (b + 1) * L, 3 * H * D + h : 3 * H * D + h + 1], beta)
                for q in T.Serial(3):
                    T.copy(P[b * L : (b + 1) * L, (q * H + h) * D : (q * H + h + 1) * D], raw[q, :, :])
                    T.copy(CS[b + 1, :, (q * H + h) * D : (q * H + h + 1) * D], cache[q, :, :])
                with T.SimtVF(threads=128):
                    for q, t, j in T.Parallel(3, 3, D):
                        hist[q, t, j] = cache[q, ac[0] - 1 + t, j]
                with T.SimtVF(threads=128):
                    cv = T.alloc_fragment((3, L, D), "float32")
                    for q, t, j in T.Parallel(3, L, D):
                        cv[q, t, j] = 0
                    for tap in taps(4):
                        for q, t, j in T.Parallel(3, L, D):
                            if t + tap < 3:
                                cv[q, t, j] += T.cast(hist[q, t + tap, j], "float32") * T.cast(cw[q, j, tap], "float32")
                            else:
                                cv[q, t, j] += T.cast(raw[q, t + tap - 3, j], "float32") * T.cast(
                                    cw[q, j, tap], "float32"
                                )
                    for q, t, j in T.Parallel(3, L, D):
                        conv[q, t, j] = cv[q, t, j] / (1 + T.exp(-cv[q, t, j]))
                    for q, t, j in T.Parallel(3, L + 2, D):
                        if t < 2:
                            cache[q, t, j] = hist[q, t + 1, j]
                        else:
                            cache[q, t, j] = raw[q, t - 2, j]
                for q in T.Serial(3):
                    T.copy(cache[q, :, :], CS[b + 1, :, (q * H + h) * D : (q * H + h + 1) * D])
                with T.SimtVF(threads=128):
                    for q, t, j in T.Parallel(3, L, D):
                        cf[q, t, j] = conv[q, t, j]
                    for t, j in T.Parallel(L, 2 * D):
                        gatef[t, j] = g[t, j]
                    for t, j in T.Parallel(L, D):
                        kg[t, j] = conv[1, t, j]
                        kg[t, D + j] = g[t, j]
                        val[t, j] = conv[2, t, j]
                    for t in T.Parallel(L):
                        betaf[t] = beta[t, 0]
                    for j in T.Parallel(D):
                        nwf[j] = nw[j]
                with T.SimdVF():
                    one = T.simd.vdup(T.float32(1), "float32")
                    zero = T.simd.vdup(T.float32(0), "float32")
                    scalar_mask = T.simd.pset(32, "PAT_VL1")
                    av = T.simd.vexp(T.simd.vld(alog[0], dist="BRC_B32"))
                    for t in T.Serial(L):
                        for q in T.Serial(2):
                            c0 = T.simd.vld(cf[q, t, 0])
                            c1 = T.simd.vld(cf[q, t, 64])
                            ss = T.simd.vdupv(
                                T.simd.vadd(T.simd.vcadd(T.simd.vmul(c0, c0)), T.simd.vcadd(T.simd.vmul(c1, c1)))
                            )
                            safe = T.simd.vsel(ss, one, T.simd.vcmp(ss, zero, op="gt"))
                            inv = T.simd.vdiv(one, T.simd.vsqrt(safe))
                            scale = T.simd.vmuls(inv, T.if_then_else(q == 0, T.float32(D**-0.5), T.float32(1)))
                            T.simd.vsts(knq[q, t, 0], T.simd.vmul(c0, scale))
                            T.simd.vsts(knq[q, t, 64], T.simd.vmul(c1, scale))
                        for j in T.Unroll(2):
                            gg = T.simd.vadd(T.simd.vld(gatef[t, j * 64]), T.simd.vld(dt[j * 64]))
                            den = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(T.simd.vmul(av, gg))))
                            gate = T.simd.vdiv(T.simd.vdup(T.float32(c.lower_bound), "float32"), den)
                            T.simd.vsts(decay[t, j * 64], T.simd.vexp(gate))
                        bv = T.simd.vld(betaf[t], dist="BRC_B32")
                        bb = T.simd.vdiv(one, T.simd.vadd(one, T.simd.vexp(T.simd.vneg(bv))))
                        T.simd.vsts(bs[t], bb, scalar_mask, dist="ONEPT_B32", extent=1)
                with T.SimtVF(threads=128):
                    for t in T.Serial(L):
                        for j in T.Parallel(D):
                            if t == 0:
                                prefix[t, j] = decay[t, j]
                            else:
                                prefix[t, j] = prefix[t - 1, j] * decay[t, j]
                    for t, j in T.Parallel(state_rows, D):
                        if t < L:
                            qk[t, j] = knq[1, t, j] * prefix[t, j]
                        elif t < 2 * L:
                            qk[t, j] = knq[0, t - L, j] * prefix[t - L, j]
                        else:
                            qk[t, j] = 0
                T.dual_copy(qk, qkl1)
                T.copy(State[((b + 1) * H + hp * 2) * D : ((b + 1) * H + hp * 2 + 2) * D, :], sl1)
                T.gemm(qkl1, sl1, cp, transpose_B=True, clear_accum=True)
                T.dual_copy(cp, proj)
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    ratio = T.simd.alloc_local((2,), "float32")
                    for t in T.Serial(L):
                        qt0 = T.simd.vld(knq[0, t, 0])
                        qt1 = T.simd.vld(knq[0, t, 64])
                        kt0 = T.simd.vld(knq[1, t, 0])
                        kt1 = T.simd.vld(knq[1, t, 64])
                        selfq = T.simd.vadd(T.simd.vcadd(T.simd.vmul(kt0, qt0)), T.simd.vcadd(T.simd.vmul(kt1, qt1)))
                        T.simd.vsts(cq[t, t], selfq, one, dist="ONEPT_B32", extent=1)
                        ratio[0] = T.simd.vdup(T.float32(1), "float32")
                        ratio[1] = T.simd.vdup(T.float32(1), "float32")
                        for back in T.Serial(t):
                            u = t - 1 - back
                            ratio[0] = T.simd.vmul(ratio[0], T.simd.vld(decay[u + 1, 0]))
                            ratio[1] = T.simd.vmul(ratio[1], T.simd.vld(decay[u + 1, 64]))
                            k0 = T.simd.vmul(T.simd.vld(knq[1, u, 0]), ratio[0])
                            k1 = T.simd.vmul(T.simd.vld(knq[1, u, 64]), ratio[1])
                            kk = T.simd.vadd(T.simd.vcadd(T.simd.vmul(k0, kt0)), T.simd.vcadd(T.simd.vmul(k1, kt1)))
                            qq = T.simd.vadd(T.simd.vcadd(T.simd.vmul(k0, qt0)), T.simd.vcadd(T.simd.vmul(k1, qt1)))
                            T.simd.vsts(ck[t, u], kk, one, dist="ONEPT_B32", extent=1)
                            T.simd.vsts(cq[t, u], qq, one, dist="ONEPT_B32", extent=1)
                with T.SimdVF():
                    acc = T.simd.alloc_local((4,), "float32")
                    for t in T.Serial(L):
                        acc[0] = T.simd.vld(proj[t, sid * D])
                        acc[1] = T.simd.vld(proj[t, sid * D + 64])
                        acc[2] = T.simd.vld(proj[t + L, sid * D])
                        acc[3] = T.simd.vld(proj[t + L, sid * D + 64])
                        for u in T.Serial(t):
                            c0 = T.simd.vld(corr[u, 0])
                            c1 = T.simd.vld(corr[u, 64])
                            k = T.simd.vld(ck[t, u], dist="BRC_B32")
                            q = T.simd.vld(cq[t, u], dist="BRC_B32")
                            T.simd.vmula(acc[0], c0, k)
                            T.simd.vmula(acc[1], c1, k)
                            T.simd.vmula(acc[2], c0, q)
                            T.simd.vmula(acc[3], c1, q)
                        bet = T.simd.vld(bs[t], dist="BRC_B32")
                        c0 = T.simd.vmul(T.simd.vsub(T.simd.vld(val[t, 0]), acc[0]), bet)
                        c1 = T.simd.vmul(T.simd.vsub(T.simd.vld(val[t, 64]), acc[1]), bet)
                        T.simd.vsts(corr[t, 0], c0)
                        T.simd.vsts(corr[t, 64], c1)
                        selfq = T.simd.vld(cq[t, t], dist="BRC_B32")
                        T.simd.vsts(of[t, 0], T.simd.vadd(acc[2], T.simd.vmul(c0, selfq)))
                        T.simd.vsts(of[t, 64], T.simd.vadd(acc[3], T.simd.vmul(c1, selfq)))
                        T.simd.mem_bar("VST_VLD")
                with T.SimtVF(threads=128):
                    for t, v in T.Parallel(L, D):
                        core[t, v] = of[t, v]
                    for t, v in T.Parallel(L, D):
                        coref[t, v] = core[t, v]
                with T.SimdVF():
                    one = T.simd.vdup(T.float32(1), "float32")
                    for t in T.Serial(L):
                        c0 = T.simd.vld(coref[t, 0])
                        c1 = T.simd.vld(coref[t, 64])
                        ss = T.simd.vdupv(
                            T.simd.vadd(T.simd.vcadd(T.simd.vmul(c0, c0)), T.simd.vcadd(T.simd.vmul(c1, c1)))
                        )
                        inv = T.simd.vdiv(
                            one, T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(ss, T.float32(1 / D)), T.float32(c.eps)))
                        )
                        for j in T.Unroll(2):
                            cc = T.simd.vld(coref[t, j * 64])
                            gw = T.simd.vld(gatef[t, D + j * 64])
                            den = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(gw)))
                            yy = T.simd.vdiv(T.simd.vmul(T.simd.vmul(cc, inv), T.simd.vld(nwf[j * 64])), den)
                            T.simd.vsts(yf[t, j * 64], yy)
                with T.SimtVF(threads=128):
                    for t, v in T.Parallel(L, D):
                        y[t, v] = yf[t, v]
                T.copy(y, Y[b * L : (b + 1) * L, h * D : (h + 1) * D])
                T.copy(corr, Correction[b + 1, h, :, :])
                T.copy(kg, KG[b + 1, h, :, :])

    return main


# Six-token BF16 input shapes whose next six rows select a faster GEMM.
EXTRA_REQUEST_BATCHES = frozenset((14, 17, 19, 22))

WEIGHT_REUSE_BATCHES = frozenset((1, 3, 5, 6, 7, 8, 9, 10))


DT_PROJECTION_ROWS = {(5, 20): (105, 100), (13, 10): (143, 160)}

GATE_TILE_ROWS = 32


def gate_projection_program(c):
    """Compute both strided gate projections while reusing each weight tile across tokens."""
    rows = T.dynamic("rows")
    width = c.heads * c.dim
    pitch = 3 * width + c.heads + 2 * c.dim
    tile_rows, tile_cols = GATE_TILE_ROWS, 512
    tiles = width // tile_cols
    padded_rows = T.ceildiv(rows, tile_rows) * tile_rows

    @T.prim_func
    def main(
        P: T.Tensor((rows, pitch), "bfloat16"),
        WF: T.Tensor((width, c.dim), "bfloat16"),
        WG: T.Tensor((width, c.dim), "bfloat16"),
        GF: T.Tensor((padded_rows, width), "bfloat16"),
        GG: T.Tensor((padded_rows, width), "bfloat16"),
    ):
        with T.Kernel(2 * tiles) as block:
            column = (block % tiles) * tile_cols
            gate = block // tiles
            start = pitch - 2 * c.dim + gate * c.dim
            activation = T.alloc_l1((tile_rows, c.dim), "bfloat16")
            weight = T.alloc_l1((tile_cols, c.dim), "bfloat16")
            accum = T.alloc_l0c((tile_rows, tile_cols), "float32")
            if gate == 0:
                T.copy(WF[column : column + tile_cols, :], weight)
            else:
                T.copy(WG[column : column + tile_cols, :], weight)
            for tile in T.Pipelined(T.ceildiv(rows, tile_rows), num_stages=2):
                row = tile * tile_rows
                count = T.min(tile_rows, rows - row)
                T.copy(P[row : row + count, start : start + c.dim], activation[:count, :])
                T.gemm(activation, weight, accum, transpose_B=True, clear_accum=True)
                # Cube stores a complete tile. The caller passes only valid rows to the consumer.
                if gate == 0:
                    T.copy(accum, GF[row : row + tile_rows, column : column + tile_cols])
                else:
                    T.copy(accum, GG[row : row + tile_rows, column : column + tile_cols])

    return main


class FusedMiddle:
    def __init__(self, baseline, *, tuned=False, reuse_weights=False):
        self.n = baseline
        c, i = baseline.c, baseline.i
        h, d = c.heads, c.dim
        # Keep the original general schedule as the six-token benchmark baseline.
        if (
            not tuned
            and c.tokens != 6
            and 1 <= c.tokens <= 16
            and 1 <= c.batch <= 80
            and (c.hidden, h, d) == (4096, 64, 128)
        ):
            tuned = True
            reuse_weights = True
        rows = c.batch * c.tokens
        verify_shape = tuned and 1 <= c.batch <= 80 and (c.hidden, h, d) == (4096, 64, 128)
        board = torch.npu.get_device_name(baseline.x.device) if verify_shape and 1 <= c.tokens <= 16 else ""
        dt_board = board == "Ascend950DT_9581"
        pr_board = board == "Ascend950PR_958b"
        fast_math = False
        self.batch_gates = False
        if verify_shape and c.tokens == 6:
            self.batch_gates = c.batch >= 5
        elif c.tokens == 8 and c.batch >= 5:
            self.batch_gates = dt_board or (pr_board and c.batch in (9, 10))
            fast_math = self.batch_gates or (pr_board and c.batch >= 11)
        elif c.tokens != 8 and (dt_board or pr_board):
            fast_math = True
            self.batch_gates = c.tokens > 8 or c.batch >= (5 if dt_board else 3) or (c.tokens == 5 and c.batch == 4)
        self.projection_rows = rows
        self.output_rows = rows
        if c.tokens in (6, 8) and c.batch in (6, 7, 8):
            self.projection_rows = max(rows, 54)
        elif verify_shape and c.tokens == 6 and c.batch in EXTRA_REQUEST_BATCHES:
            self.projection_rows += c.tokens
        elif dt_board:
            # Paired end-to-end measurements include both padding and the GEMMs.
            self.projection_rows, self.output_rows = DT_PROJECTION_ROWS.get((c.tokens, c.batch), (rows, rows))
        self.wf = i["w_f"].reshape(h // 2, 2 * d, d)
        self.wg = i["w_g"].reshape(h // 2, 2 * d, d)
        if tuned and c.batch <= 80:
            # The sliding window reads all channels of each tap contiguously.
            self.cw = i["conv_w"].permute(1, 0, 3, 2).contiguous()
            kernel = small_batch_program(
                c,
                reuse_weights=reuse_weights and c.batch in WEIGHT_REUSE_BATCHES,
                batch_gates=self.batch_gates,
                fast_math=fast_math,
                # With a lower bound of -5 the reciprocal stays finite through 16 tokens.
                precompute_keys=(dt_board or pr_board)
                and -5 <= c.lower_bound <= 0
                and ((c.tokens == 9 and c.batch >= 11) or (c.tokens == 16 and c.batch >= 32)),
            )
        else:
            self.cw = i["conv_w"].permute(1, 0, 2, 3).contiguous()
            kernel = program(c, unroll_taps=tuned and c.batch >= 41)
        if self.batch_gates:
            self.gate_kernel = tilelang.compile(gate_projection_program(c), target="ascend", execution_backend="cython")
        self.kernel = tilelang.compile(kernel, target="ascend", execution_backend="cython")

    def __call__(self):
        n = self.n
        c = n.c
        i = n.i
        rows = c.batch * c.tokens
        if self.projection_rows > rows:
            p = F.linear(F.pad(n.x, (0, 0, 0, self.projection_rows - rows)), i["w_in"])[:rows]
        else:
            p = n.project()
        gate_f, gate_g = self.wf, self.wg
        if self.batch_gates:
            padded_rows = (rows + GATE_TILE_ROWS - 1) // GATE_TILE_ROWS * GATE_TILE_ROWS
            shape = (padded_rows, c.heads * c.dim)
            gate_f, gate_g = p.new_empty(shape), p.new_empty(shape)
            self.gate_kernel(p, i["w_f"], i["w_g"], gate_f, gate_g)
            gate_f, gate_g = gate_f[:rows], gate_g[:rows]
        self.kernel(
            p,
            gate_f,
            gate_g,
            self.cw,
            n.cs,
            n.state.reshape(-1, c.dim),
            n.accepted,
            i["a_log"],
            i["dt_bias"],
            i["norm_w"],
            n.normalized.reshape(c.batch * c.tokens, -1),
            n.correction,
            n.kg,
        )
        normalized = n.normalized.reshape(rows, -1)
        if self.output_rows > rows:
            normalized = F.pad(normalized, (0, 0, 0, self.output_rows - rows))
            return F.linear(normalized, i["w_o"])[:rows]
        return F.linear(normalized, i["w_o"])
