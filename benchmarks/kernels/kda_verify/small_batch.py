# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contiguous-window KDA verify schedules for six-token BF16 requests on Ascend 950."""

import tilelang.ascend.language as T


def small_batch_program(c, *, reuse_weights=False, batch_gates=False):
    # Use the hardware FP32 SFU for the measured six-token verify shape.
    sfu_precision = "ftz_true" if (c.tokens, c.hidden, c.heads, c.dim) == (6, 4096, 64, 128) else None
    direct_history = sfu_precision is not None and c.batch == 1
    single_wave = c.batch == 1
    packed_output_gate = batch_gates
    register_prefix = c.batch != 2
    B, L, H, D = T.dynamic("B"), c.tokens, c.heads, c.dim
    gate_shape = (B * L, H * D) if batch_gates else (H // 2, 2 * D, D)
    stages = 3 if batch_gates else (6 if single_wave else 4)
    taps = T.Unroll if batch_gates else T.Serial
    PITCH = 3 * H * D + H + 2 * D

    def jobs(core):
        if single_wave:
            return T.Unroll(1)
        return T.Pipelined(T.ceildiv(B * (H // 2) - core, 32), num_stages=stages, annotations={"enable_offset": True})

    @T.prim_func
    def main(
        P: T.Tensor((B * L, PITCH), "bfloat16"),
        WF: T.Tensor(gate_shape, "bfloat16"),
        WG: T.Tensor(gate_shape, "bfloat16"),
        CW: T.Tensor((H, 3, 4, D), "bfloat16"),
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
            g0 = T.alloc_shared((L, D), "bfloat16")
            g1 = T.alloc_shared((L, D), "bfloat16")
            window = T.alloc_shared((3, L + 3, D), "bfloat16")
            cache = T.alloc_shared((3, L + 2, D), "bfloat16")
            cw = T.alloc_shared((3, 4, D), "bfloat16")
            conv = T.alloc_shared((3, L, D), "bfloat16")
            cf = T.alloc_shared((3, L, D), "float32")
            gate_weights = T.alloc_shared((L, D), "float32")
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
            qk = T.alloc_shared((16, D), "float32")
            qkl1 = T.alloc_l1((32, D), "float32")
            sl1 = T.alloc_l1((2 * D, D), "float32")
            cp = T.alloc_l0c((32, 2 * D), "float32")
            proj = T.alloc_shared((16, 2 * D), "float32")
            ck = T.alloc_shared((L, L), "float32")
            cq = T.alloc_shared((L, L), "float32")
            T.set_hf32_mode(None)
            corr = T.alloc_shared((L, D), "float32")
            kg = T.alloc_shared((L, 2 * D), "bfloat16")
            of = T.alloc_shared((L, D), "float32")
            core = T.alloc_shared((L, D), "bfloat16")
            y = T.alloc_shared((L, D), "bfloat16")
            T.copy(NW, nw)
            # Each core owns a fixed head pair across all batch items.
            if reuse_weights:
                owner = 2 * bx + sid
                if not batch_gates:
                    T.copy(WF[bx, :, :], wl1)
                    T.copy(WG[bx, :, :], vl1)
                T.copy(CW[owner, :, :, :], cw)
                T.copy(A[owner : owner + 1], alog)
                T.copy(DT[owner, :], dt)
            # Each Cube core owns one head pair in the single-wave case.
            for it in jobs(bx):
                job = it * 32 + bx
                b = it if reuse_weights else (0 if single_wave else job // (H // 2))
                hp = bx if single_wave or reuse_weights else job % (H // 2)
                h = hp * 2 + sid
                if not direct_history:
                    T.copy(Accepted[b : b + 1], ac)
                if batch_gates:
                    T.copy(WF[b * L : (b + 1) * L, h * D : (h + 1) * D], g0)
                    T.copy(WG[b * L : (b + 1) * L, h * D : (h + 1) * D], g1)
                else:
                    # Only the first L rows feed the gate outputs for this request.
                    T.copy(P[b * L : (b + 1) * L, 3 * H * D + H : 3 * H * D + H + D], al1[:L, :])
                    T.copy(P[b * L : (b + 1) * L, 3 * H * D + H + D : PITCH], bl1[:L, :])
                    if not reuse_weights:
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
                if not reuse_weights:
                    T.copy(CW[h, :, :, :], cw)
                    T.copy(A[h : h + 1], alog)
                    T.copy(DT[h, :], dt)
                T.copy(P[b * L : (b + 1) * L, 3 * H * D + h : 3 * H * D + h + 1], beta)
                for q in T.Serial(3):
                    T.copy(P[b * L : (b + 1) * L, (q * H + h) * D : (q * H + h + 1) * D], window[q, 3 : L + 3, :])
                    if direct_history:
                        # The accepted prefix selects three consecutive rows from the rollback cache.
                        start = Accepted[b] - 1
                        T.copy(CS[b + 1, start : start + 3, (q * H + h) * D : (q * H + h + 1) * D], window[q, :3, :])
                    else:
                        T.copy(CS[b + 1, :, (q * H + h) * D : (q * H + h + 1) * D], cache[q, :, :])
                if not direct_history:
                    with T.SimtVF(threads=128):
                        for q, t, j in T.Parallel(3, 3, D):
                            window[q, t, j] = cache[q, ac[0] - 1 + t, j]
                with T.SimtVF(threads=128):
                    cv = T.alloc_fragment((3, L, D), "float32")
                    for q, t, j in T.Parallel(3, L, D):
                        cv[q, t, j] = 0
                    for tap in taps(4):
                        for q, t, j in T.Parallel(3, L, D):
                            cv[q, t, j] += T.cast(window[q, t + tap, j], "float32") * T.cast(cw[q, tap, j], "float32")
                    for q, t, j in T.Parallel(3, L, D):
                        conv[q, t, j] = cv[q, t, j] / (1 + T.exp(-cv[q, t, j]))
                for q in T.Serial(3):
                    T.copy(window[q, 1 : L + 3, :], CS[b + 1, :, (q * H + h) * D : (q * H + h + 1) * D])
                with T.SimtVF(threads=128):
                    if not packed_output_gate:
                        for q, t, j in T.Parallel(3, L, D):
                            cf[q, t, j] = conv[q, t, j]
                        if batch_gates:
                            for t, j in T.Parallel(L, D):
                                gatef[t, j] = g0[t, j]
                                gatef[t, D + j] = g1[t, j]
                        else:
                            for t, j in T.Parallel(L, 2 * D):
                                gatef[t, j] = g[t, j]
                    for t, j in T.Parallel(L, D):
                        kg[t, j] = conv[1, t, j]
                        if batch_gates:
                            kg[t, D + j] = g0[t, j]
                        else:
                            kg[t, D + j] = g[t, j]
                        if not packed_output_gate:
                            val[t, j] = conv[2, t, j]
                    for t in T.Parallel(L):
                        betaf[t] = beta[t, 0]
                    if not packed_output_gate:
                        for j in T.Parallel(D):
                            nwf[j] = nw[j]
                with T.SimdVF():
                    one = T.simd.vdup(T.float32(1), "float32")
                    zero = T.simd.vdup(T.float32(0), "float32")
                    scalar_mask = T.simd.pset(32, "PAT_VL1")
                    av = T.simd.vexp(T.simd.vld(alog[0], dist="BRC_B32"), precision=sfu_precision)
                    if register_prefix:
                        running = T.simd.alloc_local((2,), "float32")
                        running[0] = one
                        running[1] = one
                        for tail in T.Serial(16 - 2 * L):
                            T.simd.vsts(qk[2 * L + tail, 0], zero)
                            T.simd.vsts(qk[2 * L + tail, 64], zero)
                    for t in T.Serial(L):
                        if register_prefix:
                            for j in T.Unroll(2):
                                if packed_output_gate:
                                    gate_input = T.simd.vcvt(T.simd.vld(g1[t, j * 64], dist="UNPK_B16"), "float32")
                                    denominator = T.simd.vadd(
                                        one, T.simd.vexp(T.simd.vneg(gate_input), precision=sfu_precision)
                                    )
                                    norm_weight = T.simd.vcvt(T.simd.vld(nw[j * 64], dist="UNPK_B16"), "float32")
                                    weight = T.simd.vdiv(norm_weight, denominator, precision=sfu_precision)
                                    T.simd.vsts(gate_weights[t, j * 64], weight)
                                    gate_f = T.simd.vcvt(T.simd.vld(g0[t, j * 64], dist="UNPK_B16"), "float32")
                                else:
                                    gate_f = T.simd.vld(gatef[t, j * 64])
                                gg = T.simd.vadd(gate_f, T.simd.vld(dt[j * 64]))
                                den = T.simd.vadd(
                                    one, T.simd.vexp(T.simd.vneg(T.simd.vmul(av, gg)), precision=sfu_precision)
                                )
                                gate = T.simd.vdiv(
                                    T.simd.vdup(T.float32(c.lower_bound), "float32"), den, precision=sfu_precision
                                )
                                decay_value = T.simd.vexp(gate, precision=sfu_precision)
                                T.simd.vsts(decay[t, j * 64], decay_value)
                                running[j] = T.simd.vmul(running[j], decay_value)
                        for q in T.Serial(2):
                            if packed_output_gate:
                                c0 = T.simd.vcvt(T.simd.vld(conv[q, t, 0], dist="UNPK_B16"), "float32")
                                c1 = T.simd.vcvt(T.simd.vld(conv[q, t, 64], dist="UNPK_B16"), "float32")
                            else:
                                c0 = T.simd.vld(cf[q, t, 0])
                                c1 = T.simd.vld(cf[q, t, 64])
                            ss = T.simd.vdupv(
                                T.simd.vadd(T.simd.vcadd(T.simd.vmul(c0, c0)), T.simd.vcadd(T.simd.vmul(c1, c1)))
                            )
                            safe = T.simd.vsel(ss, one, T.simd.vcmp(ss, zero, op="gt"))
                            inv = T.simd.vdiv(one, T.simd.vsqrt(safe, precision=sfu_precision), precision=sfu_precision)
                            scale = T.simd.vmuls(inv, T.if_then_else(q == 0, T.float32(D**-0.5), T.float32(1)))
                            T.simd.vsts(knq[q, t, 0], T.simd.vmul(c0, scale))
                            T.simd.vsts(knq[q, t, 64], T.simd.vmul(c1, scale))
                            if register_prefix:
                                T.simd.vsts(qk[(1 - q) * L + t, 0], T.simd.vmul(T.simd.vmul(c0, scale), running[0]))
                                T.simd.vsts(qk[(1 - q) * L + t, 64], T.simd.vmul(T.simd.vmul(c1, scale), running[1]))
                        if not register_prefix:
                            for j in T.Unroll(2):
                                gg = T.simd.vadd(T.simd.vld(gatef[t, j * 64]), T.simd.vld(dt[j * 64]))
                                den = T.simd.vadd(
                                    one, T.simd.vexp(T.simd.vneg(T.simd.vmul(av, gg)), precision=sfu_precision)
                                )
                                gate = T.simd.vdiv(
                                    T.simd.vdup(T.float32(c.lower_bound), "float32"), den, precision=sfu_precision
                                )
                                T.simd.vsts(decay[t, j * 64], T.simd.vexp(gate, precision=sfu_precision))
                        bv = T.simd.vld(betaf[t], dist="BRC_B32")
                        bb = T.simd.vdiv(
                            one,
                            T.simd.vadd(one, T.simd.vexp(T.simd.vneg(bv), precision=sfu_precision)),
                            precision=sfu_precision,
                        )
                        T.simd.vsts(bs[t], bb, scalar_mask, dist="ONEPT_B32", extent=1)
                if not register_prefix:
                    with T.SimtVF(threads=128):
                        for t in T.Serial(L):
                            for j in T.Parallel(D):
                                if t == 0:
                                    prefix[t, j] = decay[t, j]
                                else:
                                    prefix[t, j] = prefix[t - 1, j] * decay[t, j]
                        for t, j in T.Parallel(16, D):
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
                        if packed_output_gate:
                            c0 = T.simd.vmul(
                                T.simd.vsub(T.simd.vcvt(T.simd.vld(conv[2, t, 0], dist="UNPK_B16"), "float32"), acc[0]),
                                bet,
                            )
                            c1 = T.simd.vmul(
                                T.simd.vsub(
                                    T.simd.vcvt(T.simd.vld(conv[2, t, 64], dist="UNPK_B16"), "float32"), acc[1]
                                ),
                                bet,
                            )
                        else:
                            c0 = T.simd.vmul(T.simd.vsub(T.simd.vld(val[t, 0]), acc[0]), bet)
                            c1 = T.simd.vmul(T.simd.vsub(T.simd.vld(val[t, 64]), acc[1]), bet)
                        T.simd.vsts(corr[t, 0], c0)
                        T.simd.vsts(corr[t, 64], c1)
                        selfq = T.simd.vld(cq[t, t], dist="BRC_B32")
                        if register_prefix:
                            out0 = T.simd.vadd(acc[2], T.simd.vmul(c0, selfq))
                            out1 = T.simd.vadd(acc[3], T.simd.vmul(c1, selfq))
                            # Match the BF16 core-output boundary before RMS normalization.
                            rr0 = T.simd.vcvt(T.simd.vcvt(out0, "bfloat16"), "float32")
                            rr1 = T.simd.vcvt(T.simd.vcvt(out1, "bfloat16"), "float32")
                            ss = T.simd.vdupv(
                                T.simd.vadd(T.simd.vcadd(T.simd.vmul(rr0, rr0)), T.simd.vcadd(T.simd.vmul(rr1, rr1)))
                            )
                            onef = T.simd.vdup(T.float32(1), "float32")
                            inv = T.simd.vdiv(
                                onef,
                                T.simd.vsqrt(
                                    T.simd.vadds(T.simd.vmuls(ss, T.float32(1 / D)), T.float32(c.eps)),
                                    precision=sfu_precision,
                                ),
                                precision=sfu_precision,
                            )
                            if packed_output_gate:
                                yy0 = T.simd.vmul(T.simd.vmul(rr0, inv), T.simd.vld(gate_weights[t, 0]))
                                yy1 = T.simd.vmul(T.simd.vmul(rr1, inv), T.simd.vld(gate_weights[t, 64]))
                                T.simd.vsts(y[t, 0], T.simd.vcvt(yy0, "bfloat16"), dist="PK_B32")
                                T.simd.vsts(y[t, 64], T.simd.vcvt(yy1, "bfloat16"), dist="PK_B32")
                            else:
                                gw0 = T.simd.vld(gatef[t, D])
                                gw1 = T.simd.vld(gatef[t, D + 64])
                                yy0 = T.simd.vdiv(
                                    T.simd.vmul(T.simd.vmul(rr0, inv), T.simd.vld(nwf[0])),
                                    T.simd.vadd(onef, T.simd.vexp(T.simd.vneg(gw0), precision=sfu_precision)),
                                    precision=sfu_precision,
                                )
                                yy1 = T.simd.vdiv(
                                    T.simd.vmul(T.simd.vmul(rr1, inv), T.simd.vld(nwf[64])),
                                    T.simd.vadd(onef, T.simd.vexp(T.simd.vneg(gw1), precision=sfu_precision)),
                                    precision=sfu_precision,
                                )
                                T.simd.vsts(yf[t, 0], yy0)
                                T.simd.vsts(yf[t, 64], yy1)
                        else:
                            T.simd.vsts(of[t, 0], T.simd.vadd(acc[2], T.simd.vmul(c0, selfq)))
                            T.simd.vsts(of[t, 64], T.simd.vadd(acc[3], T.simd.vmul(c1, selfq)))
                        T.simd.mem_bar("VST_VLD")
                if not register_prefix:
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
                                one,
                                T.simd.vsqrt(
                                    T.simd.vadds(T.simd.vmuls(ss, T.float32(1 / D)), T.float32(c.eps)),
                                    precision=sfu_precision,
                                ),
                                precision=sfu_precision,
                            )
                            for j in T.Unroll(2):
                                cc = T.simd.vld(coref[t, j * 64])
                                gw = T.simd.vld(gatef[t, D + j * 64])
                                den = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(gw), precision=sfu_precision))
                                yy = T.simd.vdiv(
                                    T.simd.vmul(T.simd.vmul(cc, inv), T.simd.vld(nwf[j * 64])),
                                    den,
                                    precision=sfu_precision,
                                )
                                T.simd.vsts(yf[t, j * 64], yy)
                if not packed_output_gate:
                    with T.SimtVF(threads=128):
                        for t, v in T.Parallel(L, D):
                            y[t, v] = yf[t, v]
                T.copy(y, Y[b * L : (b + 1) * L, h * D : (h + 1) * D])
                T.copy(corr, Correction[b + 1, h, :, :])
                T.copy(kg, KG[b + 1, h, :, :])

    return main
