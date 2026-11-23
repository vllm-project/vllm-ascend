# SPDX-License-Identifier: Apache-2.0
"""Single-launch BF16 KDA verify for six-token GLM-5.3 requests on Ascend."""
import tilelang.ascend.language as T


def whole_program(c, page_layout):
    B, L, I, H, D = T.dynamic("B"), 6, 4096, 64, 128
    PITCH = 3 * H * D + H + 2 * D
    state_rows = 16
    gate_shape = (H // 2, 2 * D, D)
    conv_stride, conv_row, state_stride, corr_stride, kg_stride = page_layout
    cs_shape = (T.dynamic("conv_rows"), D)
    state_shape = (T.dynamic("state_rows"), D)
    corr_shape = (T.dynamic("correction_rows"), D)
    kg_shape = (T.dynamic("kg_rows"), 2 * D)
    metadata_shape = (3, B)
    sfu_precision = "ftz_true"
    M, K, N, CORES = 64, 512, 64, 32
    taps = T.Serial

    @T.prim_func
    def main(
        X: T.Tensor((B * L, I), "bfloat16"),
        Wi: T.Tensor((PITCH, I), "bfloat16"),
        Wo: T.Tensor((I, H * D), "bfloat16"),
        Out: T.Tensor((T.ceildiv(B * L, M) * M, I), "bfloat16"),
        P: T.Tensor((T.ceildiv(B * L, M) * M, PITCH), "bfloat16"),
        WF: T.Tensor(gate_shape, "bfloat16"),
        WG: T.Tensor(gate_shape, "bfloat16"),
        CW: T.Tensor((H, 3, 4, D), "bfloat16"),
        CS: T.Tensor(cs_shape, "bfloat16"),
        State: T.Tensor(state_shape, "float32"),
        Accepted: T.Tensor(metadata_shape, "int32"),
        A: T.Tensor((H,), "float32"),
        DT: T.Tensor((H, D), "float32"),
        NW: T.Tensor((D,), "bfloat16"),
        Y: T.Tensor((B * L, H * D), "bfloat16"),
        Correction: T.Tensor(corr_shape, "float32"),
        KG: T.Tensor(kg_shape, "bfloat16"),
    ):
        with T.MixedKernel(CORES, sids=2) as (bx, sid):
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
            window = T.alloc_shared((3, L + 3, D), "bfloat16")
            cw = T.alloc_shared((3, 4, D), "bfloat16")
            conv = T.alloc_shared((3, L, D), "bfloat16")
            cf = T.alloc_shared((3, L, D), "float32")
            gatef = T.alloc_shared((L, 2 * D), "float32")
            betaf = T.alloc_shared((L,), "float32")
            nwf = T.alloc_shared((D,), "float32")
            yf = T.alloc_shared((L, D), "float32")
            beta = T.alloc_shared((L, 1), "bfloat16")
            bs = T.alloc_shared((L,), "float32")
            alog = T.alloc_shared((1,), "float32")
            dt = T.alloc_shared((D,), "float32")
            nw = T.alloc_shared((D,), "bfloat16")
            knq = T.alloc_shared((2, L, D), "float32")
            decay = T.alloc_shared((L, D), "float32")
            val = T.alloc_shared((L, D), "float32")
            qk = T.alloc_shared((state_rows, D), "float32")
            qkl1 = T.alloc_l1((2 * state_rows, D), "float32")
            sl1 = T.alloc_l1((2 * D, D), "float32")
            cp = T.alloc_l0c((2 * state_rows, 2 * D), "float32")
            proj = T.alloc_shared((state_rows, 2 * D), "float32")
            T.set_hf32_mode(None)
            corr = T.alloc_shared((L, D), "float32")
            kg = T.alloc_shared((L, 2 * D), "bfloat16")
            y = T.alloc_shared((L, D), "bfloat16")
            # Each mixed core owns a pair of KDA heads across all requests.
            T.copy(NW, nw)
            ax = T.alloc_l1((M, K), "bfloat16")
            bw = T.alloc_l1((N, K), "bfloat16")
            accum = T.alloc_l0c((M, N), "float32")
            for it in T.Serial(T.ceildiv(T.ceildiv(B * L, M) * (PITCH // N) - bx, CORES)):
                task = it * CORES + bx
                row = task // (PITCH // N) * M
                col = task % (PITCH // N) * N
                rows = T.min(M, B * L - row)
                for kk in T.Serial(I // K):
                    T.copy(X[row:row + rows, kk * K:(kk + 1) * K], ax[:rows, :])
                    T.copy(Wi[col:col + N, kk * K:(kk + 1) * K], bw)
                    T.gemm(ax, bw, accum, transpose_B=True, clear_accum=(kk == 0))
                T.copy(accum, P[row:row + M, col:col + N])
            # Publish all projected columns before the two Vector lanes read their heads.
            # A read on each Vector MTE2 makes the scheduler wait for its
            # paired Cube to publish P before any Vector enters the barrier.
            T.copy(P[0:1, bx * N:(bx + 1) * N], g0[:1, :N])
            T.copy(g0[:1, :N], Y[0:1, (2 * bx + sid) * D:(2 * bx + sid) * D + N])
            T.ascend_sync_inter_arrive("PIPE_MTE3", 9)
            T.ascend_sync_inter_wait("PIPE_MTE2", 9)
            T.ascend_sync_inter_arrive("PIPE_FIX", 12)
            T.ascend_sync_inter_wait("PIPE_MTE2", 12)
            for b in T.Serial(B):
                hp = bx
                h = hp * 2 + sid
                # Expand both low-rank gate projections for this pair of heads.
                T.copy(P[b * L:(b + 1) * L, 3 * H * D + H:3 * H * D + H + D], al1[:L, :])
                T.copy(P[b * L:(b + 1) * L, 3 * H * D + H + D:PITCH], bl1[:L, :])
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
                T.copy(A[h:h + 1], alog)
                T.copy(DT[h, :], dt)
                T.copy(P[b * L:(b + 1) * L, 3 * H * D + h:3 * H * D + h + 1], beta)
                # Reconstruct the convolution window from accepted history and Q/K/V.
                for q in T.Serial(3):
                    T.copy(P[b * L:(b + 1) * L, (q * H + h) * D:(q * H + h + 1) * D], window[q, 3:L + 3, :])
                    for tap in T.Unroll(3):
                        offset = Accepted[0, b] * conv_stride + (Accepted[2, b] - 1 + tap) * conv_row + q * H + h
                        T.copy(CS[offset, :], window[q, tap, :])
                with T.SimtVF(threads=128):
                    cv = T.alloc_fragment((3, L, D), 'float32')
                    for q, t, j in T.Parallel(3, L, D):
                        cv[q, t, j] = 0
                    for tap in taps(4):
                        for q, t, j in T.Parallel(3, L, D):
                            cv[q, t, j] += T.cast(window[q, t + tap, j], 'float32') * T.cast(cw[q, tap, j], 'float32')
                    for q, t, j in T.Parallel(3, L, D):
                        conv[q, t, j] = cv[q, t, j] / (1 + T.exp(-cv[q, t, j]))
                for q in T.Serial(3):
                    for row in T.Unroll(L + 2):
                        offset = Accepted[0, b] * conv_stride + row * conv_row + q * H + h
                        T.copy(window[q, row + 1, :], CS[offset, :])
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
                    one = T.simd.vdup(T.float32(1), 'float32')
                    zero = T.simd.vdup(T.float32(0), 'float32')
                    scalar_mask = T.simd.pset(32, 'PAT_VL1')
                    av = T.simd.vexp(T.simd.vld(alog[0], dist='BRC_B32'), precision=sfu_precision)
                    running = T.simd.alloc_local((2,), 'float32')
                    running[0] = one
                    running[1] = one
                    for tail in T.Serial(state_rows - 2 * L):
                        T.simd.vsts(qk[2 * L + tail, 0], zero)
                        T.simd.vsts(qk[2 * L + tail, 64], zero)
                    for t in T.Serial(L):
                        for j in T.Unroll(2):
                            gate_f = T.simd.vld(gatef[t, j * 64])
                            gg = T.simd.vadd(gate_f, T.simd.vld(dt[j * 64]))
                            den = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(T.simd.vmul(av, gg)), precision=sfu_precision))
                            gate = T.simd.vdiv(T.simd.vdup(T.float32(c.lower_bound), 'float32'), den, precision=sfu_precision)
                            decay_value = T.simd.vexp(gate, precision=sfu_precision)
                            T.simd.vsts(decay[t, j * 64], decay_value)
                            running[j] = T.simd.vmul(running[j], decay_value)
                        for q in T.Serial(2):
                            c0 = T.simd.vld(cf[q, t, 0])
                            c1 = T.simd.vld(cf[q, t, 64])
                            ss = T.simd.vdupv(T.simd.vadd(T.simd.vcadd(T.simd.vmul(c0, c0)), T.simd.vcadd(T.simd.vmul(c1, c1))))
                            safe = T.simd.vsel(ss, one, T.simd.vcmp(ss, zero, op='gt'))
                            inv = T.simd.vdiv(one, T.simd.vsqrt(safe, precision=sfu_precision), precision=sfu_precision)
                            scale = T.simd.vmuls(inv, T.if_then_else(q == 0, T.float32(D ** (-0.5)), T.float32(1)))
                            T.simd.vsts(knq[q, t, 0], T.simd.vmul(c0, scale))
                            T.simd.vsts(knq[q, t, 64], T.simd.vmul(c1, scale))
                            T.simd.vsts(qk[(1 - q) * L + t, 0], T.simd.vmul(T.simd.vmul(c0, scale), running[0]))
                            T.simd.vsts(qk[(1 - q) * L + t, 64], T.simd.vmul(T.simd.vmul(c1, scale), running[1]))
                        bv = T.simd.vld(betaf[t], dist='BRC_B32')
                        bb = T.simd.vdiv(one, T.simd.vadd(one, T.simd.vexp(T.simd.vneg(bv), precision=sfu_precision)), precision=sfu_precision)
                        T.simd.vsts(bs[t], bb, scalar_mask, dist='ONEPT_B32', extent=1)
                # Multiply the decay-weighted keys and queries by the checkpoint.
                T.dual_copy(qk, qkl1)
                start_row = Accepted[1, b] * state_stride + hp * 2 * D
                T.copy(State[start_row:start_row + 2 * D, :], sl1)
                T.gemm(qkl1, sl1, cp, transpose_B=True, clear_accum=True)
                T.dual_copy(cp, proj)
                # Apply token-ordered delta corrections, then gated RMSNorm.
                with T.SimdVF():
                    ratio = T.simd.alloc_local((2,), 'float32')
                    acc = T.simd.alloc_local((4,), 'float32')
                    for t in T.Serial(L):
                        qt0 = T.simd.vld(knq[0, t, 0])
                        qt1 = T.simd.vld(knq[0, t, 64])
                        kt0 = T.simd.vld(knq[1, t, 0])
                        kt1 = T.simd.vld(knq[1, t, 64])
                        selfq = T.simd.vadd(T.simd.vcadd(T.simd.vmul(kt0, qt0)), T.simd.vcadd(T.simd.vmul(kt1, qt1)))
                        acc[0] = T.simd.vld(proj[t, sid * D])
                        acc[1] = T.simd.vld(proj[t, sid * D + 64])
                        acc[2] = T.simd.vld(proj[t + L, sid * D])
                        acc[3] = T.simd.vld(proj[t + L, sid * D + 64])
                        ratio[0] = T.simd.vdup(T.float32(1), 'float32')
                        ratio[1] = T.simd.vdup(T.float32(1), 'float32')
                        for back in T.Serial(t):
                            u = t - 1 - back
                            ratio[0] = T.simd.vmul(ratio[0], T.simd.vld(decay[u + 1, 0]))
                            ratio[1] = T.simd.vmul(ratio[1], T.simd.vld(decay[u + 1, 64]))
                            k0 = T.simd.vmul(T.simd.vld(knq[1, u, 0]), ratio[0])
                            k1 = T.simd.vmul(T.simd.vld(knq[1, u, 64]), ratio[1])
                            kk = T.simd.vadd(T.simd.vcadd(T.simd.vmul(k0, kt0)), T.simd.vcadd(T.simd.vmul(k1, kt1)))
                            qq = T.simd.vadd(T.simd.vcadd(T.simd.vmul(k0, qt0)), T.simd.vcadd(T.simd.vmul(k1, qt1)))
                            c0 = T.simd.vld(corr[u, 0])
                            c1 = T.simd.vld(corr[u, 64])
                            k = T.simd.vdupv(kk)
                            q = T.simd.vdupv(qq)
                            T.simd.vmula(acc[0], c0, k)
                            T.simd.vmula(acc[1], c1, k)
                            T.simd.vmula(acc[2], c0, q)
                            T.simd.vmula(acc[3], c1, q)
                        bet = T.simd.vld(bs[t], dist='BRC_B32')
                        c0 = T.simd.vmul(T.simd.vsub(T.simd.vld(val[t, 0]), acc[0]), bet)
                        c1 = T.simd.vmul(T.simd.vsub(T.simd.vld(val[t, 64]), acc[1]), bet)
                        T.simd.vsts(corr[t, 0], c0)
                        T.simd.vsts(corr[t, 64], c1)
                        selfq = T.simd.vdupv(selfq)
                        out0 = T.simd.vadd(acc[2], T.simd.vmul(c0, selfq))
                        out1 = T.simd.vadd(acc[3], T.simd.vmul(c1, selfq))
                        rr0 = T.simd.vcvt(T.simd.vcvt(out0, 'bfloat16'), 'float32')
                        rr1 = T.simd.vcvt(T.simd.vcvt(out1, 'bfloat16'), 'float32')
                        ss = T.simd.vdupv(T.simd.vadd(T.simd.vcadd(T.simd.vmul(rr0, rr0)), T.simd.vcadd(T.simd.vmul(rr1, rr1))))
                        onef = T.simd.vdup(T.float32(1), 'float32')
                        inv = T.simd.vdiv(onef, T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(ss, T.float32(1 / D)), T.float32(c.eps)), precision=sfu_precision), precision=sfu_precision)
                        gw0 = T.simd.vld(gatef[t, D])
                        gw1 = T.simd.vld(gatef[t, D + 64])
                        yy0 = T.simd.vdiv(T.simd.vmul(T.simd.vmul(rr0, inv), T.simd.vld(nwf[0])), T.simd.vadd(onef, T.simd.vexp(T.simd.vneg(gw0), precision=sfu_precision)), precision=sfu_precision)
                        yy1 = T.simd.vdiv(T.simd.vmul(T.simd.vmul(rr1, inv), T.simd.vld(nwf[64])), T.simd.vadd(onef, T.simd.vexp(T.simd.vneg(gw1), precision=sfu_precision)), precision=sfu_precision)
                        T.simd.vsts(yf[t, 0], yy0)
                        T.simd.vsts(yf[t, 64], yy1)
                        T.simd.mem_bar('VST_VLD')
                with T.SimtVF(threads=128):
                    for t, v in T.Parallel(L, D):
                        y[t, v] = yf[t, v]
                T.copy(y, Y[b * L:(b + 1) * L, h * D:(h + 1) * D])
                cr = Accepted[1, b] * corr_stride + h * L
                kr = Accepted[1, b] * kg_stride + h * L
                T.copy(corr, Correction[cr:cr + L, :])
                T.copy(kg, KG[kr:kr + L, :])
            # The output projection reduces over every head.
            # Force the paired Vector-to-Cube handoff before the Cube barrier.
            T.copy(Y[0:1, bx * 2 * D:(bx * 2 + 1) * D], al1[:1, :])
            T.gemm(al1, wl1, ga, transpose_B=True, clear_accum=True)
            T.copy(ga[:1, :16], Out[0:1, bx * 16:(bx + 1) * 16])
            T.ascend_sync_inter_arrive("PIPE_FIX", 14)
            T.ascend_sync_inter_wait("PIPE_MTE2", 14)
            T.ascend_sync_inter_arrive("PIPE_MTE3", 13)
            T.ascend_sync_inter_wait("PIPE_MTE3", 13)
            for it in T.Serial(T.ceildiv(T.ceildiv(B * L, M) * (I // N) - bx, CORES)):
                task = it * CORES + bx
                row = task // (I // N) * M
                col = task % (I // N) * N
                rows = T.min(M, B * L - row)
                for kk in T.Serial((H * D) // K):
                    T.copy(Y[row:row + rows, kk * K:(kk + 1) * K], ax[:rows, :])
                    T.copy(Wo[col:col + N, kk * K:(kk + 1) * K], bw)
                    T.gemm(ax, bw, accum, transpose_B=True, clear_accum=(kk == 0))
                T.copy(accum, Out[row:row + M, col:col + N])
    return main

class WholeKDA:
    """Bind BF16 verify weights and paged ReplaySSM storage for a single layer.

    metadata contains conv page IDs, checkpoint page IDs, and accepted lengths.
    Prepare metadata with the scheduler before calling the layer.
    """

    def __init__(self, layer):
        import tilelang
        import torch
        import torch_npu
        from types import SimpleNamespace

        conv, state, corr, kg = layer.kv_cache
        if layer._conv_state_dim_first:
            conv = conv.transpose(-1, -2)
        if conv.stride(2) != 1 or any(not t[0].is_contiguous() for t in (state, corr, kg)):
            raise ValueError("Unsupported ReplaySSM page layout")
        strides = (conv.stride(0), conv.stride(1), state.stride(0), corr.stride(0), kg.stride(0))
        widths = (128, 128, 128, 128, 256)
        if any(stride % width for stride, width in zip(strides, widths)):
            raise ValueError("Cache strides must be multiples of the corresponding head width")
        layout = tuple(stride // width for stride, width in zip(strides, widths))
        config = SimpleNamespace(eps=layer.o_norm.eps, lower_bound=layer.kda_lower_bound)
        self.kernel = tilelang.compile(whole_program(config, layout), target="ascend", execution_backend="cython")

        def flat(t, width):
            size = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
            if size % width:
                raise ValueError("Cache ends in an incomplete head row")
            return t.as_strided((size // width, width), (width, 1))

        def weight(module):
            return torch_npu.npu_format_cast(module.weight, 2).contiguous()

        self.win = weight(layer.in_proj_qkvbfg_a)
        self.wo = weight(layer.o_proj)
        self.wf = weight(layer.f_b_proj).reshape(32, 256, 128)
        self.wg = weight(layer.g_b_proj).reshape(32, 256, 128)
        self.cw = torch.stack(
            [m.weight.reshape(64, 128, 4) for m in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)], dim=1
        ).permute(0, 1, 3, 2).bfloat16().contiguous()
        self.a = layer.A_log.reshape(64).float().contiguous()
        self.dt = layer.dt_bias.reshape(64, 128).float().contiguous()
        self.nw = layer.o_norm.weight.reshape(128).bfloat16().contiguous()
        self.conv, self.state, self.corr, self.kg = [
            flat(t, w) for t, w in zip((conv, state, corr, kg), (128, 128, 128, 256))
        ]

    @staticmethod
    def allocate(x):
        rows = x.shape[0]
        padded = (rows + 63) // 64 * 64
        return x.new_empty((padded, 24896)), x.new_empty((rows, 8192)), x.new_empty((padded, 4096))

    def __call__(self, x, metadata, workspace=None):
        import torch
        rows = x.shape[0]
        if x.shape != (rows, 4096) or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("Expected contiguous BF16 hidden states with width 4096")
        if rows == 0 or rows % 6 or metadata.shape != (3, rows // 6):
            raise ValueError("Expected six tokens per request and three metadata fields")
        if metadata.dtype != torch.int32 or not metadata.is_contiguous():
            raise ValueError("Expected contiguous int32 metadata")
        p, y, out = self.allocate(x) if workspace is None else workspace
        self.kernel(x, self.win, self.wo, out, p, self.wf, self.wg, self.cw, self.conv,
                    self.state, metadata, self.a, self.dt, self.nw, y, self.corr, self.kg)
        return out[:rows]
