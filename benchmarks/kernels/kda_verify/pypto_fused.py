# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped-head PyPTO implementation of the six-token factorized recurrence."""

import pypto as p
import torch
import torch.nn.functional as F

BF = p.DT_BF16
FP = p.DT_FP32
N = p.DYNAMIC
PITCH = 24896
HD = 8192
D = 128
G = 16
W = G * D


@p.frontend.jit
def verify_middle(
    proj: p.Tensor([N, PITCH], BF),
    wf: p.Tensor([HD, D], BF),
    wg: p.Tensor([HD, D], BF),
    hist: p.Tensor([N, 3 * HD], BF),
    cw: p.Tensor([4, 3 * HD], BF),
    checkpoint: p.Tensor([N, D], FP),
    alog: p.Tensor([64, 1], FP),
    bias: p.Tensor([64, D], FP),
    weight: p.Tensor([1, D], BF),
    cache: p.Tensor([N, 3 * HD], BF),
    correction: p.Tensor([N, D], FP),
    kg: p.Tensor([N, 2 * D], BF),
    norm: p.Tensor([N, HD], BF),
):
    p.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
    # Each work item handles one contiguous group of heads from one request.
    for job in p.loop(proj.shape[0] // 6 * (64 // G), name="batch_head_group"):
        b = job // (64 // G)
        h = (job % (64 // G)) * G
        slot = job * G + 64
        p.set_vec_tile_shapes(8, 128)
        f = p.cast(
            p.matmul(p.view(proj, [6, D], [b * 6, 3 * HD + 64]), p.view(wf, [W, D], [h * D, 0]), FP, b_trans=True), BF
        )
        g = p.cast(
            p.matmul(p.view(proj, [6, D], [b * 6, 3 * HD + 64 + D]), p.view(wg, [W, D], [h * D, 0]), FP, b_trans=True),
            BF,
        )
        conv = p.Tensor([18, W], BF)
        for part in range(3):
            col = part * HD + h * D
            window = p.concat([p.view(hist, [3, W], [3 * b, col]), p.view(proj, [6, W], [6 * b, col])], 0)
            z = p.cast(p.view(window, [6, W], [0, 0]), FP) * p.cast(p.view(cw, [1, W], [0, col]), FP)
            for tap in range(1, 4):
                z = z + p.cast(p.view(window, [6, W], [tap, 0]), FP) * p.cast(p.view(cw, [1, W], [tap, col]), FP)
            # Match the reference SiLU instruction before BF16 rounding.
            p.assemble(p.cast(p.div(z, 1.0 + p.exp(-z), p.PrecisionType.INTRINSIC), BF), [part * 6, 0], conv)
            p.assemble(p.view(window, [8, W], [1, 0]), [(b + 1) * 8, col], cache)
        beta2 = p.transpose(p.sigmoid(p.cast(p.view(proj, [6, G], [6 * b, 3 * HD + h]), FP)), 0, 1)
        p.set_vec_tile_shapes(1, 8, 128)
        gate = p.transpose(p.reshape(p.cast(f, FP), [6, G, D]), 0, 1)
        second = p.transpose(p.reshape(p.cast(g, FP), [6, G, D]), 0, 1)
        q0 = p.transpose(p.reshape(p.cast(p.view(conv, [6, W], [0, 0]), FP), [6, G, D]), 0, 1)
        k0 = p.transpose(p.reshape(p.cast(p.view(conv, [6, W], [6, 0]), FP), [6, G, D]), 0, 1)
        v = p.transpose(p.reshape(p.cast(p.view(conv, [6, W], [12, 0]), FP), [6, G, D]), 0, 1)
        ks = p.sum(k0 * k0, -1, True)
        qs = p.sum(q0 * q0, -1, True)
        k = k0 * p.rsqrt(p.where(ks > 0.0, ks, 1.0))
        q = q0 * p.rsqrt(p.where(qs > 0.0, qs, 1.0)) * (D**-0.5)
        av = p.reshape(p.exp(p.view(alog, [G, 1], [h, 0])), [G, 1, 1])
        df = p.reshape(p.view(bias, [G, D], [h, 0]), [G, 1, D])
        decay = p.exp(p.sigmoid((gate + df) * av) * -5.0)
        beta = p.reshape(beta2, [G, 6, 1])
        running = p.view(decay, [G, 1, D], [0, 0, 0])
        prefix = running
        for t in range(1, 6):
            running = running * p.view(decay, [G, 1, D], [0, t, 0])
            prefix = p.concat([prefix, running], 1)
        stacked = p.concat([k * prefix, q * prefix], 1)
        state = p.reshape(p.view(checkpoint, [G * D, D], [slot * D, 0]), [G, D, D])
        projected = p.matmul(stacked, state, FP, b_trans=True)
        # Earlier key contributions use the ratio of the two decay prefixes.
        prior = k / prefix
        # Row zero seeds the SSA accumulator. New corrections start at one.
        deltas = p.view(v, [G, 1, D], [0, 0, 0])
        nw = p.reshape(p.cast(weight, FP), [1, 1, D])
        outputs = p.view(v, [G, 1, D], [0, 0, 0])
        for t in range(6):
            prediction = p.view(projected, [G, 1, D], [0, t, 0])
            accumulated = p.view(projected, [G, 1, D], [0, 6 + t, 0])
            ck = p.sum(prior * p.view(stacked, [G, 1, D], [0, t, 0]), -1, True)
            cq = p.sum(prior * p.view(stacked, [G, 1, D], [0, 6 + t, 0]), -1, True)
            for u in range(t):
                old = p.view(deltas, [G, 1, D], [0, u + 1, 0])
                prediction = prediction + old * p.view(ck, [G, 1, 1], [0, u, 0])
                accumulated = accumulated + old * p.view(cq, [G, 1, 1], [0, u, 0])
            delta = (p.view(v, [G, 1, D], [0, t, 0]) - prediction) * p.view(beta, [G, 1, 1], [0, t, 0])
            deltas = p.concat([deltas, delta], 1)
            sq = p.sum(p.view(k, [G, 1, D], [0, t, 0]) * p.view(q, [G, 1, D], [0, t, 0]), -1, True)
            core = p.cast(p.cast(accumulated + delta * sq, BF), FP)
            inv = p.rsqrt(p.sum(core * core, -1, True) * (1.0 / D) + 1e-5)
            out = core * inv * nw * p.sigmoid(p.view(second, [G, 1, D], [0, t, 0]))
            outputs = p.concat([outputs, out], 1)
        report = p.reshape(p.view(deltas, [G, 6, D], [0, 1, 0]), [G * 6, D])
        record = p.reshape(p.cast(p.concat([k0, gate], 2), BF), [G * 6, 2 * D])
        result = p.reshape(p.cast(p.transpose(p.view(outputs, [G, 6, D], [0, 1, 0]), 0, 1), BF), [6, W])
        p.set_vec_tile_shapes(8, 128)
        p.assemble(report, [slot * 6, 0], correction)
        p.assemble(record, [slot * 6, 0], kg)
        p.assemble(result, [6 * b, h * D], norm)


class PyPTOMiddle:
    def __init__(self, native):
        self.n = native
        self.slot = native.indices.long()[:, None]
        self.tail = torch.arange(3, device=native.x.device)[None, :]

    def __call__(self):
        n = self.n
        b = n.c.batch
        i = n.i
        if b in (6, 7, 8):
            proj = F.linear(F.pad(n.x, (0, 0, 0, 54 - 6 * b)), i["w_in"])[: 6 * b]
        else:
            proj = n.project()
        hist = n.cs[self.slot, n.accepted.long()[:, None] - 1 + self.tail].reshape(-1, 3 * HD)
        verify_middle(
            proj,
            i["w_f"],
            i["w_g"],
            hist,
            n.cw,
            n.state.view(-1, D),
            i["a_log"].view(64, 1),
            i["dt_bias"],
            i["norm_w"].view(1, D),
            n.cs.view(-1, 3 * HD),
            n.correction.view(-1, D),
            n.kg.view(-1, 2 * D),
            n.normalized.view(-1, HD),
        )
        return F.linear(n.normalized.view(-1, HD), i["w_o"])
