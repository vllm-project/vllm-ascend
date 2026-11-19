# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyPTO six-token verification with four independent heads per work item."""

import pypto as p
import torch
import torch.nn.functional as F

BF = p.DT_BF16
FP = p.DT_FP32
N = p.DYNAMIC
PITCH = 24896
HD = 8192
D = 128


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
    p.set_vec_tile_shapes(16, D)
    p.set_cube_tile_shapes([16, 16], [128, 128], [128, 128])
    for bundle in p.loop(proj.shape[0] // 6 * 16, name="batch_heads"):
        for lane in range(4):
            job = bundle * 4 + lane
            b = job // 64
            h = job % 64
            f = p.cast(
                p.cast(
                    p.matmul(
                        p.view(proj, [6, D], [b * 6, 3 * HD + 64]), p.view(wf, [D, D], [h * D, 0]), FP, b_trans=True
                    ),
                    BF,
                ),
                FP,
            )
            g = p.cast(
                p.cast(
                    p.matmul(
                        p.view(proj, [6, D], [b * 6, 3 * HD + 64 + D]), p.view(wg, [D, D], [h * D, 0]), FP, b_trans=True
                    ),
                    BF,
                ),
                FP,
            )
            conv = p.Tensor([18, D], BF)
            for part in range(3):
                col = part * HD + h * D
                history = p.view(hist, [3, D], [3 * b, col])
                raw = p.view(proj, [6, D], [6 * b, col])
                window = p.concat([history, raw], 0)
                z = p.cast(p.view(window, [6, D], [0, 0]), FP) * p.cast(p.view(cw, [1, D], [0, col]), FP)
                for tap in range(1, 4):
                    z = z + p.cast(p.view(window, [6, D], [tap, 0]), FP) * p.cast(p.view(cw, [1, D], [tap, col]), FP)
                p.assemble(p.cast(p.div(z, 1.0 + p.exp(-z), p.PrecisionType.INTRINSIC), BF), [part * 6, 0], conv)
                p.assemble(p.view(window, [8, D], [1, 0]), [(b + 1) * 8, col], cache)
            beta = p.sigmoid(p.cast(p.view(proj, [6, 1], [6 * b, 3 * HD + h]), FP))
            v = p.cast(p.view(conv, [6, D], [12, 0]), FP)
            q_raw = p.cast(p.view(conv, [6, D], [0, 0]), FP)
            k_raw = p.cast(p.view(conv, [6, D], [6, 0]), FP)
            q_ss = p.sum(q_raw * q_raw, -1, True)
            k_ss = p.sum(k_raw * k_raw, -1, True)
            q = q_raw * p.rsqrt(p.where(q_ss > 0.0, q_ss, 1.0)) * (D**-0.5)
            k = k_raw * p.rsqrt(p.where(k_ss > 0.0, k_ss, 1.0))
            a = p.exp(p.view(alog, [1, 1], [h, 0]))
            decay = p.exp(p.sigmoid((f + p.view(bias, [1, D], [h, 0])) * a) * -5.0)
            s = p.view(checkpoint, [D, D], [(job + 64) * D, 0])
            p.assemble(p.view(conv, [6, D], [6, 0]), [(job + 64) * 6, 0], kg)
            p.assemble(p.cast(f, BF), [(job + 64) * 6, D], kg)
            prefix = p.Tensor([6, D], FP)
            running = p.view(decay, [1, D], [0, 0])
            p.assemble(running, [0, 0], prefix)
            for t in range(1, 6):
                running = running * p.view(decay, [1, D], [t, 0])
                p.assemble(running, [t, 0], prefix)
            scaled = p.concat([k * prefix, q * prefix], 0)
            projected = p.matmul(scaled, s, FP, b_trans=True)
            # Normalize earlier keys by their own prefixes. The ratio for any
            # valid (u,t) is exactly the product of decays from u+1 through t.
            past = k / prefix
            deltas = p.view(v, [1, D], [0, 0])
            for t in range(6):
                prediction = p.view(projected, [1, D], [t, 0])
                accumulated = p.view(projected, [1, D], [6 + t, 0])
                coeff_k = p.sum(past * p.view(scaled, [1, D], [t, 0]), -1, True)
                coeff_q = p.sum(past * p.view(scaled, [1, D], [6 + t, 0]), -1, True)
                for u in range(t):
                    old = p.view(deltas, [1, D], [u + 1, 0])
                    prediction = prediction + old * p.view(coeff_k, [1, 1], [u, 0])
                    accumulated = accumulated + old * p.view(coeff_q, [1, 1], [u, 0])
                delta = (p.view(v, [1, D], [t, 0]) - prediction) * p.view(beta, [1, 1], [t, 0])
                deltas = p.concat([deltas, delta], 0)
                self_dot = p.sum(p.view(k, [1, D], [t, 0]) * p.view(q, [1, D], [t, 0]), -1, True)
                core = p.cast(p.cast(accumulated + delta * self_dot, BF), FP)
                inv = p.rsqrt(p.sum(core * core, -1, True) * (1.0 / D) + 1e-5)
                y = p.cast(core * inv * p.cast(weight, FP) * p.sigmoid(p.view(g, [1, D], [t, 0])), BF)
                p.assemble(y, [b * 6 + t, h * D], norm)
                p.assemble(delta, [(job + 64) * 6 + t, 0], correction)


class PyPTOMiddle:
    def __init__(self, native):
        self.n = native
        self.slot = native.indices.long()[:, None]
        self.tail = torch.arange(3, device=native.x.device)[None, :]

    def __call__(self):
        n = self.n
        b = n.c.batch
        inputs = n.i
        if b in (6, 7, 8):
            proj = F.linear(F.pad(n.x, (0, 0, 0, 54 - 6 * b)), inputs["w_in"])[: 6 * b]
        else:
            proj = n.project()
        history = n.cs[self.slot, n.accepted.long()[:, None] - 1 + self.tail].reshape(-1, 3 * HD)
        verify_middle(
            proj,
            inputs["w_f"],
            inputs["w_g"],
            history,
            n.cw,
            n.state.view(-1, D),
            inputs["a_log"].view(64, 1),
            inputs["dt_bias"],
            inputs["norm_w"].view(1, D),
            n.cs.view(-1, 3 * HD),
            n.correction.view(-1, D),
            n.kg.view(-1, 2 * D),
            n.normalized.view(-1, HD),
        )
        return F.linear(n.normalized.view(-1, HD), inputs["w_o"])
