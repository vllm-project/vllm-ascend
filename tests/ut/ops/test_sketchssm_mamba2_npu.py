# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM Mamba-2 decode on Ascend NPU (Triton kernels) vs FP64 references.

Port of the upstream CUDA oracle (vllm-project/vllm#61124,
tests/kernels/mamba/test_sketchssm_mamba2.py @ 7e6f8d8): random orthogonal
per-group frames, unrotated B/C with FP64-rotated references, key-major
strided state view, plus the rotate_states roundtrip. This NPU port uses the
model's real geometry (H=96, P=80, G=8, N=128), runs two full windows per
row, forces the triton backend, and computes the FP64 references on CPU
(910B3 downcasts float64 to float32 on device).
"""

import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.mamba import sketchssm_mamba2_npu as sk
from vllm_ascend.ops.triton.mamba import sketchssm_mamba2_npu_triton as skt


def _npu_available() -> bool:
    """Whether a real NPU is present (the UT conftest mocks torch_npu on CPU)."""
    try:
        subprocess.run(["npu-smi", "info"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


pytestmark = pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU")

H, P, G, N, NULL = 96, 80, 8, 128, 0
RANKS = ([0, 1, 2, 3, 4, 5, 9, 20, 38, 49] * H)[:H]  # dense, maps, pivots
BOUNDARY_RANKS = ([0, 1, 4, 5, 8, 9, 16, 17, 32, 49] * H)[:H]


def sketch_read(s: torch.Tensor, q: torch.Tensor, m: int) -> torch.Tensor:
    """Four-pivot residual-diagonal read of state ``s`` (P, N) with ``q``."""
    if m == 0:
        return s @ q
    norm = s / s.square().sum(0).mean().sqrt()
    dirs: list[torch.Tensor] = []
    for u in norm[:, : min(m, sk.SKETCH_PIVOTS)].T:
        v = u.clone()
        for d in dirs + dirs:
            v -= d * (d @ v)
        keep = v.square().sum() > 1e-12 * u.square().sum()
        dirs.append(v / v.norm() if keep else torch.zeros_like(v))
    z = torch.stack(dirs) @ norm
    residual = (norm.square().sum(0) - z.square().sum(0)).clamp_min(0)
    residual[: len(dirs)] = 0
    metric = torch.diag(residual) + z.T @ z
    ridge = sk.SKETCH_RIDGE * torch.eye(m, dtype=s.dtype, device=s.device)
    return s[:, :m] @ torch.linalg.solve(metric[:m, :m] + ridge, metric[:m] @ q)


def stored_read(sketch: sk.SketchArgs, meta: int, h: int, q: torch.Tensor):
    """Read of head ``h`` with ``q`` from the stored sketch row ``meta``."""
    t, m = sketch.tables, RANKS[h]
    U = sketch.u[meta, int(t.u_offsets[h]) :].cpu().double()
    if m == 0:
        return U[:N].T @ q
    w_off, ag_off = int(t.w_offsets[h]), int(t.ag_offsets[h])
    coeff = sketch.w[meta, w_off : w_off + min(m, sk.SKETCH_PIVOTS)].cpu().double() @ q
    if m > sk.SKETCH_PIVOTS:
        ag = sketch.ag[meta, :, ag_off : ag_off + m].cpu().double()
        coeff = ag[0] * q[:m] + (ag[1:] * coeff[:, None]).sum(0)
    return U[:m].T @ coeff


class Layer:
    """One layer's state, rings and sketch; the last row pads the batch."""

    def __init__(self, W: int, batch: int = 5):
        bf16, S = torch.bfloat16, batch + 1
        self.W, self.batch = W, batch
        g = self.g = torch.Generator().manual_seed(0)
        rand = lambda *s: torch.rand(*s, generator=g).npu()  # noqa: E731
        randn = lambda *s: torch.randn(*s, generator=g).npu() * 0.1  # noqa: E731
        # Key-major FP32 state; kernels see the (slot, H, dim, dstate) view.
        self.state = randn(S, H, N, P).transpose(-1, -2)
        self.x_cache, self.dt_cache = randn(S, H, W, P).to(bf16), rand(S, H, W) * 0.1
        self.B_cache, self.A = randn(S, G, W, N).to(bf16), -rand(H) - 0.5
        self.dt_bias, self.D = (rand(H) * 0.1).to(bf16), rand(H).to(bf16)
        self.slots = torch.tensor([*range(1, batch), NULL], dtype=torch.int32).npu()
        self.meta = torch.arange(batch, dtype=torch.int32).flip(0).npu()
        ranks = torch.tensor(RANKS, dtype=torch.int32)
        shapes = zip(sk.sketch_shapes(ranks, P, N), sk.SKETCH_DTYPES)
        u, w, ag = (torch.zeros(batch, *s, dtype=d).npu() for s, d in shapes)
        frames = torch.linalg.qr(torch.randn(G, N, N, generator=g))[0]
        frames_t = frames.transpose(-1, -2).contiguous().npu()
        self.rot = frames_t.cpu().double()  # row vectors rotate as v @ frames_t
        tables = sk.SketchTables(ranks, N).npu()
        self.sketch = sk.SketchArgs(u, w, ag, tables, frames_t)
        # x, B and C are slices of one row buffer, as in the model.
        self.conv = torch.empty(batch, H * P + 2 * G * N, dtype=bf16).npu()
        self.out = torch.empty(batch, H, P, dtype=bf16).npu()
        self.bc_pre = torch.empty(batch, G, W).npu()

    def inputs(self):
        self.conv.copy_(torch.randn(self.conv.shape, generator=self.g) * 0.3)
        x, B, C = self.conv.split([H * P, G * N, G * N], 1)
        dt = (torch.randn(self.batch, H, generator=self.g) * 0.5).npu().bfloat16()
        return x.view(-1, H, P), dt, B.view(-1, G, N), C.view(-1, G, N)

    def build(self) -> None:
        flags = torch.ones(self.batch, dtype=torch.int8).npu()
        sk.sketch_build(self.state, flags, self.slots, self.meta, self.sketch, NULL)

    def step(self, x, dt, B, C, pos, flush):
        skt.sketch_triton_decode(
            self.state, x, dt[..., None].expand(-1, -1, P),
            self.A[:, None, None].expand(-1, P, N), B, C,
            self.D[:, None].expand(-1, P), self.dt_bias[:, None].expand(-1, P),
            self.x_cache, self.dt_cache, self.B_cache, self.bc_pre, pos,
            flush, sk.flush_row_list(flush), self.slots, self.meta, self.out,
            self.sketch, NULL,
        )  # fmt: skip
        return self.out

    def checked_step(self, pos: torch.Tensor) -> None:
        """Decode at ring positions ``pos`` and check against FP64 references."""
        x, dt, B, C = self.inputs()
        s0 = self.state.cpu().double()
        flush = (pos == self.W - 1).to(torch.int8).npu()
        got = self.step(x, dt, B, C, pos.int().npu(), flush).cpu().double()
        assert torch.equal(self.state[NULL].cpu().double(), s0[NULL])
        for row, slot in enumerate(self.slots.tolist()[:-1]):
            p, r = int(pos[row]), H // G
            dt_cur = F.softplus(dt[row].cpu().float() + self.dt_bias.cpu().float())[:, None]
            dts = torch.cat([self.dt_cache[slot, :, :p].cpu(), dt_cur], 1).double()
            cs, a = dts.cumsum(1), self.A.cpu().double()[:, None]
            w = dts * torch.exp(a * (cs[:, -1:] - cs))
            decay = torch.exp(a * cs[:, -1:])
            xs = torch.cat([self.x_cache[slot, :, :p].cpu(), x[row, :, None].cpu()], 1).double()
            ks = torch.cat([self.B_cache[slot, :, :p].cpu(), B[row, :, None].cpu()], 1).double()
            ks = torch.einsum("gtn,gnk->gtk", ks, self.rot)
            q = (C[row].cpu().double()[:, None] @ self.rot)[:, 0]
            ks, q = ks.repeat_interleave(r, 0), q.repeat_interleave(r, 0)
            skip = self.D.cpu().double()[:, None] * xs[:, -1]
            if p == self.W - 1:
                s = decay[..., None] * s0[slot]
                s += torch.einsum("htp,ht,htn->hpn", xs, w, ks)
                got_s = self.state[slot].cpu().double()
                torch.testing.assert_close(got_s, s, rtol=1e-5, atol=1e-6)
                want, tol = torch.einsum("hpn,hn->hp", s, q) + skip, [1e-2] * H
            else:
                ring = torch.einsum("htp,ht,htn,hn->hp", xs, w, ks, q)
                # The kernel reads its stored sketch (BF16 output rounding) ...
                meta = int(self.meta[row])
                read = [stored_read(self.sketch, meta, h, q[h]) for h in range(H)]
                want = decay * torch.stack(read) + ring + skip
                err = (got[row] - want).abs().amax(1) / want.abs().amax(1)
                # NPU-specific 1% gate, not upstream's 0.5% CUDA precision gate.
                assert (err < 1e-2).all(), (row, p, err)
                # Same-geometry control: isolate BF16 output rounding from the
                # remaining kernel arithmetic error (bounded by upstream's 0.5%).
                rounded = want.bfloat16().double()
                roundoff = (rounded - want).abs().amax(1) / want.abs().amax(1)
                assert (err < roundoff + 5e-3).all(), (row, p, err, roundoff)
                # ... which holds the four-pivot read up to BF16 storage.
                read = [sketch_read(s0[slot, h], q[h], m) for h, m in enumerate(RANKS)]
                want = decay * torch.stack(read) + ring + skip
                # H=96/P=80 NPU approximation gate; upstream also uses random
                # orthogonal frames, with H=32/P=32 and a 5% CUDA gate.
                tol = [2e-2 if m == 0 else 1e-1 for m in RANKS]
                assert torch.equal(self.state[slot].cpu().double(), s0[slot])
            err = (got[row] - want).abs().amax(1) / want.abs().amax(1)
            assert torch.isfinite(err).all(), (row, p, torch.nonzero(~torch.isfinite(err)).flatten())
            bad = err >= torch.tensor(tol)
            assert not bad.any(), (row, p, err[bad])


def test_sketchssm_mamba2_decode_npu():
    """Sketches built from the prefilled states serve rows at mixed window
    positions for two full windows, each row flushing and rebuilding twice."""
    W = 16
    layer = Layer(W)
    layer.build()
    assert not layer.sketch.u[int(layer.meta[-1])].any()  # padding row
    for t in range(2 * W + 1):
        layer.checked_step((t + torch.tensor([0, W // 2 + 1, W - 1, 3, 0])) % W)


def test_sketchssm_mamba2_decode_npu_rank_boundaries(monkeypatch):
    """Every readout branch: dense, the 1..4 and 5..8 static blocks, and the
    dynamic tail, all in one layer so the shared geometry matches the model."""
    monkeypatch.setattr(sys.modules[__name__], "RANKS", BOUNDARY_RANKS)
    W = 16
    layer = Layer(W)
    layer.build()
    for t in range(W + 1):
        layer.checked_step((t + torch.tensor([0, W // 2 + 1, W - 1, 3, 0])) % W)


def test_sketchssm_rotate_states_npu():
    """Prefill states rotate into the cache frame and back."""
    g = torch.Generator().manual_seed(0)
    states = torch.randn(3, H, P, N, generator=g).npu()
    frames_t = torch.linalg.qr(torch.randn(G, N, N, generator=g))[0].contiguous().npu()
    rotated = sk.sketch_rotate_states(states, frames_t)
    want = states.cpu().double().unflatten(1, (G, -1)) @ frames_t.cpu().double()[None, :, None]
    torch.testing.assert_close(rotated.cpu(), want.flatten(1, 2).float(), rtol=1e-5, atol=1e-5)
    # The cache's initial states arrive as a key-major (transposed) view.
    key_major = rotated.transpose(-1, -2).contiguous().transpose(-1, -2)
    back = sk.sketch_rotate_states(key_major, frames_t, inverse=True)
    torch.testing.assert_close(back.cpu(), states.cpu(), rtol=1e-5, atol=1e-5)


def test_sketchssm_window_keys_requires_flush_flags():
    """Legacy row-list-only calls fail instead of processing every row."""
    with pytest.raises(TypeError, match="is_flush"):
        sk.sketch_window_keys(None, None, None, None, None, None, None)
