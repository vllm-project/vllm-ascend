# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference for the A5 benchmark's complete QKV-to-output path.

Fixture and native operation order follow vllm-ascend PR 22370 at
f16f61a9509a4a609d88942dc7109c671d9197ac, benchmarks/kernels/kda_verify.
Unlike the earlier small core demo, the checkpoint is FP32 [B,H,value,key].
All linear projections, convolution outputs, core outputs, normalized outputs,
and KG records round to BF16 at the same public boundaries as the A5 baseline.
The Torch implementation is an independent mathematical reference, not a claim
of bitwise equivalence with CANN convolution/GEMM or Triton transcendental math.

``accepted[b]`` selects the previous round's convolution-history prefix. It is
an int32/int64 tensor in [1, L], not a commit length for this round. Inputs and
the recurrent checkpoint remain unchanged. Each invocation starts from the
same supplied history; optional ``inputs['rollback']`` holds the extra L-1
history rows used in the A5 paired comparison. No recurrent commit is timed.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    batch: int = 1
    tokens: int = 6
    hidden: int = 4096
    heads: int = 64
    dim: int = 128
    lower_bound: float | None = -5.0
    eps: float = 1e-5


class ReplayResult(NamedTuple):
    output: torch.Tensor  # [B*L, hidden], BF16
    normalized: torch.Tensor  # [B*L, H, D], BF16
    correction: torch.Tensor  # [B, H, L, D], FP32
    kg: torch.Tensor  # [B, H, L, 2*D], BF16; raw convolved K and raw gate
    history: torch.Tensor  # [B, L+2, 3*H*D], BF16


class MiddleInputs(NamedTuple):
    q: torch.Tensor  # [B*H, L, D], FP32, normalized and scaled
    k: torch.Tensor  # [B*H, L, D], FP32, normalized
    v: torch.Tensor  # [B*H, L, D], FP32, widened BF16 convolution output
    decay: torch.Tensor  # [B*H, L, D], FP32, channel-wise multiplier
    beta: torch.Tensor  # [B*H, L], FP32, sigmoid activated
    g2: torch.Tensor  # [B*L, H, D], BF16, output gate before sigmoid
    kg: torch.Tensor
    history: torch.Tensor


def make_inputs(c, device="cpu", seed=7):
    """Match A5 inputs.py, including random draw order, scales and dtypes."""
    torch.manual_seed(seed)
    b, tokens, h, d, k = c.batch, c.tokens, c.heads, c.dim, c.hidden

    def rand(shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, device=device) * scale).to(dtype)

    return {
        "x": rand((b, tokens, k), 0.3),
        "w_in": rand((3 * h * d + h + 2 * d, k), k**-0.5),
        "w_f": rand((h * d, d), d**-0.5),
        "w_g": rand((h * d, d), d**-0.5),
        "w_o": rand((k, h * d), (h * d) ** -0.5),
        "conv_w": rand((3, h, d, 4), 0.3),
        "history": rand((b, 3, h, d, 3), 0.2),
        "state": rand((b, h, d, d), 0.02, torch.float32),
        "a_log": rand((h,), 0.2, torch.float32),
        "dt_bias": rand((h, d), 0.2, torch.float32),
        "norm_w": rand((d,), 0.2) + 1,
    }


def make_rollback(c, inputs):
    """Draw the additional BF16 history rows after make_inputs, as in A5."""
    shape = (c.batch, c.tokens - 1, 3 * c.heads * c.dim)
    return torch.randn(shape, device=inputs["x"].device, dtype=torch.bfloat16) * 0.25


def make_cache(c, inputs):
    """Build a private initial [B,L+2,3*H*D] BF16 convolution cache."""
    first = inputs["history"].permute(0, 4, 1, 2, 3).reshape(c.batch, 3, 3 * c.heads * c.dim)
    tail = inputs.get("rollback")
    if tail is None:
        tail = first.new_zeros((c.batch, c.tokens - 1, 3 * c.heads * c.dim))
    return torch.cat((first, tail), dim=1)


def project_inputs(c, inputs, projection_rows=0, gate_projector=None):
    """Return BF16 qkv[B,L,3,H,D], beta[B,L,H], and two gates[B,L,H,D]."""
    b, length, h, d = c.batch, c.tokens, c.heads, c.dim
    x = inputs["x"].reshape(b * length, c.hidden)
    if projection_rows > b * length:
        x = F.pad(x, (0, 0, 0, projection_rows - b * length))
    projected = F.linear(x, inputs["w_in"])[: b * length]
    qkv, beta, fa, ga = projected.split([3 * h * d, h, d, d], dim=-1)
    if gate_projector is None:
        gate_f = F.linear(fa, inputs["w_f"])
        gate_g = F.linear(ga, inputs["w_g"])
    else:
        # Keep merged-projection strides: the callback sees only valid rows.
        gate_f, gate_g = gate_projector(projected)
    gate_f = gate_f.reshape(b, length, h, d)
    gate_g = gate_g.reshape(b, length, h, d)
    return qkv.reshape(b, length, 3, h, d), beta.reshape(b, length, h), gate_f, gate_g


def convolve(c, qkv, inputs, accepted, cache=None):
    """Select previous history, then run four-tap causal depthwise conv+SiLU.

    The returned rollback cache is [selected_history[1:], current_raw_qkv].
    Using tensor gather keeps accepted values dynamic during graph replay.
    """
    b, length, h, d = c.batch, c.tokens, c.heads, c.dim
    if cache is None:
        cache = make_cache(c, inputs)
    indices = accepted.to(dtype=torch.long)[:, None] - 1
    indices = indices + torch.arange(3, device=qkv.device)[None, :]
    selected = cache.gather(1, indices[:, :, None].expand(b, 3, 3 * h * d))
    window = torch.cat((selected, qkv.reshape(b, length, 3 * h * d)), dim=1)
    weights = inputs["conv_w"].permute(3, 0, 1, 2).reshape(4, 3 * h * d).float()
    accum = torch.zeros((b, length, 3 * h * d), dtype=torch.float32, device=qkv.device)
    for tap in range(4):
        accum = accum + window[:, tap : tap + length].float() * weights[tap]
    convolved = F.silu(accum).to(torch.bfloat16).reshape(b, length, 3, h, d)
    return convolved, window[:, 1:].contiguous()


def prepare_recurrence(c, convolved, gate_f, beta_raw, a_log, dt_bias):
    """Return q/k/v/decay [B*H,L,D] and beta [B*H,L], all FP32."""
    b, length, h, d = c.batch, c.tokens, c.heads, c.dim

    def jobs(tensor):
        return tensor.permute(0, 2, 1, 3).reshape(b * h, length, d).contiguous()

    def normalize(tensor):
        square = tensor.square().sum(dim=-1, keepdim=True)
        safe = torch.where(square > 0, square, torch.ones_like(square))
        return tensor * torch.rsqrt(safe)

    q, k, v = convolved.float().unbind(dim=2)
    q, k = normalize(q) * d**-0.5, normalize(k)
    gate_input = gate_f.float() + dt_bias[None, None].float()
    a = a_log.float().exp()[None, None, :, None]
    if c.lower_bound is None:
        gate = -a * F.softplus(gate_input, threshold=20)
    else:
        gate = c.lower_bound * torch.sigmoid(a * gate_input)
    beta = torch.sigmoid(beta_raw.float()).permute(0, 2, 1).reshape(b * h, length).contiguous()
    return jobs(q), jobs(k), jobs(v), jobs(gate.exp()), beta


def prepare_middle(c, inputs, accepted, projection_rows=0, gate_projector=None, cache=None):
    """Reusable device-only projection/convolution preparation for a candidate."""
    qkv, beta_raw, gate_f, gate_g = project_inputs(c, inputs, projection_rows, gate_projector)
    convolved, history = convolve(c, qkv, inputs, accepted, cache)
    recurrence = prepare_recurrence(c, convolved, gate_f, beta_raw, inputs["a_log"], inputs["dt_bias"])
    kg = torch.cat((convolved[:, :, 1], gate_f), dim=-1).permute(0, 2, 1, 3).contiguous()
    return MiddleInputs(*recurrence, gate_g.reshape(c.batch * c.tokens, c.heads, c.dim), kg, history)


def recurrent_reference(c, q, k, v, decay, beta, state):
    """Direct FP32 recurrence against a private [job,value,key] state copy."""
    current = state.reshape(c.batch * c.heads, c.dim, c.dim).float().clone()
    corrections, outputs = [], []
    for t in range(c.tokens):
        decayed = current * decay[:, t, None, :]
        predicted = torch.bmm(decayed, k[:, t, :, None]).squeeze(-1)
        correction = beta[:, t, None] * (v[:, t] - predicted)
        current = decayed + correction[:, :, None] * k[:, t, None, :]
        output = torch.bmm(current, q[:, t, :, None]).squeeze(-1)
        corrections.append(correction)
        outputs.append(output)
    correction = torch.stack(corrections, dim=1).reshape(c.batch, c.heads, c.tokens, c.dim)
    core = torch.stack(outputs, dim=1).reshape(c.batch, c.heads, c.tokens, c.dim)
    return core.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous(), correction


def normalize_output(c, core, gate_g, norm_w):
    """RMS-normalize rounded BF16 core, then multiply scale and sigmoid gate."""
    values = core.reshape(c.batch * c.tokens, c.heads, c.dim).float()
    inv_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + c.eps)
    gate = torch.sigmoid(gate_g.reshape_as(values).float())
    return (values * inv_rms * norm_w.float() * gate).to(torch.bfloat16)


@torch.no_grad()
def full_reference(c, inputs, accepted, cache=None):
    """Evaluate all projections, convolution, verification and normalization."""
    middle = prepare_middle(c, inputs, accepted, cache=cache)
    core, correction = recurrent_reference(c, *middle[:5], inputs["state"])
    normalized = normalize_output(c, core, middle.g2, inputs["norm_w"])
    output = F.linear(normalized.reshape(c.batch * c.tokens, c.heads * c.dim), inputs["w_o"])
    return ReplayResult(output, normalized, correction, middle.kg, middle.history)
