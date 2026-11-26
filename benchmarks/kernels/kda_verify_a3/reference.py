# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent references for the small A3 KDA verification-core fixture.

This fixture starts *after* convolution, Q/K normalization, and gate activation.
It measures only the verification core, not a complete KDA layer or serving step.
Jobs flatten independent (request, head) pairs; each job has 1--8 tokens and
128 key/value channels. The checkpoint is FP16 in [job, key, value] order.
Q already includes its normalization and attention scale, K is normalized,
decay is the per-key-channel multiplier (not a log gate), and beta is activated.

Verification returns each candidate's correction and output while leaving the
checkpoint read-only. There is no accepted-prefix commit in this fixture. In a
serving implementation, rejection sampling chooses an accepted prefix and a
separate operation commits only that prefix to the persistent checkpoint.

The direct oracle advances a full state in CPU FP64. The prepared reference
instead uses an FP32 matrix multiplication of FP16 operands followed by a short
correction recurrence. Their difference includes intentional FP16 rounding of
the decay-adjusted Q/K rows used by the Cube projection.
"""

import torch

DIM = 128
MAX_TOKENS = 8
PACKED_ROWS = 2 * MAX_TOKENS


def _validate(q, k, v, decay, beta, state):
    """Check metadata only; do not synchronize an accelerator for value checks."""
    if q.ndim != 3:
        raise ValueError("q must have shape [jobs, tokens, 128]")
    jobs, tokens, dim = q.shape
    if jobs < 1 or not 1 <= tokens <= MAX_TOKENS or dim != DIM:
        raise ValueError("expected jobs >= 1, 1 <= tokens <= 8, and dim = 128")
    shapes = {
        "q": (q, (jobs, tokens, DIM)),
        "k": (k, (jobs, tokens, DIM)),
        "v": (v, (jobs, tokens, DIM)),
        "decay": (decay, (jobs, tokens, DIM)),
        "beta": (beta, (jobs, tokens)),
        "state": (state, (jobs, DIM, DIM)),
    }
    for name, (tensor, shape) in shapes.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if tensor.device != q.device:
            raise ValueError("all inputs must be on the same device")
    if state.dtype != torch.float16:
        raise TypeError("state must use the fixture's FP16 checkpoint storage")
    return jobs, tokens


@torch.no_grad()
def prepare(q, k, v, decay, beta, state):
    """Prepare the kernel's five contiguous tensors on the input device.

    Returns (packed, state, coeff, beta_pad, v_pad):
      packed:   FP16 [jobs, 16, 128], rows t=P_t*k_t, rows 8+t=P_t*q_t.
      state:    FP16 [jobs, 128, 128], key-major checkpoint, read-only.
      coeff:    FP32 [jobs, 16, 16], nonzero only for 0 <= u <= t < L:
                  coeff[t,u]   = <k_t, k_u * product(decay[u+1:t+1])>
                  coeff[8+t,u] = <q_t, k_u * product(decay[u+1:t+1])>.
      beta_pad: FP32 [jobs, 16], activated beta in the first L entries.
      v_pad:    FP32 [jobs, 8, 128], V in the first L rows.

    P_t is the inclusive product of decay[0:t+1]. All unused rows/columns
    are zero. The key half's diagonal is populated but is not consumed by
    the correction recurrence; the query half's diagonal contributes to out.
    Products are formed directly, never by dividing cumulative products, so
    zero decay is valid. The returned state may alias the input checkpoint.
    """
    jobs, tokens = _validate(q, k, v, decay, beta, state)
    q32, k32, decay32 = q.float(), k.float(), decay.float()
    prefix = torch.cumprod(decay32, dim=1)
    packed = torch.zeros((jobs, PACKED_ROWS, DIM), device=q.device, dtype=torch.float16)
    packed[:, :tokens] = (prefix * k32).to(torch.float16)
    packed[:, MAX_TOKENS : MAX_TOKENS + tokens] = (prefix * q32).to(torch.float16)

    coeff = torch.zeros((jobs, PACKED_ROWS, PACKED_ROWS), device=q.device, dtype=torch.float32)
    for u in range(tokens):
        historical_key = k32[:, u]
        for t in range(u, tokens):
            if t > u:
                historical_key = historical_key * decay32[:, t]
            coeff[:, t, u] = (k32[:, t] * historical_key).sum(dim=-1)
            coeff[:, MAX_TOKENS + t, u] = (q32[:, t] * historical_key).sum(dim=-1)

    beta_pad = torch.zeros((jobs, PACKED_ROWS), device=q.device, dtype=torch.float32)
    beta_pad[:, :tokens] = beta.float()
    v_pad = torch.zeros((jobs, MAX_TOKENS, DIM), device=q.device, dtype=torch.float32)
    v_pad[:, :tokens] = v.float()
    return packed, state.contiguous(), coeff, beta_pad, v_pad


@torch.no_grad()
def oracle(q, k, v, decay, beta, state):
    """Return (correction, output), CPU FP64 [jobs, L, 128], by direct recurrence.

    With S in [key, value] order:
      S_decay = diag(decay_t) @ S
      corr_t  = beta_t * (v_t - k_t @ S_decay)
      S       = S_decay + outer(k_t, corr_t)
      out_t   = q_t @ S

    A private state copy is advanced; every caller-owned input stays unchanged.
    FP64 runs explicitly on CPU because the accelerator need not support it.
    """
    jobs, tokens = _validate(q, k, v, decay, beta, state)
    q64, k64, v64, decay64, beta64 = [tensor.to(device="cpu", dtype=torch.float64) for tensor in (q, k, v, decay, beta)]
    current = state.to(device="cpu", dtype=torch.float64).clone()
    corrections = torch.empty((jobs, tokens, DIM), dtype=torch.float64)
    outputs = torch.empty_like(corrections)
    for t in range(tokens):
        current = current * decay64[:, t, :, None]
        predicted = torch.bmm(k64[:, t, None, :], current).squeeze(1)
        correction = beta64[:, t, None] * (v64[:, t] - predicted)
        current = current + k64[:, t, :, None] * correction[:, None, :]
        corrections[:, t] = correction
        outputs[:, t] = torch.bmm(q64[:, t, None, :], current).squeeze(1)
    return corrections, outputs


@torch.no_grad()
def prepared_reference(packed, state, coeff, beta_pad, v_pad, tokens):
    """Return FP32 (correction, output) using the prepared kernel contract.

    FP16 packed rows and checkpoint are widened before an FP32 bmm. This is
    the algebraic reference for the Cube projection plus Vector recurrence,
    not a promise of bitwise equivalence with a particular hardware GEMM.
    """
    if not 1 <= tokens <= MAX_TOKENS:
        raise ValueError("tokens must be between 1 and 8")
    projection = torch.bmm(packed.float(), state.float())
    corrections = []
    outputs = []
    for t in range(tokens):
        predicted = projection[:, t].clone()
        for u in range(t):
            predicted = predicted + coeff[:, t, u, None].float() * corrections[u]
        correction = beta_pad[:, t, None].float() * (v_pad[:, t].float() - predicted)
        corrections.append(correction)
        output = projection[:, MAX_TOKENS + t].clone()
        for u in range(t + 1):
            output = output + coeff[:, MAX_TOKENS + t, u, None].float() * corrections[u]
        outputs.append(output)
    return torch.stack(corrections, dim=1), torch.stack(outputs, dim=1)
