# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-to-projection TileLang variants for A3 KDA verification."""

import torch
import torch.nn.functional as F
from full_kernel import HEAD_DIM, TOKEN_ALIGNMENT, make_middle_kernel
from full_reference import ReplayResult, make_cache, prepare_middle
from native_adapter import NativeFrontend


def _split_bfloat16(x):
    high = x.to(torch.bfloat16)
    return high, (x - high.float()).to(torch.bfloat16)


class Candidate:
    """Compile a full-path schedule and reuse it on identical replay inputs.

    A zero gate tile uses the two native linear calls. A positive tile runs
    both gate projections in one Cube kernel with resident weight tiles.
    ``projection_pad`` aligns the merged input projection's token rows.
    ``prepare_fused`` packs the short decay prefix and projection rows in
    one Vector launch. ``cv_slots`` selects the mixed middle's GM ring size.
    """

    def __init__(self, config, inputs, row_tile=0, projection_pad=0, *, prepare_fused=False, cv_slots=0, core_count=24):
        if config.dim != HEAD_DIM or not 1 <= config.tokens <= 16:
            raise ValueError("the A3 middle supports D=128 and 1–16 candidates")
        if not -5 <= config.lower_bound <= 0:
            raise ValueError("the packed recurrence requires a gate lower bound in [-5, 0]")
        if projection_pad < 0:
            raise ValueError("the projection row alignment must be nonnegative")

        self.config = config
        self.inputs = inputs
        rows = config.batch * config.tokens
        self.projection_rows = (
            (rows + projection_pad - 1) // projection_pad * projection_pad if projection_pad else rows
        )
        self.token_rows = (config.tokens + TOKEN_ALIGNMENT - 1) // TOKEN_ALIGNMENT * TOKEN_ALIGNMENT
        self.middle = make_middle_kernel(config.tokens, config.eps, cv_slots=cv_slots, core_count=core_count)
        self.prepare = None
        if prepare_fused:
            from full_prepare import compile_full_prepare

            self.prepare = compile_full_prepare(config.batch * config.heads, config.tokens, config.dim)
        self.cache_base = make_cache(config, inputs)
        self.cache = self.cache_base.clone()

        self.gates = None
        if row_tile:
            from full_gates import make_gate_kernel

            self.gates = make_gate_kernel(rows, config.heads, row_tile)

    def reset(self):
        self.cache.copy_(self.cache_base)

    def _project_gates(self, projected):
        if self.gates.padded_rows != projected.shape[0]:
            projected = F.pad(projected, (0, 0, 0, self.gates.padded_rows - projected.shape[0]))
        return self.gates(projected, self.inputs["w_f"], self.inputs["w_g"])

    def _prepare_rows(self, middle):
        # The log gate is bounded below by -5. At L<=16, every FP32 prefix
        # is positive, so the interval product is prefix[t] / prefix[u].
        prefix = middle.decay.cumprod(dim=1)
        padding = self.token_rows - self.config.tokens
        key_rows = F.pad(prefix * middle.k, (0, 0, 0, padding))
        query_rows = F.pad(prefix * middle.q, (0, 0, 0, padding))
        rows = torch.cat((key_rows, query_rows), dim=1)
        inverse_rows = F.pad(middle.k / prefix, (0, 0, 0, padding))
        return rows, inverse_rows

    def __call__(self, accepted):
        config = self.config
        inputs = self.inputs
        tokens = config.tokens
        gate_projector = self._project_gates if self.gates is not None else None
        middle = prepare_middle(
            config, inputs, accepted, self.projection_rows, gate_projector=gate_projector, cache=self.cache
        )
        self.cache.copy_(middle.history)

        if self.prepare is None:
            rows, inverse_rows = self._prepare_rows(middle)
        else:
            high, residual, rows, inverse_rows = self.prepare(middle.decay, middle.k, middle.q)
        coeff = torch.bmm(rows, inverse_rows.transpose(1, 2))
        beta = F.pad(middle.beta, (0, self.token_rows - tokens))
        checkpoint = inputs["state"].reshape(-1, config.dim, config.dim)
        gate = middle.g2.reshape(config.batch, tokens, config.heads, config.dim)
        gate = gate.transpose(1, 2).reshape(-1, tokens, config.dim).contiguous()
        if self.prepare is None:
            high, residual = _split_bfloat16(rows)
        correction, norm = self.middle(
            high,
            residual,
            *_split_bfloat16(checkpoint),
            coeff,
            beta,
            middle.v.contiguous(),
            gate,
            inputs["norm_w"],
        )

        correction = correction.reshape(config.batch, config.heads, tokens, config.dim)
        normalized = norm.reshape(config.batch, config.heads, tokens, config.dim).transpose(1, 2)
        normalized = normalized.reshape(-1, config.heads, config.dim).contiguous()
        output = F.linear(normalized.reshape(config.batch * tokens, -1), inputs["w_o"])
        return ReplayResult(output, normalized, correction, middle.kg, self.cache)


class A5StyleCandidate(NativeFrontend):
    """Keep the FP32 recurrence and gated normalization in one Vector kernel."""

    def __init__(self, config, inputs):
        from full_a5_middle import make_a5_middle_kernel

        if config.heads != 64 or config.dim != HEAD_DIM or not 1 <= config.tokens <= 16:
            raise ValueError("the A5-style middle supports 64 heads, D=128 and 1–16 candidates")
        super().__init__(config, inputs)
        self.middle = make_a5_middle_kernel(config.tokens, config.eps, config.lower_bound)

    def __call__(self, accepted):
        c = self.config
        i = self.inputs
        projected, _, convolved, gate_f, gate_g = self.project_and_convolve(accepted)
        normalized, correction, kg = self.middle(
            convolved, projected, gate_f, gate_g, i["state"], i["a_log"], i["dt_bias"], i["norm_w"]
        )
        output = F.linear(normalized, i["w_o"])
        return ReplayResult(output, normalized.reshape(-1, c.heads, c.dim), correction, kg, self.cache[1:])


class ParallelA5StyleCandidate(A5StyleCandidate):
    """Overlap the gate projections and causal convolution on two NPU streams."""

    def __init__(self, config, inputs):
        super().__init__(config, inputs)
        self.gate_stream = torch.npu.Stream(device=self.x.device)

    def _project_gates_and_convolve(self, padded, fa, ga, qkv, accepted):
        main = torch.npu.current_stream(self.x.device)
        projection_ready = main.record_event()
        padded.record_stream(self.gate_stream)
        with torch.npu.stream(self.gate_stream):
            self.gate_stream.wait_event(projection_ready)
            gate_f, gate_g = self._project_gates(padded, fa, ga)
            gates_ready = self.gate_stream.record_event()

        convolved = self._convolve(qkv, accepted)
        gate_f.record_stream(main)
        gate_g.record_stream(main)
        # Join the auxiliary stream tail into the same ACL Graph before recurrence.
        main.wait_event(gates_ready)
        return convolved, gate_f, gate_g
