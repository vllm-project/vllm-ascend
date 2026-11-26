# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native comparator matching the unfused path of vllm-ascend PR #22370."""

import torch
import torch.nn.functional as F
from full_reference import ReplayResult, make_cache


class NativeFrontend:
    """Shared projections and causal convolution for the local recurrence paths."""

    def __init__(self, config, inputs, row_tile=0, projection_pad=0):
        if projection_pad < 0:
            raise ValueError("the projection row alignment must be nonnegative")
        from fla_npu.ops.ascendc import causal_conv1d_update

        self.conv = causal_conv1d_update
        self.config = config
        self.inputs = inputs
        b, length, h, d = config.batch, config.tokens, config.heads, config.dim
        rows = b * length
        self.projection_rows = (
            (rows + projection_pad - 1) // projection_pad * projection_pad if projection_pad else rows
        )
        self.gates = None
        if row_tile:
            from full_gates import HEAD_DIM, make_gate_kernel

            if d != HEAD_DIM:
                raise ValueError(f"the fused gate projection requires D={HEAD_DIM}")
            self.gates = make_gate_kernel(rows, h, row_tile)
        device = inputs["x"].device
        self.x = inputs["x"].reshape(b * length, config.hidden)
        self.conv_weight = inputs["conv_w"].permute(3, 0, 1, 2).reshape(4, 3 * h * d).contiguous()
        self.cache = self.x.new_zeros((b + 1, length + 2, 3 * h * d))
        self.cache[1:] = make_cache(config, inputs)
        self.cache_base = self.cache.clone()
        self.starts = torch.arange(b + 1, device=device, dtype=torch.int32) * length
        self.indices = torch.arange(1, b + 1, device=device, dtype=torch.int32)
        self.conv_out = self.x.new_empty((b * length, 3 * h * d))

    def reset(self):
        self.cache.copy_(self.cache_base)

    def project_and_convolve(self, accepted):
        c = self.config
        i = self.inputs
        b, length, h, d = c.batch, c.tokens, c.heads, c.dim
        x = self.x
        if self.projection_rows > x.shape[0]:
            x = F.pad(x, (0, 0, 0, self.projection_rows - x.shape[0]))
        padded = F.linear(x, i["w_in"])
        projected = padded[: b * length]
        qkv, beta, fa, ga = projected.split((3 * h * d, h, d, d), dim=-1)
        convolved, gate_f, gate_g = self._project_gates_and_convolve(padded, fa, ga, qkv, accepted)
        return projected, beta, convolved, gate_f, gate_g

    def _project_gates_and_convolve(self, padded, fa, ga, qkv, accepted):
        gate_f, gate_g = self._project_gates(padded, fa, ga)
        return self._convolve(qkv, accepted), gate_f, gate_g

    def _project_gates(self, padded, fa, ga):
        i = self.inputs
        if self.gates is None:
            return F.linear(fa, i["w_f"]), F.linear(ga, i["w_g"])
        # Reuse input-projection padding where available. Only row padding
        # is allowed: the gate kernel expects the original merged pitch.
        gate_input = padded[: self.gates.padded_rows]
        if gate_input.shape[0] < self.gates.padded_rows:
            gate_input = F.pad(gate_input, (0, 0, 0, self.gates.padded_rows - gate_input.shape[0]))
        return self.gates(gate_input, i["w_f"], i["w_g"])

    def _convolve(self, qkv, accepted):
        return self.conv(
            qkv,
            self.cache,
            self.conv_weight,
            bias=None,
            activation="silu",
            conv_state_indices=self.indices,
            num_accepted_tokens=accepted,
            query_start_loc=self.starts,
            null_block_id=0,
            out=self.conv_out,
        )


class Native(NativeFrontend):
    def __init__(self, config, inputs, row_tile=0, projection_pad=0):
        super().__init__(config, inputs, row_tile, projection_pad)
        from vllm_ascend.ops.triton.kda.kda import rms_norm_gated
        from vllm_ascend.ops.triton.kda.recoverssm import kda_recoverssm_verify

        b, length, h, d = config.batch, config.tokens, config.heads, config.dim
        self.verify = kda_recoverssm_verify
        self.norm = rms_norm_gated
        self.state = inputs["state"].new_zeros((b + 1, h, d, d))
        self.state[1:] = inputs["state"]
        self.correction = self.state.new_full((b + 1, h, length, d), float("nan"))
        self.kg = self.x.new_empty((b + 1, h, length, 2 * d))
        self.record_offset = None
        self.core = self.x.new_empty((1, b * length, h, d))
        self.normalized = torch.empty_like(self.core)

    def checkpoint_unchanged(self):
        return torch.equal(self.state[1:], self.inputs["state"]) and not self.state[0].count_nonzero().item()

    def __call__(self, accepted):
        c = self.config
        i = self.inputs
        b, length, h, d = c.batch, c.tokens, c.heads, c.dim
        _, beta, convolved, gate_f, gate_g = self.project_and_convolve(accepted)
        gate_f = gate_f.reshape(1, b * length, h, d)
        gate_g = gate_g.reshape(b * length, h, d)
        q, k, v = (tensor.reshape(1, b * length, h, d) for tensor in convolved.split(h * d, dim=-1))
        core = self.verify(
            q,
            k,
            v,
            gate_f,
            beta.unsqueeze(0),
            i["a_log"],
            i["dt_bias"],
            c.lower_bound,
            self.state,
            self.correction,
            self.kg,
            self.starts,
            self.indices,
            length,
            out=self.core,
        )
        # The A5 reference uses cache-slot indices, while newer RecoverSSM
        # versions write request indices. Resolve the layout on eager warmup
        # by checking which of the two edges still has its original sentinel.
        if self.record_offset is None:
            first_written = torch.isfinite(self.correction[0, 0, 0, 0]).item()
            last_written = torch.isfinite(self.correction[b, 0, 0, 0]).item()
            if first_written == last_written:
                raise RuntimeError("could not identify the native RecoverSSM record layout")
            self.record_offset = int(last_written)
        records = slice(self.record_offset, self.record_offset + b)
        norm = self.norm(core, gate_g, i["norm_w"], None, activation="sigmoid", eps=c.eps, out=self.normalized)
        output = F.linear(norm.reshape(b * length, h * d), i["w_o"])
        return ReplayResult(
            output, norm.reshape(b * length, h, d), self.correction[records], self.kg[records], self.cache[1:]
        )
