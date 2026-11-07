# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU integration smoke: AscendC recurrent op + LeapQuant boundary flush.

Runs the real ``fla_npu.ops.ascendc.recurrent_gated_delta_rule`` decode op
token by token against a state pool (two live slots), flushing window-boundary
rows through LeapQuant, and compares against the same trajectory with the
flush disabled (the exact fp32 reference). Pins the index plumbing: the flush
must touch exactly the rows the recurrent op updated, and a bf16
gated-delta-rule stream must tolerate four int8 window flushes with only
small-multiple-of-quantization drift.

Run on an Atlas A3 NPU (directory suffix routes to a3_2 CI runners).
"""

import unittest

import torch

from vllm_ascend.ascend_config import LeapQuantConfig
from vllm_ascend.quantization.methods.mamba_state.leapquant import leapquant_flush_state_rows

# The AscendC recurrent op requires head_dim == 128 on both K and V sides.
H, DK, DV = 4, 128, 128
STEPS, WINDOW = 64, 16
SLOTS = 8
SEQS = 2


def _run_trajectory(flush: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    torch.manual_seed(42)
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    # Per-sequence token streams: q/k [STEPS, SEQS, H, D], one row per step.
    q = torch.randn(STEPS, SEQS, H, DK, device=device, dtype=torch.float32)
    k = torch.randn(STEPS, SEQS, H, DK, device=device, dtype=torch.float32)
    v = torch.randn(STEPS, SEQS, H, DV, device=device, dtype=torch.float32)
    g = -torch.rand(STEPS, SEQS, H, device=device, dtype=torch.float32) * 0.1  # fp32 log-decay
    beta = (torch.rand(STEPS, SEQS, H, device=device, dtype=torch.float32) * 0.5 + 0.5).to(torch.bfloat16)
    q_n = torch.nn.functional.normalize(q, dim=-1).to(torch.bfloat16)
    k_n = torch.nn.functional.normalize(k, dim=-1).to(torch.bfloat16)

    state = torch.zeros(SLOTS, H, DV, DK, device=device, dtype=torch.float32)
    indices = torch.tensor([1, 3], device=device, dtype=torch.int32)  # two live slots
    seq_lens = torch.zeros(SEQS, dtype=torch.int64)  # post-step lengths per request
    config = LeapQuantConfig(enabled=True, window=WINDOW, compensator_tokens=4)
    outputs = []

    for t in range(STEPS):
        # T = SEQS tokens (one per sequence), matching len(ssm_state_indices).
        # actual_seq_lengths is the cumulative query_start_loc format
        # [0, len0, len1, ...] used by the GDN metadata builder.
        out = recurrent_gated_delta_rule(
            q_n[t].contiguous(),
            k_n[t].contiguous(),
            v[t].to(torch.bfloat16).contiguous(),
            state,
            g=g[t].contiguous(),
            beta=beta[t].contiguous(),
            scale=DK**-0.5,
            actual_seq_lengths=torch.tensor([0] + [1] * SEQS, device=device, dtype=torch.int32),
            ssm_state_indices=indices,
        )
        outputs.append(out.detach().clone())
        seq_lens += 1
        if flush:
            leapquant_flush_state_rows(state, indices, seq_lens, config=config)
    return torch.stack(outputs), state


class TestLeapQuantGDNIntegration(unittest.TestCase):
    # Hold the reference tensors as class attributes: keeps the allocation
    # alive for the comparison and avoids interleaved NPU-memory reuse.
    reference_output: torch.Tensor
    reference_state: torch.Tensor

    @classmethod
    def setUpClass(cls):
        cls.reference_output, cls.reference_state = _run_trajectory(flush=False)

    def test_flushed_trajectory_stays_close_to_reference(self):
        flushed_output, flushed_state = _run_trajectory(flush=True)
        out_rel = ((flushed_output - self.reference_output).norm() / self.reference_output.norm()).item()
        state_rel = ((flushed_state - self.reference_state).norm() / self.reference_state.norm()).item()
        # 4 flushes at int8+r=4 must keep the 64-token trajectory within a
        # small multiple of the single-flush reconstruction error (bf16 op
        # noise included).
        self.assertLess(out_rel, 0.05, f"output drift too large: {out_rel}")
        self.assertLess(state_rel, 0.05, f"state drift too large: {state_rel}")
        # Untouched slots must remain exactly zero.
        self.assertEqual(int(flushed_state[[0, 2, 4, 5, 6, 7]].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
