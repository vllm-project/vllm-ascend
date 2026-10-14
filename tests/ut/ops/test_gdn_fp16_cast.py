# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


class TestChunkGatedDeltaRuleFusedCast:
    def test_fp16_inputs_are_cast_to_bf16_and_output_cast_back(self):
        # Regression test for #16359: npu_chunk_gated_delta_rule only accepts
        # bfloat16, but runtime q/k/v follow the model dtype (float16). The
        # helper must cast q/k/v/beta to bf16 for the op call and cast the
        # output o back to the original dtype. CPU-only: the op is mocked.
        t, nk, nv, dk = 8, 2, 4, 128
        q = torch.zeros((1, t, nk, dk), dtype=torch.float16)
        k = torch.zeros((1, t, nk, dk), dtype=torch.float16)
        v = torch.zeros((1, t, nv, dk), dtype=torch.float16)
        g = torch.full((1, t, nv), -0.1, dtype=torch.float32)
        beta = torch.full((1, t, nv), 0.5, dtype=torch.float16)
        initial_state = torch.zeros((1, nv, dk, dk), dtype=torch.float32)
        cu_seqlens = torch.tensor([0, t], dtype=torch.int32)

        captured = {}

        def fake_op(q_in, k_in, v_in, *, beta, initial_state, actual_seq_lengths, scale, g):
            captured["q_dtype"] = q_in.dtype
            captured["k_dtype"] = k_in.dtype
            captured["v_dtype"] = v_in.dtype
            captured["beta_dtype"] = beta.dtype
            o = torch.zeros((t, nv, dk), dtype=torch.bfloat16)
            final_state = torch.zeros((1, nv, dk, dk), dtype=torch.bfloat16)
            return o, final_state

        with patch("torch_npu.npu_chunk_gated_delta_rule", side_effect=fake_op):
            o, final_state = AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                scale=dk**-0.5,
            )

        assert captured["q_dtype"] == torch.bfloat16
        assert captured["k_dtype"] == torch.bfloat16
        assert captured["v_dtype"] == torch.bfloat16
        assert captured["beta_dtype"] == torch.bfloat16
        # Output is cast back to the caller's dtype (float16), so it can be
        # concatenated with the recurrent-path output of the same dtype.
        assert o.dtype == torch.float16

    def test_bf16_inputs_pass_through_unchanged(self):
        # bf16 models are unaffected: .to(bfloat16) on a bf16 tensor is a
        # no-op and the output dtype stays bfloat16.
        t, nk, nv, dk = 8, 2, 4, 128
        q = torch.zeros((1, t, nk, dk), dtype=torch.bfloat16)
        k = torch.zeros((1, t, nk, dk), dtype=torch.bfloat16)
        v = torch.zeros((1, t, nv, dk), dtype=torch.bfloat16)
        g = torch.full((1, t, nv), -0.1, dtype=torch.float32)
        beta = torch.full((1, t, nv), 0.5, dtype=torch.bfloat16)
        initial_state = torch.zeros((1, nv, dk, dk), dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, t], dtype=torch.int32)

        def fake_op(q_in, k_in, v_in, *, beta, initial_state, actual_seq_lengths, scale, g):
            o = torch.zeros((t, nv, dk), dtype=torch.bfloat16)
            final_state = torch.zeros((1, nv, dk, dk), dtype=torch.bfloat16)
            return o, final_state

        with patch("torch_npu.npu_chunk_gated_delta_rule", side_effect=fake_op):
            o, _ = AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                scale=dk**-0.5,
            )

        assert o.dtype == torch.bfloat16
