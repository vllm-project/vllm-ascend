from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import PytestBase
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule


class TestChunkGatedDeltaRule(PytestBase):
    def test_triton_fusion_ops(self):
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.num_decodes = 1
        mock_forward_context = MagicMock()
        mock_forward_context.attn_metadata = mock_attn_metadata

        # Use logsigmoid for g (forget gate in log space, must be <= 0) and
        # sigmoid for beta (in [0, 1]). Plain randn produces positive g values
        # whose exp() explodes the recurrent state, amplifying bf16/fp32
        # divergences between the AscendC kernel and the fp32 reference.
        q = torch.randn(1, 192, 4, 128, dtype=torch.bfloat16).npu()
        k = torch.randn(1, 192, 4, 128, dtype=torch.bfloat16).npu()
        v = torch.randn(1, 192, 8, 128, dtype=torch.bfloat16).npu()
        g = torch.nn.functional.logsigmoid(torch.randn(1, 192, 8, dtype=torch.float32)).npu()
        beta = torch.rand(1, 192, 8, dtype=torch.bfloat16).sigmoid().npu()
        initial_state = torch.randn(3, 8, 128, 128, dtype=torch.bfloat16).npu()
        # 3 sequences with length 64 each, cumulative sum = 192 == seqlen.
        # Each sequence length >= chunk_size (64) to satisfy the AscendC
        # kernel's memory access contract: the kernel reads chunk_size (64)
        # elements per chunk and does not guard against seq_len < chunk_size.
        # The previous [0, 1, 2, 3] input triggered MTE out-of-bounds reads
        # because the kernel stepped through a full 64-token chunk while only
        # 1 token was allocated, causing an AICore ffttsplus error.
        q_start_loc = torch.tensor([0, 64, 128, 192], dtype=torch.int).npu()

        # Single-rank PCP group: skips the inter-rank state recursion block.
        mock_pcp_group = SimpleNamespace(world_size=1, rank_in_group=0)
        with (
            patch("vllm_ascend.ops.triton.fla.chunk.get_forward_context", return_value=mock_forward_context),
            patch("vllm_ascend.ops.triton.fla.chunk.get_pcp_group", return_value=mock_pcp_group),
        ):
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=q_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )

        assert core_attn_out_non_spec.shape == (1, 192, 8, 128)
        assert last_recurrent_state.shape == (3, 8, 128, 128)

        # Numerical correctness against the PyTorch reference implementation.
        # Shape-only assertions silently pass even when the AscendC kernel
        # crashes on the device (NPU execution is asynchronous), so compare
        # values to make a kernel fault actually fail the test.
        ref_out, ref_state = chunk_gated_delta_rule_pytorch(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=q_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        # Tolerances accommodate bf16 (AscendC) vs fp32 (reference) divergence:
        # the kernel accumulates in bf16 while the reference uses fp32, so a
        # small fraction of elements near chunk boundaries can deviate by
        # ~0.3 in absolute terms. rtol=5e-2 / atol=5e-2 keeps the test
        # meaningful while avoiding false positives from dtype differences.
        torch.testing.assert_close(core_attn_out_non_spec, ref_out, rtol=5e-2, atol=5e-2)
        torch.testing.assert_close(last_recurrent_state, ref_state, rtol=5e-2, atol=5e-2)


def test_chunk_gated_delta_rule_310_state_layout_matches_vllm():
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    k = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    v = torch.tensor([[[[10.0, 20.0, 30.0]]]], dtype=torch.float32)
    g = torch.zeros(1, 1, 1, dtype=torch.float32)
    beta = torch.ones(1, 1, 1, dtype=torch.float32)
    initial_state = torch.tensor(
        [[[[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]]],
        dtype=torch.float32,
    )

    out, final_state = chunk_gated_delta_rule_pytorch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )

    expected_out = torch.tensor([[[[10.0, 20.0, 30.0]]]], dtype=torch.float32) / (2.0**0.5)
    expected_state = torch.tensor(
        [[[[10.0, 2.0], [20.0, 8.0], [30.0, 32.0]]]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    assert final_state is not None
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)
