from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import PytestBase
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule


class TestChunkGatedDeltaRule(PytestBase):
    @pytest.mark.parametrize("seq_lens", [(1, 8, 8), (17,), (1, 64, 65)])
    def test_triton_fusion_ops(self, seq_lens):
        torch.manual_seed(42)
        mock_attn_metadata = MagicMock()
        mock_attn_metadata.num_decodes = 1
        mock_forward_context = MagicMock()
        mock_forward_context.attn_metadata = mock_attn_metadata

        num_tokens = sum(seq_lens)
        q = torch.randn(1, num_tokens, 4, 128, dtype=torch.bfloat16).npu()
        k = torch.randn_like(q)
        v = torch.randn(1, num_tokens, 8, 128, dtype=torch.bfloat16).npu()
        g = torch.nn.functional.logsigmoid(torch.randn(1, num_tokens, 8, dtype=torch.float32)).npu()
        beta = torch.rand(1, num_tokens, 8, dtype=torch.bfloat16).npu()
        initial_state = (torch.randn(len(seq_lens), 8, 128, 128, dtype=torch.bfloat16) * 0.1).npu()
        # Packed sequence boundaries must cover every input token. Each sequence
        # rounds up its chunk count independently, including partial chunks.
        q_start_loc = torch.tensor([0, *seq_lens], dtype=torch.int64, device="npu").cumsum(0)

        with (
            patch("vllm_ascend.ops.triton.fla.chunk.get_forward_context", return_value=mock_forward_context),
            patch("vllm_ascend.ops.triton.fla.chunk.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
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

        assert core_attn_out_non_spec.shape == (1, num_tokens, 8, 128)
        assert last_recurrent_state.shape == (len(seq_lens), 8, 128, 128)
        # Normalize independently on CPU: the 310P reference's optional L2 norm
        # uses npu_rms_norm, whereas the recurrence itself supports CPU tensors.
        q_cpu, k_cpu = q.cpu().float(), k.cpu().float()
        q_cpu = (q_cpu * torch.rsqrt(q_cpu.square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        k_cpu = (k_cpu * torch.rsqrt(k_cpu.square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
        expected_out, expected_state = chunk_gated_delta_rule_pytorch(
            q=q_cpu,
            k=k_cpu,
            v=v.cpu(),
            g=g.cpu(),
            beta=beta.cpu(),
            # The PyTorch reference follows vLLM's [V, K] state layout.
            initial_state=initial_state.cpu().transpose(-1, -2).contiguous(),
            output_final_state=True,
            cu_seqlens=q_start_loc.cpu(),
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        torch.testing.assert_close(core_attn_out_non_spec.cpu().float(), expected_out.float(), rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(
            last_recurrent_state.cpu().float(), expected_state.transpose(-1, -2).float(), rtol=1e-2, atol=1e-2
        )


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
