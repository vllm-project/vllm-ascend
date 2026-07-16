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

        q = torch.randn(1, 4, 17, 128, dtype=torch.bfloat16).npu()
        k = torch.randn(1, 4, 17, 128, dtype=torch.bfloat16).npu()
        v = torch.randn(1, 8, 17, 128, dtype=torch.bfloat16).npu()
        g = torch.randn(1, 17, 8, dtype=torch.float32).npu()
        beta = torch.randn(1, 17, 8, dtype=torch.bfloat16).npu()
        initial_state = torch.randn(3, 8, 128, 128, dtype=torch.bfloat16).npu()
        q_start_loc = torch.range(0, 3, dtype=torch.int).npu()

        mock_pcp_group = MagicMock()
        mock_pcp_group.world_size = 1
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
                use_qk_l2norm_in_kernel=True,
            )

        assert core_attn_out_non_spec.shape == (1, 17, 8, 128)
        assert last_recurrent_state.shape == (3, 8, 128, 128)


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


def test_chunk_gated_delta_rule_head_first_varlen():
    mock_attn_metadata = MagicMock()
    mock_attn_metadata.num_decodes = 0
    mock_forward_context = MagicMock()
    mock_forward_context.attn_metadata = mock_attn_metadata
    mock_pcp_group = MagicMock()
    mock_pcp_group.world_size = 1

    torch.manual_seed(0)
    q = torch.randn(1, 17, 4, 128, dtype=torch.bfloat16).npu()
    k = torch.randn(1, 17, 4, 128, dtype=torch.bfloat16).npu()
    v = torch.randn(1, 17, 8, 128, dtype=torch.bfloat16).npu()
    g = torch.randn(1, 17, 8, dtype=torch.float32).npu()
    beta = torch.randn(1, 17, 8, dtype=torch.bfloat16).npu()
    initial_state = torch.randn(2, 8, 128, 128, dtype=torch.bfloat16).npu()
    cu_seqlens = torch.tensor([0, 1, 17], dtype=torch.int64).npu()

    with (
        patch("vllm_ascend.ops.triton.fla.chunk.get_forward_context", return_value=mock_forward_context),
        patch("vllm_ascend.ops.triton.fla.chunk.get_pcp_group", return_value=mock_pcp_group),
    ):
        head_major_out, head_major_state = chunk_gated_delta_rule(
            q=q.movedim(1, 2).contiguous(),
            k=k.movedim(1, 2).contiguous(),
            v=v.movedim(1, 2).contiguous(),
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    assert head_major_out.shape == (1, 17, 8, 128)
    assert head_major_state.shape == (2, 8, 128, 128)
    assert torch.isfinite(head_major_out).all()
    assert torch.isfinite(head_major_state).all()


def test_chunk_gated_delta_rule_head_first_varlen_matches_reference():
    mock_attn_metadata = MagicMock()
    mock_attn_metadata.num_decodes = 0
    mock_forward_context = MagicMock()
    mock_forward_context.attn_metadata = mock_attn_metadata
    mock_pcp_group = MagicMock()
    mock_pcp_group.world_size = 1

    torch.manual_seed(11)
    tokens, qk_heads, value_heads, head_dim = 97, 4, 8, 128
    cu_seqlens = torch.tensor([0, 1, 17, tokens], dtype=torch.int64).npu()
    q_bthd = torch.randn(1, tokens, qk_heads, head_dim, dtype=torch.bfloat16).npu()
    k_bthd = torch.randn(1, tokens, qk_heads, head_dim, dtype=torch.bfloat16).npu()
    v_bthd = torch.randn(1, tokens, value_heads, head_dim, dtype=torch.bfloat16).npu()
    g = -torch.rand(1, tokens, value_heads, dtype=torch.float32).npu()
    beta = torch.rand(1, tokens, value_heads, dtype=torch.bfloat16).npu()
    initial_state_kv = torch.randn(3, value_heads, head_dim, head_dim, dtype=torch.bfloat16).npu()

    with (
        patch("vllm_ascend.ops.triton.fla.chunk.get_forward_context", return_value=mock_forward_context),
        patch("vllm_ascend.ops.triton.fla.chunk.get_pcp_group", return_value=mock_pcp_group),
    ):
        actual_out, actual_state_kv = chunk_gated_delta_rule(
            q=q_bthd.movedim(1, 2).contiguous(),
            k=k_bthd.movedim(1, 2).contiguous(),
            v=v_bthd.movedim(1, 2).contiguous(),
            g=g,
            beta=beta,
            initial_state=initial_state_kv,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    expected_out, expected_state_vk = chunk_gated_delta_rule_pytorch(
        q=q_bthd,
        k=k_bthd,
        v=v_bthd,
        g=g,
        beta=beta,
        initial_state=initial_state_kv.transpose(-1, -2).contiguous(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=3e-2, atol=1e-2)
    torch.testing.assert_close(
        actual_state_kv,
        expected_state_vk.transpose(-1, -2).to(actual_state_kv.dtype),
        rtol=3e-2,
        atol=1e-2,
    )
