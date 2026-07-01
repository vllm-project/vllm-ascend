import gc

import torch

from tests.ut.base import PytestBase
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule


class TestChunkGatedDeltaRule(PytestBase):
    def test_triton_fusion_ops(self, mock_moe_env):
        q = torch.randn(1, 17, 4, 128, dtype=torch.bfloat16).npu()
        k = torch.randn(1, 17, 4, 128, dtype=torch.bfloat16).npu()
        v = torch.randn(1, 17, 8, 128, dtype=torch.bfloat16).npu()
        g = torch.randn(1, 17, 8, dtype=torch.float32).npu()
        beta = torch.randn(1, 17, 8, dtype=torch.bfloat16).npu()
        initial_state = torch.randn(3, 8, 128, 128, dtype=torch.bfloat16).npu()
        q_start_loc = torch.range(0, 3, dtype=torch.int).npu()

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

        assert core_attn_out_non_spec.shape == (1, 17, 8, 128)
        assert last_recurrent_state.shape == (3, 8, 128, 128)
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()


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
