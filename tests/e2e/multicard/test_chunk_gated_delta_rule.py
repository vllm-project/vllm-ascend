import torch

from tests.ut.base import PytestBase
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
        ) = chunk_gated_delta_rule(q=q,
                                   k=k,
                                   v=v,
                                   g=g,
                                   beta=beta,
                                   initial_state=initial_state,
                                   output_final_state=True,
                                   cu_seqlens=q_start_loc,
                                   head_first=False,
                                   use_qk_l2norm_in_kernel=True)

        assert core_attn_out_non_spec.shape == (1, 17, 8, 128)
        assert last_recurrent_state.shape == (3, 8, 128, 128)
