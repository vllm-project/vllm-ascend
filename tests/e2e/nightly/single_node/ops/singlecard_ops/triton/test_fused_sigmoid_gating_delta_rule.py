import torch
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule
from vllm.model_executor.models.qwen3_next import fused_gdn_gating

from vllm_ascend.ops.triton.fla.sigmoid_gating import \
    fused_sigmoid_gating_delta_rule_update


def test_triton_fusion_ops():
    q = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16).npu()
    k = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16).npu()
    v = torch.randn(1, 1, 8, 128, dtype=torch.bfloat16).npu()
    a = torch.tensor([[
        -2.6094, -0.2617, -0.3848, 2.2656, 3.6250, -0.7383, -1.0938, -0.0505
    ]]).bfloat16().npu()
    b = torch.tensor(
        [[0.4277, 0.8906, 1.6875, 2.3750, 4.1562, 0.3809, 1.0625,
          3.6719]]).bfloat16().npu()
    non_spec_state_indices_tensor = torch.tensor([2]).int().npu()
    non_spec_query_start_loc = torch.tensor([0, 1]).int().npu()
    a_log = torch.tensor([
        -2.6875, -3.2031, -3.3438, -2.7812, -3.0625, -4.0312, -5.3750, 5.7188
    ]).bfloat16().npu()
    dt_bias = torch.tensor(
        [-4.7812, -5.0938, -5.5000, 9.4375, 7.6250, -4.3750, -3.0938,
         0.9688]).bfloat16().npu()
    ssm_state1 = torch.ones(1, 8, 128, 128, dtype=torch.bfloat16).npu()

    core_attn_out_non_spec_fused = fused_sigmoid_gating_delta_rule_update(
        A_log=a_log.contiguous(),
        dt_bias=dt_bias.contiguous(),
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        a=a.contiguous(),
        b=b.contiguous(),
        initial_state_source=ssm_state1,
        initial_state_indices=non_spec_state_indices_tensor,
        cu_seqlens=non_spec_query_start_loc,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )

    ssm_state2 = torch.ones(1, 8, 128, 128, dtype=torch.bfloat16).npu()
    g, beta = fused_gdn_gating(a_log, a, b, dt_bias)
    g_non_spec = g
    beta_non_spec = beta
    core_attn_out_non_spec_split, last_recurrent_state = (
        fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g_non_spec,
            beta=beta_non_spec,
            initial_state=ssm_state2,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc,
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
        ))
    torch.testing.assert_close(core_attn_out_non_spec_fused,
                               core_attn_out_non_spec_split,
                               rtol=1e-02,
                               atol=1e-02,
                               equal_nan=True)
