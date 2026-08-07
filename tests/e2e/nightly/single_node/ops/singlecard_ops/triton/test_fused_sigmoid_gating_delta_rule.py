import gc

import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_sigmoid_gating_delta_rule_update
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch


def test_triton_fusion_ops():
    q = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16).npu()
    k = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16).npu()
    v = torch.randn(1, 1, 8, 128, dtype=torch.bfloat16).npu()
    a = torch.tensor([[-2.6094, -0.2617, -0.3848, 2.2656, 3.6250, -0.7383, -1.0938, -0.0505]]).bfloat16().npu()
    b = torch.tensor([[0.4277, 0.8906, 1.6875, 2.3750, 4.1562, 0.3809, 1.0625, 3.6719]]).bfloat16().npu()
    non_spec_state_indices_tensor = torch.tensor([2]).int().npu()
    non_spec_query_start_loc = torch.tensor([0, 1]).int().npu()
    a_log = torch.tensor([-2.6875, -3.2031, -3.3438, -2.7812, -3.0625, -4.0312, -5.3750, 5.7188]).bfloat16().npu()
    dt_bias = torch.tensor([-4.7812, -5.0938, -5.5000, 9.4375, 7.6250, -4.3750, -3.0938, 0.9688]).bfloat16().npu()
    # The state slot 2 is referenced by non_spec_state_indices_tensor, so the
    # state buffer must have at least 3 slots (a 1-slot buffer would be an
    # out-of-bounds access for the AscendC custom op).
    ssm_state1 = torch.ones(3, 8, 128, 128, dtype=torch.bfloat16).npu()

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

    # Reference: the AscendC custom op npu_recurrent_gated_delta_rule, which is
    # what GatedDeltaNet attention dispatches to on the decode path. It expects
    # 3D tensors, BF16 q/k/v/beta and FP32 g, and requires l2norm to be applied
    # externally (mirroring the production path in gdn.py).
    ssm_state2 = torch.ones(3, 8, 128, 128, dtype=torch.bfloat16).npu()
    g, beta = fused_gdn_gating_patch(a_log, a, b, dt_bias)
    # non_spec_query_start_loc is cumulative cu_seqlens ([0, len0, len0+len1, ...]);
    # the AscendC op expects per-sequence lengths with a leading 0
    # ([0, len0, len1, ...]), so convert it with torch.diff.
    seq_lengths = torch.diff(non_spec_query_start_loc.cpu()).tolist()
    actual_seq_lengths = torch.tensor([0] + seq_lengths, dtype=torch.int32, device=q.device)
    core_attn_out_non_spec_split = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=l2norm_fwd(q.squeeze(0)),
        key=l2norm_fwd(k.squeeze(0)),
        value=v.squeeze(0),
        g=g.squeeze(0),
        beta=beta.squeeze(0),
        state=ssm_state2,
        scale=q.shape[-1] ** -0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=non_spec_state_indices_tensor,
    ).unsqueeze(0)
    torch.testing.assert_close(
        core_attn_out_non_spec_fused, core_attn_out_non_spec_split, rtol=1e-02, atol=1e-02, equal_nan=True
    )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
