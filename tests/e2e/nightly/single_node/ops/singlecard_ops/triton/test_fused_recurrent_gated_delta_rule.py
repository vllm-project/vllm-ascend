# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule import fused_recurrent_gated_delta_rule_pytorch


def _run_ascendc_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    use_qk_l2norm_in_kernel,
):
    """Run the AscendC custom op npu_recurrent_gated_delta_rule as a reference.

    Adapts the fla-style inputs (batch-first) to the op's calling convention:
    3D tensors, BF16 q/k/v/beta, FP32 g, external l2norm, per-sequence
    actual_seq_lengths with a leading 0, and a flat per-token ssm_state_indices
    array. The op updates the state tensor in place.
    """
    # The op is BF16-only for q/k/v/beta and FP32-only for g.
    q_a = q.squeeze(0).to(torch.bfloat16)
    k_a = k.squeeze(0).to(torch.bfloat16)
    v_a = v.squeeze(0).to(torch.bfloat16)
    g_a = g.squeeze(0)  # already float32
    beta_a = beta.squeeze(0).to(torch.bfloat16)
    state_a = initial_state.to(torch.bfloat16).clone()

    # The op does not apply l2norm internally; mirror the production path in
    # gdn.py, which calls l2norm_fwd before dispatching to the op. l2norm_fwd
    # matches the in-kernel math used by fused_recurrent_gated_delta_rule_pytorch.
    if use_qk_l2norm_in_kernel:
        q_a = l2norm_fwd(q_a)
        k_a = l2norm_fwd(k_a)

    # fla cu_seqlens is cumulative ([0, 4, 9]); the op expects per-sequence
    # lengths with a leading 0 ([0, 4, 5]).
    seq_lengths = torch.diff(cu_seqlens.cpu()).tolist()
    actual_seq_lengths = torch.tensor([0] + seq_lengths, dtype=torch.int32, device=q.device)

    # fla ssm_state_indices is [n_seq, T]; the op expects a flat per-token
    # array. This test uses one constant slot per sequence, so expand the
    # first column by each sequence's length.
    flat_state_indices = (
        torch.repeat_interleave(ssm_state_indices[:, 0], torch.tensor(seq_lengths, device=ssm_state_indices.device))
        .to(torch.int32)
        .contiguous()
    )

    out_a = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=q_a,
        key=k_a,
        value=v_a,
        g=g_a,
        beta=beta_a,
        state=state_a,
        scale=k_a.shape[-1] ** -0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=flat_state_indices,
    ).unsqueeze(0)
    return out_a, state_a


@pytest.mark.skip("Probabilistic failure, need zengtian after fix")
def test_fused_recurrent_gated_delta_rule_310p_parity_precision():
    torch.manual_seed(0)
    device = "npu"

    bsz = 1
    total_tokens = 9
    num_qk_heads = 2
    num_v_heads = 4
    kdim = 64
    vdim = 48

    # The AscendC reference op is BF16-only, so the parity inputs are generated
    # in BF16 (both sides accumulate in fp32 and round the state to the same dtype).
    dtype = torch.bfloat16

    q = torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=dtype, device=device)
    k = torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=dtype, device=device)
    v = torch.randn(bsz, total_tokens, num_v_heads, vdim, dtype=dtype, device=device)
    g = torch.randn(bsz, total_tokens, num_v_heads, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(bsz, total_tokens, num_v_heads, dtype=torch.float32, device=device)).to(dtype)

    initial_state = torch.randn(2, num_v_heads, vdim, kdim, dtype=dtype, device=device)
    cu_seqlens = torch.tensor([0, 4, 9], dtype=torch.long, device=device)
    # For inplace_final_state=True, the Ascend triton kernel expects explicit per-token state indices.
    # seq0 (len=4) -> state 0, seq1 (len=5) -> state 1.
    ssm_state_indices = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
        device=device,
    )

    ref_out, ref_state = _run_ascendc_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    py_out, py_state = fused_recurrent_gated_delta_rule_pytorch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(
        ref_out.to(torch.float32).cpu(),
        py_out.to(torch.float32).cpu(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        ref_state.to(torch.float32).cpu(),
        py_state.to(torch.float32).cpu(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
    )


def test_fused_recurrent_gated_delta_rule_310_state_layout_matches_vllm():
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    k = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    v = torch.tensor([[[[10.0, 20.0, 30.0]]]], dtype=torch.float32)
    g = torch.zeros(1, 1, 1, dtype=torch.float32)
    beta = torch.ones(1, 1, 1, dtype=torch.float32)
    initial_state = torch.tensor(
        [[[[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]]],
        dtype=torch.float32,
    )

    out, final_state = fused_recurrent_gated_delta_rule_pytorch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        inplace_final_state=False,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    )

    expected_out = torch.tensor([[[[10.0, 20.0, 30.0]]]], dtype=torch.float32) / (2.0**0.5)
    expected_state = torch.tensor(
        [[[[10.0, 2.0], [20.0, 8.0], [30.0, 32.0]]]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)
