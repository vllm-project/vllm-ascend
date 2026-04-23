import torch
from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule

from vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule import fused_recurrent_gated_delta_rule_pytorch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def test_fused_recurrent_gated_delta_rule_310p_parity_precision():
    init_device_properties_triton()
    torch.manual_seed(0)
    device = "npu"

    bsz = 1
    total_tokens = 9
    num_qk_heads = 2
    num_v_heads = 4
    kdim = 64
    vdim = 48

    q = torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=torch.float16, device=device)
    k = torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=torch.float16, device=device)
    v = torch.randn(bsz, total_tokens, num_v_heads, vdim, dtype=torch.float16, device=device)
    g = torch.randn(bsz, total_tokens, num_v_heads, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(bsz, total_tokens, num_v_heads, dtype=torch.float32, device=device)).to(
        torch.float16
    )

    initial_state = torch.randn(2, num_v_heads, vdim, kdim, dtype=torch.float16, device=device)
    cu_seqlens = torch.tensor([0, 4, 9], dtype=torch.long, device=device)
    # For inplace_final_state=True, Ascend triton kernel expects explicit per-token state indices.
    # seq0 (len=4) -> state 0, seq1 (len=5) -> state 1.
    ssm_state_indices = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
        device=device,
    )

    triton_out, triton_state = fused_recurrent_gated_delta_rule(
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
    ref_out, ref_state = fused_recurrent_gated_delta_rule_pytorch(
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
        triton_out.to(torch.float32).cpu(),
        ref_out.to(torch.float32).cpu(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        triton_state.to(torch.float32).cpu(),
        ref_state.to(torch.float32).cpu(),
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
