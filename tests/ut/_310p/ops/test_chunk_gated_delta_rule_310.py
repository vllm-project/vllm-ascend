import torch

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch


def test_chunk_gated_delta_rule_310_output_shape_and_dtype():
    torch.manual_seed(0)

    bsz = 2
    total_tokens = 7
    num_qk_heads = 2
    num_v_heads = 4
    kdim = 16
    vdim = 12

    q = torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=torch.float16)
    k = torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=torch.float16)
    v = torch.randn(bsz, total_tokens, num_v_heads, vdim, dtype=torch.float16)
    g = -0.2 * torch.rand(bsz, total_tokens, num_v_heads, dtype=torch.float32)
    beta = (0.15 + 0.35 * torch.rand(bsz, total_tokens, num_v_heads, dtype=torch.float32)).to(torch.float16)
    initial_state = torch.randn(bsz, num_v_heads, kdim, vdim, dtype=torch.float16)

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
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == v.shape
    assert out.dtype == v.dtype
    assert final_state is not None
    assert final_state.shape == initial_state.shape
    assert final_state.dtype == torch.float32


def test_chunk_gated_delta_rule_310_varlen_path():
    torch.manual_seed(0)

    bsz = 1
    total_tokens = 9
    num_qk_heads = 2
    num_v_heads = 4
    kdim = 16
    vdim = 12

    q = torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=torch.float16)
    k = torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=torch.float16)
    v = torch.randn(bsz, total_tokens, num_v_heads, vdim, dtype=torch.float16)
    g = -0.2 * torch.rand(bsz, total_tokens, num_v_heads, dtype=torch.float32)
    beta = (0.15 + 0.35 * torch.rand(bsz, total_tokens, num_v_heads, dtype=torch.float32)).to(torch.float16)
    cu_seqlens = torch.tensor([0, 4, 9], dtype=torch.long)
    initial_state = torch.randn(2, num_v_heads, kdim, vdim, dtype=torch.float16)

    out, final_state = chunk_gated_delta_rule_pytorch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )

    assert out.shape == v.shape
    assert final_state is not None
    assert final_state.shape == initial_state.shape
