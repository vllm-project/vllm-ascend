from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_pytorch
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def test_chunk_gated_delta_rule_310p_parity_precision():
    init_device_properties_triton()
    torch.manual_seed(0)
    device = "npu"

    bsz = 1
    total_tokens = 9
    num_qk_heads = 2
    num_v_heads = 4
    kdim = 128
    vdim = 128

    q = (0.25 * torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=torch.float32, device=device)).to(
        torch.float16
    )
    k = (0.25 * torch.randn(bsz, total_tokens, num_qk_heads, kdim, dtype=torch.float32, device=device)).to(
        torch.float16
    )
    v = (0.25 * torch.randn(bsz, total_tokens, num_v_heads, vdim, dtype=torch.float32, device=device)).to(
        torch.float16
    )
    g = -0.2 * torch.rand(bsz, total_tokens, num_v_heads, dtype=torch.float32, device=device)
    beta = (0.15 + 0.35 * torch.rand(bsz, total_tokens, num_v_heads, dtype=torch.float32, device=device)).to(
        torch.float16
    )

    initial_state = (0.05 * torch.randn(2, num_v_heads, kdim, vdim, dtype=torch.float32, device=device)).to(
        torch.float16
    )
    cu_seqlens = torch.tensor([0, 4, 9], dtype=torch.long, device=device)

    q = l2norm_fwd(q)
    k = l2norm_fwd(k)

    with (
        patch(
            "vllm_ascend.ops.triton.fla.chunk.get_forward_context",
            return_value=SimpleNamespace(attn_metadata=None),
        ),
        patch(
            "vllm_ascend.ops.triton.fla.chunk.get_pcp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
    ):
        triton_out, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )

    ref_out, _ = chunk_gated_delta_rule_pytorch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )

    torch.testing.assert_close(
        triton_out.to(torch.float32).cpu(),
        ref_out.to(torch.float32).cpu(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
    )
