# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.fla.fused_gdn_prepare import fused_gdn_prepare


@pytest.mark.parametrize("num_tokens", [0, 1, 4, 16, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_gdn_prepare(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    device = "npu"
    num_v_heads = 24
    head_v_dim = 128
    qkv_width = 5120
    z_width = num_v_heads * head_v_dim

    mixed_qkvz = torch.randn(
        (num_tokens, qkv_width + z_width),
        dtype=dtype,
        device=device,
    )
    mixed_ba = torch.randn(
        (num_tokens, 2 * num_v_heads),
        dtype=dtype,
        device=device,
    )

    mixed_qkv, z, b, a, core_attn_out = fused_gdn_prepare(
        mixed_qkvz,
        mixed_ba,
        qkv_width,
        num_v_heads,
        head_v_dim,
    )

    torch.testing.assert_close(mixed_qkv, mixed_qkvz[:, :qkv_width], rtol=0, atol=0)
    torch.testing.assert_close(z, mixed_qkvz[:, qkv_width:].reshape_as(z), rtol=0, atol=0)
    torch.testing.assert_close(b, mixed_ba[:, :num_v_heads], rtol=0, atol=0)
    torch.testing.assert_close(a, mixed_ba[:, num_v_heads:], rtol=0, atol=0)
    torch.testing.assert_close(core_attn_out, torch.zeros_like(core_attn_out), rtol=0, atol=0)

    assert mixed_qkv.is_contiguous()
    assert b.is_contiguous()
    assert a.is_contiguous()
    assert core_attn_out.is_contiguous()
    if num_tokens > 0:
        assert z.untyped_storage().data_ptr() == mixed_qkvz.untyped_storage().data_ptr()
        assert z.storage_offset() == qkv_width
        if num_tokens > 1:
            assert z.stride(0) == mixed_qkvz.stride(0)

    supplied_core_attn_out = torch.full_like(core_attn_out, float("nan"))
    *_, reused_core_attn_out = fused_gdn_prepare(
        mixed_qkvz,
        mixed_ba,
        qkv_width,
        num_v_heads,
        head_v_dim,
        core_attn_out=supplied_core_attn_out,
    )
    assert reused_core_attn_out.data_ptr() == supplied_core_attn_out.data_ptr()
    torch.testing.assert_close(reused_core_attn_out, torch.zeros_like(reused_core_attn_out), rtol=0, atol=0)
