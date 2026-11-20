# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


@pytest.fixture(params=[False, True], ids=["ND", "NZ"])
def use_nz(request):
    was_enabled = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
    torch.npu.config.allow_internal_format = request.param
    try:
        yield request.param
    finally:
        torch.npu.config.allow_internal_format = was_enabled


@pytest.mark.skipif(not hasattr(torch_npu, "npu_transpose_batchmatmul"), reason="fused transpose bmm unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 16])
@torch.inference_mode()
def test_sfa_q_projection_eager_and_graph(dtype, use_nz, tokens):
    # GLM-5.1 with TP16: keep the strided NoPE slice produced by the Q/RoPE split.
    heads, nope, rope, latent = 4, 192, 64, 512
    projected = torch.randn(tokens, heads * (nope + rope), device="npu", dtype=dtype)
    weight = torch.randn(heads, nope, latent, device="npu", dtype=dtype) * (nope**-0.5)
    if use_nz:
        weight = torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
        assert torch_npu.get_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ

    layer = SimpleNamespace(
        q_proj=lambda x: (x, None),
        local_num_heads=heads,
        qk_head_dim=nope + rope,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        W_UK_T=weight,
    )

    def check(outputs):
        q_nope, q_pe = projected.view(tokens, heads, nope + rope).split([nope, rope], dim=-1)
        reference = torch.bmm(q_nope.transpose(0, 1), weight).transpose(0, 1)
        assert outputs[0].is_contiguous()
        torch.testing.assert_close(outputs[0], reference, atol=0.03, rtol=0.02)
        torch.testing.assert_close(outputs[1], q_pe, atol=0, rtol=0)

    with patch.object(torch_npu, "npu_transpose_batchmatmul", wraps=torch_npu.npu_transpose_batchmatmul) as fused:
        check(AscendSFAImpl._q_proj_and_k_up_proj(layer, projected))
        fused.assert_called_once()
        fused.reset_mock()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = AscendSFAImpl._q_proj_and_k_up_proj(layer, projected)
        fused.assert_called_once()
    projected.add_(0.5)
    graph.replay()
    check(outputs)
