# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU numerical coverage for folded SFA projections around actual TQ4 attention."""

import pytest
import torch
import torch_npu

from vllm_ascend.attention.sfa_v1 import _fold_turboquant_projection_weights
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import TurboQuantLatent


@pytest.mark.parametrize("query_tokens", [1, 83, 256])
@pytest.mark.parametrize("page", [128, 256])
@torch.inference_mode()
def test_folded_projections_with_tq4_attention(query_tokens, page):
    init_device_properties_triton()
    torch.manual_seed(103)
    heads, context = 64, 2048
    transform = TurboQuantLatent()
    latent = torch.randn(context, 512, device="npu", dtype=torch.bfloat16)
    packed = transform.compress(latent).reshape(context, 258)
    scales = packed[:, 256:].contiguous().view(torch.float16)
    rope = torch.randn(context, 64, device="npu", dtype=torch.bfloat16)
    rope = (rope.float() / (scales.float() + 1e-20)).bfloat16().contiguous()
    cache = torch.cat((packed[:, :256], rope.view(torch.uint8), packed[:, 256:]), dim=-1)
    cache = cache.view(context // page, page, 1, 386).view(torch.int8)
    uk = (torch.randn(heads, 192, 512, device="npu") / 192**0.5).bfloat16()
    uv = (torch.randn(heads, 512, 256, device="npu") / 512**0.5).bfloat16()
    folded_uk, folded_uv = _fold_turboquant_projection_weights(uk, uv, transform.rotation)
    q = torch.randn(query_tokens, heads, 192, device="npu", dtype=torch.bfloat16)
    qr = torch.randn(query_tokens, heads, 64, device="npu", dtype=torch.bfloat16)
    indices = torch.arange(context, dtype=torch.int32, device="npu").repeat(query_tokens, 1)
    causal_end = context - query_tokens + torch.arange(query_tokens, device="npu") + 1
    indices.masked_fill_(indices >= causal_end[:, None], -1)
    block_table = torch.arange(context // page, dtype=torch.int32, device="npu").view(1, -1)
    qlens = torch.tensor([query_tokens], dtype=torch.int32, device="npu")
    klens = torch.tensor([context], dtype=torch.int32, device="npu")

    def project(x, weight):
        return torch_npu.npu_transpose_batchmatmul(
            x, weight, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
        )

    def attend(q_nope):
        result = torch_npu.npu_kv_quant_sparse_flash_attention(
            torch.cat((q_nope, qr), dim=-1),
            cache,
            cache,
            indices.unsqueeze(1),
            block_table=block_table,
            actual_seq_lengths_query=qlens,
            actual_seq_lengths_kv=klens,
            scale_value=576**-0.5,
            key_quant_mode=3,
            value_quant_mode=3,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=64,
        )
        return result[0] if isinstance(result, tuple) else result

    expected = project(transform.inverse(attend(transform.forward(project(q, uk)))), uv)
    actual = project(attend(project(q, folded_uk)), folded_uv)
    expected, actual = expected.float().cpu(), actual.float().cpu()
    assert torch.isfinite(actual).all()
    # This bounds one-layer BF16 perturbation; it is not a model-quality gate.
    assert ((actual - expected).norm() / expected.norm()).item() < 0.015
