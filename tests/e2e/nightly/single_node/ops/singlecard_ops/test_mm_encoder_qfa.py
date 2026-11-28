# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 numerical check for non-causal, variable-length encoder QuantFlashAttn."""

import pytest
import torch

from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention


@pytest.mark.parametrize("lengths", [(67, 61), (128,)])
@pytest.mark.parametrize("dim", [72, 128])
def test_mm_encoder_qfa_matches_bf16_attention(lengths, dim):
    device = torch.device("npu")
    torch.manual_seed(42)
    heads = 4
    total = sum(lengths)
    query = torch.randn(total, heads, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)

    layer = AscendMMEncoderAttention.__new__(AscendMMEncoderAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads = heads
    layer.num_kv_heads = heads
    layer.head_size = dim
    layer.scale = dim**-0.5
    layer.enable_pad = 64 < dim < 128
    actual = layer._forward_eager_qfa(
        query,
        key,
        value,
        cu_seqlens=cu,
        is_reshaped=True,
        bsz=1,
        q_len=total,
    ).reshape(total, heads, dim)

    expected = []
    start = 0
    for length in lengths:
        q = query[start : start + length].float().transpose(0, 1)
        k = key[start : start + length].float().transpose(0, 1)
        v = value[start : start + length].float().transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * layer.scale
        expected.append(torch.matmul(scores.softmax(dim=-1), v).transpose(0, 1))
        start += length
    reference = torch.cat(expected).cpu()
    error = (actual.float().cpu() - reference).abs()
    assert error.mean() < 0.02
    assert error.max() < 0.25
