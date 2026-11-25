# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.ops.triton.gdn.split_normalized_qkv import split_normalized_decode_qkv


@pytest.mark.parametrize(
    "tokens,key_heads,value_heads,key_dim,value_dim,stride,dtype",
    [(4, 16, 32, 128, 128, 1, torch.bfloat16), (120, 4, 8, 64, 128, 2, torch.float16)],
)
def test_split_normalized_decode_qkv(tokens, key_heads, value_heads, key_dim, value_dim, stride, dtype):
    sizes = (key_heads * key_dim, key_heads * key_dim, value_heads * value_dim)
    width = sum(sizes)
    data = torch.randn(tokens, 1 + stride * width, device="npu", dtype=dtype)
    packed = data[:, 1::stride]
    packed[0].zero_()
    q, k, v = packed.split(sizes, dim=-1)
    expected = (
        l2norm_fwd(q.contiguous().view(1, tokens, key_heads, key_dim)),
        l2norm_fwd(k.contiguous().view(1, tokens, key_heads, key_dim)),
        v.contiguous().view(1, tokens, value_heads, value_dim),
    )
    actual = split_normalized_decode_qkv(packed, sizes[0], sizes[2], key_dim, value_dim)
    for got, want in zip(actual, expected):
        assert got.is_contiguous()
        torch.testing.assert_close(got, want, rtol=0, atol=0)
