# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.gdn import _rearrange_decode_qkv


@pytest.mark.parametrize("tokens", [0, 1, 4, 17])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_qkv_matches_concat_layout(tokens, strided, dtype):
    key_dim, value_dim, head_k_dim, head_v_dim = 32, 48, 16, 8
    width = 2 * key_dim + value_dim
    packed = torch.arange(tokens * width, dtype=torch.float32).reshape(tokens, width).to(dtype)
    if strided:
        backing = torch.empty(tokens, width * 2, dtype=dtype)
        backing[:, ::2] = packed
        packed = backing[:, ::2]
    parts = packed.split((key_dim, key_dim, value_dim), dim=-1)
    # Preserve the old concat layout as the numerical reference.
    fused = torch.cat([x.reshape(-1) for x in parts])
    q_size = tokens * key_dim
    expected = (
        fused[:q_size].view(1, tokens, 2, head_k_dim),
        fused[q_size : 2 * q_size].view(1, tokens, 2, head_k_dim),
        fused[2 * q_size :].view(1, tokens, 6, head_v_dim),
    )
    with patch("torch.cat", side_effect=AssertionError("decode must not concatenate")):
        actual = _rearrange_decode_qkv(packed, key_dim, value_dim, head_k_dim, head_v_dim)
    for got, want, source in zip(actual, expected, parts):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        assert got.is_contiguous()
        if tokens == 1 and not strided:
            assert got.data_ptr() == source.data_ptr()
