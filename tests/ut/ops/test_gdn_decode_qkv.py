# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.gdn import _rearrange_decode_qkv


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


@pytest.mark.parametrize("tokens", [1, 4, 144])
@pytest.mark.parametrize("padded_rows", [False, True])
def test_native_decode_qkv_preserves_strided_storage(tokens, padded_rows):
    key_dim, value_dim = 16 * 128, 32 * 128
    width = 2 * key_dim + value_dim
    backing = torch.arange((2 * tokens + 3) * (width + 128), dtype=torch.float32).to(torch.bfloat16)
    backing = backing.reshape(2 * tokens + 3, width + 128)
    packed = backing[1 : 1 + tokens * 2 : 2, :width] if padded_rows else backing[1 : 1 + tokens, :width]
    parts = packed.split((key_dim, key_dim, value_dim), dim=-1)
    with patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("native QKV must remain views")):
        actual = _rearrange_decode_qkv(packed, key_dim, value_dim, 128, 128, allow_strided=True)
    for got, source, heads in zip(actual, parts, (16, 16, 32)):
        torch.testing.assert_close(got, source.reshape(1, tokens, heads, 128), rtol=0, atol=0)
        assert got.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
        assert got.storage_offset() == source.storage_offset()
        assert got.stride()[-2:] == (128, 1)
        if tokens > 1:
            assert got.stride(-3) == packed.stride(0)


@pytest.mark.parametrize("case", ["disabled", "inner_stride", "transposed", "fp16", "head_dim", "empty"])
def test_native_decode_qkv_retains_contiguous_fallback(case):
    tokens, head_dim = (0 if case == "empty" else 4), (64 if case == "head_dim" else 128)
    key_dim, value_dim = 2 * head_dim, 4 * head_dim
    width = 2 * key_dim + value_dim
    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    packed = torch.arange(tokens * width, dtype=torch.float32).reshape(tokens, width).to(dtype)
    if case == "inner_stride":
        packed = packed.repeat_interleave(2, dim=-1)[:, ::2]
    elif case == "transposed":
        packed = packed.t().contiguous().t()
    actual = _rearrange_decode_qkv(packed, key_dim, value_dim, head_dim, head_dim, allow_strided=case != "disabled")
    for got, source, heads in zip(actual, packed.split((key_dim, key_dim, value_dim), -1), (2, 2, 4)):
        torch.testing.assert_close(got, source.reshape(1, tokens, heads, head_dim), rtol=0, atol=0)
        assert got.is_contiguous()
