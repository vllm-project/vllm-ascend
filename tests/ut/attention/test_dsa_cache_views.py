# SPDX-License-Identifier: Apache-2.0
"""CPU byte-oracle tests for DeepSeek V4 backend component views."""

import pytest
import torch

from vllm_ascend.attention.utils import combine_cache_components
from vllm_ascend.worker.v2.attn_utils import create_runtime_component_views


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("padding", [0, 64])
def test_indexer_views_preserve_page_components_and_alias(scale_dtype, combined, padding):
    blocks, heads, tokens, width = 3, 1, 4, 8
    data_bytes = tokens * heads * width
    scale_bytes = tokens * heads * scale_dtype.itemsize
    payload = data_bytes + scale_bytes
    stride = payload + padding
    span = (blocks - 1) * stride + payload
    offset = 64
    backing = torch.zeros(offset + span + 64, dtype=torch.int8)
    raw = backing[offset : offset + span]
    views = create_runtime_component_views(
        raw,
        ((blocks, heads, tokens, width), (blocks, heads, tokens, scale_dtype.itemsize)),
        (torch.int8, scale_dtype),
        payload,
        stride,
    )
    data, scale = views[:2]
    assert data.shape == (blocks, tokens, heads, width)
    assert scale.shape == (blocks, tokens, heads, 1)
    assert data.stride(0) == stride
    assert scale.stride(0) == stride // scale_dtype.itemsize
    assert scale.data_ptr() - data.data_ptr() == data_bytes
    data[1].fill_(3)
    scale[2].fill_(5)
    expected = torch.zeros_like(backing)
    expected[offset + stride : offset + stride + data_bytes].fill_(3)
    start = offset + 2 * stride + data_bytes
    expected[start : start + scale_bytes].view(scale_dtype).fill_(5)
    torch.testing.assert_close(backing, expected)
    if combined:
        packed = combine_cache_components(data, scale)
        assert packed.shape == (blocks, tokens, heads, width + scale_dtype.itemsize)
        assert packed.data_ptr() == data.data_ptr()
        torch.testing.assert_close(packed[1].flatten()[:data_bytes], data[1].flatten())
        packed[0].fill_(7)
        torch.testing.assert_close(backing[offset : offset + payload], torch.full((payload,), 7, dtype=torch.int8))


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 512), (torch.uint8, 258), (torch.float32, 32)])
def test_single_component_retains_storage_tokens_and_strided_pages(dtype, width):
    blocks, tokens, heads = 2, 4, 1
    payload = tokens * heads * width * dtype.itemsize
    stride = 2 * payload
    backing = torch.zeros(64 + stride + payload, dtype=torch.int8)
    raw = backing[64:]
    (cache,) = create_runtime_component_views(
        raw,
        ((blocks, heads, tokens, width * dtype.itemsize),),
        (dtype,),
        payload,
        stride,
    )
    assert cache.shape == (blocks, tokens, heads, width)
    assert cache.stride(0) == stride // dtype.itemsize
    cache[1].fill_(2)
    expected = torch.zeros_like(backing)
    expected[64 + stride :].view(dtype).fill_(2)
    torch.testing.assert_close(backing, expected)
