# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.dsv41_a5.quantization import (
    _mxfp4_quantize_e8m0_reference,
)
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import (
    write_mxfp4_indexer_cache,
)


def make_caches():
    raw = torch.full((4 * 140000,), 0x5A, dtype=torch.uint8, device="npu")
    data = torch.as_strided(raw, (4, 128, 1, 64), (140000, 64, 64, 1))
    scale = torch.as_strided(
        raw,
        (4, 128, 1, 4),
        (140000, 4, 4, 1),
        storage_offset=70000,
    )
    return data, scale


@pytest.mark.parametrize("tokens", [0, 1, 6, 129, 4080])
@torch.inference_mode()
def test_write_mxfp4_indexer_cache_matches_reference(tokens):
    torch.manual_seed(41)
    values = torch.randn(tokens, 128, dtype=torch.bfloat16, device="npu")
    slots = torch.full((tokens, 2), -1, dtype=torch.int32, device="npu")
    valid = min(tokens, 128)
    if valid:
        slots[:valid, 0] = torch.arange(valid, device="npu") // 32
        slots[:valid, 1] = torch.arange(valid, device="npu") % 32
    data, scale = make_caches()
    write_mxfp4_indexer_cache(values, slots, data, scale)
    expected_data, expected_scale = _mxfp4_quantize_e8m0_reference(values[:valid])
    if valid:
        actual_data = data[slots[:valid, 0].long(), slots[:valid, 1].long(), 0]
        actual_scale = scale[slots[:valid, 0].long(), slots[:valid, 1].long(), 0]
        torch.testing.assert_close(actual_data.cpu(), expected_data.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(actual_scale.cpu(), expected_scale.cpu(), rtol=0, atol=0)
    assert data[0, 127, 0].eq(0x5A).all()
    assert scale[0, 127, 0].eq(0x5A).all()


@torch.inference_mode()
def test_write_mxfp4_indexer_cache_graph_replay():
    values = torch.randn(6, 128, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor(
        [[0, 0], [0, 1], [1, 4], [2, 8], [3, 31], [-1, -1]],
        dtype=torch.int32,
        device="npu",
    )
    data, scale = make_caches()
    write_mxfp4_indexer_cache(values, slots, data, scale)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        write_mxfp4_indexer_cache(values, slots, data, scale)
    values.mul_(0.125)
    expected_data, expected_scale = _mxfp4_quantize_e8m0_reference(values[:5])
    graph.replay()
    torch.npu.synchronize()
    actual_data = data[slots[:5, 0].long(), slots[:5, 1].long(), 0]
    actual_scale = scale[slots[:5, 0].long(), slots[:5, 1].long(), 0]
    torch.testing.assert_close(actual_data.cpu(), expected_data.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale.cpu(), rtol=0, atol=0)


@torch.inference_mode()
def test_write_mxfp4_indexer_cache_crosses_int32_byte_offset_boundary():
    """Page 16384 at a 128-KiB stride starts exactly at byte 2**31."""
    page_stride = 128 * 1024
    boundary_page = 16384
    num_pages = boundary_page + 1
    raw = torch.empty(num_pages * page_stride, dtype=torch.uint8, device="npu")
    data = torch.as_strided(
        raw,
        (num_pages, 128, 1, 64),
        (page_stride, 64, 64, 1),
    )
    scale = torch.as_strided(
        raw,
        (num_pages, 128, 1, 4),
        (page_stride, 4, 4, 1),
        storage_offset=64 * 1024,
    )
    values = torch.randn(2, 128, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor(
        [[boundary_page - 1, 0], [boundary_page, 0]],
        dtype=torch.int32,
        device="npu",
    )

    write_mxfp4_indexer_cache(values, slots, data, scale)
    torch.npu.synchronize()

    expected_data, expected_scale = _mxfp4_quantize_e8m0_reference(values)
    actual_data = data[slots[:, 0].long(), slots[:, 1].long(), 0]
    actual_scale = scale[slots[:, 0].long(), slots[:, 1].long(), 0]
    torch.testing.assert_close(actual_data.cpu(), expected_data.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale.cpu(), rtol=0, atol=0)
