# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices


def reference(selected, positions, compress_ratio):
    visible = ((positions + 1) // compress_ratio).unsqueeze(-1)
    valid = (selected >= 0) & (selected < visible)
    sentinel = torch.iinfo(torch.int32).max
    selected = torch.where(valid, selected, sentinel).sort(dim=-1).values
    return torch.where(selected == sentinel, -1, selected)


@pytest.mark.parametrize("topk", [1, 7, 8, 33, 128, 512, 2047, 2048])
@pytest.mark.parametrize("tokens", [0, 1, 3, 41, 129])
@pytest.mark.parametrize("compress_ratio", [1, 2])
@torch.inference_mode()
def test_prepare_indexer_indices(topk, tokens, compress_ratio):
    torch.manual_seed(41)
    selected = torch.randint(-3, 1000, (tokens, topk), dtype=torch.int32, device="npu")
    positions = torch.randint(-1, 2000, (tokens,), dtype=torch.int64, device="npu")
    original = selected.clone()
    actual = prepare_indexer_indices(selected, positions, compress_ratio)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), compress_ratio), rtol=0, atol=0)
    torch.testing.assert_close(selected, original, rtol=0, atol=0)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_prepare_indexer_indices_boundaries(compress_ratio, position_dtype):
    # Preserve distinct INT32 indices above FP32's exact-integer range, ties,
    # negative sentinels, causal boundaries and all-invalid rows.
    row = torch.tensor([0, -1, -3, 9, 9, 10, 11, 2**24 - 1, 2**24, 2**24 + 1, 2**24 + 2, 2**31 - 2, 2**31 - 1])
    selected = row.int().repeat(5, 1).npu()
    positions = torch.tensor([-1, 0, 19, 2**25 + 3, 2**31 - 1], dtype=position_dtype, device="npu")
    actual = prepare_indexer_indices(selected, positions, compress_ratio)
    torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), compress_ratio), rtol=0, atol=0)


@torch.inference_mode()
def test_prepare_indexer_indices_full_int32_range():
    torch.manual_seed(42)
    selected = torch.randint(0, 2**31 - 1, (41, 2048), dtype=torch.int32, device="npu")
    positions = torch.full((41,), 2**32, dtype=torch.int64, device="npu")
    actual = prepare_indexer_indices(selected, positions, 2)
    torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), 2), rtol=0, atol=0)


@torch.inference_mode()
def test_prepare_indexer_indices_noncontiguous():
    selected = torch.randint(-1, 2000, (41, 256), dtype=torch.int32, device="npu")[:, ::2]
    positions = torch.arange(82, dtype=torch.int64, device="npu")[::2]
    actual = prepare_indexer_indices(selected, positions, 2)
    torch.testing.assert_close(actual.cpu(), reference(selected.cpu(), positions.cpu(), 2), rtol=0, atol=0)


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("tokens", [41, 129])
@torch.inference_mode()
def test_prepare_indexer_indices_graph_replay(compress_ratio, tokens):
    selected = torch.randint(-1, 4096, (tokens, 2048), dtype=torch.int32, device="npu")
    positions = torch.full((tokens,), 4095, dtype=torch.int64, device="npu")
    prepare_indexer_indices(selected, positions, compress_ratio)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = prepare_indexer_indices(selected, positions, compress_ratio)
    pointer = actual.data_ptr()
    for last_position in (0, 127, 8191):
        selected.copy_(torch.randint_like(selected, -1, 4096))
        positions.fill_(last_position)
        graph.replay()
        torch.npu.synchronize()
        assert actual.data_ptr() == pointer
        torch.testing.assert_close(
            actual.cpu(), reference(selected.cpu(), positions.cpu(), compress_ratio), rtol=0, atol=0
        )
