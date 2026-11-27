# SPDX-License-Identifier: Apache-2.0
"""Exact block-selection and buffer-contract checks for fused decode ordering."""

import pytest
import torch
import torch_npu
from vllm.triton_utils import triton

# Initialize the operator registry before importing the model package, as the
# service does. Importing the model first can create a DeviceOperator cycle.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.device.device_config import check_ascend_device_type, is_950
from vllm_ascend.ops.triton.msa.msa_m3_index_topk import (
    _decode_index_postprocess_sorted,
)


@pytest.fixture(scope="module", autouse=True)
def require_a5():
    if not torch.npu.is_available():
        pytest.skip("MiniMax M3 A5 decode ordering requires Ascend 950")
    check_ascend_device_type()
    if not is_950():
        pytest.skip("MiniMax M3 A5 decode ordering requires Ascend 950")


def _inputs(heads, query_len, topk, lengths):
    lengths = torch.tensor(lengths, dtype=torch.int64)
    tokens = lengths.numel() * query_len
    visible = ((lengths[:, None] - query_len + torch.arange(query_len) + 128) // 128).clamp_min(0)
    width = max(2064, triton.next_power_of_2(topk))
    scores = torch.randn(heads, tokens, width)
    scores.masked_fill_(torch.arange(width)[None, None, :] >= visible.reshape(1, tokens, 1), -torch.inf)
    raw = scores.topk(topk, dim=-1).indices
    slots = torch.arange(topk)
    visible = visible.reshape(1, tokens, 1)
    valid = (slots < visible.clamp_max(topk)) & (raw >= 0) & (raw < visible)
    counts = valid.sum(-1).to(torch.int32)
    keys = torch.where(valid, raw.to(torch.float32), torch.inf)
    ordered = keys.sort(-1).values
    expected = torch.where(torch.isinf(ordered), -1, ordered).to(torch.int32)
    return raw, lengths, expected, counts


def _launch(raw, lengths, output, counts, query_len):
    _decode_index_postprocess_sorted(raw, lengths, output, counts, query_len)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("query_len", [1, 4, 8])
@pytest.mark.parametrize("topk", [1, 3, 16, 31, 32, 64])
@pytest.mark.parametrize("strided", [False, True])
def test_decode_sorted_postprocess_exact_selection(heads, query_len, topk, strided):
    raw, lengths, expected, expected_counts = _inputs(
        heads, query_len, topk, [0, 1, 127, 128, 129, 256, 65536, 131072, 263000]
    )
    raw, lengths = raw.npu(), lengths.npu()
    tokens = raw.shape[1]
    backing = torch.full((heads, tokens + 3, topk * (2 if strided else 1)), -99, device="npu", dtype=torch.int32)
    output = backing[:, :tokens, ::2] if strided else backing[:, :tokens]
    count_backing = torch.full((heads, tokens * 2), -99, device="npu", dtype=torch.int32)
    counts = count_backing[:, ::2]
    if strided:
        raw_backing = torch.empty((heads, tokens, topk * 2), device="npu", dtype=raw.dtype)
        raw_backing[..., ::2].copy_(raw)
        raw = raw_backing[..., ::2]
        seq_backing = torch.empty(lengths.numel() * 2, device="npu", dtype=lengths.dtype)
        seq_backing[::2].copy_(lengths)
        lengths = seq_backing[::2]
    raw_before, lengths_before = raw.clone(), lengths.clone()
    metadata = (raw.dtype, raw.stride(), raw.data_ptr(), lengths.dtype, lengths.stride(), lengths.data_ptr())
    _launch(raw, lengths, output, counts, query_len)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(raw, raw_before, rtol=0, atol=0)
    torch.testing.assert_close(lengths, lengths_before, rtol=0, atol=0)
    assert (raw.dtype, raw.stride(), raw.data_ptr(), lengths.dtype, lengths.stride(), lengths.data_ptr()) == metadata
    assert torch.all(backing[:, tokens:] == -99)
    assert torch.all(count_backing[:, 1::2] == -99)
    if strided:
        assert torch.all(backing[..., 1::2] == -99)


def test_decode_sorted_postprocess_graph_replay_updates_inputs():
    raw_cpu, lengths_cpu, _, _ = _inputs(1, 4, 16, [0, 129, 65536, 263000])
    raw, lengths = raw_cpu.npu(), lengths_cpu.npu()
    output = torch.empty_like(raw, dtype=torch.int32)
    counts = torch.empty(raw.shape[:2], device="npu", dtype=torch.int32)
    for _ in range(2):
        _launch(raw, lengths, output, counts, 4)
    torch_npu.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _launch(raw, lengths, output, counts, 4)
    for changed in [[263000, 0, 128, 257], [0, 0, 0, 0], [65536] * 4]:
        new_raw, new_lengths, expected, expected_counts = _inputs(1, 4, 16, changed)
        raw.copy_(new_raw)
        lengths.copy_(new_lengths)
        graph.replay()
        torch_npu.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)


@pytest.mark.parametrize("raw_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_decode_sorted_postprocess_integer_input_types(raw_dtype, length_dtype):
    raw, lengths, expected, expected_counts = _inputs(1, 8, 16, [0, 131072])
    raw, lengths = raw.to(raw_dtype).npu(), lengths.to(length_dtype).npu()
    raw_before, lengths_before = raw.clone(), lengths.clone()
    metadata = (raw.dtype, raw.stride(), raw.data_ptr(), lengths.dtype, lengths.stride(), lengths.data_ptr())
    output = torch.empty(raw.shape, device="npu", dtype=torch.int32)
    counts = torch.empty(raw.shape[:2], device="npu", dtype=torch.int32)
    _launch(raw, lengths, output, counts, 8)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(raw, raw_before, rtol=0, atol=0)
    torch.testing.assert_close(lengths, lengths_before, rtol=0, atol=0)
    assert (raw.dtype, raw.stride(), raw.data_ptr(), lengths.dtype, lengths.stride(), lengths.data_ptr()) == metadata


def test_decode_sorted_postprocess_empty_batch_keeps_output_guards():
    raw = torch.empty((1, 0, 16), device="npu", dtype=torch.int64)
    lengths = torch.empty((0,), device="npu", dtype=torch.int32)
    output = torch.full((1, 2, 16), -99, device="npu", dtype=torch.int32)
    counts = torch.full((1, 2), -99, device="npu", dtype=torch.int32)
    _launch(raw, lengths, output, counts, 8)
    assert torch.all(output == -99)
    assert torch.all(counts == -99)
