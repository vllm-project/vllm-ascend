# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU correctness and graph-replay coverage for shared-expert tail padding."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe import shared_experts as shared_experts_module
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts


@pytest.fixture
def pad_cache(monkeypatch):
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=16))
    cache = {}
    monkeypatch.setattr(moe_utils, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(moe_utils, "_PAD_ZERO_BLOCKS", cache)
    return cache


@pytest.mark.parametrize("rows", [1, 8, 15, 16, 17, 31, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_padding_matches_reference_on_npu(pad_cache, rows, dtype):
    value = torch.randn((rows, 7168), device="npu", dtype=dtype)
    padded_rows = ((rows + 15) // 16) * 16
    result = moe_utils._pad_tokens_with_cat(value, padded_rows) if padded_rows != rows else value
    reference = F.pad(value, (0, 0, 0, padded_rows - rows)) if padded_rows != rows else value
    torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=0, atol=0)


def test_cold_graphs_do_not_publish_zero_cache_and_replay_in_reverse(pad_cache):
    first = torch.randn((8, 7168), device="npu", dtype=torch.bfloat16)
    second = torch.randn((17, 7168), device="npu", dtype=torch.bfloat16)
    # Compile/warm the existing fallback without populating the candidate cache.
    F.pad(first, (0, 0, 0, 8))
    F.pad(second, (0, 0, 0, 15))
    torch.npu.synchronize()
    graph_a = torch.npu.NPUGraph()
    with torch.npu.graph(graph_a):
        out_a = moe_utils._pad_tokens_with_cat(first, 16)
    graph_b = torch.npu.NPUGraph()
    with torch.npu.graph(graph_b):
        out_b = moe_utils._pad_tokens_with_cat(second, 32)
    assert not pad_cache
    first.fill_(2)
    second.fill_(3)
    torch.npu.synchronize()
    graph_b.replay()
    graph_a.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out_a.cpu(), F.pad(first, (0, 0, 0, 8)).cpu(), rtol=0, atol=0)
    torch.testing.assert_close(out_b.cpu(), F.pad(second, (0, 0, 0, 15)).cpu(), rtol=0, atol=0)


def test_cold_capture_on_auxiliary_stream_falls_back(pad_cache):
    value = torch.randn((8, 7168), device="npu", dtype=torch.bfloat16)
    F.pad(value, (0, 0, 0, 8))
    torch.npu.synchronize()
    auxiliary = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        capture_stream = torch.npu.current_stream()
        auxiliary.wait_stream(capture_stream)
        with torch.npu.stream(auxiliary):
            result = moe_utils._pad_tokens_with_cat(value, 16)
        capture_stream.wait_stream(auxiliary)
    assert not pad_cache
    value.fill_(5)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), F.pad(value, (0, 0, 0, 8)).cpu(), rtol=0, atol=0)


def test_warm_graphs_reuse_zero_block_with_fresh_inputs(pad_cache):
    first = torch.randn((8, 7168), device="npu", dtype=torch.bfloat16)
    second = torch.randn((17, 7168), device="npu", dtype=torch.bfloat16)
    moe_utils._pad_tokens_with_cat(first, 16)
    moe_utils._pad_tokens_with_cat(second, 32)
    torch.npu.synchronize()
    assert len(pad_cache) == 1
    block = next(iter(pad_cache.values()))
    graph_a = torch.npu.NPUGraph()
    with torch.npu.graph(graph_a):
        out_a = moe_utils._pad_tokens_with_cat(first, 16)
    graph_b = torch.npu.NPUGraph()
    with torch.npu.graph(graph_b):
        out_b = moe_utils._pad_tokens_with_cat(second, 32)
    for iteration in range(3):
        first.fill_(iteration + 1)
        second.fill_(iteration + 4)
        torch.npu.synchronize()
        graph_b.replay()
        graph_a.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out_a.cpu(), F.pad(first, (0, 0, 0, 8)).cpu(), rtol=0, atol=0)
        torch.testing.assert_close(out_b.cpu(), F.pad(second, (0, 0, 0, 15)).cpu(), rtol=0, atol=0)
    assert next(iter(pad_cache.values())) is block


@pytest.mark.parametrize("reverse", [False, True])
def test_zero_block_can_be_reused_after_stream_dependency(pad_cache, reverse):
    value = torch.randn((8, 7168), device="npu", dtype=torch.bfloat16)
    torch.npu.synchronize()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    producer, consumer = streams[::-1] if reverse else streams
    with torch.npu.stream(producer):
        moe_utils._pad_tokens_with_cat(value, 16)
        ready = producer.record_event()
    with torch.npu.stream(consumer):
        consumer.wait_event(ready)
        result = moe_utils._pad_tokens_with_cat(value, 16)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), F.pad(value, (0, 0, 0, 8)).cpu(), rtol=0, atol=0)
    assert len(pad_cache) == 1


def test_shared_tail_warm_graph_uses_group_tp_without_current_config(monkeypatch):
    cache = {}
    monkeypatch.setattr(moe_utils, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(moe_utils, "_PAD_ZERO_BLOCKS", cache)
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_reduce_scatter", lambda value, dim: value)
    shared = SimpleNamespace(moe_config=SimpleNamespace(tp_group=SimpleNamespace(world_size=16)))
    value = torch.randn((8, 7168), device="npu", dtype=torch.bfloat16)
    AscendSharedExperts._pad_and_reduce_scatter(shared, value)
    torch.npu.synchronize()
    assert len(cache) == 1 and next(iter(cache))[0] == 16
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = AscendSharedExperts._pad_and_reduce_scatter(shared, value)
    value.fill_(3)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), F.pad(value, (0, 0, 0, 8)).cpu(), rtol=0, atol=0)


def test_shared_tail_cold_graph_falls_back_without_current_config(monkeypatch):
    cache = {}
    monkeypatch.setattr(moe_utils, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(moe_utils, "_PAD_ZERO_BLOCKS", cache)
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_reduce_scatter", lambda value, dim: value)
    shared = SimpleNamespace(moe_config=SimpleNamespace(tp_group=SimpleNamespace(world_size=16)))
    value = torch.randn((8, 7168), device="npu", dtype=torch.bfloat16)
    F.pad(value, (0, 0, 0, 8))
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = AscendSharedExperts._pad_and_reduce_scatter(shared, value)
    assert not cache
    value.fill_(4)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), F.pad(value, (0, 0, 0, 8)).cpu(), rtol=0, atol=0)
