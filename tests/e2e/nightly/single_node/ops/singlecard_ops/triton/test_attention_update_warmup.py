# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""NPU cache reuse and correctness after attention-update JIT warmup."""

import pytest
import torch
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from vllm_ascend.ops.triton import attention_update as au


def _cache_keys():
    device = torch.npu.current_device()
    return set(au._prepare_attention_update_kernel.device_caches[device][0])


@pytest.fixture(scope="module")
def warmed_keys():
    registry = JitWarmupRegistry(None)
    with registry.activate():
        au._PREPARE_ATTENTION_UPDATE_KERNEL.register_warmup()
    registry.warmup()
    return _cache_keys()


def _inputs(count, rows, has_anchor):
    generator = torch.Generator().manual_seed(19012)
    lses = [torch.randn(rows, generator=generator) * 5 + 1000 for _ in range(count)]
    previous = torch.randn(rows, generator=generator) + 1000 if has_anchor else None
    if has_anchor:
        lses[0] = torch.randn(rows, generator=generator)
    if rows > 1:
        lses[-1][0] = -torch.inf
        lses[-1][1] = torch.inf
    return [lse.npu() for lse in lses], previous.npu() if previous is not None else None


def _check_result(lses, previous, result):
    scores = torch.stack([lse.cpu().double() for lse in lses])
    candidates = scores.clone()
    if previous is not None:
        candidates[0] = previous.cpu().double()
    expected_anchor = candidates.masked_fill(torch.isinf(candidates), -torch.inf).amax(0)
    expected_relative = scores - expected_anchor
    if previous is not None:
        expected_relative[0] = scores[0] + previous.cpu().double() - expected_anchor
    relative, anchor = result
    torch.testing.assert_close(anchor.cpu().double(), expected_anchor, atol=0, rtol=0)
    torch.testing.assert_close(torch.stack(relative).cpu().double(), expected_relative, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("count,has_anchor", [(16, False), *[(count, True) for count in range(2, 17)]])
@pytest.mark.parametrize("rows", [1, 17, 32, 129, 1025, 77288])
def test_warmup_covers_runtime_rows(warmed_keys, count, has_anchor, rows):
    lses, previous = _inputs(count, rows, has_anchor)
    _check_result(lses, previous, au.prepare_attention_update(lses, previous))
    assert _cache_keys() == warmed_keys


@pytest.mark.parametrize("count,has_anchor", [(16, False), (16, True), (2, True), (15, True)])
def test_warmup_covers_graph_replay(warmed_keys, count, has_anchor):
    lses, previous = _inputs(count, 129, has_anchor)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = au.prepare_attention_update(lses, previous)
    for _ in range(2):
        for lse in lses:
            lse.add_(0.25)
        if previous is not None:
            previous.add_(0.5)
        graph.replay()
        torch.npu.synchronize()
        _check_result(lses, previous, result)
        assert _cache_keys() == warmed_keys
