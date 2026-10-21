# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

adapter = importlib.import_module("vllm_ascend.ops.fused_moe.shared_dequant_swiglu_quant")


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 7, 8, 9, 64])
@pytest.mark.parametrize("width", [256, 384, 4096, 6144])
@pytest.mark.parametrize("limit", [1.0, 2.0, 10.0, 100.0])
@pytest.mark.parametrize("use_cache", [False, True])
def test_shared_adapter_dispatch_and_parameters(monkeypatch, num_tokens, width, limit, use_cache):
    x = torch.arange(num_tokens * width, dtype=torch.int32).reshape(num_tokens, width)
    weight_scale = torch.arange(width, dtype=torch.float32) + 1
    activation_scale = torch.arange(num_tokens, dtype=torch.float32) + 1
    original_x, original_weight = x.clone(), weight_scale.clone()
    output = torch.ones((num_tokens, width // 2), dtype=torch.int8)
    scale = torch.ones(num_tokens, dtype=torch.float32)
    operator = MagicMock(return_value=(output, scale))
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(official_shared_dequant_swiglu_quant=operator))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    cache = adapter.SharedSwiGLUCache() if use_cache else None

    actual, actual_scale = adapter.shared_dequant_swiglu_quant(x, weight_scale, activation_scale, limit, cache=cache)

    assert actual.shape == (num_tokens, width // 2)
    assert actual_scale.shape == (num_tokens,)
    assert actual.dtype == torch.int8
    assert actual_scale.dtype == torch.float32
    assert torch.equal(x, original_x)
    assert torch.equal(weight_scale, original_weight)
    if num_tokens == 0:
        operator.assert_not_called()
        return

    args = operator.call_args.args
    assert args[0] is x and args[2] is activation_scale
    assert args[3] == limit
    if use_cache:
        assert torch.equal(args[1][0, 0::2], weight_scale[: width // 2])
        assert torch.equal(args[1][0, 1::2], weight_scale[width // 2 :])
        assert args[4][max(num_tokens, 8)] == max(num_tokens, 8)
        assert args[4].dtype == torch.int64 and args[4].device == x.device
        assert args[5] is True
    else:
        assert args[1] is weight_scale and len(args) == 4
    assert actual is output and actual_scale is scale


@pytest.fixture
def eager_cache(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    return adapter.SharedSwiGLUCache()


def test_cache_reuses_constants_and_tracks_weight_updates(eager_cache):
    weight = torch.arange(256, dtype=torch.float32)
    prepared, groups = eager_cache.get_parameters(weight, 1)
    again, same_groups = eager_cache.get_parameters(weight, 64)
    assert again is prepared and same_groups is groups
    weight.add_(1)
    changed, _ = eager_cache.get_parameters(weight, 1)
    assert changed is not prepared
    assert torch.equal(changed[0, 0::2], weight[:128])
    weight.data = weight.clone().add_(2)
    replaced, _ = eager_cache.get_parameters(weight, 1)
    assert replaced is not changed


def test_cache_retains_old_graph_ranges_when_capacity_grows(eager_cache):
    weight = torch.ones(256)
    _, old = eager_cache.get_parameters(weight, 64)
    _, new = eager_cache.get_parameters(weight, 9000)
    assert old is not new and new[9000] == 9000
    assert any(t is old for t in eager_cache._retired_groups)
    assert old[64] == 64


def test_cold_capture_falls_back_without_mutating_cache(monkeypatch, eager_cache):
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    assert eager_cache.get_parameters(torch.ones(256), 64) is None
    assert eager_cache._source is None and eager_cache._groups is None


def test_warm_capture_reuses_cache(monkeypatch, eager_cache):
    weight = torch.ones(256)
    first = eager_cache.get_parameters(weight, 64)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    second = eager_cache.get_parameters(weight, 64)
    assert first[0] is second[0] and first[1] is second[1]
    assert eager_cache.get_parameters(weight, 9000) is None


def test_inference_weight_can_be_explicitly_invalidated(eager_cache):
    with torch.inference_mode():
        weight = torch.ones(256)
        original, _ = eager_cache.get_parameters(weight, 1)
        weight.add_(1)
        eager_cache.invalidate()
        changed, _ = eager_cache.get_parameters(weight, 1)
    assert changed is not original and torch.all(changed == 2)
