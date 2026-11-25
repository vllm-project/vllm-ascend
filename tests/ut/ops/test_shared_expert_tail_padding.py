# SPDX-License-Identifier: Apache-2.0
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.nn import functional as F

from vllm_ascend.ops.fused_moe import moe_utils as moe_utils_module
from vllm_ascend.ops.fused_moe import shared_experts as shared_experts_module
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts


@pytest.mark.parametrize("num_tokens", [0, 1, 8, 15, 16, 17, 31, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_shared_expert_tail_padding_matches_reference(monkeypatch, num_tokens, dtype, noncontiguous):
    tp_size = 16
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp_size))
    monkeypatch.setattr(moe_utils_module, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(moe_utils_module, "_PAD_ZERO_BLOCKS", {})
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=tp_size))
    reduce_scatter = MagicMock(side_effect=lambda value, dim: value)
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_reduce_scatter", reduce_scatter)

    source = torch.arange(max(num_tokens, 1) * 12, dtype=torch.float32).reshape(-1, 12).to(dtype)
    value = source[:num_tokens, ::2] if noncontiguous else source[:num_tokens, :6].contiguous()
    original = value.clone()
    pad_rows = (tp_size - num_tokens % tp_size) % tp_size
    expected = F.pad(value, (0, 0, 0, pad_rows)) if pad_rows else value
    cat = MagicMock(wraps=torch.cat)
    monkeypatch.setattr(moe_utils_module.torch, "cat", cat)

    result = shared_experts._pad_and_reduce_scatter(value)
    reduce_scatter.assert_called_once_with(result, dim=0)
    assert cat.call_count == (1 if pad_rows else 0)
    assert result.shape == (num_tokens + pad_rows, 6)
    if not pad_rows:
        assert result is value
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    if pad_rows:
        assert len(moe_utils_module._PAD_ZERO_BLOCKS) == 1
        cached = next(iter(moe_utils_module._PAD_ZERO_BLOCKS.values()))
        changed = value + 1
        changed_expected = F.pad(changed, (0, 0, 0, pad_rows))
        changed_result = shared_experts._pad_and_reduce_scatter(changed)
        assert next(iter(moe_utils_module._PAD_ZERO_BLOCKS.values())) is cached
        assert cat.call_count == 2
        torch.testing.assert_close(changed_result, changed_expected, rtol=0, atol=0)


def test_padding_zero_cache_keeps_distinct_tp_sizes(monkeypatch):
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2))
    monkeypatch.setattr(moe_utils_module, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(moe_utils_module, "_PAD_ZERO_BLOCKS", {})
    value = torch.tensor([[1.0, 2.0, 3.0]])
    first = moe_utils_module._pad_tokens_with_cat(value, 2)
    first_block = next(iter(moe_utils_module._PAD_ZERO_BLOCKS.values()))
    config.parallel_config.tensor_parallel_size = 16
    second = moe_utils_module._pad_tokens_with_cat(value, 16)
    config.parallel_config.tensor_parallel_size = 2
    third = moe_utils_module._pad_tokens_with_cat(value, 2)
    assert len(moe_utils_module._PAD_ZERO_BLOCKS) == 2
    assert any(block is first_block for block in moe_utils_module._PAD_ZERO_BLOCKS.values())
    torch.testing.assert_close(first, F.pad(value, (0, 0, 0, 1)), rtol=0, atol=0)
    torch.testing.assert_close(second, F.pad(value, (0, 0, 0, 15)), rtol=0, atol=0)
    torch.testing.assert_close(third, first, rtol=0, atol=0)


@pytest.mark.parametrize("capturing", [False, True])
def test_padding_zero_cache_detects_npu_graph_capture(monkeypatch, capturing):
    capture_probe = MagicMock(return_value=capturing)
    monkeypatch.setattr(moe_utils_module.torch.npu, "is_current_stream_capturing", capture_probe)
    npu_tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
    cpu_tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    assert moe_utils_module._pad_zero_cache_can_initialize(npu_tensor) is (not capturing)
    capture_probe.assert_called_once_with()
    assert moe_utils_module._pad_zero_cache_can_initialize(cpu_tensor)
    capture_probe.assert_called_once_with()


def test_padding_cold_graph_capture_falls_back_without_publishing(monkeypatch):
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=16))
    monkeypatch.setattr(moe_utils_module, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(moe_utils_module, "_PAD_ZERO_BLOCKS", {})
    monkeypatch.setattr(moe_utils_module, "_pad_zero_cache_can_initialize", lambda value: False)
    value = torch.tensor([[float("nan"), float("inf"), -float("inf")]])
    expected = F.pad(value, (0, 0, 0, 15))
    cat = MagicMock(wraps=torch.cat)
    monkeypatch.setattr(moe_utils_module.torch, "cat", cat)
    result = moe_utils_module._pad_tokens_with_cat(value, 16)
    cat.assert_not_called()
    assert not moe_utils_module._PAD_ZERO_BLOCKS
    torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("fallback", ["cold_capture", "capacity"])
def test_padding_fallback_always_pads_token_dimension(monkeypatch, shape, fallback):
    tp_size = 16 if fallback == "cold_capture" else 2
    padded_rows = 16 if fallback == "cold_capture" else 5
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp_size))
    monkeypatch.setattr(moe_utils_module, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(moe_utils_module, "_PAD_ZERO_BLOCKS", {})
    monkeypatch.setattr(moe_utils_module, "_pad_zero_cache_can_initialize", lambda value: False)
    value = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    expected = torch.zeros((padded_rows, *shape[1:]), dtype=value.dtype)
    expected[: shape[0]] = value
    result = moe_utils_module._pad_tokens_with_cat(value, padded_rows)
    assert result.shape == expected.shape
    assert not moe_utils_module._PAD_ZERO_BLOCKS
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [8, 15, 16, 17])
@pytest.mark.parametrize("global_tp", [None, 2])
def test_shared_expert_padding_uses_tp_group_when_global_tp_differs(monkeypatch, num_tokens, global_tp):
    config = (
        None if global_tp is None else SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=global_tp))
    )
    monkeypatch.setattr(moe_utils_module, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(moe_utils_module, "_PAD_ZERO_BLOCKS", {})
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=16))
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_reduce_scatter", lambda value, dim: value)
    value = torch.arange(num_tokens * 3, dtype=torch.float32).reshape(num_tokens, 3)
    pad_rows = (16 - num_tokens % 16) % 16
    expected = F.pad(value, (0, 0, 0, pad_rows)) if pad_rows else value
    cat = MagicMock(wraps=torch.cat)
    monkeypatch.setattr(moe_utils_module.torch, "cat", cat)
    result = shared_experts._pad_and_reduce_scatter(value)
    assert result.shape == expected.shape
    assert cat.call_count == (1 if pad_rows else 0)
    assert [key[0] for key in moe_utils_module._PAD_ZERO_BLOCKS] == ([16] if pad_rows else [])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_shared_expert_cold_graph_capture_still_falls_back_without_config(monkeypatch):
    monkeypatch.setattr(moe_utils_module, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(moe_utils_module, "_PAD_ZERO_BLOCKS", {})
    monkeypatch.setattr(moe_utils_module, "_pad_zero_cache_can_initialize", lambda value: False)
    monkeypatch.setattr(shared_experts_module, "tensor_model_parallel_reduce_scatter", lambda value, dim: value)
    shared_experts = AscendSharedExperts.__new__(AscendSharedExperts)
    shared_experts.moe_config = SimpleNamespace(tp_group=SimpleNamespace(world_size=16))
    value = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    cat = MagicMock(wraps=torch.cat)
    monkeypatch.setattr(moe_utils_module.torch, "cat", cat)

    result = shared_experts._pad_and_reduce_scatter(value)

    cat.assert_not_called()
    assert not moe_utils_module._PAD_ZERO_BLOCKS
    torch.testing.assert_close(result, F.pad(value, (0, 0, 0, 8)), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_padding_cat_preserves_nonfinite_values(monkeypatch, dtype):
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=16))
    monkeypatch.setattr(moe_utils_module, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(moe_utils_module, "_PAD_ZERO_BLOCKS", {})
    value = torch.tensor([[float("nan"), float("inf"), -float("inf")]], dtype=dtype)
    result = moe_utils_module._pad_tokens_with_cat(value, 16)
    expected = F.pad(value, (0, 0, 0, 15))
    torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.count_nonzero(result[1:]) == 0
