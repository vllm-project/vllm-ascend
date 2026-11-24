# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.dspark_decode_inputs import (
    DSparkDecodeIndexCache,
    DSparkTargetAttention,
    supports_dspark_decode_reuse,
)


def _config():
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark", enable_adaptive_verification=False),
        model_config=SimpleNamespace(architecture="MiniMaxM3SparseForCausalLM"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
    )


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("speculative_config", "method", "eagle3"),
        ("speculative_config", "enable_adaptive_verification", True),
        ("model_config", "architecture", "OtherForCausalLM"),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
    ],
)
def test_reuse_config_guards(owner, field, value):
    config = _config()
    assert supports_dspark_decode_reuse(config, sparse_offload_enabled=False)
    setattr(getattr(config, owner), field, value)
    assert not supports_dspark_decode_reuse(config, sparse_offload_enabled=False)


def test_reuse_requires_speculation_and_resident_kv():
    config = _config()
    assert not supports_dspark_decode_reuse(config, sparse_offload_enabled=True)
    config.speculative_config = None
    assert not supports_dspark_decode_reuse(config, sparse_offload_enabled=False)


def test_target_groups_preserve_kv_indices_and_capture_ownership():
    def group(draft, name):
        return SimpleNamespace(kv_cache_spec=SimpleNamespace(is_draft=draft), layer_names=[name])

    target = group(False, "target")
    indexer = group(False, "indexer")
    draft = group(True, "draft")
    groups = [[draft], [target, draft], [], [indexer]]
    selected = DSparkTargetAttention(groups)
    assert selected.select(groups) == [[], [target], [], [indexer]]
    assert groups == [[draft], [target, draft], [], [indexer]]
    replacement = [[draft], [target]]
    assert selected.select(replacement) is replacement


@pytest.mark.parametrize("size", [0, 1, 2, 4, 16])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_unchanged_indices_skip_copies_and_keep_old_storage(size, dtype):
    cache = DSparkDecodeIndexCache()
    copy = Mock(side_effect=lambda values: torch.from_numpy(values.copy()))
    values = np.arange(size, dtype=dtype)
    original = cache.copy("indices", values, copy)
    assert cache.copy("indices", values.copy(), copy) is original
    assert copy.call_count == 1
    expected = original.clone()
    if size:
        values[0] += 42
    else:
        values = np.array([42], dtype=dtype)
    changed = cache.copy("indices", values, copy)
    assert changed is not original
    torch.testing.assert_close(original, expected)
    torch.testing.assert_close(changed, torch.from_numpy(values))
    cache.clear()
    assert cache.copy("indices", values, copy) is not changed


def test_copy_key_covers_shape_dtype_and_full_values():
    cache = DSparkDecodeIndexCache()
    copy = Mock(side_effect=lambda values: torch.from_numpy(values.copy()))
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    previous = cache.copy("indices", values, copy)
    for changed in (values.reshape(2, 2), values.astype(np.int64), values[::-1]):
        result = cache.copy("indices", changed, copy)
        assert result is not previous
        torch.testing.assert_close(result, torch.from_numpy(changed.copy()))
        previous = result
    assert copy.call_count == 4


@pytest.mark.parametrize("size", [1, 2, 4, 16])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_expansion_matches_reference_and_reordered_requests(size, width):
    cache = DSparkDecodeIndexCache()
    copy = lambda values: torch.from_numpy(values.copy())

    def expand(indices, total, boundaries, query_len):
        assert query_len == width
        counts = torch.diff(boundaries).to(torch.int64)
        expanded = torch.repeat_interleave(indices, counts)
        positions = torch.cat([torch.arange(int(n), dtype=torch.int32) for n in counts])
        assert len(expanded) == total
        return expanded, positions

    expander = Mock(side_effect=expand)

    def prepare(values, boundaries):
        indices = cache.copy("indices", values, copy)
        cumulative = cache.copy("logits", boundaries, copy)
        return cache.expand(indices, int(boundaries[-1]), cumulative, width, expander)

    values = np.arange(size, dtype=np.int32) + 10
    boundaries = np.arange(size + 1, dtype=np.int32) * width
    result = prepare(values, boundaries)
    assert prepare(values.copy(), boundaries.copy()) is result
    assert expander.call_count == 1
    expected = np.repeat(values, width)
    np.testing.assert_array_equal(result[0], expected)
    np.testing.assert_array_equal(result[1], np.tile(np.arange(width), size))
    reordered = values[::-1].copy() + 1
    changed = prepare(reordered, boundaries)
    assert changed is not result
    np.testing.assert_array_equal(changed[0], np.repeat(reordered, width))
    np.testing.assert_array_equal(result[0], expected)
    shorter = boundaries.copy()
    shorter[-1] -= 1
    assert prepare(reordered, shorter) is not changed
    assert expander.call_count == 3


def test_expansion_key_includes_query_width_and_total_logits():
    cache = DSparkDecodeIndexCache()
    copy = lambda values: torch.from_numpy(values.copy())
    indices = cache.copy("indices", np.array([3], np.int32), copy)
    logits = cache.copy("logits", np.array([0, 8], np.int32), copy)
    expand = Mock(side_effect=lambda *args: (torch.tensor([1]), torch.tensor([0])))
    first = cache.expand(indices, 8, logits, 8, expand)
    assert cache.expand(indices, 8, logits, 8, expand) is first
    assert cache.expand(indices, 8, logits, 4, expand) is not first
    cache.expand(indices, 7, logits, 4, expand)
    assert expand.call_count == 3
