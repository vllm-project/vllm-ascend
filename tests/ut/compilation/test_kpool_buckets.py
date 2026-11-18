# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# This policy is host-only; no NPU or serving stack is needed to test bounds.
_PATH = Path(__file__).parents[3] / "vllm_ascend/compilation/kpool_buckets.py"
_SPEC = importlib.util.spec_from_file_location("kpool_buckets", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("length", [None, 0, 1, 512, 2047, 2048, 2049, 8193, 262144, 262208])
def test_bucket_covers_history_and_topk(length):
    capacity = 262208
    bucket = _MODULE.kpool_score_bucket(length, capacity, 512)
    assert max(length or 0, 512) <= bucket <= capacity
    if length is None or length > 262144:
        assert bucket == capacity
    else:
        assert bucket & (bucket - 1) == 0


def test_provider_changes_buckets_without_retaining_previous_length():
    provider = _MODULE.KPoolBucketProvider("layer", 262208, 512)
    for length, expected in [(1024, 2048), (2049, 4096), (262144, 262144), (1, 2048), (None, 262208)]:
        assert provider.resolve({"layer": SimpleNamespace(max_pool_seq_len=length)}) == {"max_pool_seq_len": expected}


@pytest.mark.parametrize("length,capacity,selected", [(-1, 2048, 512), (2049, 2048, 512), (1, 128, 512), (1, 2048, 0)])
def test_invalid_bounds_fail_instead_of_truncating(length, capacity, selected):
    with pytest.raises(ValueError):
        _MODULE.kpool_score_bucket(length, capacity, selected)


def test_small_model_capacity_remains_supported():
    assert _MODULE.kpool_score_bucket(128, 192, 192) == 192


def test_chunks_have_distinct_provider_identity():
    first = _MODULE.KPoolBucketProvider("layer", 262208, 512, token_offset=0)
    second = _MODULE.KPoolBucketProvider("layer", 262208, 512, token_offset=2)
    assert first != second
    context = {"layer": SimpleNamespace(max_pool_seq_len=2049)}
    assert first.resolve(context) == second.resolve(context)
