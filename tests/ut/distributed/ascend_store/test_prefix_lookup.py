# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefix_lookup import readable_prefix_ends
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.restore_planner import CacheSlice


def ref(start, end, *, state=False):
    cache = CacheSlice("layer", "0", "mamba-v2:{}" if state else "kv", start, end, 0, 1, 4, 4, 4)
    return SimpleNamespace(slices=(SimpleNamespace(cache=cache),))


def ready(refs, flags, *, window=None, start=0):
    return readable_prefix_ends(
        refs, flags, {"layer": SimpleNamespace(sliding_window=window)}, start=start, end=12, granularity=4
    )


def test_attention_hole_stops_reuse_even_if_later_objects_exist():
    assert ready([ref(0, 4), ref(4, 8), ref(8, 12)], [True, False, True]) == [4]


def test_state_checkpoint_reachability_is_not_assumed_monotonic():
    refs = [ref(0, 12), ref(3, 4, state=True), ref(7, 8, state=True), ref(11, 12, state=True)]
    assert ready(refs, [True, False, True, False]) == [8]


def test_window_expiry_makes_an_old_missing_page_irrelevant():
    assert ready([ref(0, 4), ref(4, 8), ref(8, 12)], [False, True, True], window=4) == [8, 12]


def test_hbm_prefix_does_not_require_reloading_earlier_pool_pages():
    assert ready([ref(0, 4), ref(4, 12)], [False, True], start=4) == [8, 12]


@pytest.mark.parametrize("flags", [[], [1], [None], [True, False]])
def test_invalid_readability_cannot_become_a_prefix_hit(flags):
    with pytest.raises(ValueError, match="not aligned"):
        ready([ref(0, 4)], flags)
