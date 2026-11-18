# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise byte ownership on real CPU tensor storages, without NPU imports."""

import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def allocation_modules():
    logger = ModuleType("vllm.logger")
    logger.logger = Mock()
    with patch.dict(sys.modules, {"vllm.logger": logger}):
        allocation = runpy.run_path(str(ROOT / "vllm_ascend/worker/sentinel/kv_cache_region_registry.py"))
        registry = runpy.run_path(str(ROOT / "vllm_ascend/worker/sentinel/kv_cache_region_registry.py"))
    return allocation, registry


def test_uce_crossing_page_boundary_only_zeros_affected_blocks(allocation_modules):
    allocation, registry_mod = allocation_modules
    raw = torch.ones(8 * 4 * 4)
    cache = raw.view(8, 4, 4)
    allocation["register_kv_allocations"]({"layer": raw}, {"layer": cache}, 8)
    registry = registry_mod["KvCacheRegionRegistry"]()
    registry.register({"layer": cache})
    registry.validate_block_space(8)
    registry.uce_ranges = [(raw.data_ptr() + 64, 65)]
    assert registry.partial_block_ids(8) == {1, 2}
    registry.invalidate_blocks([1, 2])
    assert cache[1:3].count_nonzero() == 0
    assert torch.all(cache[0] == 1) and torch.all(cache[3:] == 1)


def test_alias_closure_includes_all_owners_before_zeroing(allocation_modules):
    allocation, registry_mod = allocation_modules
    raw = torch.ones(8, 4)
    shifted = raw.flip(0)  # Independent storage must not be accepted as an alias.
    mapping = allocation["CacheAllocationMap"](8)
    mapping.own(raw)
    with pytest.raises(ValueError, match="allocator-owned"):
        mapping.block_view(shifted, rows_per_block=1)
    # A broadcast page is shared by every logical block.
    broadcast = raw[:1].expand(8, 4)
    mapping.block_view(raw, rows_per_block=1)
    mapping.block_view(broadcast, rows_per_block=1)
    mapping.seal()
    registry = registry_mod["KvCacheRegionRegistry"]()
    registry.register({"normal": raw, "shared": broadcast})
    registry.uce_ranges = [(raw.data_ptr(), 1)]
    assert registry.partial_block_ids(8) == set(range(8))


def test_unknown_allocation_and_out_of_range_uce_fail_closed(allocation_modules):
    allocation, registry_mod = allocation_modules
    registry = registry_mod["KvCacheRegionRegistry"]()
    with pytest.raises(ValueError, match="did not register"):
        registry.register({"layer": torch.ones(8, 4)})
    raw = torch.ones(8, 4)
    allocation["register_kv_allocations"]({"layer": raw}, {"layer": raw}, 8)
    registry.register({"layer": raw})
    assert registry.recover_uce_addresses([{"ptr": raw.data_ptr() - 1, "size": 2}]) is None
    assert not registry.uce_ranges


def test_current_tuple_cache_adapter_preserves_connector_mapping():
    from functools import wraps

    from .test_fault_tolerance_port import load_definitions

    ns = load_definitions(
        "vllm_ascend/patch/worker/patch_attn_utils.py",
        ["RunnerKVCacheView", "with_runner_cache_view", "with_layer_cache_mapping"],
        wraps=wraps,
    )
    mapping = {"layer": (torch.ones(8, 2), torch.ones(8, 2))}
    cache_view = ns["with_runner_cache_view"](lambda: mapping)()
    assert list(cache_view.values()) == list(mapping["layer"])
    connector = Mock()
    ns["with_layer_cache_mapping"](connector)("config", cache_view)
    connector.assert_called_once_with("config", mapping)
