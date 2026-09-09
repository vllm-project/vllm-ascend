# SPDX-License-Identifier: Apache-2.0
"""CPU address contracts, independent of vLLM/NPU package initialization."""

import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).parents[5] / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/cache_layout.py"
_spec = importlib.util.spec_from_file_location("sfa_offload_cache_layout", _path)
_layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_layout)


def test_source_remapping_preserves_parent_tokens_and_unused_tail():
    parent = torch.arange(3 * 4 * 12, dtype=torch.float32).to(torch.bfloat16).reshape(3, 4, 1, 12)
    k, r = parent[..., :8], parent[..., 8:]
    slots = torch.tensor([0, 1, 3, 4, 11], dtype=torch.int64)
    descriptors = torch.cat((k.data_ptr() + slots * 16, r.data_ptr() + slots * 8, torch.tensor([-1, -1])))
    original_ptr = descriptors.data_ptr()
    _layout.remap_sfa_source_addresses(
        descriptors,
        slots.numel(),
        k_base=k.data_ptr(),
        rope_base=r.data_ptr(),
        k_token_bytes=16,
        rope_token_bytes=8,
    )
    expected = torch.cat((parent.data_ptr() + slots * 24, parent.data_ptr() + slots * 24 + 16))
    assert torch.equal(descriptors[:10], expected)
    assert descriptors[10:].tolist() == [-1, -1]
    assert descriptors.data_ptr() == original_ptr


@pytest.mark.parametrize("width", [8, 4])
def test_paged_addresses_follow_view_stride_across_permuted_slots(width):
    parent = torch.zeros(3, 4, 1, 12, dtype=torch.bfloat16)
    cache = parent[..., :8] if width == 8 else parent[..., 8:]
    slots = torch.tensor([11, 4, 3, 1, 0], dtype=torch.int64)
    addresses = _layout.paged_cache_token_addresses(cache, slots)
    expected = [cache[b // 4, b % 4, 0].data_ptr() for b in slots.tolist()]
    assert addresses.tolist() == expected


def test_paged_addresses_also_preserve_legacy_contiguous_layout():
    cache = torch.zeros(3, 4, 1, 8, dtype=torch.bfloat16)
    slots = torch.tensor([0, 3, 4, 11], dtype=torch.int64)
    assert torch.equal(_layout.paged_cache_token_addresses(cache, slots), cache.data_ptr() + slots * 16)


def test_no_misses_does_not_touch_descriptor_buffer():
    addresses = torch.full((4,), 123, dtype=torch.int64)
    _layout.remap_sfa_source_addresses(addresses, 0, k_base=100, rope_base=116, k_token_bytes=16, rope_token_bytes=8)
    assert addresses.tolist() == [123] * 4


@pytest.mark.parametrize("count", [-1, 3])
def test_remap_rejects_out_of_range_descriptor_count(count):
    with pytest.raises(ValueError):
        _layout.remap_sfa_source_addresses(
            torch.zeros(4, dtype=torch.int64), count, k_base=100, rope_base=116, k_token_bytes=16, rope_token_bytes=8
        )
