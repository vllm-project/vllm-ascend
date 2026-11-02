# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for retained packed-cache alias and compaction helpers."""

import importlib.util
from pathlib import Path

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/hisparse_layout.py"
_SPEC = importlib.util.spec_from_file_location("hisparse_layout_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
layout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(layout)


@pytest.mark.parametrize("layer", [0, 1, 3])
def test_raw_alias_preserves_layer_offset_and_page_gaps(layer):
    backing = torch.arange(3 * 4 * 128 * 576, dtype=torch.float32).to(torch.bfloat16).view(3, 4, 128, 576)
    cache = backing[:, layer]
    raw, block, stride, capacity = layout.packed_cache_storage(cache)
    assert raw.data_ptr() == cache.data_ptr()
    assert raw.storage_offset() == cache.storage_offset()
    assert raw.numel() == 2 * backing.stride(0) + 128 * 576
    for row in (0, 127, 128, 255, capacity - 1):
        address = row // block * stride + row % block * 576
        torch.testing.assert_close(raw[address : address + 576], cache[row // block, row % block], rtol=0, atol=0)
    assert raw.storage_offset() + raw.numel() <= backing.numel()


def test_expanded_attention_view_alias_stays_within_tail_storage():
    backing = torch.zeros((4, 6, 128, 576), dtype=torch.bfloat16)
    cache = backing.view(-1, 128, 576)[3:]
    raw, block, stride, capacity = layout.packed_cache_storage(cache)
    assert block == 128 and stride == 128 * 576
    assert raw.numel() == capacity * 576
    assert raw.storage_offset() + raw.numel() == backing.numel()


@pytest.mark.parametrize("bad", ["dtype", "row_stride", "overlap", "empty"])
def test_reject_unsupported_cache_layouts(bad):
    cache = torch.zeros((2, 128, 576), dtype=torch.bfloat16)
    if bad == "dtype":
        cache = cache.float()
    elif bad == "row_stride":
        cache = cache[:, ::2]
    elif bad == "overlap":
        cache = cache.as_strided(cache.shape, (576, 576, 1))
    else:
        cache = cache[:0]
    with pytest.raises(ValueError):
        layout.packed_cache_storage(cache)


def test_compaction_preserves_order_duplicates_and_high_physical_ids():
    indices = torch.full((2, 128), -1, dtype=torch.int32)
    values = [2097151, -1, 0, 262144, 262144, 9, 3, 2097152, -7]
    indices[0, : len(values)] = torch.tensor(values, dtype=torch.int32)
    positions = torch.arange(128, dtype=torch.float32).view(1, -1)
    compact, counts = layout.compact_physical_indices(indices, 2097152, positions)
    assert counts.tolist() == [6, 0]
    assert compact[0, :6].tolist() == [2097151, 0, 262144, 262144, 9, 3]
    assert (compact[0, 6:] == -1).all() and (compact[1] == -1).all()
