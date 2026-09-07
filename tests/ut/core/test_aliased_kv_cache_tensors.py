# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm_ascend.core.kv_cache_interface import aliased_kv_cache_tensors


@dataclass
class _NewStyleTensor:
    """A placement into the shared backing allocation, as vLLM emits today."""

    size: int
    layers: list[str]
    layer_stride: int
    block_stride: int
    offset: int = 0


@dataclass
class _OldStyleTensor:
    size: int
    shared_by: list[str]


def _config(num_blocks, tensors):
    return SimpleNamespace(num_blocks=num_blocks, kv_cache_tensors=tensors)


def test_old_style_config_is_returned_untouched():
    tensors = [_OldStyleTensor(size=128, shared_by=["a", "b"])]

    assert aliased_kv_cache_tensors(_config(4, tensors)) is tensors


def test_empty_config_is_returned_untouched():
    tensors: list[_OldStyleTensor] = []

    assert aliased_kv_cache_tensors(_config(4, tensors)) is tensors


def test_one_layer_per_entry_becomes_one_allocation_each():
    num_blocks, page = 8, 512
    tensors = [
        _NewStyleTensor(
            size=2 * page * num_blocks,
            layers=[name],
            layer_stride=page * num_blocks,
            block_stride=page,
            offset=index * page * num_blocks,
        )
        for index, name in enumerate(("layer.0", "layer.1"))
    ]

    placements = aliased_kv_cache_tensors(_config(num_blocks, tensors))

    assert [p.shared_by for p in placements] == [["layer.0"], ["layer.1"]]
    # ``size`` on the entry is the whole backing allocation; each layer only
    # owns the slice its own pages occupy.
    assert [p.size for p in placements] == [page * num_blocks] * 2


def test_placements_at_one_address_are_grouped():
    """GLM-5.3-Flash overlays each Mamba layer onto its MLA partner."""
    num_blocks, mla_page, mamba_page = 4, 512, 480
    offset = 0
    tensors = [
        _NewStyleTensor(
            size=mla_page * num_blocks,
            layers=["mla.0"],
            layer_stride=mla_page * num_blocks,
            block_stride=mla_page,
            offset=offset,
        ),
        _NewStyleTensor(
            size=mla_page * num_blocks,
            layers=["mamba.0"],
            layer_stride=mamba_page * num_blocks,
            block_stride=mamba_page,
            offset=offset,
        ),
    ]

    (placement,) = aliased_kv_cache_tensors(_config(num_blocks, tensors))

    assert placement.shared_by == ["mla.0", "mamba.0"]
    # The buffer has to hold the larger of the two overlaid pages.
    assert placement.size == mla_page * num_blocks


def test_grouping_keeps_the_attention_layer_first():
    """The allocator picks its branch from the first name in the group."""
    num_blocks, page = 2, 256
    tensors = [
        _NewStyleTensor(
            size=page * num_blocks,
            layers=[name],
            layer_stride=page * num_blocks,
            block_stride=page,
            offset=0,
        )
        for name in ("attn.3", "linear_attn.3")
    ]

    (placement,) = aliased_kv_cache_tensors(_config(num_blocks, tensors))

    assert placement.shared_by[0] == "attn.3"


def test_multi_layer_entry_expands_to_one_allocation_per_layer():
    num_blocks, page = 4, 128
    tensors = [
        _NewStyleTensor(
            size=3 * page * num_blocks,
            layers=["layer.0", "layer.1", "layer.2"],
            layer_stride=page * num_blocks,
            block_stride=page,
            offset=64,
        )
    ]

    placements = aliased_kv_cache_tensors(_config(num_blocks, tensors))

    assert [p.shared_by for p in placements] == [["layer.0"], ["layer.1"], ["layer.2"]]
    assert {p.size for p in placements} == {page * num_blocks}


def test_block_outermost_layout_is_rejected():
    """Ascend reshapes assume each layer owns a contiguous run of pages."""
    num_blocks, page = 4, 128
    tensors = [
        _NewStyleTensor(
            size=2 * page * num_blocks,
            layers=["layer.0", "layer.1"],
            layer_stride=page,
            block_stride=2 * page,
            offset=0,
        )
    ]

    with pytest.raises(AssertionError, match="layer-outermost"):
        aliased_kv_cache_tensors(_config(num_blocks, tensors))
