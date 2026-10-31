# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread
from vllm_ascend.distributed.kv_transfer.utils.cache_layout import (
    expand_cache_block_ids,
    kernel_cache_pages,
    kv_head_interval,
)


@pytest.mark.parametrize("tp_size,heads", [(1, 8), (2, 8), (4, 8), (8, 2)])
def test_head_intervals_match_original_v2_entry_point(tp_size, heads):
    for rank in range(tp_size):
        assert kv_head_interval(rank, tp_size, heads) == MooncakePullRecvingThread._get_head_interval(
            rank, tp_size, heads
        )
    intervals = {kv_head_interval(rank, tp_size, heads) for rank in range(tp_size)}
    assert set().union(*(set(range(start, end)) for start, end in intervals)) == set(range(heads))


def test_dcp_kernel_pages_cover_virtual_sequence_without_overlap():
    spans = []
    for rank in range(4):
        pages = kernel_cache_pages([7, 11], logical_block_size=32, physical_scale=2, dcp_size=4, dcp_rank=rank)
        assert [page.physical_block_id for page in pages] == [14, 15, 22, 23]
        spans.extend((page.token_start, page.token_end) for page in pages)
    ordered = sorted(spans)
    assert ordered[0][0] == 0
    assert ordered[-1][1] == 256
    assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:]))


def test_prefix_skipped_blocks_preserve_global_token_origin():
    pages = kernel_cache_pages(
        [19],
        logical_block_size=32,
        physical_scale=2,
        dcp_size=2,
        dcp_rank=1,
        first_logical_block=3,
    )
    assert [(page.token_start, page.token_end) for page in pages] == [(224, 240), (240, 256)]
    assert all(page.logical_block_index == 3 for page in pages)


def test_different_dcp_and_logical_page_sizes_have_same_kernel_coordinates():
    source = [
        page
        for rank in range(2)
        for page in kernel_cache_pages(
            [10],
            logical_block_size=64,
            physical_scale=4,
            dcp_size=2,
            dcp_rank=rank,
        )
    ]
    target = kernel_cache_pages([20, 21, 22, 23], logical_block_size=32, physical_scale=2)
    assert sorted((page.token_start, page.token_end) for page in source) == [
        (page.token_start, page.token_end) for page in target
    ]


def test_block_expansion_preserves_v2_alias_and_numbering_contract():
    blocks = [2, 7]
    assert expand_cache_block_ids(blocks, 1) is blocks
    assert expand_cache_block_ids(blocks, 3) == [6, 7, 8, 21, 22, 23]
    assert MooncakePullRecvingThread._expand_block_ids(blocks, 3) == [6, 7, 8, 21, 22, 23]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"logical_block_size": 15, "physical_scale": 2},
        {"logical_block_size": 16, "physical_scale": 1, "dcp_size": 0},
        {"logical_block_size": 16, "physical_scale": 1, "dcp_rank": 2},
        {"logical_block_size": 16, "physical_scale": 1, "first_logical_block": -1},
    ],
)
def test_invalid_coordinates_are_rejected(kwargs):
    with pytest.raises(ValueError):
        kernel_cache_pages([1], **kwargs)
