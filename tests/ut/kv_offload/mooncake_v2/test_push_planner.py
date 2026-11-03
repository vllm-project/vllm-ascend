# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_planner import plan_push

from .helpers import make_transfer_metadata


def test_merge_only_when_source_and_destination_are_contiguous():
    source = make_transfer_metadata(base_addrs=[[1000]])
    destination = make_transfer_metadata(engine_id="d", base_addrs=[[10000]])
    assert plan_push(source, destination, ([1, 2, 4],), ([3, 4, 8],), 48) == (
        [1128, 1512],
        [10384, 11024],
        [256, 128],
    )
    assert plan_push(source, destination, ([1, 2],), ([3, 6],), 31)[2] == [128, 128]


def test_pair_layers_by_name_with_independent_page_strides():
    source = make_transfer_metadata(layer_names=["a", "b"], base_addrs=[[1000], [2000]], block_strides=[[256], [256]])
    destination = make_transfer_metadata(
        layer_names=["b", "a"], base_addrs=[[20000], [10000]], block_strides=[[512], [512]]
    )
    assert plan_push(source, destination, ([1],), ([2],), 7) == (
        [1256, 2256],
        [11024, 21024],
        [128, 128],
    )


@pytest.mark.parametrize("blocks", [([-1],), ([32],), ([1, 1],), ([True],)])
def test_reject_invalid_or_aliased_destination_pages(blocks):
    source = make_transfer_metadata()
    with pytest.raises(ValueError):
        plan_push(source, source, ([1],), blocks, 16)


@pytest.mark.parametrize(
    "change",
    [
        {"block_size": 32},
        {"layer_names": ["different"]},
        {"block_lens": [[64]]},
        {"block_strides": [[64]]},
        {"block_size_scales": [[2]]},
        {"block_shapes": [[(2, 8, 4)]]},
        {"group_indices": [1]},
    ],
)
def test_reject_layout_mismatch_before_write(change):
    source = make_transfer_metadata()
    with pytest.raises(ValueError):
        plan_push(source, replace(source, **change), ([1],), ([2],), 16)


def test_one_token_prompt_authorizes_no_page_write():
    metadata = make_transfer_metadata()
    assert plan_push(metadata, metadata, ([],), ([],), 0) == ([], [], [])
