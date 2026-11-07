# SPDX-License-Identifier: Apache-2.0
"""Port of the recorded external-state regression to native manager fixtures."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


def _make_manager(mtp, mode="align"):
    pool = BlockPool(64, False, 128, False, MagicMock())
    manager = AscendMambaManager(
        kv_cache_spec=MambaSpec(
            block_size=8704,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode=mode,
            num_speculative_blocks=mtp,
        ),
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=8704,
    )
    return manager, pool


@pytest.mark.parametrize("mtp", [0, 5])
@pytest.mark.parametrize("async_load", [False, True])
@pytest.mark.parametrize("cached_tokens,target_tokens", [(8498, 8499), (8704, 8705)])
def test_external_state_prediction_matches_pool_consumption(
    monkeypatch, mtp, async_load, cached_tokens, target_tokens, record_property
):
    manager, pool = _make_manager(mtp)
    allocations = []
    get_new_blocks = pool.get_new_blocks

    def allocate(count):
        blocks = get_new_blocks(count)
        allocations.append(len(blocks))
        return blocks

    monkeypatch.setattr(pool, "get_new_blocks", allocate)
    before = pool.get_num_free_blocks()
    predicted = manager.get_num_blocks_to_allocate(
        "external",
        target_tokens + mtp,
        [],
        cached_tokens,
        0,
        target_tokens,
        apply_admission_cap=True,
    )
    manager.add_local_computed_blocks("external", [], 0, cached_tokens)
    manager.allocate_external_computed_blocks("external", 0, cached_tokens)
    manager.allocate_new_blocks(
        "external",
        cached_tokens if async_load else target_tokens + mtp,
        cached_tokens if async_load else target_tokens,
    )
    if async_load:
        manager.allocate_new_blocks("external", target_tokens + mtp, target_tokens)
    record_property("predicted_blocks", predicted)
    record_property("allocated_blocks", sum(allocations))
    assert predicted == sum(allocations)
    assert before - pool.get_num_free_blocks() == sum(allocations)
    manager.free("external")
    assert pool.get_num_free_blocks() == before


@pytest.mark.parametrize("mtp", [0, 5])
@pytest.mark.parametrize("async_load", [False, True])
@pytest.mark.parametrize("cached_tokens,target_tokens", [(8498, 8499), (8704, 8705)])
def test_non_align_capacity_prediction_keeps_existing_behavior(
    mtp, async_load, cached_tokens, target_tokens, record_property
):
    manager, pool = _make_manager(mtp, mode="none")
    before = pool.get_num_free_blocks()
    predicted = manager.get_num_blocks_to_allocate(
        "external",
        target_tokens + mtp,
        [],
        cached_tokens,
        0,
        target_tokens,
        apply_admission_cap=True,
    )
    manager.add_local_computed_blocks("external", [], 0, cached_tokens)
    manager.allocate_external_computed_blocks("external", 0, cached_tokens)
    manager.allocate_new_blocks(
        "external",
        cached_tokens if async_load else target_tokens + mtp,
        cached_tokens if async_load else target_tokens,
    )
    if async_load:
        manager.allocate_new_blocks("external", target_tokens + mtp, target_tokens)
    actual = before - pool.get_num_free_blocks()
    record_property("predicted_blocks", predicted)
    record_property("allocated_blocks", actual)
    # Preserve the existing non-align extra headroom; this PR changes align only.
    assert predicted == actual + 1
    assert actual == (1 if cached_tokens == 8498 else 2) + mtp
    manager.free("external")
    assert pool.get_num_free_blocks() == before
