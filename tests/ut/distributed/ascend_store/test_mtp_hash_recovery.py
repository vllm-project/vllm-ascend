# SPDX-License-Identifier: Apache-2.0
"""MTP proof data must describe the same boundary as the Mamba checkpoint."""

import pytest

from tests.ut.distributed.ascend_store.test_hash_recovery import groups
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import BoundaryPut, ReqMeta
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler


@pytest.mark.parametrize("partial,margin", [(False, 16), (True, 4)])
def test_eagle_attention_proof_margin_excludes_mamba(partial, margin):
    coord = AscendStoreCoordinator(
        groups(),
        16,
        4,
        group_block_sizes=[16, 16],
        group_cache_families=["default", "default"],
        use_eagle=True,
        enable_partial_hash_hits=partial,
    )
    assert coord.eagle_proof_margin_by_group == {0: margin}


@pytest.mark.parametrize("completed,expected", [(11, []), (12, [BoundaryPut(0, 1, 12)])])
def test_mtp_tail_saves_only_completed_attention_proof(completed, expected):
    scheduler = object.__new__(KVPoolScheduler)
    scheduler.cache_coordinator = AscendStoreCoordinator(
        groups(),
        16,
        4,
        group_block_sizes=[16, 16],
        group_cache_families=["default", "default"],
        use_eagle=True,
        enable_partial_hash_hits=True,
    )
    scheduler.hash_block_size = 4
    scheduler.lcm_block_size = 16
    scheduler.grouped_block_size = [16, 16]
    scheduler.cacheable_group_ids = [0, 1]
    scheduler.use_eagle = True
    req = ReqMeta(
        "mtp-tail",
        token_len_chunk=0,
        can_save=True,
        block_ids_by_group=[[1], [2]],
        block_hashes=[b"a", b"b", b"c"],
        num_prompt_tokens=13,
        completed_token_len=completed,
        boundary_puts=[BoundaryPut(1, 9, 8)],
    )
    # Prompt 13, PMU 4: MTP checkpoint is 8; Attention proof ends at 12.
    assert scheduler._partial_tail_puts(req) == expected
