# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager


@pytest.mark.parametrize("record_new_blocks", [False, True])
@pytest.mark.parametrize("async_load", [False, True])
def test_tail_external_allocation_obeys_coordinator_zeroing(record_new_blocks, async_load):
    spec = AscendIndexerKPoolTailSpec(
        block_size=9, sliding_window=4, compress_ratio=4, num_kv_heads=1, head_size=128, dtype=torch.float32
    )
    pool = BlockPool(8, True, 4)
    manager = KpoolTailManager(spec, pool, True, 0, scheduler_block_size=16)
    manager._record_new_block_ids = record_new_blocks
    coordinator = SimpleNamespace(single_type_managers=[manager])
    initially_free = pool.get_num_free_blocks()

    # Use the real upstream coordinator, including its keyword argument and
    # per-group decision for asynchronous PD transfers.
    KVCacheCoordinator.allocate_new_computed_blocks(
        coordinator,
        "remote",
        [[]],
        num_local_computed_tokens=0,
        num_external_computed_tokens=1024,
        skip_zeroing_group_ids={0} if async_load else set(),
    )
    remote = manager.req_to_blocks["remote"]
    assert len(remote) == 1
    assert manager.take_new_block_ids() == ([remote[0].block_id] if record_new_blocks and not async_load else [])

    # Verification and repeated allocation must preserve the received ring.
    assert manager.allocate_new_blocks("remote", 2048, 2043) == []
    manager.allocate_external_computed_blocks("remote", 1024, 1024, record_for_zeroing=True)
    assert manager.take_new_block_ids() == []
    assert pool.get_num_free_blocks() == initially_free - 1

    # Suppression for a transferred request must not leak to local requests.
    local = manager.allocate_new_blocks("local", 16, 11)
    assert len(local) == 1 and local[0].block_id != remote[0].block_id
    assert manager.take_new_block_ids() == ([local[0].block_id] if record_new_blocks else [])
    manager.free("remote")
    manager.free("local")
    assert pool.get_num_free_blocks() == initially_free
