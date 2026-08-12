# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkScheduler
from vllm_ascend.core.scheduler_utils import take_pending_kv_cache_block_copies
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler

pytestmark = pytest.mark.cpu_test


def test_take_pending_kv_cache_block_copies_defers_retained_blocks() -> None:
    copies = [KVCacheBlockCopy(src_block_id=3, dst_block_id=7)]
    retained_blocks = [object()]
    scheduler = SimpleNamespace(
        kv_cache_manager=MagicMock(),
        sched_step_seq=11,
        _free_cow_retained_blocks=MagicMock(),
    )
    scheduler.kv_cache_manager.take_kv_cache_block_copies.return_value = (
        copies,
        retained_blocks,
    )

    assert take_pending_kv_cache_block_copies(scheduler) == copies
    scheduler._free_cow_retained_blocks.assert_called_once_with(
        retained_blocks,
        12,
    )


def test_take_pending_kv_cache_block_copies_ignores_empty_queue() -> None:
    scheduler = SimpleNamespace(
        kv_cache_manager=MagicMock(),
        sched_step_seq=2,
        _free_cow_retained_blocks=MagicMock(),
    )
    scheduler.kv_cache_manager.take_kv_cache_block_copies.return_value = ([], [])

    assert take_pending_kv_cache_block_copies(scheduler) is None
    scheduler._free_cow_retained_blocks.assert_not_called()


@pytest.mark.parametrize(
    "scheduler_cls",
    [RecomputeScheduler, ProfilingChunkScheduler, BalanceScheduler],
)
def test_custom_scheduler_preserves_cow_handoff_contract(scheduler_cls) -> None:
    source = inspect.getsource(scheduler_cls.schedule)

    assert "take_pending_kv_cache_block_copies(self)" in source
    assert "kv_cache_block_copies=kv_cache_block_copies" in source
