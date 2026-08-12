# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy


def take_pending_kv_cache_block_copies(scheduler: Any) -> list[KVCacheBlockCopy] | None:
    """Drain pending CoW copies and defer retained-block release.

    Ascend schedulers that override the upstream scheduling loop must preserve
    its copy-on-write handoff contract explicitly.
    """
    copies, retained_blocks = scheduler.kv_cache_manager.take_kv_cache_block_copies()
    if not copies:
        return None

    scheduler._free_cow_retained_blocks(
        retained_blocks,
        scheduler.sched_step_seq + 1,
    )
    return copies
