# SPDX-License-Identifier: Apache-2.0
"""Backpressure for PD requests awaiting a stable fused-copy top-k row."""

from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import SFAPDRD2HScheduler
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import CopySfaTopkSlotAllocator


def _scheduler(capacity):
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler.block_size = [128, 128]
    scheduler._copy_sfa_slot_allocator = CopySfaTopkSlotAllocator(capacity) if capacity else None
    return scheduler


def _request(req_id, remote=True):
    return SimpleNamespace(
        request_id=req_id,
        kv_transfer_params={"do_remote_prefill": remote},
        prompt_token_ids=list(range(598)),
    )


def test_full_pool_defers_without_reservation_or_recompute():
    scheduler = _scheduler(6)
    allocator = scheduler._copy_sfa_slot_allocator
    for index in range(6):
        allocator.bind(f"owner-{index}")
    request = _request("waiting")
    for _ in range(3):
        assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
        assert allocator.get("waiting") is None
        assert request.kv_transfer_params["do_remote_prefill"]
    allocator.release("owner-0")
    assert scheduler.get_num_new_matched_tokens(request, 128) == (470, True)
    assert allocator.get("waiting") is None
    assert allocator.bind("waiting") == 0


def test_existing_owner_can_be_queried_when_pool_is_full():
    scheduler = _scheduler(1)
    scheduler._copy_sfa_slot_allocator.bind("owner")
    assert scheduler.get_num_new_matched_tokens(_request("owner"), 0) == (598, True)


def test_nonremote_request_is_not_blocked_by_full_pool():
    scheduler = _scheduler(1)
    scheduler._copy_sfa_slot_allocator.bind("owner")
    assert scheduler.get_num_new_matched_tokens(_request("local", remote=False), 0) == (0, False)


def test_other_offload_modes_keep_remote_matching_behavior():
    scheduler = _scheduler(0)
    assert scheduler.get_num_new_matched_tokens(_request("remote"), 0) == (598, True)
