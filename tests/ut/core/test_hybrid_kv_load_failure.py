# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import KVCacheRecvingThread


@pytest.mark.parametrize(
    "groups,invalid,evict,expected",
    [
        (([1, 2], [8], [10, 11]), {10}, False, ({"bad"}, 24, set())),
        (([], [8]), {8}, False, ({"bad"}, 24, set())),
        (([1, 2], [8], [10, 11]), {8}, True, ({"bad"}, 24, {1, 2, 8, 10, 11})),
        (([1, 2], [8]), {99}, True, (set(), 0, set())),
        (([1, 2],), {2}, True, ({"bad"}, 24, {1, 2})),
    ],
)
def test_fail_policy_handles_all_groups(groups, invalid, evict, expected):
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.recompute_kv_load_failures = False
    scheduler.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda req_id: groups if req_id == "bad" else ([91], [92])
    )
    bad = SimpleNamespace(request_id="bad", num_computed_tokens=32)
    good = SimpleNamespace(request_id="good", num_computed_tokens=16)

    result = scheduler._update_requests_with_invalid_blocks(iter([bad, good]), invalid, {"bad": 8}, evict)

    assert result == expected
    # The existing error finalization owns request teardown; do not partially
    # rewind hybrid/Mamba state or mutate unrelated requests here.
    assert bad.num_computed_tokens == 32
    assert good.num_computed_tokens == 16


def test_recompute_policy_delegates_to_upstream():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.recompute_kv_load_failures = True
    requests = iter([])
    expected = ({"r"}, 16, {2, 3})
    with patch.object(Scheduler, "_update_requests_with_invalid_blocks", return_value=expected) as upstream:
        assert scheduler._update_requests_with_invalid_blocks(requests, {2}, {}, False) == expected
    upstream.assert_called_once_with(requests, {2}, {}, False)


def test_upstream_failure_handler_uses_hybrid_hook_for_waiting_and_running():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.recompute_kv_load_failures = False
    waiting = SimpleNamespace(request_id="waiting", num_computed_tokens=24, status=RequestStatus.WAITING_FOR_REMOTE_KVS)
    running = SimpleNamespace(request_id="running", num_computed_tokens=32)
    good = SimpleNamespace(request_id="good", num_computed_tokens=16)
    groups = {"waiting": ([], [8]), "running": ([1, 2], [10, 11]), "good": ([91], [92])}
    scheduler.skipped_waiting = [waiting]
    scheduler.running = [running, good]
    scheduler.kv_cache_manager = SimpleNamespace(get_block_ids=groups.__getitem__, evict_blocks=Mock())

    assert scheduler._handle_invalid_blocks({8, 10}, {"running": 8}) == {"waiting", "running"}
    scheduler.kv_cache_manager.evict_blocks.assert_called_once_with({1, 2, 10, 11})
    assert [request.num_computed_tokens for request in (waiting, running, good)] == [24, 32, 16]


def test_failed_transfer_reports_all_groups_and_completes_cleanup():
    receiver = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    receiver.failed_recv_requests_lock = threading.Lock()
    receiver.failed_recv_requests = set()
    receiver.invalid_block_ids = set()
    receiver.pending_reformat_lock = threading.Lock()
    receiver.pending_reformat = {"r": object()}
    receiver.proc_not_transfer_request_lock = threading.Lock()
    receiver.proc_not_transfer_request = {"remote-r": True}
    receiver._transfer_kv_cache_all_groups = Mock(side_effect=RuntimeError("CreateChannel failed: 503900"))
    receiver._mark_request_task_done = Mock(return_value=True)
    receiver._reformat_pending_kv_caches = Mock()
    receiver.task_tracker = Mock()
    receiver.request_queue = Mock()
    receiver._send_done_signal_to_free_remote_port = Mock()
    receiver._send_done_recv_signal = Mock()
    req = {
        "request_id": "r",
        "remote_request_id": "remote-r",
        "remote_host": "producer",
        "remote_handshake_port": 1234,
        "remote_port_send_num": {1234: 1},
        "all_task_done": True,
        "local_block_ids": ([], [8], [10, 11]),
    }
    receiver._handle_request(req)

    assert receiver.get_and_clear_invalid_block_ids() == {8, 10, 11}
    assert receiver.get_and_clear_invalid_block_ids() == set()
    receiver._reformat_pending_kv_caches.assert_not_called()
    assert receiver.pending_reformat == {}
    assert receiver.proc_not_transfer_request == {}
    assert receiver.failed_recv_requests == set()
    receiver.task_tracker.update_done_task_count.assert_called_once_with("r")
    receiver.request_queue.task_done.assert_called_once_with()
    receiver._send_done_signal_to_free_remote_port.assert_called_once_with("remote-r", "producer", {1234: 1})
    receiver._send_done_recv_signal.assert_called_once_with("remote-r", "producer", 1234, {1234: 1})
