# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Health probes must complete locally without contacting a producer."""

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    MooncakePullConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)


def params(virtual=False):
    return dict(
        do_remote_prefill=True,
        do_virtual=virtual,
        remote_block_ids=([0],),
        remote_host="127.0.0.1",
        remote_port=1,
        remote_engine_id="absent-p",
        remote_request_id="p-request",
        remote_num_prompt_tokens=256,
    )


def scheduler():
    s = MooncakePullConnectorScheduler.__new__(MooncakePullConnectorScheduler)
    s.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    s._heartbeat_thread = MagicMock()
    s._recving_thread = MagicMock()
    s._sending_thread = None
    s._reqs_need_recv = {}
    s._reqs_recv_info = {}
    s._reqs_in_batch = set()
    s._log_mamba_request_allocation = MagicMock()
    s._validated_producer_leases = {}
    s._kv_lease_duration = 60
    return s


def request(virtual=True):
    return SimpleNamespace(request_id="d-request", kv_transfer_params=params(virtual), num_prompt_tokens=256)


@pytest.mark.parametrize("virtual", [False, True])
def test_metadata_flag(virtual):
    meta = MooncakeConnectorMetadata()
    p = params(virtual)
    if not virtual:
        p.pop("do_virtual")
    meta.add_new_req("d", ([1],), 256, 255, p)
    assert meta.requests["d"].do_virtual is virtual


def test_virtual_admission_needs_no_producer_lease():
    s = scheduler()
    with patch.object(MooncakeBaseConnectorScheduler, "on_new_request"):
        s.on_new_request(request())
    s._heartbeat_thread.add_request.assert_not_called()
    assert not s._validated_producer_leases
    with patch.object(MooncakeBaseConnectorScheduler, "on_new_request"), pytest.raises(ValueError):
        s.on_new_request(request(False))


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.parametrize("external", [0, 255])
def test_allocation_and_completion_ack(virtual, external):
    s, r, blocks = scheduler(), request(virtual), MagicMock()
    blocks.get_unhashed_block_ids_all_groups.return_value = ([1, 2],)
    blocks.get_block_ids.return_value = ([1, 2],)
    s.update_state_after_alloc(r, blocks, external)
    assert r.kv_transfer_params["do_remote_prefill"] is False
    assert (r.request_id in s._reqs_need_recv) is bool(external)
    assert (r.request_id in s._reqs_recv_info) is (bool(external) and not virtual)
    if external:
        m = s.build_connector_meta(MagicMock())
        assert m.requests[r.request_id].do_virtual is virtual
        s.update_connector_output(KVConnectorOutput(finished_recving={r.request_id}))
    if virtual:
        s._recving_thread.add_request.assert_not_called()
    else:
        s._recving_thread.add_request.assert_called_once_with("127.0.0.1", 1, "p-request")


@pytest.mark.parametrize("allocated", [False, True])
def test_virtual_finish_never_acknowledges_producer(allocated):
    s, r = scheduler(), request()
    if allocated:
        s.update_state_after_alloc(r, MagicMock(), 255)
    assert s.request_finished(r, ([1],)) == (False, None)
    assert s.request_finished(r, ([1],)) == (False, None)
    assert not s._reqs_need_recv and not s._reqs_recv_info and not s._reqs_in_batch
    s._recving_thread.add_request.assert_not_called()


def test_mixed_worker_batch_and_once_only_completion():
    worker = MooncakePullConnectorWorker.__new__(MooncakePullConnectorWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    receiver = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    receiver.finished_requests = queue.SimpleQueue()
    receiver.request_queue = queue.SimpleQueue()
    worker._recving_thread = receiver
    meta = MooncakeConnectorMetadata()
    for name, virtual in [("virtual", True), ("normal", False)]:
        meta.add_new_req(name, ([1],), 256, 255, params(virtual))
    worker.start_load_kv(meta)
    engine, real_requests = receiver.request_queue.get_nowait()
    assert engine == "absent-p" and set(real_requests) == {"normal"}
    assert receiver.request_queue.empty()
    result = worker.get_transfer_results()
    assert result.finished_recving == {"virtual"} and not result.failed_recving
    assert not worker.get_transfer_results().finished_recving
