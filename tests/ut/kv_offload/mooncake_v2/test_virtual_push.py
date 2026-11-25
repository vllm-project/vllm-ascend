# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Layerwise health probes must neither dispatch P nor await its DONE messages."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.ut.kv_offload.mooncake_v2 import test_push_scheduler, test_push_worker
from tests.ut.kv_offload.mooncake_v2.test_push_worker import (
    completion,
    send,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata

scheduler = test_push_scheduler.scheduler
worker = test_push_worker.worker


def request(virtual=True):
    return SimpleNamespace(
        request_id="d-request-12345678",
        num_prompt_tokens=32,
        kv_transfer_params={
            "do_remote_prefill": True,
            "do_virtual": virtual,
            "num_computed_tokens": 0,
        },
    )


@pytest.mark.parametrize("virtual", [False, True])
def test_metadata_flag(virtual):
    meta = MooncakeConnectorMetadata()
    params = {"num_computed_tokens": 0}
    if virtual:
        params["do_virtual"] = True
    meta.add_new_push_req("probe", ([1],), 32, params)
    assert meta.push_requests["probe"].do_virtual is virtual


@pytest.mark.parametrize("external", [0, 32])
@pytest.mark.parametrize("virtual", [False, True])
def test_allocation_dispatches_only_real_requests(scheduler, external, virtual):
    req = request(virtual)
    blocks = Mock()
    blocks.get_block_ids.return_value = ([1, 2], [3, 4])
    scheduler.metaserver_client = Mock()
    scheduler.metaserver_client.get_failed_requests.return_value = {}
    scheduler.update_state_after_alloc(req, blocks, external)
    assert not req.kv_transfer_params["do_remote_prefill"]
    if virtual:
        scheduler.metaserver_client.submit.assert_not_called()
    else:
        scheduler.metaserver_client.submit.assert_called_once()
    meta = scheduler.build_connector_meta(Mock())
    assert bool(meta.push_requests) is bool(external)
    if external:
        assert meta.push_requests[req.request_id].do_virtual is virtual
    assert not scheduler._reqs_pending_recv


@pytest.mark.parametrize("allocated", [False, True])
def test_cancel_virtual_probe_clears_pending_allocation(scheduler, allocated):
    req = request()
    scheduler.metaserver_client = Mock()
    if allocated:
        blocks = Mock()
        blocks.get_block_ids.return_value = ([1],)
        scheduler.update_state_after_alloc(req, blocks, 32)
    assert scheduler.request_finished(req, ([1],)) == (False, None)
    assert scheduler.request_finished(req, ([1],)) == (False, None)
    assert not scheduler._reqs_pending_recv
    assert not req.kv_transfer_params["do_remote_prefill"]
    scheduler.metaserver_client.submit.assert_not_called()


def test_virtual_completion_without_producer_and_duplicate_metadata(worker):
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_req("probe", ([1],), 32, {"num_computed_tokens": 0, "do_virtual": True})
    worker.start_load_kv(meta)
    worker.start_load_kv(meta)
    result = worker.get_transfer_results()
    assert result.finished_recving == {"probe"}
    assert not result.failed_recving
    assert not worker._reqs_recv_state
    worker.start_load_kv(meta)
    assert not worker.get_transfer_results().finished_recving


def test_mixed_batch_keeps_real_receive_waiting_for_producer(worker):
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_req("probe", ([1],), 32, {"num_computed_tokens": 0, "do_virtual": True})
    meta.add_new_push_req("d-request", ([2],), 32, {"num_computed_tokens": 0})
    worker.start_load_kv(meta)
    result = worker.get_transfer_results()
    assert result.finished_recving == {"probe"}
    assert not result.failed_recving
    assert set(worker._reqs_recv_state) == {"d-request"}
    send(worker, completion())
    result = worker.get_transfer_results()
    assert result.finished_recving == {"d-request"}
    assert not result.failed_recving
    assert not worker.get_transfer_results().finished_recving


def test_empty_metadata_has_no_completion(worker):
    worker.start_load_kv(MooncakeConnectorMetadata())
    result = worker.get_transfer_results()
    assert not result.finished_recving and not result.failed_recving


def test_virtual_allocation_discards_early_producer_completion(worker):
    # A stale control message must not leave pending state behind a virtual probe.
    send(worker, completion())
    assert not worker.get_transfer_results().finished_recving
    assert "d-request" in worker._pending_completions
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_req("d-request", ([1],), 32, {"num_computed_tokens": 0, "do_virtual": True})
    worker.start_load_kv(meta)
    result = worker.get_transfer_results()
    assert result.finished_recving == {"d-request"}
    assert not result.failed_recving
    assert not worker._pending_completions
