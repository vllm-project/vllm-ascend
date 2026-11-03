# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_scheduler import MooncakePushConnectorScheduler

from .helpers import make_blocks, make_full_spec, make_request


def scheduler():
    instance = MooncakePushConnectorScheduler.__new__(MooncakePushConnectorScheduler)
    instance.group_block_size = [16]
    instance.group_unique_specs = [[make_full_spec()]]
    instance.dcp_size = instance.tp_size = 1
    instance._new_receives = {}
    instance._receiving = set()
    instance._sending = set()
    instance._new_sources = []
    instance._cancellations = []
    return instance


def request(**changes):
    params = {
        "do_remote_prefill": True,
        "push_transfer_id": str(uuid4()),
        "remote_host": "127.0.0.1",
        "remote_port": 5600,
        "remote_engine_id": "p",
        "remote_tp_size": 1,
    }
    return make_request(kv_transfer_params=params, num_computed_tokens=0, **changes)


def test_decode_can_allocate_without_source_block_ids_and_recomputes_last_token():
    instance = scheduler()
    req = request()
    assert instance.get_num_new_matched_tokens(req, 0) == (31, True)
    instance.update_state_after_alloc(req, make_blocks(full=([3, 4, 5],)), 31)
    meta = instance.build_connector_meta(MagicMock())
    assert meta.push_receives[0].block_ids == ([3, 4],)
    assert meta.push_receives[0].num_tokens == 31
    assert "remote_block_ids" not in req.kv_transfer_params
    assert instance.has_pending_push_work()


def test_cancellation_before_worker_publication_can_release_immediately():
    instance = scheduler()
    req = request()
    instance.update_state_after_alloc(req, make_blocks(), 31)
    assert instance.request_finished(req, ([1, 2],)) == (False, None)
    assert not instance.build_connector_meta(MagicMock()).push_receives
    assert not instance.has_pending_push_work()


def test_published_target_retained_until_fenced_completion():
    instance = scheduler()
    req = request()
    instance.update_state_after_alloc(req, make_blocks(), 31)
    instance.build_connector_meta(MagicMock())
    assert instance.request_finished(req, ([1, 2],)) == (True, None)
    assert instance.build_connector_meta(MagicMock()).push_cancellations == [req.request_id]
    assert instance.has_pending_push_work()
    instance.update_connector_output(KVConnectorTransferResults(finished_recving={req.request_id}))
    assert not instance.has_pending_push_work()


def test_source_retained_and_engine_kept_alive_until_write_completion():
    instance = scheduler()
    req = request(status=RequestStatus.FINISHED_LENGTH_CAPPED)
    req.kv_transfer_params = {"do_remote_decode": True, "push_transfer_id": str(uuid4())}
    assert instance.request_finished(req, ([1, 2, 3],)) == (True, None)
    meta = instance.build_connector_meta(MagicMock())
    assert meta.push_sources[0].block_ids == ([1, 2],)
    assert instance.has_pending_push_work()
    instance.update_connector_output(KVConnectorTransferResults(finished_sending={req.request_id}))
    assert not instance.has_pending_push_work()


def test_one_token_decode_has_no_remote_dependency():
    instance = scheduler()
    req = request(num_prompt_tokens=1, prompt_token_ids=[7])
    assert instance.get_num_new_matched_tokens(req, 0) == (0, False)
    instance.update_state_after_alloc(req, make_blocks(full=([2],)), 0)
    receive = instance.build_connector_meta(MagicMock()).push_receives[0]
    assert receive.block_ids == ([],)
    assert receive.num_tokens == 0
    assert not instance.has_pending_push_work()


def test_cancelled_prefill_does_not_publish_live_source_blocks():
    instance = scheduler()
    req = request(status=RequestStatus.FINISHED_ABORTED)
    req.kv_transfer_params = {"do_remote_decode": True, "push_transfer_id": str(uuid4())}
    assert instance.request_finished(req, ([1, 2],)) == (False, None)
    source = instance.build_connector_meta(MagicMock()).push_sources[0]
    assert source.cancelled
    assert not source.delayed_free
    assert not source.block_ids
