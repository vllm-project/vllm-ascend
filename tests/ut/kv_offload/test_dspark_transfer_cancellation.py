# SPDX-License-Identifier: Apache-2.0
"""Cancellation must drain remote target/draft reads before releasing their owners."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.connector import SfaRemoteD2HConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextDescriptor,
    DSparkContextReceiver,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_kv import DraftKVCacheMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import DSPARK_DRAFT_KV, DSPARK_DRAFT_KV_ACK
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import ConsumerReadState, MembPullReadThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import SFAPDRD2HScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import SFAPDRD2HConsumerWorker
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import CopySfaTopkSlotAllocator


class _TransferEngine:
    """Gate the actual read method's engine call, without writing any device memory."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.gated = False
        self.result = 0
        self.calls = []

    def batch_transfer_sync_read(self, session, local, peer, lengths):
        self.calls.append((session, list(local), list(peer), list(lengths)))
        if self.gated:
            self.entered.set()
            assert self.release.wait(5), "test transfer gate was never released"
        return self.result


def _metadata(group_id, base):
    return DraftKVCacheMetadata(
        group_id=group_id,
        block_size=4,
        num_blocks=16,
        base_addrs=(base,),
        block_strides=(16,),
        block_lens=(16,),
        block_scales=(1,),
        shapes=((4, 2),),
        dtypes=("torch.bfloat16",),
    )


def _request(name):
    request = Request.__new__(Request)
    request.request_id = name + "-00000000"
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.client_index = 0
    request.num_computed_tokens = 0
    request.num_in_flight_tokens = 0
    request.num_prompt_tokens = 6
    request.prompt_token_ids = [1] * 6
    request.kv_transfer_params = {"do_remote_prefill": False}
    return request


def _fixture(names=("cancel",), *, admit=True):
    requests = {name: _request(name) for name in names}
    descriptors = {name: DSparkContextDescriptor(name, "generation-1", 6, (2, 22, 38, 58, 74), 4) for name in names}
    receiver = DSparkContextReceiver(max_requests=len(names))
    engine = _TransferEngine()
    worker = SFAPDRD2HConsumerWorker.__new__(SFAPDRD2HConsumerWorker)
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.request_map = {}
    worker._dest_blocks_by_req = {}
    worker._dspark_draft_blocks_by_req = {}
    worker._cpu_blocks_by_req = {}
    worker.copy_sfa_slots_by_req = {}
    worker._copy_sfa_tail_by_req = {}
    worker._pending_done = set()
    worker._terminal_ext_ids = set()
    worker._invalid_block_ids = set()
    worker._dspark_context_receiver = receiver
    worker._dspark_pending_recv = set()
    worker._deferred_finished_req_ids = set()
    state = ConsumerReadState(
        num_blocks=16,
        tp_size=1,
        layer_metadata={"target": {}},
        main_name_to_idx={"target": 0},
        cpu_pools=[],
        main_gva_bases=[(10000, 20000)],
        main_block_lens=[(16, 16)],
        indexer_tensors=[None],
        indexer_scale_tensors=[None],
        dest_blocks_by_req=worker._dest_blocks_by_req,
        get_offload_layer_id=lambda name: 0,
        dspark_context_receiver=receiver,
        dspark_draft_kv_metadata={"draft": _metadata(2, 4000)},
        dspark_draft_blocks_by_req=worker._dspark_draft_blocks_by_req,
    )
    reader = MembPullReadThread(0, 0, engine, state)
    reader._p_sessions = {b"p": "p-session"}
    reader._p_pp_topology = {b"p": (0, 1)}
    reader._p_layer_meta = {"target": {"base_addrs": [1000, 2000], "block_len": [16, 16]}}
    worker._mf_read_thread = reader
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler._request_trackers = {}
    scheduler._dspark_context_requests = {}
    scheduler._dspark_context_groups = (2,)
    scheduler._reqs_need_recv = set()
    scheduler._dspark_pending_recv = set()
    scheduler._deferred_finished_req_ids = set()
    scheduler._copy_sfa_slot_allocator = CopySfaTopkSlotAllocator(len(names))
    scheduler._copy_sfa_bindings = {}
    scheduler._metaserver_lock = threading.Lock()
    scheduler._cancelled_metaserver_requests = set()
    scheduler._metaserver_futures = {}
    scheduler._metaserver_retry_timers = {}
    for index, name in enumerate(names):
        internal = requests[name].request_id
        blocks = [4 + 2 * index, 5 + 2 * index]
        slot = scheduler._copy_sfa_slot_allocator.bind(internal)
        scheduler._request_trackers[internal] = (blocks, [])
        scheduler._dspark_context_requests[internal] = (descriptors[name], {2: tuple(blocks)})
        scheduler._reqs_need_recv.add(internal)
        scheduler._dspark_pending_recv.add(internal)
        scheduler._copy_sfa_bindings[internal] = (slot, 0, 0, 6, False)
        scheduler._metaserver_futures[internal] = Mock()
        scheduler._metaserver_retry_timers[internal] = Mock()
    connector = SfaRemoteD2HConnector.__new__(SfaRemoteD2HConnector)
    connector.connector_scheduler = scheduler
    connector.is_consumer = True
    connector.is_producer = False
    upstream = Scheduler.__new__(Scheduler)
    upstream.requests = {request.request_id: request for request in requests.values()}
    upstream.running = []
    upstream.waiting = Mock()
    upstream.skipped_waiting = Mock()
    upstream.kv_holding_waiting = Mock()
    upstream.deferred_waiting = set()
    upstream.finished_recving_kv_req_ids = set()
    upstream.failed_recving_kv_req_ids = set()
    upstream.finished_req_ids = set()
    upstream.finished_req_ids_dict = None
    upstream._inflight_prefills = set()
    upstream._kv_fetch_stages = None
    upstream.aux_output_connector = None
    upstream.ec_connector = None
    upstream.encoder_cache_manager = Mock()
    upstream.connector = connector
    upstream.defer_block_free = False
    upstream.vllm_config = SimpleNamespace(kv_transfer_config=None)
    upstream.kv_cache_config = SimpleNamespace(kv_cache_groups=[None, None, None])
    upstream.kv_cache_manager = Mock()
    upstream.kv_cache_manager.get_block_ids_for_computed_tokens.return_value = ([4, 5], [], [4, 5])
    if admit:
        worker.start_load_kv(scheduler.build_connector_meta(Mock()))
    return SimpleNamespace(
        requests=requests,
        descriptors=descriptors,
        receiver=receiver,
        engine=engine,
        worker=worker,
        reader=reader,
        scheduler=scheduler,
        upstream=upstream,
    )


def _draft_transfer(fixture, name, descriptor=None):
    descriptor = fixture.descriptors[name] if descriptor is None else descriptor
    source = _metadata(7, 3000)
    source_wire = {key: getattr(source, key) for key in source.__dataclass_fields__}
    message = (
        DSPARK_DRAFT_KV,
        name,
        descriptor.generation,
        descriptor.prompt_tokens,
        descriptor.aux_layer_ids,
        descriptor.hidden_size,
        {"draft": source_wire},
        {7: (1, 2)},
    )
    socket = Mock()
    fixture.reader._handle_dspark_draft_kv(b"p", message, socket, msgspec.msgpack.Encoder())
    return msgspec.msgpack.decode(socket.send_multipart.call_args.args[0][2])


def _target_transfer(fixture, names):
    fixture.reader._do_read_batch("target", [(name, [1, 2], [], 0, 0) for name in names], p_session="p-session")
    # This is the existing read-loop's successful final-contributor callback.
    fixture.reader._record_chunk_done(list(names), 0, 1)


def _complete(fixture, name):
    assert _draft_transfer(fixture, name) == [DSPARK_DRAFT_KV_ACK, name, "generation-1", b"accepted"]
    _target_transfer(fixture, [name])


def _assert_held(fixture, name):
    internal = fixture.requests[name].request_id
    assert fixture.worker.request_map[name] == internal
    assert fixture.worker._dest_blocks_by_req[name]
    assert fixture.worker._dspark_draft_blocks_by_req[name]
    assert fixture.worker.copy_sfa_slots_by_req[internal] >= 0
    assert fixture.scheduler._copy_sfa_slot_allocator.get(internal) is not None
    assert not fixture.scheduler._copy_sfa_slot_allocator.can_bind("replacement")
    assert internal in fixture.upstream.requests
    fixture.upstream.kv_cache_manager.free.assert_not_called()


def _publish(fixture, finished):
    fixture.upstream._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving=finished))


@pytest.mark.parametrize("in_flight", ["draft", "target"])
def test_cancel_in_flight_keeps_destinations_and_slots_until_real_join(in_flight):
    fixture = _fixture()
    internal = fixture.requests["cancel"].request_id
    fixture.engine.gated = True
    errors: list[BaseException] = []

    def transfer():
        try:
            if in_flight == "draft":
                _draft_transfer(fixture, "cancel")
            else:
                _target_transfer(fixture, ["cancel"])
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=transfer)
    thread.start()
    try:
        assert fixture.engine.entered.wait(5)
        fixture.upstream.finish_requests(internal, RequestStatus.FINISHED_ABORTED)
        assert fixture.worker.get_finished({internal}) == (set(), set())
        _assert_held(fixture, "cancel")
        # The scheduler sends finished IDs once, not on every subsequent tick.
        assert fixture.worker.get_finished(set()) == (set(), set())
        _assert_held(fixture, "cancel")
    finally:
        fixture.engine.release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    fixture.engine.gated = False
    assert fixture.worker.get_finished(set()) == (set(), set())
    _assert_held(fixture, "cancel")
    if in_flight == "draft":
        _target_transfer(fixture, ["cancel"])
    else:
        _draft_transfer(fixture, "cancel")
    assert fixture.worker.get_finished(set()) == (set(), {internal})
    assert "cancel" not in fixture.worker._dest_blocks_by_req
    assert "cancel" not in fixture.worker._dspark_draft_blocks_by_req
    # Worker completion alone cannot release the scheduler-owned slot.
    assert fixture.scheduler._copy_sfa_slot_allocator.get(internal) == 0
    _publish(fixture, {internal})
    fixture.upstream.kv_cache_manager.free.assert_called_once_with(fixture.requests["cancel"])
    assert internal not in fixture.upstream.requests
    assert fixture.scheduler._copy_sfa_slot_allocator.bind("replacement") == 0
    assert fixture.worker.get_finished(set()) == (set(), set())


def test_cancel_before_dispatch_keeps_seed_and_rendezvous_until_receive_finishes():
    fixture = _fixture(admit=False)
    internal = fixture.requests["cancel"].request_id
    future = fixture.scheduler._metaserver_futures[internal]
    timer = fixture.scheduler._metaserver_retry_timers[internal]

    fixture.upstream.finish_requests(internal, RequestStatus.FINISHED_ABORTED)

    assert internal in fixture.scheduler._reqs_need_recv
    assert internal in fixture.scheduler._dspark_context_requests
    assert internal in fixture.scheduler._request_trackers
    future.cancel.assert_not_called()
    timer.cancel.assert_not_called()
    fixture.worker.start_load_kv(fixture.scheduler.build_connector_meta(Mock()))
    assert fixture.worker.get_finished({internal}) == (set(), set())
    _assert_held(fixture, "cancel")
    _complete(fixture, "cancel")
    assert fixture.worker.get_finished(set()) == (set(), {internal})
    _publish(fixture, {internal})
    future.cancel.assert_called_once()
    timer.cancel.assert_called_once()
    fixture.upstream.kv_cache_manager.free.assert_called_once()


def test_cancel_waits_for_all_tp_ranks_not_only_local_ready(monkeypatch):
    fixture = _fixture()
    internal = fixture.requests["cancel"].request_id
    fixture.worker.tp_size = 2
    peer_terminal: set[str] = set()
    collective_calls = []

    def gather(output, local, group):
        collective_calls.append(local)
        output[:] = [local, (set(peer_terminal), set())]

    monkeypatch.setattr(worker_module, "get_tp_group", lambda: SimpleNamespace(world_size=2, cpu_group="cpu"))
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    _complete(fixture, "cancel")
    fixture.upstream.finish_requests(internal, RequestStatus.FINISHED_ABORTED)
    assert fixture.worker.get_finished({internal}) == (set(), set())
    _assert_held(fixture, "cancel")
    assert fixture.receiver.get_descriptor("cancel") is not None
    peer_terminal.add("cancel")
    assert fixture.worker.get_finished(set()) == (set(), {internal})
    assert len(collective_calls) == 2
    _publish(fixture, {internal})
    fixture.upstream.kv_cache_manager.free.assert_called_once()


def test_normal_decode_finish_does_not_report_receive_completion_twice():
    fixture = _fixture()
    internal = fixture.requests["cancel"].request_id
    _complete(fixture, "cancel")
    assert fixture.worker.get_finished(set()) == (set(), {internal})
    _publish(fixture, {internal})
    assert fixture.scheduler._copy_sfa_slot_allocator.get(internal) == 0
    fixture.requests["cancel"].status = RequestStatus.RUNNING
    fixture.upstream.finish_requests(internal, RequestStatus.FINISHED_STOPPED)
    assert fixture.worker.get_finished({internal}) == (set(), set())
    assert fixture.worker.get_finished(set()) == (set(), set())
    assert "cancel" not in fixture.worker.request_map
    assert fixture.scheduler._copy_sfa_slot_allocator.get(internal) is None
    fixture.upstream.kv_cache_manager.free.assert_called_once()


def test_cancelled_and_live_requests_drain_independently_in_one_target_batch():
    fixture = _fixture(("cancel", "live"))
    cancelled = fixture.requests["cancel"].request_id
    live = fixture.requests["live"].request_id
    fixture.upstream.finish_requests(cancelled, RequestStatus.FINISHED_ABORTED)
    assert fixture.worker.get_finished({cancelled}) == (set(), set())
    _draft_transfer(fixture, "live")
    _target_transfer(fixture, ["cancel", "live"])
    assert fixture.worker.get_finished(set()) == (set(), {live})
    _publish(fixture, {live})
    assert "cancel" in fixture.worker.request_map
    assert "live" in fixture.worker.request_map
    assert fixture.scheduler._copy_sfa_slot_allocator.get(live) == 1
    fixture.upstream.kv_cache_manager.free.assert_not_called()
    _draft_transfer(fixture, "cancel")
    assert fixture.worker.get_finished(set()) == (set(), {cancelled})
    _publish(fixture, {cancelled})
    fixture.upstream.kv_cache_manager.free.assert_called_once_with(fixture.requests["cancel"])
    assert live in fixture.upstream.requests
    assert "live" in fixture.worker._dspark_draft_blocks_by_req
    assert fixture.scheduler._copy_sfa_slot_allocator.get(live) == 1


def test_failed_transfer_still_quarantines_without_fabricating_completion():
    fixture = _fixture()
    internal = fixture.requests["cancel"].request_id
    fixture.upstream.finish_requests(internal, RequestStatus.FINISHED_ABORTED)
    assert fixture.worker.get_finished({internal}) == (set(), set())
    fixture.engine.result = 1
    assert _draft_transfer(fixture, "cancel")[-1] == b"failed"
    with pytest.raises(RuntimeError, match="refusing local recomputation"):
        fixture.worker.get_finished(set())
    assert fixture.receiver.ready_requests() == set()
    _assert_held(fixture, "cancel")
    assert internal in fixture.worker._dspark_pending_recv


def test_cancelled_retired_generation_ack_is_idempotent_and_cannot_overwrite_replacement():
    fixture = _fixture()
    internal = fixture.requests["cancel"].request_id
    fixture.upstream.finish_requests(internal, RequestStatus.FINISHED_ABORTED)
    fixture.worker.get_finished({internal})
    _complete(fixture, "cancel")
    assert fixture.worker.get_finished(set()) == (set(), {internal})
    _publish(fixture, {internal})
    calls = len(fixture.engine.calls)
    assert _draft_transfer(fixture, "cancel")[-1] == b"accepted"
    assert len(fixture.engine.calls) == calls
    replacement = DSparkContextDescriptor("cancel", "generation-2", 6, (2, 22, 38, 58, 74), 4)
    fixture.receiver.register_request(replacement)
    assert _draft_transfer(fixture, "cancel")[-1] == b"stale"
    assert len(fixture.engine.calls) == calls
    assert fixture.receiver.get_descriptor("cancel") == replacement
