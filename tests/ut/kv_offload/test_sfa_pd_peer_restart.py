# SPDX-License-Identifier: Apache-2.0
"""Decode process-incarnation regression tests using the real send paths."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import send_thread as sending
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextDescriptor,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_kv import DraftKVCacheMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
    DSPARK_DRAFT_KV,
    DSPARK_DRAFT_KV_ACK,
    MF_META,
    MF_META_ACK,
    READ_DONE,
    READ_READY_BATCH,
    SFAPD_PROTOCOL_VERSION,
    LayerMetadata,
    SendTask,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import SFAPDRD2HProducerWorker

PATH = "tcp://127.0.0.1:1234"
LAYER = "model.layers.0.self_attn"


@pytest.fixture
def sender(monkeypatch):
    context = MagicMock()
    monkeypatch.setattr(sending.zmq, "Context", MagicMock(return_value=context))
    thread = sending.MembPullSendingThread(
        ready_event=threading.Event(),
        state=sending.ProducerSendState(
            last_layer_idx=0,
            main_group_idx=0,
            indexer_group_idx=1,
            block_sizes=(16, 32),
            layer_metadata={
                LAYER: LayerMetadata(
                    tensor_group_idx=[0],
                    kv_caches_base_addr=[1000],
                    block_len=[10],
                    block_size_scale=[1],
                    main_tensor_count=1,
                    has_indexer=False,
                )
            },
            layer_storage_slots={0: (0,)},
            p_session="p-session",
            producer_pp_size=2,
        ),
    )
    first, replacement = MagicMock(), MagicMock()
    for dealer in (first, replacement):
        dealer.poll.return_value = True
        dealer.recv_multipart.return_value = [msgspec.msgpack.encode((MF_META_ACK, SFAPD_PROTOCOL_VERSION))]
    context.socket.side_effect = [first, replacement]
    return thread, first, replacement


def _task(engine_id, request_id="request", **extra):
    meta = SimpleNamespace(
        remote_engine_id=engine_id,
        remote_host="127.0.0.1",
        remote_port=1234,
        local_block_ids=[[1], []],
        local_transed_tokens=0,
        local_computed_tokens=16,
        chunk_finish=True,
        group_member_idx=0,
        tp_ratio=1,
    )
    return SendTask(send_request={request_id: meta, **extra}, layer_idx=0, layer_name=LAYER)


def _send(thread, task):
    thread._process_send_task(task, msgspec.msgpack.Encoder())


def _messages(dealer):
    return [msgspec.msgpack.decode(call.args[0])[0] for call in dealer.send.call_args_list]


@pytest.mark.parametrize("engine_id", ["decode-a", None])
def test_same_process_reuses_metadata(sender, engine_id):
    thread, first, replacement = sender
    _send(thread, _task(engine_id))
    thread._signal_layer_done(0)
    _send(thread, _task(engine_id))
    assert _messages(first) == [MF_META, READ_READY_BATCH, READ_READY_BATCH]
    assert not replacement.send.called
    assert not first.close.called


@pytest.mark.parametrize("previous", ["decode-a", None])
def test_idle_restart_resends_metadata_before_read_ready(sender, previous):
    thread, first, replacement = sender
    _send(thread, _task(previous))
    thread._signal_layer_done(0)
    first.poll.return_value = False
    _send(thread, _task("decode-b"))
    first.close.assert_called_once_with(linger=0)
    assert _messages(replacement) == [MF_META, READ_READY_BATCH]
    assert thread._peer_engine_ids[PATH] == "decode-b"
    assert thread._pending_reads_by_layer == {0: 1}


def test_restart_with_unfinished_read_fails_storage_closed(sender):
    thread, first, replacement = sender
    _send(thread, _task("decode-a"))
    first.poll.return_value = False
    with pytest.raises(RuntimeError, match="outstanding KV reads"):
        _send(thread, _task("decode-b"))
    assert not replacement.send.called
    assert not first.close.called
    assert thread._peer_engine_ids[PATH] == "decode-a"
    assert thread._pending_reads_by_layer == {}
    # The event wakes the producer to report the error, not to reuse storage.
    assert thread.storage_send_done_events[0].is_set()
    assert "refusing storage reuse" in thread.get_storage_error(0)


def test_restart_drains_old_completion_before_changing_socket(sender):
    thread, first, replacement = sender
    _send(thread, _task("decode-a"))
    first.poll.side_effect = [True, False]
    first.recv_multipart.return_value = [msgspec.msgpack.encode((READ_DONE, 0))]
    _send(thread, _task("decode-b"))
    assert _messages(replacement) == [MF_META, READ_READY_BATCH]
    assert thread.get_storage_error(0) is None
    assert not thread.storage_send_done_events[0].is_set()


def test_batch_cannot_mix_process_incarnations_at_one_endpoint(sender):
    thread, first, replacement = sender
    extra = _task("decode-b").send_request["request"]
    with pytest.raises(RuntimeError, match="different Decode processes"):
        _send(thread, _task("decode-a", other=extra))
    assert not first.send.called
    assert not replacement.send.called
    assert thread._pending_reads_by_layer == {}


def test_known_process_cannot_silently_lose_identity(sender):
    thread, first, replacement = sender
    _send(thread, _task("decode-a"))
    thread._signal_layer_done(0)
    with pytest.raises(RuntimeError, match="engine ID disappeared"):
        _send(thread, _task(None))
    assert _messages(first) == [MF_META, READ_READY_BATCH]
    assert not replacement.send.called


def test_idle_peer_can_restart_while_another_peer_has_pending_reads(sender):
    thread, first, replacement = sender
    _send(thread, _task("decode-a"))
    thread._signal_layer_done(0)
    first.poll.return_value = False
    other_path = "tcp://127.0.0.2:1234"
    thread._pending_reads_by_layer[1] = 1
    thread._pending_read_paths_by_layer[1] = {other_path}
    _send(thread, _task("decode-b"))
    assert _messages(replacement) == [MF_META, READ_READY_BATCH]
    assert thread._pending_reads_by_layer == {0: 1, 1: 1}
    assert thread._pending_read_paths_by_layer[1] == {other_path}


def test_old_or_duplicate_reply_cannot_complete_another_peer_read(sender):
    thread, _, _ = sender
    _send(thread, _task("decode-a"))
    other_path = "tcp://127.0.0.2:1234"
    thread._pending_reads_by_layer[0] = 2
    thread._pending_read_paths_by_layer[0].add(other_path)
    thread._complete_peer_read(PATH, 0)
    thread._complete_peer_read(PATH, 0)
    thread._complete_peer_read("tcp://old-peer:1234", 0, "stale failure")
    assert thread._pending_reads_by_layer == {0: 1}
    assert not thread.storage_send_done_events[0].is_set()
    assert thread.get_storage_error(0) is None
    thread._complete_peer_read(other_path, 0)
    assert thread._pending_reads_by_layer == {}
    assert thread.storage_send_done_events[0].is_set()


def test_draft_kv_path_renews_the_same_peer_metadata(sender):
    thread, first, replacement = sender
    _send(thread, _task("decode-a"))
    thread._signal_layer_done(0)
    first.poll.return_value = False
    descriptor = DSparkContextDescriptor("request", "generation", 4, (2, 22, 38, 58, 74), 4)
    replacement.recv_multipart.side_effect = [
        [msgspec.msgpack.encode((MF_META_ACK, SFAPD_PROTOCOL_VERSION))],
        [msgspec.msgpack.encode((DSPARK_DRAFT_KV_ACK, "request", "generation", b"accepted"))],
    ]
    task = sending.DSparkDraftKVSendTask(
        host="127.0.0.1",
        port=1234,
        descriptor=descriptor,
        draft_kv_metadata={"draft": {"group_id": 0}},
        source_blocks_by_group={0: (2,)},
        remote_engine_id="decode-b",
    )
    thread._process_dspark_draft_kv_task(task, msgspec.msgpack.Encoder(), msgspec.msgpack.Decoder(type=tuple))
    assert _messages(replacement) == [MF_META, DSPARK_DRAFT_KV]


def test_draft_kv_worker_passes_process_identity():
    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    descriptor = DSparkContextDescriptor("request", "generation", 4, (2, 22, 38, 58, 74), 4)
    worker.pp_rank, worker.pp_size = 1, 2
    worker.tp_rank = 0
    worker.kv_send_layer_thread = MagicMock()
    worker.dspark_aux_layer_ids = descriptor.aux_layer_ids
    worker.kv_cache_specs = [SimpleNamespace(block_size=16)]
    worker.dspark_draft_kv_metadata = {
        "draft": DraftKVCacheMetadata(
            group_id=0,
            block_size=16,
            num_blocks=8,
            base_addrs=(4096,),
            block_strides=(32,),
            block_lens=(32,),
            block_scales=(1,),
            shapes=((16, 2),),
            dtypes=("torch.bfloat16",),
        )
    }
    worker._active_dspark_requests = {
        "request": SimpleNamespace(
            remote_host="127.0.0.1",
            remote_port=1234,
            remote_engine_id="decode-b",
            dspark_context_generation="generation",
            local_block_ids=[[7]],
        )
    }
    worker.send_dspark_draft_kv("request", descriptor, {0: (7,)})
    assert worker.kv_send_layer_thread.send_dspark_draft_kv.call_args.kwargs["remote_engine_id"] == "decode-b"
