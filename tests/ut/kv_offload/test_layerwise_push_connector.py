# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the push transfer mode of the layerwise pull connector.

Mirrors the mock-based style of test_layerwise_pull_connector.py: no NPU, no
network — backends, dealers and streams are faked. Covers the p-push.md v2
deltas: extended DEST_LAYOUT_META handshake, tp_shared write slicing, deferred
DEST_BLOCKS retries, and async write submission with event-based reaping.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.protocol import (
    DEST_BLOCKS,
    DEST_BLOCKS_REQUEST,
    DEST_LAYOUT_META,
    PUSH_META,
    WRITE_DONE,
    WRITE_FAILED,
    ComponentLayout,
    LayerwisePullProducerReqMeta,
    SendTask,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread import (
    ConsumerReadState,
    LayerwisePullReadThread,
    PullBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread import (
    LayerwisePullSendingThread,
    ProducerSendState,
)


def _component(name: str, group_index: int, base: int, length: int) -> ComponentLayout:
    return ComponentLayout(
        name=name,
        group_index=group_index,
        block_size=16,
        dtypes=("torch.bfloat16",),
        base_addrs=(base,),
        block_strides=(length,),
        block_lengths=(length,),
        block_shapes=((length,),),
        block_size_scales=(1,),
    )


def _ok_engine() -> MagicMock:
    engine = MagicMock()
    engine.batch_transfer_sync_write.return_value = 0
    engine.batch_transfer_async_write_submit.return_value = 0
    return engine


_PUSH_PATH = "tcp://decode:20050"


def _push_thread(
    mode: str = "async",
    engine: MagicMock | None = None,
    slots: dict[int, tuple[int, ...]] | None = None,
) -> LayerwisePullSendingThread:
    state = ProducerSendState(
        last_layer_idx=0,
        layer_layouts={0: (_component("layer.0.main", 0, 1000, 16),)},
        p_session="p-session",
        block_sizes=(16,),
        layer_storage_slots=slots if slots is not None else {0: (0,)},
        num_blocks=64,
        tp_rank=0,
        tp_size=1,
        transfer_mode="push",
        backend=PullBackend.memfabric(engine if engine is not None else _ok_engine()),
        push_write_mode=mode,
    )
    thread = LayerwisePullSendingThread(ready_event=threading.Event(), state=state)
    # Skip the handshake: tests register the D side directly.
    thread._push_ready_paths.add(_PUSH_PATH)
    thread._d_sessions[_PUSH_PATH] = "d-session"
    thread._d_layouts_by_path[_PUSH_PATH] = {0: (_component("layer.0.main", 0, 2000, 16),)}
    thread._d_topology[_PUSH_PATH] = (0, 1, frozenset())
    dealer = MagicMock()
    thread._ensure_dealer = MagicMock(return_value=dealer)
    thread._dealer_mock = dealer
    return thread


def _push_task(
    ext: str = "req",
    blocks=(0, 1, 2, 3),
    terminal: bool = False,
    member: int = 0,
    ratio: int = 1,
    layer: int = 0,
) -> SendTask:
    req_meta = LayerwisePullProducerReqMeta(
        local_block_ids=[list(blocks)],
        remote_tp_size=1,
        chunk_finish=terminal,
        chunk_start_blocks=[0],
        tp_ratio=ratio,
        group_member_idx=member,
        layer_endpoints={layer: ("decode", 20050)},
        terminal_layers=frozenset({layer}) if terminal else frozenset(),
    )
    # vLLM appends a 9-character EngineCore suffix; get_external_request_id
    # strips it, so the internal id must carry one.
    return SendTask(
        send_request={f"{ext}-12345678": req_meta},
        wait_event=None,
        layer_idx=layer,
        layer_name="layer.0.main",
    )


def _layouts_blob() -> bytes:
    return msgspec.msgpack.encode(
        {
            0: {
                "components": [
                    {
                        "name": "layer.0.main",
                        "group_index": 0,
                        "block_size": 16,
                        "dtypes": ["torch.bfloat16"],
                        "base_addrs": [2000],
                        "block_strides": [16],
                        "block_lengths": [16],
                        "block_shapes": [[16]],
                        "block_size_scales": [1],
                    }
                ]
            }
        }
    )


def _new_read_thread(tp_size: int = 1, tp_shared: frozenset[str] = frozenset()) -> LayerwisePullReadThread:
    thread = LayerwisePullReadThread.__new__(LayerwisePullReadThread)
    thread.tp_rank = 0
    thread._state = ConsumerReadState(
        tp_size=tp_size,
        layer_layouts={},
        dest_blocks_by_req={},
        tp_shared_components=tp_shared,
        d_session="d-session",
        transfer_mode="push",
    )
    thread._pending_dest_queries = {}
    thread._lock = threading.Lock()
    thread._done_requests = set()
    thread._failed_requests = set()
    thread._failed_request_ids = set()
    thread._done_contributors = {}
    thread._expected_sources = {}
    thread._terminal_requests = set()
    thread._p_completion_sources = {}
    return thread


# ---------------------------------------------------------------------------
# Handshake (extended DEST_LAYOUT_META)
# ---------------------------------------------------------------------------


def test_push_handshake_parses_extended_reply():
    thread = _push_thread()
    thread._push_ready_paths.clear()
    dealer = MagicMock()
    dealer.poll.return_value = True
    reply = (DEST_LAYOUT_META, _layouts_blob(), "d-session", 1, 4, ["layer.0.main"])
    dealer.recv_multipart.return_value = [b"", msgspec.msgpack.encode(reply)]

    thread._push_handshake(_PUSH_PATH, dealer, msgspec.msgpack.Encoder())

    assert thread._d_sessions[_PUSH_PATH] == "d-session"
    assert thread._d_topology[_PUSH_PATH] == (1, 4, frozenset({"layer.0.main"}))
    assert 0 in thread._d_layouts_by_path[_PUSH_PATH]
    sent = msgspec.msgpack.decode(dealer.send.call_args.args[0])
    assert sent[0] == PUSH_META


def test_push_handshake_accepts_legacy_three_field_reply():
    thread = _push_thread()
    thread._push_ready_paths.clear()
    dealer = MagicMock()
    dealer.poll.return_value = True
    reply = (DEST_LAYOUT_META, _layouts_blob(), "d-session")
    dealer.recv_multipart.return_value = [b"", msgspec.msgpack.encode(reply)]

    thread._push_handshake(_PUSH_PATH, dealer, msgspec.msgpack.Encoder())

    assert thread._d_sessions[_PUSH_PATH] == "d-session"
    assert thread._d_topology[_PUSH_PATH] == (0, 1, frozenset())


def test_push_handshake_rejected_by_decode_raises():
    thread = _push_thread()
    thread._push_ready_paths.clear()
    dealer = MagicMock()
    dealer.poll.return_value = True
    reply = (WRITE_FAILED, -1, "sparse offload rejected", [])
    dealer.recv_multipart.return_value = [b"", msgspec.msgpack.encode(reply)]

    with pytest.raises(RuntimeError, match="rejected"):
        thread._push_handshake(_PUSH_PATH, dealer, msgspec.msgpack.Encoder())
    assert _PUSH_PATH not in thread._push_ready_paths


# ---------------------------------------------------------------------------
# DEST_BLOCKS query / defer lifecycle
# ---------------------------------------------------------------------------


def test_dest_blocks_query_is_sent_once_per_request():
    thread = _push_thread()
    encoder = msgspec.msgpack.Encoder()

    thread._query_dest_blocks(thread._dealer_mock, encoder, _PUSH_PATH, "req")
    thread._query_dest_blocks(thread._dealer_mock, encoder, _PUSH_PATH, "req")

    thread._dealer_mock.send.assert_called_once()
    msg = msgspec.msgpack.decode(thread._dealer_mock.send.call_args.args[0])
    assert msg == [DEST_BLOCKS_REQUEST, "req"]


def test_push_task_defers_when_dest_blocks_missing():
    engine = _ok_engine()
    thread = _push_thread(engine=engine)

    thread._process_push_task(_push_task(), msgspec.msgpack.Encoder())

    assert len(thread._deferred_tasks) == 1
    engine.batch_transfer_sync_write.assert_not_called()
    engine.batch_transfer_async_write_submit.assert_not_called()
    # A DEST_BLOCKS_REQUEST went out for the pending request.
    msg = msgspec.msgpack.decode(thread._dealer_mock.send.call_args.args[0])
    assert "req" in msg


def test_decode_flushes_pending_dest_blocks_once_allocated():
    thread = _new_read_thread()
    sock = MagicMock()
    encoder = msgspec.msgpack.Encoder()

    thread._handle_dest_blocks_request(sock, b"prefill", "req", encoder)
    sock.send_multipart.assert_not_called()
    assert thread._pending_dest_queries["req"] == {b"prefill"}

    thread._state.dest_blocks_by_req["req"] = [[7, 8]]
    thread._flush_pending_dest_queries(sock, encoder)

    sock.send_multipart.assert_called_once()
    frames = sock.send_multipart.call_args.args[0]
    assert frames[0] == b"prefill"
    msg = msgspec.msgpack.decode(frames[2])
    assert msg[0] == DEST_BLOCKS
    assert msg[1] == "req"
    assert msg[2] == [[7, 8]]
    assert not thread._pending_dest_queries


def test_deferred_task_writes_once_blocks_arrive():
    engine = _ok_engine()
    thread = _push_thread(mode="sync", engine=engine)
    task = _push_task()
    thread._process_push_task(task, msgspec.msgpack.Encoder())
    assert len(thread._deferred_tasks) == 1

    thread._dest_blocks_by_req["req"] = [[10, 11, 12, 13]]
    thread._retry_deferred(msgspec.msgpack.Encoder())

    assert not thread._deferred_tasks
    engine.batch_transfer_sync_write.assert_called_once_with("d-session", [1000], [2160], [64])


def test_deferred_task_stays_pending_with_same_deadline_without_blocks():
    thread = _push_thread(mode="sync")
    task = _push_task()
    thread._process_push_task(task, msgspec.msgpack.Encoder())
    _, deadline = thread._deferred_tasks[0]

    thread._retry_deferred(msgspec.msgpack.Encoder())

    assert len(thread._deferred_tasks) == 1
    assert thread._deferred_tasks[0][1] == deadline


def test_deferred_task_times_out_with_write_failed():
    thread = _push_thread(mode="sync")
    task = _push_task()
    thread._process_push_task(task, msgspec.msgpack.Encoder())
    thread._deferred_tasks[0] = (thread._deferred_tasks[0][0], 0.0)  # expired

    thread._retry_deferred(msgspec.msgpack.Encoder())

    assert not thread._deferred_tasks
    msg = msgspec.msgpack.decode(thread._dealer_mock.send.call_args.args[0])
    assert msg[0] == WRITE_FAILED
    assert "dest blocks timeout" in msg[2]
    assert msg[3] == ["req"]


# ---------------------------------------------------------------------------
# D-side WRITE_DONE / WRITE_FAILED accounting
# ---------------------------------------------------------------------------


def test_write_done_counts_every_contributor_before_completing():
    thread = _new_read_thread()
    expected = frozenset({(0, 0), (0, 1)})
    thread._p_completion_sources[b"p0"] = ((0, 0), 2, expected)
    thread._p_completion_sources[b"p1"] = ((0, 1), 2, expected)

    thread._handle_write_done((WRITE_DONE, 0, ["req"], 1), b"p0")
    assert "req" not in thread._done_requests
    thread._handle_write_done((WRITE_DONE, 0, ["req"], 2), b"p1")
    assert thread.get_and_clear_done() == {"req"}


def test_write_failed_marks_request_failed_and_clears_done():
    thread = _new_read_thread()
    expected = frozenset({(0, 0)})
    thread._p_completion_sources[b"p0"] = ((0, 0), 1, expected)
    thread._handle_write_done((WRITE_DONE, 0, ["req"], 1), b"p0")
    assert thread.get_and_clear_done() == {"req"}

    thread._handle_write_failed((WRITE_FAILED, 0, "boom", ["req"]))

    assert thread.get_and_clear_failed() == {"req"}
    assert "req" in thread._failed_request_ids
    assert not thread._done_contributors


# ---------------------------------------------------------------------------
# Write path failures and slicing
# ---------------------------------------------------------------------------


def test_write_failure_marks_request_and_notifies_decode():
    engine = _ok_engine()
    engine.batch_transfer_sync_write.return_value = -1
    thread = _push_thread(mode="sync", engine=engine)
    thread._dest_blocks_by_req["req"] = [[10, 11, 12, 13]]

    with pytest.raises(RuntimeError, match="WRITE failed"):
        thread._process_push_task(_push_task(), msgspec.msgpack.Encoder())

    assert "req" in thread._push_failed_reqs
    msg = msgspec.msgpack.decode(thread._dealer_mock.send.call_args.args[0])
    assert msg[0] == WRITE_FAILED
    assert "req" in msg[3]


def test_push_unequal_tp_writes_only_own_slice():
    engine = _ok_engine()
    thread = _push_thread(mode="sync", engine=engine)
    thread._dest_blocks_by_req["req"] = [[10, 11, 12, 13]]

    thread._process_push_task(_push_task(member=1, ratio=2), msgspec.msgpack.Encoder())

    # _tp_block_range(4, rank=1, size=2) owns blocks [2, 4): source 2,3 → dest 12,13.
    engine.batch_transfer_sync_write.assert_called_once_with("d-session", [1032], [2192], [32])


def test_push_tp_shared_written_by_contributor_zero_with_decode_geometry():
    engine = _ok_engine()
    thread = _push_thread(mode="sync", engine=engine)
    thread._d_topology[_PUSH_PATH] = (1, 2, frozenset({"layer.0.main"}))
    thread._dest_blocks_by_req["req"] = [[10, 11, 12, 13]]

    thread._process_push_task(_push_task(member=0, ratio=2), msgspec.msgpack.Encoder())

    # Sliced by the D rank's geometry: _tp_block_range(4, d_rank=1, d_size=2).
    engine.batch_transfer_sync_write.assert_called_once_with("d-session", [1032], [2192], [32])


def test_push_tp_shared_nonzero_contributor_issues_no_write():
    engine = _ok_engine()
    thread = _push_thread(mode="sync", engine=engine)
    thread._d_topology[_PUSH_PATH] = (1, 2, frozenset({"layer.0.main"}))
    thread._dest_blocks_by_req["req"] = [[10, 11, 12, 13]]

    thread._process_push_task(_push_task(member=1, ratio=2, terminal=True), msgspec.msgpack.Encoder())

    engine.batch_transfer_sync_write.assert_not_called()
    engine.batch_transfer_async_write_submit.assert_not_called()
    # Completion still flows: the terminal-layer WRITE_DONE is delivered.
    msg = msgspec.msgpack.decode(thread._dealer_mock.send.call_args.args[0])
    assert msg[0] == WRITE_DONE
    assert msg[2] == ["req"]


# ---------------------------------------------------------------------------
# Async write submission and completion reaping
# ---------------------------------------------------------------------------


def _make_async_thread(engine: MagicMock, ready: bool):
    thread = _push_thread(mode="async", engine=engine, slots={0: (0,), 1: (0,)})
    thread._push_stream = SimpleNamespace(npu_stream=4321)
    event = SimpleNamespace(query=lambda: ready)
    thread._record_push_event = lambda: event
    return thread, event


def test_async_write_releases_slots_and_reports_done_only_after_event():
    engine = _ok_engine()
    thread, event = _make_async_thread(engine, ready=False)
    thread._dest_blocks_by_req["req"] = [[10, 11, 12, 13]]
    encoder = msgspec.msgpack.Encoder()

    thread._process_push_task(_push_task(terminal=True), encoder)

    # Submitted onto the push stream with the raw NPU handle.
    engine.batch_transfer_async_write_submit.assert_called_once_with("d-session", [1000], [2160], [64], 4321)
    assert len(thread._inflight_writes) == 1
    # Slot 0 is shared by layers 0 and 1: it must stay blocked while in flight.
    assert not thread.get_storage_send_event(0).is_set()
    thread._dealer_mock.send.assert_not_called()

    thread._reap_write_completions(encoder)
    assert not thread.get_storage_send_event(0).is_set()

    event.query = lambda: True
    thread._reap_write_completions(encoder)
    assert thread.get_storage_send_event(0).is_set()
    assert not thread._inflight_writes
    msg = msgspec.msgpack.decode(thread._dealer_mock.send.call_args.args[0])
    assert msg[0] == WRITE_DONE
    assert msg[2] == ["req"]


def test_async_submit_failure_fails_request_immediately():
    engine = _ok_engine()
    engine.batch_transfer_async_write_submit.return_value = -1
    thread, _ = _make_async_thread(engine, ready=False)
    thread._dest_blocks_by_req["req"] = [[10, 11, 12, 13]]

    with pytest.raises(RuntimeError, match="async WRITE submit failed"):
        thread._process_push_task(_push_task(), msgspec.msgpack.Encoder())

    assert "req" in thread._push_failed_reqs
    assert not thread._inflight_writes
    msg = msgspec.msgpack.decode(thread._dealer_mock.send.call_args.args[0])
    assert msg[0] == WRITE_FAILED


def test_sync_write_mode_completes_inline_without_inflight():
    engine = _ok_engine()
    thread = _push_thread(mode="sync", engine=engine, slots={0: (0,), 1: (0,)})
    thread._dest_blocks_by_req["req"] = [[10, 11, 12, 13]]

    thread._process_push_task(_push_task(terminal=True), msgspec.msgpack.Encoder())

    engine.batch_transfer_sync_write.assert_called_once_with("d-session", [1000], [2160], [64])
    engine.batch_transfer_async_write_submit.assert_not_called()
    assert not thread._inflight_writes
    # Sync return means the payload left the source buffers: slots free at once.
    assert thread.get_storage_send_event(0).is_set()
    msg = msgspec.msgpack.decode(thread._dealer_mock.send.call_args.args[0])
    assert msg[0] == WRITE_DONE
