# SPDX-License-Identifier: Apache-2.0

import socket
import threading
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from vllm_ascend.distributed.kv_transfer import register_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.connector import (
    LayerwisePushConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.protocol import (
    LAYOUT_META,
    REQUEST_DONE,
    ComponentLayout,
    LayerwisePushConsumerMetadata,
    LayerwisePushHandshakeMetadata,
    LayerwisePushProducerMetadata,
    LayerwisePushProducerReqMeta,
    SendTask,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.receive_thread import (
    ConsumerDestinationState,
    LayerwisePushReceiveThread,
    WriteBackend,
    WritePlanner,
    _tp_block_range,
    plan_block_transfers,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.scheduler import (
    LayerwisePushConsumerScheduler,
    LayerwisePushProducerScheduler,
    _SendReqInfo,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.send_thread import (
    LayerwisePushSendingThread,
    ProducerSendState,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.worker import (
    LayerwisePushConsumerWorker,
    LayerwisePushProducerWorker,
    _resolve_kv_transfer_backend,
    _resolve_push_write_mode,
)


def _wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(0.005)
    return False


@pytest.fixture
def push_channels():
    receivers, senders = [], []
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.receive_thread.get_ip", return_value="127.0.0.1"
        ),
        patch("vllm.distributed.get_world_group", return_value=SimpleNamespace(local_rank=0)),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.send_thread.torch"),
    ):

        def create(backend, ready_targets=True):
            endpoints = []
            for index in range(2):
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                receiver = LayerwisePushReceiveThread(
                    0,
                    port,
                    ConsumerDestinationState(
                        tp_size=1,
                        layer_layouts={layer: (_component(f"layer.{layer}", 0, 2000, 16),) for layer in range(2)},
                        dest_blocks_by_req={f"request{index}": [[5, 6]]} if ready_targets else {},
                        session=f"decode-{index}",
                    ),
                )
                receivers.append(receiver)
                receiver.start()
                assert receiver.ready_event.wait(2)
                assert receiver.startup_error is None
                endpoints.append(("127.0.0.1", port))
            sender = LayerwisePushSendingThread(
                ready_event=threading.Event(),
                backend=backend,
                state=ProducerSendState(
                    last_layer_idx=1,
                    layer_layouts={layer: (_component(f"layer.{layer}", 0, 1000, 16),) for layer in range(2)},
                    p_session="prefill",
                    block_sizes=(16,),
                    layer_storage_slots={0: (0,), 1: (0,)},
                    num_blocks=2,
                    pp_layers=((0, 1),),
                    write_mode="sync",
                ),
            )
            senders.append(sender)
            sender.start()
            assert sender.ready_event.wait(2)
            assert sender.startup_error is None
            requests = {
                f"request{index}": LayerwisePushProducerReqMeta(
                    local_block_ids=[[0, 1]],
                    remote_tp_size=1,
                    chunk_finish=True,
                    chunk_start_blocks=[0],
                    layer_endpoints={0: endpoint, 1: endpoint},
                    terminal_layers=frozenset({1}),
                )
                for index, endpoint in enumerate(endpoints)
            }
            for request_id, request in requests.items():
                sender.track_requests({request_id}, {request.layer_endpoints[0]})
            return sender, receivers, requests

        yield create
        for sender in senders:
            sender.stop(timeout=3)
            assert not sender.is_alive()
        for receiver in receivers:
            receiver.stop(timeout=3)
            assert not receiver.is_alive()


def test_parallel_write_keeps_other_endpoint_moving_and_preserves_order(push_channels):
    blocked, release = threading.Event(), threading.Event()
    calls = []
    lock = threading.Lock()

    def write(session, sources, targets, lengths):
        with lock:
            calls.append((session, sources, targets, lengths))
            first = sum(call[0] == session for call in calls) == 1
        if session == "decode-0" and first:
            blocked.set()
            assert release.wait(3)

    backend = MagicMock()
    backend.write.side_effect = write
    sender, receivers, requests = push_channels(backend)
    try:
        sender.enqueue(SendTask(requests, layer_idx=0))
        assert blocked.wait(2)
        assert _wait_until(lambda: any(call[0] == "decode-1" for call in calls))
        assert not sender.get_storage_send_event(0).is_set()
        # Enqueue the next layer without computing into the shared buffer:
        # test submission order only; real callers wait before overwriting it.
        sender.enqueue(SendTask(requests, layer_idx=1))
        assert _wait_until(lambda: sum(call[0] == "decode-1" for call in calls) == 2)
        assert sum(call[0] == "decode-0" for call in calls) == 1
        assert sender.get_and_clear_finished_requests(set(requests)) <= {"request1"}
        release.set()
        assert sender.get_storage_send_event(0).wait(2)
        assert _wait_until(lambda: "request0" in sender._completed_requests)
        for index, receiver in enumerate(receivers):
            assert _wait_until(lambda index=index, receiver=receiver: f"request{index}" in receiver._done_requests)
        assert all(call[1:] == ([1000], [2080], [32]) for call in calls)
        assert sender.fatal_error is None
    finally:
        release.set()


def test_same_layer_tasks_keep_their_own_source_events(push_channels):
    backend = MagicMock()
    sender, _, requests = push_channels(backend)
    worker = LayerwisePushProducerWorker.__new__(LayerwisePushProducerWorker)
    worker.kv_send_layer_thread = sender
    worker._layer_order = (0, 1)
    worker.current_layer = 0
    worker.total_base_layers = 2
    worker.layer_layouts = sender._state.layer_layouts
    source_events = [MagicMock(), MagicMock()]
    second_ready = threading.Event()
    source_events[0].query.return_value = True
    source_events[1].query.side_effect = second_ready.is_set
    queued_tasks = []

    # Hold both batches' tasks before the control thread consumes either one.
    # Only event/dispatch ordering is tested; no real shared buffer is written.
    with (
        patch.object(sender, "enqueue", side_effect=queued_tasks.append),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.worker.torch") as worker_torch,
    ):
        worker_torch.npu.Event.side_effect = source_events
        for index in range(2):
            worker._pd_dispatched_layers = set()
            metadata = LayerwisePushProducerMetadata()
            metadata.requests = {f"request{index}": requests[f"request{index}"]}
            worker.on_kv_cache_written("model.layers.0.self_attn.attn", metadata)

    assert len(queued_tasks) == 2
    for task, event in zip(queued_tasks, source_events, strict=True):
        assert task.wait_event is event
        event.record.assert_called_once_with()
        sender.enqueue(task)
    try:
        assert _wait_until(lambda: backend.write.call_count == 1 and source_events[1].query.called)
        assert backend.write.call_args.args[0] == "decode-0"
        second_ready.set()
        assert _wait_until(lambda: backend.write.call_count == 2)
        assert backend.write.call_args.args[0] == "decode-1"
        assert sender.fatal_error is None
    finally:
        second_ready.set()


def test_same_layer_queued_tasks_keep_independent_slot_reservations():
    sender = LayerwisePushSendingThread(
        ready_event=threading.Event(),
        backend=MagicMock(),
        state=ProducerSendState(
            last_layer_idx=1,
            layer_layouts={
                0: (_component("layer.0", 0, 1000, 16),),
                1: (_component("layer.1", 0, 1000, 16),),
            },
            p_session="prefill",
            block_sizes=(16,),
            layer_storage_slots={0: (0,), 1: (0,)},
            write_mode="sync",
        ),
    )
    first = SendTask({}, layer_idx=0)
    second = SendTask({}, layer_idx=0)
    sender.enqueue(first)
    sender.enqueue(second)
    assert not sender.get_storage_send_event(0).is_set()

    sender._process_send_task(sender.send_queue.get_nowait(), msgspec.msgpack.Encoder())
    assert not sender.get_storage_send_event(0).is_set()

    sender._process_send_task(sender.send_queue.get_nowait(), msgspec.msgpack.Encoder())
    assert sender.get_storage_send_event(0).is_set()


def test_target_registration_is_async_and_cached_across_layers(push_channels):
    backend = MagicMock()
    sender, receivers, requests = push_channels(backend, ready_targets=False)
    sender.enqueue(SendTask(requests, layer_idx=0))
    assert _wait_until(lambda: all(receiver._target_waiters for receiver in receivers))
    assert backend.write.call_count == 0
    # One D stays unready; the other must still make progress.
    receivers[1]._state.dest_blocks_by_req["request1"] = [[5, 6]]
    receivers[1].notify_targets()
    assert _wait_until(lambda: backend.write.call_count == 1)
    assert backend.write.call_args.args[0] == "decode-1"
    sender.enqueue(SendTask({"request1": requests["request1"]}, layer_idx=1))
    assert _wait_until(lambda: backend.write.call_count == 2)
    assert not receivers[1]._target_waiters
    assert sender.fatal_error is None


def test_only_request_completion_is_notified_without_waiting_for_decode(push_channels):
    backend = MagicMock()
    sender, receivers, requests = push_channels(backend)
    receiver = receivers[0]
    request_id = "request0"
    request = {request_id: requests[request_id]}
    receiving, release = threading.Event(), threading.Event()
    notifications = []
    record_done = receiver._record_chunk_done

    def delay_completion(done_ext_ids, contributor, expected):
        notifications.append(list(done_ext_ids))
        if done_ext_ids:
            receiving.set()
            assert release.wait(3)
        record_done(done_ext_ids, contributor, expected)

    with patch.object(receiver, "_record_chunk_done", side_effect=delay_completion):
        try:
            sender.enqueue(SendTask(request, layer_idx=0))
            assert sender.get_storage_send_event(0).wait(2)
            assert backend.write.call_count == 1
            assert receiver.get_and_clear_done() == set()
            path = next(iter(sender._planners))
            assert request_id in sender._planners[path]._state.dest_blocks_by_req

            sender.enqueue(SendTask(request, layer_idx=1))
            assert receiving.wait(2)
            # Messages on this connection are ordered: reaching REQUEST_DONE
            # also proves that no earlier layer sent an empty notification.
            assert notifications == [[request_id]]
            assert _wait_until(lambda: request_id in sender._completed_requests)
            assert sender.get_storage_send_event(0).is_set()
            assert request_id not in sender._planners[path]._state.dest_blocks_by_req
            assert request_id not in sender._requested_targets[path]
            assert receiver.get_and_clear_done() == set()
            # Even while D is handling completion, P only waits for its own
            # scheduler finish; neither slot nor request release needs an ACK.
            assert sender.get_and_clear_finished_requests(set()) == set()
            assert sender.get_and_clear_finished_requests({request_id}) == {request_id}
        finally:
            release.set()
        assert _wait_until(lambda: request_id in receiver._done_requests)
        assert sender.fatal_error is None


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_write_failure_does_not_release_requests_as_success(push_channels, layer_idx):
    backend = MagicMock()
    backend.write.side_effect = RuntimeError("WRITE timeout")
    sender, receivers, requests = push_channels(backend)
    sender.enqueue(SendTask(requests, layer_idx=layer_idx))
    assert _wait_until(lambda: sender.fatal_error is not None)
    with pytest.raises(RuntimeError, match="source blocks remain pinned"):
        sender.get_and_clear_finished_requests(set(requests))
    for receiver in receivers:
        assert _wait_until(lambda receiver=receiver: receiver.fatal_error is not None)
        assert receiver.get_and_clear_done() == set()
        assert receiver.get_and_clear_failed() == set()
        assert receiver._state.dest_blocks_by_req


def test_write_component_matching_is_cached_per_layer():
    source = {0: (_component("layer.0", 0, 1000, 16),)}
    planner = WritePlanner(
        MagicMock(),
        ConsumerDestinationState(
            tp_size=1,
            layer_layouts={0: (_component("layer.0", 0, 2000, 16),)},
            dest_blocks_by_req={"request": [[5, 6]]},
        ),
        0,
    )
    with patch.object(planner, "_match_components", wraps=planner._match_components) as match:
        for block in range(2):
            planner.write_batch(b"p", 0, [("request", [[block]], [block])], "decode", source)
        match.assert_called_once()
    assert planner.backend.write.call_args.args == ("decode", [1016], [2096], [16])


def test_async_empty_terminal_waits_behind_prior_write_and_its_own_event():
    class Event:
        ready = False
        recorded_stream = None

        def record(self, stream):
            self.recorded_stream = stream

        def query(self):
            return self.ready

    path = "tcp://decode:1234"
    write_event = Event()
    terminal_event = Event()
    stream = SimpleNamespace(npu_stream=321)
    engine = MagicMock()
    engine.batch_transfer_async_write_submit.return_value = 0
    backend = WriteBackend.memfabric(engine)
    sender = LayerwisePushSendingThread(
        ready_event=threading.Event(),
        backend=backend,
        state=ProducerSendState(
            last_layer_idx=0,
            layer_layouts={0: (_component("layer.0", 0, 1000, 16),)},
            p_session="prefill",
            block_sizes=(16,),
            layer_storage_slots={0: (0,)},
            write_mode="async",
        ),
    )
    sender._push_stream = stream
    sender._dealers[path] = MagicMock()
    sender._requested_targets[path] = {"request"}
    sender._planners[path] = WritePlanner(
        backend,
        ConsumerDestinationState(
            tp_size=1,
            layer_layouts={0: (_component("layer.0", 0, 2000, 16),)},
            dest_blocks_by_req={"request": [[5]]},
            session="decode",
        ),
        0,
    )
    sender._pending[path] = deque(
        [
            (0, 0, [("request", [[0]], [0])], [], 0, 1, None),
            (1, 0, [], ["request"], 0, 1, None),
        ]
    )

    with patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.send_thread.torch") as mock_torch:
        mock_torch.npu.Event.side_effect = [write_event, terminal_event]
        sender._dispatch_writes(msgspec.msgpack.Encoder())
        assert write_event.recorded_stream is stream
        assert path in sender._active
        sender._dealers[path].send.assert_not_called()

        sender._dispatch_writes(msgspec.msgpack.Encoder())
        assert terminal_event.recorded_stream is None
        sender._dealers[path].send.assert_not_called()

        write_event.ready = True
        sender._dispatch_writes(msgspec.msgpack.Encoder())
        assert terminal_event.recorded_stream is stream
        sender._dealers[path].send.assert_not_called()

        terminal_event.ready = True
        sender._dispatch_writes(msgspec.msgpack.Encoder())

    message = msgspec.msgpack.decode(sender._dealers[path].send.call_args.args[0])
    assert message == [REQUEST_DONE, ["request"]]
    engine.batch_transfer_async_write_submit.assert_called_once_with("decode", [1000], [2080], [16], 321)
    assert path not in sender._active


def test_layout_handshake_has_a_deadline():
    sender = LayerwisePushSendingThread(
        ready_event=threading.Event(),
        backend=MagicMock(),
        state=ProducerSendState(
            last_layer_idx=0,
            layer_layouts={},
            p_session="prefill",
            block_sizes=(16,),
            layer_storage_slots={},
        ),
    )
    sender._handshake_deadlines["tcp://decode:1234"] = time.monotonic() - 1

    with pytest.raises(TimeoutError, match="layout handshake"):
        sender._check_handshake_deadlines()


def test_poll_timeout_uses_nearest_event_or_handshake_deadline():
    sender = LayerwisePushSendingThread(
        ready_event=threading.Event(),
        backend=MagicMock(),
        state=ProducerSendState(
            last_layer_idx=0,
            layer_layouts={},
            p_session="prefill",
            block_sizes=(16,),
            layer_storage_slots={},
        ),
    )
    sender._handshake_deadlines["tcp://decode:1234"] = time.monotonic() + 0.05

    assert 0 <= sender._poll_timeout_ms() <= 50
    sender._waiting_for_event = True
    assert 0 <= sender._poll_timeout_ms() <= 1


def _component(
    name: str,
    group_index: int,
    base: int,
    length: int,
) -> ComponentLayout:
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


def test_only_generic_layerwise_push_connector_is_registered():
    with (
        patch.object(KVConnectorFactory, "_registry", {}),
        patch.object(KVConnectorFactory, "register_connector") as register,
    ):
        register_connector()

    calls = [call.args for call in register.call_args_list]
    assert (
        "LayerwisePushConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.connector",
        "LayerwisePushConnector",
    ) in calls
    assert "MooncakeLayerwiseConnector" not in {args[0] for args in calls}
    assert "SfaRemoteD2HConnector" not in {args[0] for args in calls}


def test_slot_release_wait_is_only_exposed_through_dedicated_callback():
    connector = LayerwisePushConnector.__new__(LayerwisePushConnector)
    connector.connector_worker = MagicMock()

    connector.wait_for_layer_load("model.layers.7.self_attn")

    connector.connector_worker.wait_for_slot_release.assert_not_called()

    connector.wait_for_slot_release(7)

    connector.connector_worker.wait_for_slot_release.assert_called_once_with(7)


def test_consumer_metadata_preserves_every_cache_group():
    metadata = LayerwisePushConsumerMetadata()
    block_ids = [[1, 2], [10], [20, 21, 22]]

    metadata.add_request("request", block_ids)
    block_ids[0].append(3)

    assert metadata.requests[0].block_ids_by_group == [[1, 2], [10], [20, 21, 22]]


@pytest.mark.parametrize("backend_name", ["memfabric", "mooncake"])
@pytest.mark.parametrize("sparse_enabled", [False, True])
def test_destination_registration(backend_name, sparse_enabled):
    name = "model.layers.0.self_attn.attn"
    tensors = tuple(torch.empty(4, 16, 1, dim, dtype=torch.bfloat16) for dim in (8, 4))
    worker = LayerwisePushConsumerWorker.__new__(LayerwisePushConsumerWorker)
    worker.kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[SimpleNamespace(layer_names=[name], kv_cache_spec=SimpleNamespace(block_size=16))],
    )
    worker.total_base_layers = 1
    worker.tp_rank, worker.tp_size = 0, 1
    worker.side_channel_host, worker.side_channel_port = "decode", 1234
    worker._backend_name = backend_name
    worker._dest_blocks_by_req = {}
    worker._dest_blocks_condition = threading.Condition()
    worker._ensure_engine = MagicMock(return_value=(MagicMock(), MagicMock()))
    engine_name = "global_memfabric_te" if backend_name == "memfabric" else "global_te"
    with (
        patch("vllm_ascend.ascend_config.get_ascend_config") as config,
        patch(f"vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.worker.{engine_name}") as engine,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.worker.LayerwisePushReceiveThread"
        ) as receiver,
    ):
        config.return_value.sparse_kv_offload_config.enabled = sparse_enabled
        receiver.return_value.startup_error = None
        if sparse_enabled and backend_name != "memfabric":
            with pytest.raises(ValueError, match="requires .*transfer_backend.*memfabric"):
                worker.register_kv_caches({name: (None, None)})
            worker._ensure_engine.assert_not_called()
            engine.register_buffer.assert_not_called()
            receiver.assert_not_called()
        elif sparse_enabled:
            k_tensor = torch.empty(4, 16, 1, 8, dtype=torch.bfloat16)
            v_tensor = torch.empty(4, 16, 1, 4, dtype=torch.bfloat16)
            manager = SimpleNamespace(
                offload_layer_names=[name],
                gvas_k_bases=[0x1000],
                gvas_v_bases=[0x2000],
                cpu_block_lens=[(k_tensor.numel() * 2, v_tensor.numel() * 2)],
                topk_buffers_k=[k_tensor],
                topk_buffers_v=[v_tensor],
                block_size=16,
            )
            with patch(
                "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager."
                "get_sparse_kv_offload_manager",
                return_value=manager,
            ):
                worker.register_kv_caches({name: (None, None)})
            engine.register_buffer.assert_called_once_with(
                [0x1000, 0x2000],
                [1024 * 4, 512 * 4],
            )
            component = worker.layer_layouts[0][0]
            assert component.base_addrs == (0x1000, 0x2000)
            assert receiver.call_args.kwargs["state"].tp_shared_components == frozenset({name})
            receiver.return_value.start.assert_called_once()
        else:
            worker.register_kv_caches({name: tensors})
            engine.register_buffer.assert_called_once_with(
                [tensor.data_ptr() for tensor in tensors],
                [tensor.numel() * tensor.element_size() for tensor in tensors],
            )
            assert worker.layer_layouts[0][0].base_addrs == tuple(tensor.data_ptr() for tensor in tensors)
            assert receiver.call_args.kwargs["state"].tp_shared_components == frozenset()
            receiver.return_value.start.assert_called_once()


@pytest.fixture
def consumer_scheduler():
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_port=1234),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_rank=0,
            data_parallel_size=1,
        ),
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))])
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.scheduler.get_ip",
        return_value="127.0.0.1",
    ):
        scheduler = LayerwisePushConsumerScheduler(config, True, kv_cache_config)
    scheduler._submit_metaserver_request = MagicMock()
    scheduler.remote_endpoints = [[{"host": "127.0.0.1", "port": 1234, "layer_ids": [0]}]]
    scheduler.remote_topology_id = "decode-topology"
    try:
        yield scheduler
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("cache_lookups", [(0,), (64,), (32, 64)])
def test_consumer_cache_hit_excludes_cached_blocks_from_producer_metadata(consumer_scheduler, cache_lookups):
    request = SimpleNamespace(
        request_id="request123456789",
        prompt_token_ids=list(range(96)),
        num_computed_tokens=0,
        kv_transfer_params={"do_remote_prefill": True, "metaserver": "http://metaserver"},
    )
    for hit in cache_lookups:
        assert consumer_scheduler.get_num_new_matched_tokens(request, hit) == (96 - hit, True)
    hit = cache_lookups[-1]
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([20, 21, 22, 23, 24, 25],)
    consumer_scheduler.update_state_after_alloc(request, blocks, 96 - hit)

    message = consumer_scheduler._submit_metaserver_request.call_args.kwargs["message"]
    assert request.num_computed_tokens == 0
    assert message["remote_cached_tokens"] == hit
    assert message["remote_topology_id"] == consumer_scheduler.remote_topology_id
    assert request.request_id not in consumer_scheduler._local_cached_tokens

    producer = LayerwisePushProducerScheduler.__new__(LayerwisePushProducerScheduler)
    producer.producer_pp_layers = ((0,),)
    producer.block_size = [16]
    producer._reqs_finalized_layerwise = set()
    producer._reqs_need_send_layerwise = {
        "prefill": _SendReqInfo(
            local_block_ids=[[10, 11, 12, 13, 14, 15]],
            local_transferred_tokens=hit,
            local_computed_tokens=0,
            request=SimpleNamespace(all_token_ids=list(range(96)), kv_transfer_params=message),
        )
    }
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        scheduled_new_reqs=[SimpleNamespace(req_id="prefill", num_computed_tokens=0)],
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={"prefill": 96},
    )
    metadata = producer.build_connector_meta(output).requests["prefill"]
    assert metadata.chunk_start_blocks == [hit // 16]
    assert metadata.local_block_ids == [list(range(10 + hit // 16, 16))]
    assert metadata.remote_topology_id == consumer_scheduler.remote_topology_id


def test_consumer_cleans_cached_tokens_when_cancelled_before_allocation(consumer_scheduler):
    request = SimpleNamespace(
        request_id="request123456789",
        prompt_token_ids=list(range(96)),
        kv_transfer_params={"do_remote_prefill": True},
    )
    consumer_scheduler.get_num_new_matched_tokens(request, 64)
    assert request.request_id in consumer_scheduler._local_cached_tokens

    assert consumer_scheduler.request_finished_all_groups(request, ([],)) == (False, None)

    assert request.request_id not in consumer_scheduler._local_cached_tokens
    consumer_scheduler._submit_metaserver_request.assert_not_called()


def test_write_thread_plans_arbitrary_components_in_one_backend_write():
    engine = MagicMock()
    engine.batch_transfer_sync_write.return_value = 0
    source = {
        0: (
            _component("layer.0.c0", 0, 1000, 16),
            _component("layer.0.c1", 1, 3000, 8),
        )
    }
    destination = {
        0: (
            _component("layer.0.c0", 0, 2000, 16),
            _component("layer.0.c1", 1, 4000, 8),
        )
    }
    thread = WritePlanner.__new__(WritePlanner)
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 0
    thread.backend = WriteBackend.memfabric(engine)
    thread._state = ConsumerDestinationState(
        tp_size=1,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[5, 6], [7, 8]], "second": [[9], [12]]},
    )

    thread.write_batch(
        b"prefill",
        0,
        [("request", [[1, 2], [3, 4]], [0, 0]), ("second", [[5], [8]], [0, 0])],
        "prefill-session",
        source,
    )

    engine.batch_transfer_sync_write.assert_called_once_with(
        "prefill-session",
        [1016, 1080, 3024, 3064],
        [2080, 2144, 4056, 4096],
        [32, 16, 16, 8],
    )


@pytest.mark.parametrize("contiguous_requests", [False, True])
def test_write_batch_reuses_numpy_block_ids_across_tensors(contiguous_requests):
    engine = MagicMock()
    engine.batch_transfer_sync_write.return_value = 0
    source = ComponentLayout(
        name="layer.0",
        group_index=0,
        block_size=16,
        dtypes=("torch.bfloat16",) * 2,
        base_addrs=(1000, 3000),
        block_strides=(16, 16),
        block_lengths=(16, 16),
        block_shapes=((8,), (8,)),
        block_size_scales=(1, 1),
    )
    destination = replace(source, base_addrs=(2000, 4000))
    requests = [("first", [[1, 2]], [0]), ("second", [[3, 4]], [0])]
    second_blocks = [7, 8] if contiguous_requests else [9, 11]
    destination_blocks = {"first": [[5, 6]], "second": [second_blocks]}
    thread = WritePlanner.__new__(WritePlanner)
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 0
    thread.backend = WriteBackend.memfabric(engine)
    thread._state = ConsumerDestinationState(
        tp_size=1,
        layer_layouts={0: (destination,)},
        dest_blocks_by_req=destination_blocks,
    )

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.receive_thread.plan_block_transfers",
        wraps=plan_block_transfers,
    ) as planner:
        thread.write_batch(b"prefill", 0, requests, "prefill-session", {0: (source,)})

    assert planner.call_count == 2
    first, second = planner.call_args_list
    for key in ("source_block_ids", "destination_block_ids"):
        assert isinstance(first.kwargs[key], np.ndarray)
        assert first.kwargs[key] is second.kwargs[key]
    np.testing.assert_array_equal(first.kwargs["source_block_ids"], [1, 2, 3, 4])
    np.testing.assert_array_equal(first.kwargs["destination_block_ids"], [5, 6, *second_blocks])
    assert requests == [("first", [[1, 2]], [0]), ("second", [[3, 4]], [0])]
    assert destination_blocks == {"first": [[5, 6]], "second": [second_blocks]}
    engine.batch_transfer_sync_write.assert_called_once_with(
        "prefill-session",
        [1016, 3016] if contiguous_requests else [1016, 1048, 1064, 3016, 3048, 3064],
        [2080, 4080] if contiguous_requests else [2080, 2144, 2176, 4080, 4144, 4176],
        [64, 64] if contiguous_requests else [32, 16, 16, 32, 16, 16],
    )


@pytest.mark.parametrize(
    ("shared", "member", "tp_rank", "expected_source", "expected_destination"),
    [
        (False, 0, 0, [1, 4], [5, 11]),
        (False, 1, 0, [2, 3], [6, 10]),
        (True, 0, 0, [1, 4], [5, 11]),
        (True, 0, 1, [2, 3], [6, 10]),
        (True, 1, 0, [], []),
    ],
)
def test_write_batch_splits_each_request_before_combining_blocks(
    shared, member, tp_rank, expected_source, expected_destination
):
    engine = MagicMock()
    engine.batch_transfer_sync_write.return_value = 0
    source = _component("layer.0", 0, 1000, 16)
    destination = replace(source, base_addrs=(2000,))
    thread = WritePlanner.__new__(WritePlanner)
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = tp_rank
    thread.backend = WriteBackend.memfabric(engine)
    thread._state = ConsumerDestinationState(
        tp_size=2,
        layer_layouts={0: (destination,)},
        dest_blocks_by_req={"first": [[5, 6]], "second": [[9, 10, 11]], "empty": [[]]},
        tp_shared_components=frozenset({"layer.0"}) if shared else frozenset(),
    )
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.receive_thread.plan_block_transfers",
        wraps=plan_block_transfers,
    ) as planner:
        thread.write_batch(
            b"prefill",
            0,
            [("first", [[1, 2]], [0]), ("second", [[3, 4]], [1]), ("empty", [[]], [0])],
            "prefill-session",
            {0: (source,)},
            group_member_idx=member,
            ratio=2,
        )

    if expected_source:
        planner.assert_called_once()
        np.testing.assert_array_equal(planner.call_args.kwargs["source_block_ids"], expected_source)
        np.testing.assert_array_equal(planner.call_args.kwargs["destination_block_ids"], expected_destination)
        engine.batch_transfer_sync_write.assert_called_once_with(
            "prefill-session",
            [1000 + block * 16 for block in expected_source],
            [2000 + block * 16 for block in expected_destination],
            [16, 16],
        )
    else:
        planner.assert_not_called()
        engine.batch_transfer_sync_write.assert_not_called()


def test_write_batch_rejects_incomplete_request_before_planning():
    source = _component("layer.0", 0, 1000, 16)
    thread = WritePlanner.__new__(WritePlanner)
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.backend = MagicMock()
    thread._state = ConsumerDestinationState(
        tp_size=1,
        layer_layouts={0: (replace(source, base_addrs=(2000,)),)},
        dest_blocks_by_req={"first": [[5, 6]], "second": [[7]]},
    )
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.receive_thread.plan_block_transfers"
        ) as planner,
        pytest.raises(RuntimeError, match="D block range is incomplete"),
    ):
        thread.write_batch(
            b"prefill",
            0,
            [("first", [[1, 2]], [0]), ("second", [[3, 4]], [0])],
            "prefill-session",
            {0: (source,)},
        )
    planner.assert_not_called()
    thread.backend.write.assert_not_called()


@pytest.fixture
def destination_registration():
    worker = LayerwisePushConsumerWorker.__new__(LayerwisePushConsumerWorker)
    worker._pending_recv_req_ids = set()
    worker._deferred_cleanup_req_ids = set()
    worker.request_map = {}
    worker._dest_blocks_by_req = {}
    worker._dest_blocks_condition = threading.Condition()
    state = ConsumerDestinationState(
        tp_size=1,
        layer_layouts={0: (_component("layer.0", 0, 2000, 16),)},
        dest_blocks_by_req=worker._dest_blocks_by_req,
        dest_blocks_condition=worker._dest_blocks_condition,
    )
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.receive_thread.get_ip",
        return_value="127.0.0.1",
    ):
        thread = LayerwisePushReceiveThread(0, 1234, state)
    worker.tp_size = 1
    worker._receive_thread = thread
    worker._terminal_ext_ids = set()
    worker._invalid_block_ids = set()
    metadata = LayerwisePushConsumerMetadata()
    metadata.add_request("request123456789", [[5, 6]])
    yield worker, thread, metadata
    thread._wake_reader.close()
    thread._wake_writer.close()


@pytest.mark.parametrize("blocks_registered", [False, True])
def test_write_thread_rejects_component_layout_mismatch_before_transfer(blocks_registered):
    engine = MagicMock()
    source = {0: (_component("layer.0.c0", 0, 1000, 16),)}
    destination = {0: (replace(source[0][0], base_addrs=(2000,), block_lengths=(32,)),)}
    thread = WritePlanner.__new__(WritePlanner)
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 0
    thread.backend = WriteBackend.mooncake(engine)
    thread._state = ConsumerDestinationState(
        tp_size=1,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[5]]} if blocks_registered else {},
    )

    with pytest.raises(RuntimeError, match="tensor size differs"):
        thread.write_batch(
            b"prefill",
            0,
            [("request", [[1]], [0])],
            "prefill-session",
            source,
        )
    engine.batch_transfer_sync_write.assert_not_called()


def test_write_thread_rejects_same_size_different_dtype():
    engine = MagicMock()
    source_component = _component("layer.0.c0", 0, 1000, 16)
    destination_component = ComponentLayout(
        name="layer.0.c0",
        group_index=0,
        block_size=16,
        dtypes=("torch.float16",),
        base_addrs=(2000,),
        block_strides=(16,),
        block_lengths=(16,),
        block_shapes=((16,),),
        block_size_scales=(1,),
    )
    thread = WritePlanner.__new__(WritePlanner)
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 0
    thread.backend = WriteBackend.mooncake(engine)
    thread._state = ConsumerDestinationState(
        tp_size=1,
        layer_layouts={0: (destination_component,)},
        dest_blocks_by_req={"request": [[5]]},
    )

    with pytest.raises(RuntimeError, match="dtype differs"):
        thread.write_batch(
            b"prefill",
            0,
            [("request", [[1]], [0])],
            "prefill-session",
            {0: (source_component,)},
        )
    engine.batch_transfer_sync_write.assert_not_called()


def test_tp_shared_component_is_written_once_and_split_across_decode_tp():
    engine = MagicMock()
    engine.batch_transfer_sync_write.return_value = 0
    source = {0: (_component("layer.0.main", 0, 1000, 16),)}
    destination = {0: (_component("layer.0.main", 0, 2000, 16),)}
    thread = WritePlanner.__new__(WritePlanner)
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 1
    thread.backend = WriteBackend.memfabric(engine)
    thread._state = ConsumerDestinationState(
        tp_size=2,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[10, 11, 12, 13]]},
        tp_shared_components=frozenset({"layer.0.main"}),
    )

    thread.write_batch(
        b"prefill",
        0,
        [("request", [[0, 1, 2, 3]], [0])],
        "prefill-session",
        source,
        group_member_idx=0,
        ratio=2,
    )

    engine.batch_transfer_sync_write.assert_called_once_with(
        "prefill-session",
        [1032],
        [2192],
        [32],
    )


@pytest.mark.parametrize(
    ("prompt_length", "expected_block_ids"),
    [(32, [[1], []]), (17, [[1, 2], [3]])],
)
def test_producer_scheduler_precomputes_chunk_block_ranges(prompt_length, expected_block_ids):
    scheduler = LayerwisePushProducerScheduler.__new__(LayerwisePushProducerScheduler)
    scheduler.block_size = [16, 32]
    scheduler.producer_pp_layers = ((0,),)
    scheduler._reqs_finalized_layerwise = set()
    request = SimpleNamespace(
        kv_transfer_params={
            "do_remote_decode": True,
            "remote_host": "decode",
            "remote_port": 1234,
            "remote_tp_size": 1,
            "remote_cached_tokens": 0,
        },
        all_token_ids=list(range(prompt_length)),
    )
    scheduler._reqs_need_send_layerwise = {
        "request": _SendReqInfo(
            local_block_ids=[[1, 2], [3]],
            local_transferred_tokens=0,
            local_computed_tokens=0,
            request=request,
        )
    }
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=0)],
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={"request": 17},
    )

    metadata = scheduler.build_connector_meta(scheduler_output)

    req_meta = metadata.requests["request"]
    assert req_meta.local_block_ids == expected_block_ids
    assert req_meta.chunk_start_blocks == [0, 0]


def test_producer_scheduler_delays_only_after_final_chunk_is_dispatched():
    scheduler = LayerwisePushProducerScheduler.__new__(LayerwisePushProducerScheduler)
    scheduler._reqs_need_send_layerwise = {}
    scheduler._reqs_finalized_layerwise = {"request"}
    request = SimpleNamespace(
        request_id="request",
        kv_transfer_params={"do_remote_decode": True},
    )

    assert scheduler.request_finished_all_groups(request, ([1],)) == (True, None)

    # Cancellation/preemption before a final transfer was dispatched has no
    # request-level completion waiting on the worker.
    assert scheduler.request_finished_all_groups(request, ([1],)) == (False, None)

    pending = _SendReqInfo([[1]], 0, 16, request)
    pending.has_dispatched_blocks = True
    scheduler._reqs_need_send_layerwise["request"] = pending
    with pytest.raises(RuntimeError, match="cannot cancel Prefill"):
        scheduler.request_finished_all_groups(request, ([1],))
    assert scheduler._reqs_need_send_layerwise["request"] is pending


@pytest.mark.parametrize("backend", ["memfabric", "mooncake"])
def test_backend_selection_accepts_both_push_backends(backend):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"transfer_backend": backend})
    )
    assert _resolve_kv_transfer_backend(config) == backend


def test_backend_selection_requires_explicit_supported_backend():
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}))
    with pytest.raises(ValueError, match="transfer_backend"):
        _resolve_kv_transfer_backend(config)


@pytest.mark.parametrize("configured, expected", [(None, "async"), ("sync", "sync")])
def test_push_write_mode_defaults_to_async(configured, expected):
    extra = {} if configured is None else {"push_write_mode": configured}
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra))

    assert _resolve_push_write_mode(config) == expected


def test_consumer_start_load_keeps_all_destination_groups():
    worker = LayerwisePushConsumerWorker.__new__(LayerwisePushConsumerWorker)
    worker._pending_recv_req_ids = set()
    worker._deferred_cleanup_req_ids = set()
    worker.request_map = {}
    worker._dest_blocks_by_req = {}
    worker._dest_blocks_condition = threading.Condition()
    metadata = LayerwisePushConsumerMetadata()
    metadata.add_request("request123456789", [[1], [2, 3], [4]])

    worker._receive_thread = None
    worker.start_load_kv(metadata)

    assert worker._dest_blocks_by_req["request"] == [[1], [2, 3], [4]]


def test_consumer_rejects_completed_request_without_internal_mapping():
    worker = LayerwisePushConsumerWorker.__new__(LayerwisePushConsumerWorker)
    worker._pending_recv_req_ids = set()
    worker._deferred_cleanup_req_ids = set()
    worker.tp_size = 1
    worker._receive_thread = MagicMock()
    worker._receive_thread.fatal_error = None
    worker._receive_thread.get_and_clear_done.return_value = {"request"}
    worker._receive_thread.get_and_clear_failed.return_value = set()
    worker._terminal_ext_ids = set()
    worker._dest_blocks_by_req = {}
    worker._invalid_block_ids = set()
    worker.request_map = {}

    with pytest.raises(RuntimeError, match="internal request mapping"):
        worker.get_finished()


@pytest.mark.parametrize(
    "finish_timing", ["before_first_source", "between_sources", "with_completion", "after_completion"]
)
def test_consumer_defers_cleanup_until_receive_finishes(destination_registration, finish_timing):
    worker, reader, metadata = destination_registration
    worker.start_load_kv(metadata)
    req_id = metadata.requests[0].req_id
    finished = {req_id}
    expected = frozenset({(0, 0), (1, 0)})
    if finish_timing == "before_first_source":
        assert worker.get_finished(finished) == (set(), set())
    reader._record_chunk_done(["request"], (0, 0), expected)
    if finish_timing == "between_sources":
        assert worker.get_finished(finished) == (set(), set())

    # Cancellation must preserve both completed PP contributors and the
    # destination mapping needed by the remaining reads.
    assert reader._done_contributors["request"] == {(0, 0)}
    assert worker.request_map == {"request": req_id}
    assert worker._dest_blocks_by_req == {"request": [[5, 6]]}
    reader._record_chunk_done(["request"], (1, 0), expected)

    assert worker.get_finished(finished if finish_timing == "with_completion" else None) == (set(), finished)
    if finish_timing == "after_completion":
        assert worker.request_map == {"request": req_id}
        assert worker.get_finished(finished) == (set(), set())
    assert worker.get_finished() == (set(), set())
    assert worker.request_map == {}
    assert worker._dest_blocks_by_req == {}
    assert worker._pending_recv_req_ids == set()
    assert worker._deferred_cleanup_req_ids == set()
    assert reader._done_contributors == {}
    assert reader._expected_sources == {}
    assert reader._terminal_requests == set()


def test_cancelled_receive_waits_for_other_tp_before_cleanup(destination_registration):
    worker, reader, metadata = destination_registration
    worker.start_load_kv(metadata)
    req_id = metadata.requests[0].req_id
    reader._record_chunk_done(["request"], (0, 0), frozenset({(0, 0)}))
    with patch.object(
        worker,
        "_gather_tp_write_status",
        side_effect=[
            [({"request"}, set(), None), (set(), set(), None)],
            [({"request"}, set(), None), ({"request"}, set(), None)],
        ],
    ):
        assert worker.get_finished({req_id}) == (set(), set())
        assert worker.request_map == {"request": req_id}
        assert worker._dest_blocks_by_req == {"request": [[5, 6]]}
        assert worker.get_finished() == (set(), {req_id})
    assert worker.request_map == {}
    assert worker._pending_recv_req_ids == set()
    assert worker._deferred_cleanup_req_ids == set()


def test_consumer_gathers_local_fatal_error_before_raising():
    worker = LayerwisePushConsumerWorker.__new__(LayerwisePushConsumerWorker)
    worker.tp_size = 2
    worker._receive_thread = MagicMock()
    worker._receive_thread.fatal_error = RuntimeError("receiver failed")
    worker._terminal_ext_ids = set()

    tp_group = SimpleNamespace(world_size=2, cpu_group=object())

    def gather_status(output, local_status, *, group):
        assert local_status == (set(), set(), "receiver failed")
        assert group is tp_group.cpu_group
        output[:] = [local_status, (set(), set(), None)]

    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.worker.get_tp_group",
            return_value=tp_group,
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.worker.torch.distributed.all_gather_object",
            side_effect=gather_status,
        ) as all_gather,
        pytest.raises(RuntimeError, match="receiver failed"),
    ):
        worker.get_finished()

    all_gather.assert_called_once()


@pytest.mark.parametrize(
    ("prefill_rank", "expected_decode_rank"),
    [(0, 0), (3, 0), (4, 1), (7, 1)],
)
def test_prefill_rank_maps_to_decode_contributor_group(prefill_rank, expected_decode_rank):
    assert (
        LayerwisePushProducerWorker._map_prefill_rank_to_decode_rank(
            prefill_tp_size=8,
            decode_tp_size=2,
            prefill_tp_rank=prefill_rank,
        )
        == expected_decode_rank
    )


@pytest.mark.parametrize(
    "producer_layers,decode_layers",
    [
        (((0, 1), (2, 3)), ((0, 1), (2, 3))),
        (((0, 1), (2, 3)), ((0, 1, 2, 3),)),
        (((0, 1, 2, 3),), ((0, 1), (2, 3))),
        (((0, 1), (2, 3)), ((0,), (1, 2, 3))),
        (((0, 1, 2), (3, 4, 5)), ((0, 1, 2, 3), (4, 5))),
    ],
)
@pytest.mark.parametrize("decode_tp_size", [1, 2])
def test_prefill_routes_by_layer_ownership(producer_layers, decode_layers, decode_tp_size):
    worker = LayerwisePushProducerWorker.__new__(LayerwisePushProducerWorker)
    worker.current_layer = 7
    worker._pd_dispatched_layers = {1}
    worker.tp_size = 2
    worker.tp_rank = 1
    worker.pp_size = len(producer_layers)
    worker.kv_send_layer_thread = MagicMock()
    worker._routes_by_topology = {}
    metadata = LayerwisePushProducerMetadata()
    metadata.producer_pp_layers = producer_layers
    endpoints = [
        [
            {"host": f"decode-{pp}-{tp}", "port": 4000 + pp * decode_tp_size + tp, "layer_ids": list(layers)}
            for tp in range(decode_tp_size)
        ]
        for pp, layers in enumerate(decode_layers)
    ]
    metadata.add_new_req(
        "request",
        [[1]],
        {
            "remote_tp_size": decode_tp_size,
            "remote_pp_size": len(decode_layers),
            "remote_endpoints": endpoints,
            "remote_topology_id": "decode-topology",
        },
        chunk_finish=True,
    )
    request = metadata.requests["request"]
    for pp_rank, layers in enumerate(producer_layers):
        worker.pp_rank = pp_rank
        worker._layer_order = layers
        worker._routes_by_topology.clear()
        worker.start_load_kv(metadata)
        remote_tp = worker.tp_rank // (worker.tp_size // decode_tp_size)
        expected = {
            layer: (endpoints[pp][remote_tp]["host"], endpoints[pp][remote_tp]["port"])
            for pp, owned in enumerate(decode_layers)
            for layer in layers
            if layer in owned
        }
        assert request.layer_endpoints == expected
        assert request.terminal_layers == frozenset({endpoint: layer for layer, endpoint in expected.items()}.values())
        worker.kv_send_layer_thread.track_requests.assert_called_with({"request"}, set(expected.values()))
        assert request.tp_ratio == worker.tp_size // decode_tp_size
        assert request.group_member_idx == worker.tp_rank % request.tp_ratio
        assert worker.current_layer == 0
        assert worker._pd_dispatched_layers == set()


def test_prefill_reuses_routes_across_requests_and_chunks_without_traversing_topology():
    worker = LayerwisePushProducerWorker.__new__(LayerwisePushProducerWorker)
    worker.tp_rank, worker.tp_size = 0, 1
    worker.pp_rank, worker.pp_size = 0, 1
    worker._layer_order = (0, 1)
    worker._routes_by_topology = {}
    worker.kv_send_layer_thread = MagicMock()
    metadata = LayerwisePushProducerMetadata()
    metadata.producer_pp_layers = ((0, 1),)
    first = LayerwisePushProducerReqMeta(
        local_block_ids=[[0]],
        remote_tp_size=1,
        remote_pp_size=2,
        remote_topology_id="decode-topology",
        remote_endpoints=[
            [{"host": "decode-a", "port": 4000, "layer_ids": [0]}],
            [{"host": "decode-b", "port": 5000, "layer_ids": [1]}],
        ],
    )
    second = replace(first, remote_endpoints=MagicMock())
    metadata.requests = {"first": first, "second": second}

    worker.start_load_kv(metadata)

    route = first.layer_endpoints
    assert route == {0: ("decode-a", 4000), 1: ("decode-b", 5000)}
    assert second.layer_endpoints is route
    assert second.terminal_layers == frozenset({0, 1})
    assert second.remote_endpoints.mock_calls == []

    # Later chunks arrive as fresh request metadata, but reuse the same route.
    metadata.requests = {
        req_id: replace(request, remote_endpoints=MagicMock(), layer_endpoints={}, terminal_layers=frozenset())
        for req_id, request in metadata.requests.items()
    }
    worker.start_load_kv(metadata)
    for request in metadata.requests.values():
        assert request.layer_endpoints is route
        assert request.terminal_layers == frozenset({0, 1})
        assert request.remote_endpoints.mock_calls == []


def test_tp_block_range_rotates_owner_between_chunks():
    assert _tp_block_range(1, 0, 2, 0) == (0, 1)
    assert _tp_block_range(1, 1, 2, 0) == (1, 1)
    assert _tp_block_range(1, 0, 2, 1) == (1, 1)
    assert _tp_block_range(1, 1, 2, 1) == (0, 1)


def test_request_metadata_remains_backend_agnostic():
    request = LayerwisePushProducerReqMeta(
        local_block_ids=[[1]],
        remote_tp_size=1,
    )
    assert not hasattr(request, "transfer_backend")


def test_startup_handshake_collects_real_worker_endpoints_and_layer_ownership():
    connector = LayerwisePushConnector.__new__(LayerwisePushConnector)
    connector.is_producer = False
    connector.is_consumer = True
    connector.connector_scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2, tensor_parallel_size=1))
    )
    handshake = {
        (0, 0): LayerwisePushHandshakeMetadata((0, 1, 2), "host-a", 4000),
        (1, 0): LayerwisePushHandshakeMetadata((3, 4, 5, 6), "host-b", 5000),
    }
    connector.set_xfer_handshake_metadata_pp_aware(handshake)
    assert connector.connector_scheduler.remote_endpoints == [
        [{"host": "host-a", "port": 4000, "layer_ids": [0, 1, 2]}],
        [{"host": "host-b", "port": 5000, "layer_ids": [3, 4, 5, 6]}],
    ]
    topology_id = connector.connector_scheduler.remote_topology_id
    connector.set_xfer_handshake_metadata_pp_aware(handshake)
    assert connector.connector_scheduler.remote_topology_id == topology_id
    for changed in (
        replace(handshake[1, 0], host="host-c"),
        replace(handshake[1, 0], port=5001),
        replace(handshake[1, 0], layer_ids=(3, 4, 5, 6, 7)),
    ):
        connector.set_xfer_handshake_metadata_pp_aware({**handshake, (1, 0): changed})
        assert connector.connector_scheduler.remote_topology_id != topology_id
    connector.is_producer, connector.is_consumer = True, False
    connector.set_xfer_handshake_metadata_pp_aware(handshake)
    assert connector.connector_scheduler.producer_pp_layers == ((0, 1, 2), (3, 4, 5, 6))
    with pytest.raises(ValueError, match="complete"):
        connector.set_xfer_handshake_metadata_pp_aware({(0, 0): handshake[0, 0]})
    with pytest.raises(ValueError, match="disjoint"):
        connector.set_xfer_handshake_metadata_pp_aware({**handshake, (1, 0): LayerwisePushHandshakeMetadata((2, 3))})


def test_consumer_handshake_requires_ready_listener():
    connector = LayerwisePushConnector.__new__(LayerwisePushConnector)
    connector.is_producer = False
    reader = SimpleNamespace(
        ready_event=threading.Event(),
        startup_error=None,
        _host="worker-host",
        side_channel_port=4000,
        tp_rank=1,
    )
    connector.connector_worker = SimpleNamespace(layer_layouts={5: (), 6: ()}, _receive_thread=reader)
    with pytest.raises(RuntimeError, match="listener must be ready"):
        connector.get_handshake_metadata()
    reader.ready_event.set()
    assert connector.get_handshake_metadata() == LayerwisePushHandshakeMetadata((5, 6), "worker-host", 4001)


@pytest.mark.parametrize(
    "endpoints,error",
    [
        (None, "complete D worker endpoint table"),
        ([[{"host": "decode", "port": 4000, "layer_ids": [4]}]], "missing destination layers"),
        ([[{"host": "decode", "port": 4000, "layer_ids": [4, 4, 5]}]], "duplicate ownership"),
        ([[{"host": "decode", "port": 70000, "layer_ids": [4, 5]}]], "must be in"),
    ],
)
def test_prefill_rejects_invalid_destination_topology_before_sending(endpoints, error):
    worker = LayerwisePushProducerWorker.__new__(LayerwisePushProducerWorker)
    worker.pp_rank, worker.pp_size = 1, 2
    worker.tp_rank, worker.tp_size = 0, 1
    worker._layer_order = (4, 5)
    worker._routes_by_topology = {}
    worker.kv_send_layer_thread = MagicMock()
    metadata = LayerwisePushProducerMetadata()
    metadata.producer_pp_layers = ((0, 1, 2, 3), (4, 5))
    metadata.add_new_req(
        "request",
        [[0]],
        {
            "remote_endpoints": endpoints,
            "remote_topology_id": "decode-topology",
            "remote_pp_size": 1,
            "remote_tp_size": 1,
        },
        chunk_finish=True,
    )
    with pytest.raises(ValueError, match=error):
        worker.start_load_kv(metadata)
    worker.kv_send_layer_thread.track_requests.assert_not_called()
    worker.kv_send_layer_thread.enqueue.assert_not_called()


def test_consumer_cannot_advertise_before_startup_handshake(consumer_scheduler):
    consumer_scheduler.remote_endpoints = None
    request = SimpleNamespace(kv_transfer_params={"do_remote_prefill": True})
    with pytest.raises(RuntimeError, match="topology has not been initialized"):
        consumer_scheduler.update_state_after_alloc(request, MagicMock(), 16)
    consumer_scheduler._submit_metaserver_request.assert_not_called()


@pytest.mark.parametrize("local_layers,expected_pp", [([1, 2], {0, 1}), ([3], {1})])
def test_layout_handshake_establishes_all_pp_sources_before_first_completion(local_layers, expected_pp):
    reader = LayerwisePushReceiveThread(
        tp_rank=0,
        side_channel_port=1,
        state=ConsumerDestinationState(tp_size=1, layer_layouts=dict.fromkeys(local_layers, ()), dest_blocks_by_req={}),
    )
    sources = frozenset((pp, member) for pp in expected_pp for member in range(2))
    # P has PP=2/TP=2. Even the first connection describes all required sources.
    first_pp = min(expected_pp)
    remote_layers = {layer: {"components": []} for layer in ((0, 1), (2, 3))[first_pp]}
    message = (LAYOUT_META, "p-session", msgspec.msgpack.encode(remote_layers), ((0, 1), (2, 3)), 2, first_pp, 0)
    reader._register_remote_layout(b"p", message)
    contributor, ratio, expected = reader._p_completion_sources[b"p"]
    assert expected == sources
    assert ratio == 2
    reader._record_chunk_done(["request"], contributor, expected)
    reader._record_chunk_done(["request"], contributor, expected)
    assert reader.get_and_clear_done() == set()
    remaining = sorted(sources - {contributor}, reverse=True)
    for source in remaining[:-1]:
        reader._record_chunk_done(["request"], source, expected)
        assert reader.get_and_clear_done() == set()
    reader._record_chunk_done(["request"], remaining[-1], expected)
    assert reader.get_and_clear_done() == {"request"}
    reader._record_chunk_done(["request"], remaining[-1], expected)
    assert reader.get_and_clear_done() == set()
    reader.discard_requests({"request"})
    assert "request" not in reader._terminal_requests


def test_pp_push_over_real_control_sockets_with_chunked_slot_reuse():
    # Only the data-transfer backend/NPU events are mocked. Both control
    # threads, their wire messages, and the slot gates are real.
    # P1 feeds both D stages, while D0 reads from both P stages.
    producer_layers = ((0, 1, 2), (3, 4, 5))
    decode_layers = ((0, 1, 2, 3), (4, 5))
    readers = []
    workers = []
    endpoints = []
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.receive_thread.get_ip", return_value="127.0.0.1"
        ),
        patch("vllm.distributed.get_world_group", return_value=SimpleNamespace(local_rank=0)),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_push.send_thread.torch"),
    ):
        try:
            for layers in decode_layers:
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                reader = LayerwisePushReceiveThread(
                    tp_rank=0,
                    side_channel_port=port,
                    state=ConsumerDestinationState(
                        tp_size=1,
                        layer_layouts={
                            layer: (_component(f"layer.{layer}", 0, 2000 + 100 * layer, 16),) for layer in layers
                        },
                        dest_blocks_by_req={"request": [[0, 1]]},
                    ),
                )
                readers.append(reader)
                reader.start()
                assert reader.ready_event.wait(timeout=2)
                assert reader.startup_error is None
                endpoints.append([{"host": "127.0.0.1", "port": port, "layer_ids": list(layers)}])
            for pp, layers in enumerate(producer_layers):
                worker = LayerwisePushProducerWorker.__new__(LayerwisePushProducerWorker)
                worker.pp_rank, worker.pp_size = pp, len(producer_layers)
                worker.tp_rank, worker.tp_size = 0, 1
                worker._layer_order = layers
                worker._routes_by_topology = {}
                worker.layer_storage_slots = dict.fromkeys(layers, (0,))
                worker.reused_storage_slots = frozenset({0})
                worker.kv_send_layer_thread = LayerwisePushSendingThread(
                    backend=MagicMock(),
                    ready_event=threading.Event(),
                    state=ProducerSendState(
                        last_layer_idx=layers[-1],
                        layer_layouts={layer: (_component(f"layer.{layer}", 0, 1000, 16),) for layer in layers},
                        p_session=f"producer-{pp}",
                        block_sizes=(16,),
                        layer_storage_slots=worker.layer_storage_slots,
                        num_blocks=2,
                        pp_rank=pp,
                        write_mode="sync",
                    ),
                )
                workers.append(worker)
                worker.kv_send_layer_thread.start()
                assert worker.kv_send_layer_thread.ready_event.wait(timeout=2)
                assert worker.kv_send_layer_thread.startup_error is None
            for chunk in (0, 1):
                for worker in workers:
                    meta = LayerwisePushProducerMetadata()
                    meta.producer_pp_layers = producer_layers
                    meta.add_new_req(
                        "request",
                        [[chunk]],
                        {
                            "remote_endpoints": endpoints,
                            "remote_topology_id": "decode-topology",
                            "remote_pp_size": len(decode_layers),
                            "remote_tp_size": 1,
                        },
                        chunk_finish=chunk == 1,
                        chunk_start_blocks=[chunk],
                    )
                    worker.start_load_kv(meta)
                    sender = worker.kv_send_layer_thread
                    for layer in worker._layer_order:
                        sender.enqueue(SendTask(meta.requests, layer_idx=layer, layer_name=f"layer.{layer}"))
                        # Never hang the test if routing or a completion marker
                        # is broken. Check the real gate before entering waiter.
                        assert sender.get_storage_send_event(0).wait(timeout=3)
                        worker.wait_for_slot_release(layer)
                if chunk == 0:
                    assert all(reader.get_and_clear_done() == set() for reader in readers)
            for reader, layers in zip(readers, decode_layers, strict=True):
                assert _wait_until(lambda reader=reader: "request" in reader._done_requests)
                assert reader.get_and_clear_done() == {"request"}
                assert reader.get_and_clear_failed() == set()
            assert sum(worker.kv_send_layer_thread.backend.write.call_count for worker in workers) == 12
        finally:
            for worker in workers:
                worker.kv_send_layer_thread.stop(timeout=2)
                assert not worker.kv_send_layer_thread.is_alive()
            for reader in readers:
                reader.stop(timeout=2)
                assert not reader.is_alive()
