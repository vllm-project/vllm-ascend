# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pytest
import zmq

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from vllm_ascend.distributed.kv_transfer import register_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.connector import (
    LayerwisePullConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.protocol import (
    LAYOUT_META,
    READ_DONE,
    READ_FAILED,
    ComponentLayout,
    LayerwisePullConsumerMetadata,
    LayerwisePullProducerMetadata,
    LayerwisePullProducerReqMeta,
    SendTask,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread import (
    ConsumerReadState,
    LayerwisePullReadThread,
    PullBackend,
    _tp_block_range,
    plan_block_reads,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.scheduler import (
    LayerwisePullConsumerScheduler,
    LayerwisePullProducerScheduler,
    _SendReqInfo,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread import (
    LayerwisePullSendingThread,
    ProducerSendState,
    SlotReuseTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.worker import (
    LayerwisePullConsumerWorker,
    LayerwisePullProducerWorker,
    _resolve_kv_transfer_backend,
)


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


def test_only_generic_layerwise_pull_connector_is_registered():
    with (
        patch.object(KVConnectorFactory, "_registry", {}),
        patch.object(KVConnectorFactory, "register_connector") as register,
    ):
        register_connector()

    calls = [call.args for call in register.call_args_list]
    assert (
        "LayerwisePullConnector",
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.connector",
        "LayerwisePullConnector",
    ) in calls
    assert "MooncakeLayerwiseConnector" not in {args[0] for args in calls}
    assert "SfaRemoteD2HConnector" not in {args[0] for args in calls}


def test_slot_release_wait_is_only_exposed_through_dedicated_callback():
    connector = LayerwisePullConnector.__new__(LayerwisePullConnector)
    connector.connector_worker = MagicMock()

    connector.wait_for_layer_load("model.layers.7.self_attn")

    connector.connector_worker.wait_for_slot_release.assert_not_called()

    connector.wait_for_slot_release(7)

    connector.connector_worker.wait_for_slot_release.assert_called_once_with(7)


def test_consumer_metadata_preserves_every_cache_group():
    metadata = LayerwisePullConsumerMetadata()
    block_ids = [[1, 2], [10], [20, 21, 22]]

    metadata.add_request("request", block_ids)
    block_ids[0].append(3)

    assert metadata.requests[0].block_ids_by_group == [[1, 2], [10], [20, 21, 22]]


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
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.scheduler.get_ip",
        return_value="127.0.0.1",
    ):
        scheduler = LayerwisePullConsumerScheduler(config, True, kv_cache_config)
    scheduler._submit_metaserver_request = MagicMock()
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
    assert request.request_id not in consumer_scheduler._local_cached_tokens

    producer = LayerwisePullProducerScheduler.__new__(LayerwisePullProducerScheduler)
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


def test_read_thread_plans_arbitrary_components_in_one_backend_read():
    engine = MagicMock()
    engine.batch_transfer_sync_read.return_value = 0
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
    thread = LayerwisePullReadThread.__new__(LayerwisePullReadThread)
    thread.tp_rank = 0
    thread.backend = PullBackend.memfabric(engine)
    thread._state = ConsumerReadState(
        tp_size=1,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[5, 6], [7, 8]], "second": [[9], [12]]},
    )

    thread._do_read_batch(
        0,
        [("request", [[1, 2], [3, 4]], [0, 0]), ("second", [[5], [8]], [0, 0])],
        "prefill-session",
        source,
    )

    engine.batch_transfer_sync_read.assert_called_once_with(
        "prefill-session",
        [2080, 2144, 4056, 4096],
        [1016, 1080, 3024, 3064],
        [32, 16, 16, 8],
    )


@pytest.mark.parametrize("contiguous_requests", [False, True])
def test_read_batch_reuses_numpy_block_ids_across_tensors(contiguous_requests):
    engine = MagicMock()
    engine.batch_transfer_sync_read.return_value = 0
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
    thread = LayerwisePullReadThread.__new__(LayerwisePullReadThread)
    thread.tp_rank = 0
    thread.backend = PullBackend.memfabric(engine)
    thread._state = ConsumerReadState(
        tp_size=1,
        layer_layouts={0: (destination,)},
        dest_blocks_by_req=destination_blocks,
    )

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread.plan_block_reads",
        wraps=plan_block_reads,
    ) as planner:
        thread._do_read_batch(0, requests, "prefill-session", {0: (source,)})

    assert planner.call_count == 2
    first, second = planner.call_args_list
    for key in ("remote_block_ids", "local_block_ids"):
        assert isinstance(first.kwargs[key], np.ndarray)
        assert first.kwargs[key] is second.kwargs[key]
    np.testing.assert_array_equal(first.kwargs["remote_block_ids"], [1, 2, 3, 4])
    np.testing.assert_array_equal(first.kwargs["local_block_ids"], [5, 6, *second_blocks])
    assert requests == [("first", [[1, 2]], [0]), ("second", [[3, 4]], [0])]
    assert destination_blocks == {"first": [[5, 6]], "second": [second_blocks]}
    engine.batch_transfer_sync_read.assert_called_once_with(
        "prefill-session",
        [2080, 4080] if contiguous_requests else [2080, 2144, 2176, 4080, 4144, 4176],
        [1016, 3016] if contiguous_requests else [1016, 1048, 1064, 3016, 3048, 3064],
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
def test_read_batch_splits_each_request_before_combining_blocks(
    shared, member, tp_rank, expected_source, expected_destination
):
    engine = MagicMock()
    engine.batch_transfer_sync_read.return_value = 0
    source = _component("layer.0", 0, 1000, 16)
    destination = replace(source, base_addrs=(2000,))
    thread = LayerwisePullReadThread.__new__(LayerwisePullReadThread)
    thread.tp_rank = tp_rank
    thread.backend = PullBackend.memfabric(engine)
    thread._state = ConsumerReadState(
        tp_size=2,
        layer_layouts={0: (destination,)},
        dest_blocks_by_req={"first": [[5, 6]], "second": [[9, 10, 11]], "empty": [[]]},
        tp_shared_components=frozenset({"layer.0"}) if shared else frozenset(),
    )
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread.plan_block_reads",
        wraps=plan_block_reads,
    ) as planner:
        thread._do_read_batch(
            0,
            [("first", [[1, 2]], [0]), ("second", [[3, 4]], [1]), ("empty", [[]], [0])],
            "prefill-session",
            {0: (source,)},
            group_member_idx=member,
            ratio=2,
        )

    if expected_source:
        planner.assert_called_once()
        np.testing.assert_array_equal(planner.call_args.kwargs["remote_block_ids"], expected_source)
        np.testing.assert_array_equal(planner.call_args.kwargs["local_block_ids"], expected_destination)
        engine.batch_transfer_sync_read.assert_called_once_with(
            "prefill-session",
            [2000 + block * 16 for block in expected_destination],
            [1000 + block * 16 for block in expected_source],
            [16, 16],
        )
    else:
        planner.assert_not_called()
        engine.batch_transfer_sync_read.assert_not_called()


def test_read_batch_rejects_incomplete_request_before_planning():
    source = _component("layer.0", 0, 1000, 16)
    thread = LayerwisePullReadThread.__new__(LayerwisePullReadThread)
    thread.backend = MagicMock()
    thread._state = ConsumerReadState(
        tp_size=1,
        layer_layouts={0: (replace(source, base_addrs=(2000,)),)},
        dest_blocks_by_req={"first": [[5, 6]], "second": [[7]]},
    )
    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread.plan_block_reads") as planner,
        pytest.raises(RuntimeError, match="D block range is incomplete"),
    ):
        thread._do_read_batch(
            0,
            [("first", [[1, 2]], [0]), ("second", [[3, 4]], [0])],
            "prefill-session",
            {0: (source,)},
        )
    planner.assert_not_called()
    thread.backend.read.assert_not_called()


@pytest.fixture
def destination_registration():
    worker = LayerwisePullConsumerWorker.__new__(LayerwisePullConsumerWorker)
    worker.request_map = {}
    worker._dest_blocks_by_req = {}
    worker._dest_blocks_condition = threading.Condition()
    state = ConsumerReadState(
        tp_size=1,
        layer_layouts={0: (_component("layer.0", 0, 2000, 16),)},
        dest_blocks_by_req=worker._dest_blocks_by_req,
        dest_blocks_condition=worker._dest_blocks_condition,
    )
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread.get_ip",
        return_value="127.0.0.1",
    ):
        thread = LayerwisePullReadThread(0, 1234, MagicMock(), state)
    metadata = LayerwisePullConsumerMetadata()
    metadata.add_request("request123456789", [[5, 6]])
    return worker, thread, metadata


@pytest.mark.parametrize("registered_before_read", [False, True])
def test_read_waits_for_destination_registration(destination_registration, registered_before_read):
    worker, thread, metadata = destination_registration
    source = {0: (_component("layer.0", 0, 1000, 16),)}
    waiting = threading.Event()
    condition = worker._dest_blocks_condition
    original_wait = condition.wait

    def wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    if registered_before_read:
        worker.start_load_kv(metadata)
    with patch.object(condition, "wait", side_effect=wait), ThreadPoolExecutor(1) as executor:
        future = executor.submit(thread._do_read_batch, 0, [("request", [[1, 2]], [0])], "prefill-session", source)
        try:
            if not registered_before_read:
                assert waiting.wait(timeout=1)
                thread.backend.read.assert_not_called()
                worker.start_load_kv(metadata)
            future.result(timeout=1)
        finally:
            thread.stop()
    assert waiting.is_set() == (not registered_before_read)
    thread.backend.read.assert_called_once_with("prefill-session", [2080], [1016], [32])


def test_read_times_out_without_destination_registration(destination_registration):
    _, thread, _ = destination_registration
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread.DEST_BLOCK_WAIT_TIMEOUT_SECONDS",
            0.01,
        ),
        pytest.raises(RuntimeError, match="has no D blocks"),
    ):
        thread._do_read_batch(
            0,
            [("request", [[1]], [0])],
            "prefill-session",
            {0: (_component("layer.0", 0, 1000, 16),)},
        )
    thread.backend.read.assert_not_called()


def test_stop_wakes_destination_registration_wait(destination_registration):
    worker, thread, _ = destination_registration
    waiting = threading.Event()
    condition = worker._dest_blocks_condition
    original_wait = condition.wait

    def wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    with patch.object(condition, "wait", side_effect=wait), ThreadPoolExecutor(1) as executor:
        future = executor.submit(
            thread._do_read_batch,
            0,
            [("request", [[1]], [0])],
            "prefill-session",
            {0: (_component("layer.0", 0, 1000, 16),)},
        )
        try:
            assert waiting.wait(timeout=1)
        finally:
            thread.stop()
        with pytest.raises(RuntimeError, match="stopped waiting"):
            future.result(timeout=1)
    thread.backend.read.assert_not_called()


@pytest.mark.parametrize("blocks_registered", [False, True])
def test_read_thread_rejects_component_layout_mismatch_before_transfer(blocks_registered):
    engine = MagicMock()
    source = {0: (_component("layer.0.c0", 0, 1000, 16),)}
    destination = {0: (replace(source[0][0], base_addrs=(2000,), block_lengths=(32,)),)}
    thread = LayerwisePullReadThread.__new__(LayerwisePullReadThread)
    thread.tp_rank = 0
    thread.backend = PullBackend.mooncake(engine)
    thread._state = ConsumerReadState(
        tp_size=1,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[5]]} if blocks_registered else {},
    )

    with pytest.raises(RuntimeError, match="tensor size differs"):
        thread._do_read_batch(
            0,
            [("request", [[1]], [0])],
            "prefill-session",
            source,
        )
    engine.batch_transfer_sync_read.assert_not_called()


def test_read_thread_rejects_same_size_different_dtype():
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
    thread = LayerwisePullReadThread.__new__(LayerwisePullReadThread)
    thread.tp_rank = 0
    thread.backend = PullBackend.mooncake(engine)
    thread._state = ConsumerReadState(
        tp_size=1,
        layer_layouts={0: (destination_component,)},
        dest_blocks_by_req={"request": [[5]]},
    )

    with pytest.raises(RuntimeError, match="dtype differs"):
        thread._do_read_batch(
            0,
            [("request", [[1]], [0])],
            "prefill-session",
            {0: (source_component,)},
        )
    engine.batch_transfer_sync_read.assert_not_called()


def test_tp_shared_component_is_read_once_and_split_across_decode_tp():
    engine = MagicMock()
    engine.batch_transfer_sync_read.return_value = 0
    source = {0: (_component("layer.0.main", 0, 1000, 16),)}
    destination = {0: (_component("layer.0.main", 0, 2000, 16),)}
    thread = LayerwisePullReadThread.__new__(LayerwisePullReadThread)
    thread.tp_rank = 1
    thread.backend = PullBackend.memfabric(engine)
    thread._state = ConsumerReadState(
        tp_size=2,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[10, 11, 12, 13]]},
        tp_shared_components=frozenset({"layer.0.main"}),
    )

    thread._do_read_batch(
        0,
        [("request", [[0, 1, 2, 3]], [0])],
        "prefill-session",
        source,
        group_member_idx=0,
        ratio=2,
    )

    engine.batch_transfer_sync_read.assert_called_once_with(
        "prefill-session",
        [2192],
        [1032],
        [32],
    )


def test_layout_metadata_contains_component_descriptors():
    state = ProducerSendState(
        last_layer_idx=0,
        layer_layouts={0: (_component("layer.0.c0", 2, 1000, 16),)},
        p_session="prefill-session",
        block_sizes=(16, 16, 16),
        layer_storage_slots={0: (0,)},
        num_blocks=8,
    )
    thread = LayerwisePullSendingThread.__new__(LayerwisePullSendingThread)
    thread._state = state
    thread.timeout = 1
    thread._layout_meta_sent_paths = set()
    dealer = MagicMock()
    dealer.poll.return_value = True
    dealer.recv_multipart.return_value = [b"", b"ACK"]
    encoder = msgspec.msgpack.Encoder()

    thread._send_layout_meta("tcp://decode:1", dealer, encoder)

    message = msgspec.msgpack.decode(dealer.send.call_args.args[0])
    assert message[0] == LAYOUT_META
    assert message[1] == "prefill-session"
    layouts = msgspec.msgpack.decode(message[2])
    assert layouts[0]["components"][0] == {
        "name": "layer.0.c0",
        "group_index": 2,
        "block_size": 16,
        "dtypes": ["torch.bfloat16"],
        "base_addrs": [1000],
        "block_strides": [16],
        "block_lengths": [16],
        "block_shapes": [[16]],
        "block_size_scales": [1],
    }


@pytest.fixture
def sending_thread():
    thread = LayerwisePullSendingThread(
        ready_event=threading.Event(),
        state=ProducerSendState(
            last_layer_idx=1,
            layer_layouts={},
            p_session="prefill-session",
            block_sizes=(16,),
            layer_storage_slots={0: (0,), 1: (0,)},
        ),
    )
    poll_started = queue.Queue()
    real_poll = thread._poller.poll

    def poll():
        poll_started.put(None)
        # No timeout: progress must come from a task, reply, or shutdown wakeup.
        return real_poll()

    with (
        patch("vllm.distributed.get_world_group", return_value=SimpleNamespace(local_rank=0)),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread.torch"),
        patch.object(thread._poller, "poll", side_effect=poll),
    ):
        thread.start()
        try:
            assert thread.ready_event.wait(timeout=2)
            assert thread.startup_error is None
            poll_started.get(timeout=2)
            yield thread, poll_started
        finally:
            thread.stop(timeout=2)
            assert not thread.is_alive()


@pytest.mark.parametrize("reply_type", [READ_DONE, READ_FAILED])
def test_read_reply_wakes_idle_sending_thread(sending_thread, reply_type):
    thread, poll_started = sending_thread
    with zmq.Context() as context, context.socket(zmq.ROUTER) as router:
        port = router.bind_to_random_port("tcp://127.0.0.1")
        path = f"tcp://127.0.0.1:{port}"
        reader = (7, path)
        thread._reuse_tracker.begin(reader, (0,))
        released = thread.get_storage_send_event(0)
        assert released is not None and not released.is_set()

        def connect(task, encoder):
            thread._ensure_dealer(path).send(b"ready")

        thread._process_send_task = MagicMock(side_effect=connect)
        thread.enqueue(SendTask(send_request={}))
        assert router.poll(timeout=2000)
        identity, _ = router.recv_multipart()
        poll_started.get(timeout=2)
        assert thread.send_queue.empty()

        reply = (READ_DONE, 0, 7) if reply_type == READ_DONE else (READ_FAILED, 0, "read failed", 7)
        router.send_multipart([identity, msgspec.msgpack.encode(reply)])

        assert released.wait(timeout=2)
        assert thread.get_storage_error(0) == (None if reply_type == READ_DONE else "read failed")
        assert thread.is_alive()


def test_new_tasks_and_stop_wake_idle_sending_thread(sending_thread):
    thread, poll_started = sending_thread
    processed = threading.Event()
    thread._process_send_task = MagicMock(side_effect=lambda task, encoder: processed.set())
    for layer_idx in range(3):
        processed.clear()
        task = SendTask(send_request={}, layer_idx=layer_idx)
        thread.enqueue(task)
        assert processed.wait(timeout=2)
        poll_started.get(timeout=2)
        assert thread._process_send_task.call_args.args[0] is task

    thread.stop(timeout=2)
    assert not thread.is_alive()
    assert thread._task_reader.fileno() == -1
    assert thread._task_writer.fileno() == -1


def test_send_task_keeps_all_groups_when_only_one_component_group_is_ready():
    state = ProducerSendState(
        last_layer_idx=0,
        layer_layouts={
            0: (
                _component("layer.0.c0", 0, 1000, 16),
                _component("layer.0.c2", 2, 3000, 8),
            )
        },
        p_session="prefill-session",
        block_sizes=(16, 16, 32),
        layer_storage_slots={0: (0, 1)},
    )
    request = LayerwisePullProducerReqMeta(
        local_block_ids=[[1], [], []],
        remote_host="decode",
        remote_port=1234,
        remote_tp_size=1,
        local_computed_tokens=16,
        chunk_start_blocks=[0, 0, 0],
    )
    thread = LayerwisePullSendingThread.__new__(LayerwisePullSendingThread)
    thread._state = state
    thread.last_layer_idx = 0
    thread._next_transfer_id = 0
    thread._p_save_events = {}
    thread._layout_meta_sent_paths = {"tcp://decode:1234"}
    thread._reuse_tracker = SlotReuseTracker(state.layer_storage_slots)
    thread.storage_send_done_events = thread._reuse_tracker.events
    thread._storage_read_errors = thread._reuse_tracker.errors
    thread._legacy_readers = {}
    thread._request_completion_lock = threading.Lock()
    thread._pending_completion_requests = set()
    thread._scheduler_finished_requests = set()
    thread._completed_requests = set()
    thread._completion_requests_by_reader = {}
    dealer = MagicMock()
    thread._ensure_dealer = MagicMock(return_value=dealer)

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread.make_zmq_path",
        return_value="tcp://decode:1234",
    ):
        thread._process_send_task(
            SendTask({"request123456789": request}, layer_idx=0, layer_name="layer.0"),
            msgspec.msgpack.Encoder(),
        )

    message = msgspec.msgpack.decode(dealer.send.call_args.args[0])
    read_request = message[3][0]
    assert read_request == ["request", [[1], [], []], [0, 0, 0]]


@pytest.mark.parametrize(
    ("prompt_length", "expected_block_ids"),
    [(32, [[1], []]), (17, [[1, 2], [3]])],
)
def test_producer_scheduler_precomputes_chunk_block_ranges(prompt_length, expected_block_ids):
    scheduler = LayerwisePullProducerScheduler.__new__(LayerwisePullProducerScheduler)
    scheduler.block_size = [16, 32]
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


@pytest.mark.parametrize("scheduler_finishes_first", [False, True])
def test_source_blocks_are_released_after_both_finish_signals(scheduler_finishes_first):
    state = ProducerSendState(
        last_layer_idx=0,
        layer_layouts={0: (_component("layer.0.c0", 0, 1000, 16),)},
        p_session="prefill-session",
        block_sizes=(16,),
        layer_storage_slots={0: (0,)},
    )
    request_id = "request123456789"
    request = LayerwisePullProducerReqMeta(
        local_block_ids=[[1]],
        remote_host="decode",
        remote_port=1234,
        remote_tp_size=1,
        chunk_finish=True,
        local_computed_tokens=16,
        chunk_start_blocks=[0],
    )
    thread = LayerwisePullSendingThread.__new__(LayerwisePullSendingThread)
    thread._state = state
    thread.last_layer_idx = 0
    thread._next_transfer_id = 0
    thread._p_save_events = {}
    thread._layout_meta_sent_paths = {"tcp://decode:1234"}
    thread._reuse_tracker = SlotReuseTracker(state.layer_storage_slots)
    thread.storage_send_done_events = thread._reuse_tracker.events
    thread._storage_read_errors = thread._reuse_tracker.errors
    thread._legacy_readers = {}
    thread._request_completion_lock = threading.Lock()
    thread._pending_completion_requests = {request_id}
    thread._scheduler_finished_requests = set()
    thread._completed_requests = set()
    thread._completion_requests_by_reader = {}
    dealer = MagicMock()
    dealer.poll.side_effect = [True, False]
    dealer.recv_multipart.return_value = [b"", msgspec.msgpack.encode((b"read_done", 0, 0))]
    thread._dealers = {"tcp://decode:1234": dealer}
    thread._ensure_dealer = MagicMock(return_value=dealer)

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread.make_zmq_path",
        return_value="tcp://decode:1234",
    ):
        thread._process_send_task(
            SendTask({request_id: request}, layer_idx=0, layer_name="layer.0"),
            msgspec.msgpack.Encoder(),
        )

    if scheduler_finishes_first:
        assert thread.get_and_clear_finished_requests({request_id}) == set()
        thread._drain_read_replies(msgspec.msgpack.Decoder(type=tuple))
        assert thread.get_and_clear_finished_requests(set()) == {request_id}
    else:
        # READ_DONE may arrive before scheduler finishes the request. Do not
        # report it early because vLLM only accepts finished_sending for a
        # finished request.
        thread._drain_read_replies(msgspec.msgpack.Decoder(type=tuple))
        assert thread.get_and_clear_finished_requests(set()) == set()
        assert thread.get_and_clear_finished_requests({request_id}) == {request_id}
    assert thread.get_and_clear_finished_requests({request_id}) == set()


def test_producer_scheduler_delays_only_after_final_chunk_is_dispatched():
    scheduler = LayerwisePullProducerScheduler.__new__(LayerwisePullProducerScheduler)
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


@pytest.mark.parametrize("backend", ["memfabric", "mooncake"])
def test_backend_selection_accepts_both_pull_backends(backend):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"transfer_backend": backend})
    )
    assert _resolve_kv_transfer_backend(config) == backend


def test_backend_selection_requires_explicit_supported_backend():
    config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}))
    with pytest.raises(ValueError, match="transfer_backend"):
        _resolve_kv_transfer_backend(config)


def test_consumer_start_load_keeps_all_destination_groups():
    worker = LayerwisePullConsumerWorker.__new__(LayerwisePullConsumerWorker)
    worker.request_map = {}
    worker._dest_blocks_by_req = {}
    worker._dest_blocks_condition = threading.Condition()
    metadata = LayerwisePullConsumerMetadata()
    metadata.add_request("request123456789", [[1], [2, 3], [4]])

    worker.start_load_kv(metadata)

    assert worker._dest_blocks_by_req["request"] == [[1], [2, 3], [4]]


def test_consumer_rejects_completed_request_without_internal_mapping():
    worker = LayerwisePullConsumerWorker.__new__(LayerwisePullConsumerWorker)
    worker.tp_size = 1
    worker._read_thread = MagicMock()
    worker._read_thread.get_and_clear_done.return_value = {"request"}
    worker._read_thread.get_and_clear_failed.return_value = set()
    worker._terminal_ext_ids = set()
    worker._dest_blocks_by_req = {}
    worker._invalid_block_ids = set()
    worker.request_map = {}

    with pytest.raises(RuntimeError, match="internal request mapping"):
        worker.get_finished()


@pytest.mark.parametrize(
    ("prefill_rank", "expected_decode_rank"),
    [(0, 0), (3, 0), (4, 1), (7, 1)],
)
def test_prefill_rank_maps_to_decode_contributor_group(prefill_rank, expected_decode_rank):
    assert (
        LayerwisePullProducerWorker._map_prefill_rank_to_decode_rank(
            prefill_tp_size=8,
            decode_tp_size=2,
            prefill_tp_rank=prefill_rank,
        )
        == expected_decode_rank
    )


def test_prefill_routes_to_aligned_decode_pp_and_tp_rank():
    worker = LayerwisePullProducerWorker.__new__(LayerwisePullProducerWorker)
    worker.current_layer = 7
    worker._pd_dispatched_layers = {1}
    worker.tp_size = 2
    worker.tp_rank = 1
    worker.pp_size = 2
    worker.pp_rank = 1
    metadata = LayerwisePullProducerMetadata()
    metadata.add_new_req(
        "request",
        [[1]],
        {
            "remote_host": "decode",
            "remote_port": 4000,
            "remote_tp_size": 2,
            "remote_pp_size": 2,
        },
    )

    worker.start_load_kv(metadata)

    request = metadata.requests["request"]
    assert request.remote_port == 4003
    assert request.tp_ratio == 1
    assert request.group_member_idx == 0
    assert worker.current_layer == 0
    assert worker._pd_dispatched_layers == set()


def test_prefill_rejects_unaligned_decode_pp_size():
    worker = LayerwisePullProducerWorker.__new__(LayerwisePullProducerWorker)
    worker.current_layer = 0
    worker._pd_dispatched_layers = set()
    worker.tp_size = 1
    worker.tp_rank = 0
    worker.pp_size = 2
    worker.pp_rank = 0
    metadata = LayerwisePullProducerMetadata()
    metadata.add_new_req(
        "request",
        [[1]],
        {
            "remote_host": "decode",
            "remote_port": 4000,
            "remote_tp_size": 1,
            "remote_pp_size": 1,
        },
    )

    with pytest.raises(ValueError, match="aligned P/D pipeline parallel sizes"):
        worker.start_load_kv(metadata)


def test_tp_block_range_rotates_owner_between_chunks():
    assert _tp_block_range(1, 0, 2, 0) == (0, 1)
    assert _tp_block_range(1, 1, 2, 0) == (1, 1)
    assert _tp_block_range(1, 0, 2, 1) == (1, 1)
    assert _tp_block_range(1, 1, 2, 1) == (0, 1)


def test_request_metadata_remains_backend_agnostic():
    request = LayerwisePullProducerReqMeta(
        local_block_ids=[[1]],
        remote_host="decode",
        remote_port=1234,
        remote_tp_size=1,
    )
    assert not hasattr(request, "transfer_backend")
