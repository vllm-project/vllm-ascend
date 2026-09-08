# SPDX-License-Identifier: Apache-2.0

import queue
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pytest
import zmq

pytest.importorskip("torch")
pytest.importorskip("vllm")

import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from vllm_ascend.distributed.kv_transfer import register_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.connector import (
    LayerwisePullConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.protocol import (
    LAYOUT_META,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    ComponentLayout,
    LayerwisePullConsumerMetadata,
    LayerwisePullHandshakeMetadata,
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


@pytest.mark.parametrize(
    "sparse_enabled,tp_rank,keep_device_kv_cache",
    [(True, 0, False), (True, 1, False), (True, 0, True), (False, 0, False)],
)
def test_destination_registration_with_optional_sparse_offload(sparse_enabled, tp_rank, keep_device_kv_cache):
    main_name = "model.layers.0.self_attn.attn"
    indexer_name = "model.layers.0.self_attn.indexer"
    num_blocks, block_size = 4, 16
    main_tensors = tuple(torch.empty(num_blocks, block_size, 1, dim, dtype=torch.bfloat16) for dim in (8, 4))
    # Top-k rows are not cache blocks, and need not divide num_blocks.
    topk_tensors = tuple(torch.empty(3, 32, 1, dim, dtype=torch.bfloat16) for dim in (8, 4))
    indexer_tensors = (
        torch.empty(num_blocks, block_size, 1, 8, dtype=torch.int8),
        torch.empty(num_blocks, block_size, 1, 1, dtype=torch.float16),
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[
            SimpleNamespace(layer_names=[main_name, indexer_name], kv_cache_spec=SimpleNamespace(block_size=block_size))
        ],
    )
    expected_main, expected_indexer = LayerwisePullConsumerWorker._build_hbm_layouts(
        kv_cache_config, {main_name: main_tensors, indexer_name: indexer_tensors}, 1
    )[0]
    k_base = 100000
    v_base = k_base + expected_main.block_lengths[0] * num_blocks
    manager = SimpleNamespace(
        offload_layer_names=[main_name],
        block_size=block_size,
        topk_buffers_k=[topk_tensors[0]],
        topk_buffers_v=[topk_tensors[1]],
        gvas_k_bases=[k_base],
        gvas_v_bases=[v_base],
        cpu_block_lens=[expected_main.block_lengths],
    )
    worker = LayerwisePullConsumerWorker.__new__(LayerwisePullConsumerWorker)
    worker.kv_cache_config = kv_cache_config
    worker.total_base_layers = 1
    worker.tp_rank = tp_rank
    worker.tp_size = 2
    worker.side_channel_port = 1234
    worker._backend_name = "memfabric"
    worker._dest_blocks_by_req = {}
    worker._dest_blocks_condition = threading.Condition()
    worker._ensure_engine = MagicMock(return_value=(None, MagicMock()))
    sparse_caches = {
        main_name: (
            *(main_tensors if keep_device_kv_cache else (None, None)),
            *(main_tensors if tp_rank == 0 else (None, None)),
            *topk_tensors,
        ),
        indexer_name: indexer_tensors,
    }
    get_manager = MagicMock(return_value=manager)
    with (
        patch("vllm_ascend.ascend_config.get_ascend_config") as config,
        patch.dict(
            "sys.modules",
            {
                "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager": SimpleNamespace(
                    get_sparse_kv_offload_manager=get_manager
                )
            },
        ),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.worker.global_memfabric_te") as engine,
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.worker.LayerwisePullReadThread") as reader,
    ):
        config.return_value.sparse_kv_offload_config.enabled = sparse_enabled
        reader.return_value.startup_error = None
        worker.register_kv_caches(
            sparse_caches if sparse_enabled else {main_name: main_tensors, indexer_name: indexer_tensors}
        )

    assert worker.layer_layouts[0] == (
        replace(expected_main, base_addrs=(k_base, v_base)) if sparse_enabled else expected_main,
        expected_indexer,
    )
    if sparse_enabled:
        main_ptrs = [k_base]
        main_lengths = [sum(expected_main.block_lengths) * num_blocks]
    else:
        get_manager.assert_not_called()
        main_ptrs = [tensor.data_ptr() for tensor in main_tensors]
        main_lengths = [tensor.numel() * tensor.element_size() for tensor in main_tensors]
    engine.register_buffer.assert_called_once_with(
        [*main_ptrs, *(tensor.data_ptr() for tensor in indexer_tensors)],
        [*main_lengths, *(tensor.numel() * tensor.element_size() for tensor in indexer_tensors)],
    )
    assert reader.call_args.kwargs["state"].tp_shared_components == (
        frozenset({main_name}) if sparse_enabled else frozenset()
    )
    reader.return_value.start.assert_called_once()


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

    producer = LayerwisePullProducerScheduler.__new__(LayerwisePullProducerScheduler)
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
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 0
    thread.backend = PullBackend.memfabric(engine)
    thread._state = ConsumerReadState(
        tp_size=1,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[5, 6], [7, 8]], "second": [[9], [12]]},
    )

    thread._do_read_batch(
        b"prefill",
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
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
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
        thread._do_read_batch(b"prefill", 0, requests, "prefill-session", {0: (source,)})

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
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
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
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
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
            b"prefill",
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
    worker._pending_recv_req_ids = set()
    worker._deferred_cleanup_req_ids = set()
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
    worker.tp_size = 1
    worker._read_thread = thread
    worker._terminal_ext_ids = set()
    worker._invalid_block_ids = set()
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
        future = executor.submit(
            thread._do_read_batch, b"prefill", 0, [("request", [[1, 2]], [0])], "prefill-session", source
        )
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


def test_component_pairs_are_cached_per_source_and_layer(destination_registration):
    worker, reader, metadata = destination_registration
    metadata.add_request("second123456789", [[8]])
    worker.start_load_kv(metadata)
    reader._state.layer_layouts[1] = (_component("layer.1", 0, 3000, 16),)
    source = {layer: (_component(f"layer.{layer}", 0, 1000 + 100 * layer, 16),) for layer in (0, 1)}
    with patch.object(reader, "_match_components", wraps=reader._match_components) as match:
        reader._do_read_batch(b"P0", 0, [("request", [[0]], [0])], "P0", source)
        reader.backend.read.assert_called_with("P0", [2080], [1000], [16])
        cached = reader._component_pairs_by_source[b"P0"][0]
        # A new request and a later chunk reuse only the static layouts, not
        # block IDs or the resulting transfer addresses.
        reader._do_read_batch(b"P0", 0, [("second", [[3]], [0])], "P0", source)
        reader.backend.read.assert_called_with("P0", [2128], [1048], [16])
        reader._do_read_batch(b"P0", 0, [("request", [[2]], [1])], "P0", source)
        reader.backend.read.assert_called_with("P0", [2096], [1032], [16])
        assert match.call_count == 1
        assert reader._component_pairs_by_source[b"P0"][0] is cached

        reader._do_read_batch(b"P0", 1, [("request", [[0]], [0])], "P0", source)
        reader.backend.read.assert_called_with("P0", [3080], [1100], [16])
        other_source = {0: (replace(source[0][0], base_addrs=(4000,)),)}
        reader._do_read_batch(b"P1", 0, [("request", [[0]], [0])], "P1", other_source)
        reader.backend.read.assert_called_with("P1", [2080], [4000], [16])
        assert match.call_count == 3


@pytest.mark.parametrize("incompatible_layout", [False, True])
def test_layout_update_invalidates_only_its_source_component_pairs(destination_registration, incompatible_layout):
    worker, reader, metadata = destination_registration
    worker.start_load_kv(metadata)
    source = _component("layer.0", 0, 1000, 16)
    encoded = msgspec.msgpack.encode({0: {"components": [asdict(source)]}})
    for identity in (b"P0", b"P1"):
        reader._register_remote_layout(identity, (LAYOUT_META, "session", encoded, ((0,),), 1, 0, 0))
        reader._do_read_batch(identity, 0, [("request", [[0]], [0])], "session", reader._p_layer_layouts[identity])
    unaffected_pairs = reader._component_pairs_by_source[b"P1"][0]
    replacement = replace(source, base_addrs=(4000,))
    if incompatible_layout:
        replacement = replace(replacement, dtypes=("torch.float16",))
    encoded = msgspec.msgpack.encode({0: {"components": [asdict(replacement)]}})
    reader._register_remote_layout(b"P0", (LAYOUT_META, "new-session", encoded, ((0,),), 1, 0, 0))
    assert b"P0" not in reader._component_pairs_by_source
    assert reader._component_pairs_by_source[b"P1"][0] is unaffected_pairs

    reader.backend.read.reset_mock()
    with patch.object(reader, "_match_components", wraps=reader._match_components) as match:
        if incompatible_layout:
            with pytest.raises(RuntimeError, match="dtype differs"):
                reader._do_read_batch(
                    b"P0", 0, [("request", [[0]], [0])], "new-session", reader._p_layer_layouts[b"P0"]
                )
            reader.backend.read.assert_not_called()
            assert 0 not in reader._component_pairs_by_source[b"P0"]
        else:
            reader._do_read_batch(b"P0", 0, [("request", [[0]], [0])], "new-session", reader._p_layer_layouts[b"P0"])
            reader.backend.read.assert_called_once_with("new-session", [2080], [4000], [16])
        assert match.call_count == 1
        reader._do_read_batch(b"P1", 0, [("request", [[0]], [0])], "session", reader._p_layer_layouts[b"P1"])
        reader.backend.read.assert_called_with("session", [2080], [1000], [16])
        assert match.call_count == 1


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
            b"prefill",
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
            b"prefill",
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
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 0
    thread.backend = PullBackend.mooncake(engine)
    thread._state = ConsumerReadState(
        tp_size=1,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[5]]} if blocks_registered else {},
    )

    with pytest.raises(RuntimeError, match="tensor size differs"):
        thread._do_read_batch(
            b"prefill",
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
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 0
    thread.backend = PullBackend.mooncake(engine)
    thread._state = ConsumerReadState(
        tp_size=1,
        layer_layouts={0: (destination_component,)},
        dest_blocks_by_req={"request": [[5]]},
    )

    with pytest.raises(RuntimeError, match="dtype differs"):
        thread._do_read_batch(
            b"prefill",
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
    thread._component_pairs_by_source = {}
    thread._failed_request_ids = set()
    thread.tp_rank = 1
    thread.backend = PullBackend.memfabric(engine)
    thread._state = ConsumerReadState(
        tp_size=2,
        layer_layouts=destination,
        dest_blocks_by_req={"request": [[10, 11, 12, 13]]},
        tp_shared_components=frozenset({"layer.0.main"}),
    )

    thread._do_read_batch(
        b"prefill",
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
        remote_tp_size=1,
        local_computed_tokens=16,
        chunk_start_blocks=[0, 0, 0],
        layer_endpoints={0: ("decode", 1234)},
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
        remote_tp_size=1,
        chunk_finish=True,
        local_computed_tokens=16,
        chunk_start_blocks=[0],
        layer_endpoints={0: ("decode", 1234)},
        terminal_layers=frozenset({0}),
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
    thread._pending_completion_paths = {request_id: {"tcp://decode:1234"}}
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
    worker._pending_recv_req_ids = set()
    worker._deferred_cleanup_req_ids = set()
    worker.request_map = {}
    worker._dest_blocks_by_req = {}
    worker._dest_blocks_condition = threading.Condition()
    metadata = LayerwisePullConsumerMetadata()
    metadata.add_request("request123456789", [[1], [2, 3], [4]])

    worker.start_load_kv(metadata)

    assert worker._dest_blocks_by_req["request"] == [[1], [2, 3], [4]]


def test_consumer_rejects_completed_request_without_internal_mapping():
    worker = LayerwisePullConsumerWorker.__new__(LayerwisePullConsumerWorker)
    worker._pending_recv_req_ids = set()
    worker._deferred_cleanup_req_ids = set()
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
    reader._do_read_batch(b"P1", 0, [("request", [[0, 1]], [0])], "P1", {0: (_component("layer.0", 0, 1000, 16),)})
    reader.backend.read.assert_called_once_with("P1", [2080], [1000], [32])
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
        "_gather_tp_read_status",
        side_effect=[
            [({"request"}, set()), (set(), set())],
            [({"request"}, set()), ({"request"}, set())],
        ],
    ):
        assert worker.get_finished({req_id}) == (set(), set())
        assert worker.request_map == {"request": req_id}
        assert worker._dest_blocks_by_req == {"request": [[5, 6]]}
        assert worker.get_finished() == (set(), {req_id})
    assert worker.request_map == {}
    assert worker._pending_recv_req_ids == set()
    assert worker._deferred_cleanup_req_ids == set()


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
    worker = LayerwisePullProducerWorker.__new__(LayerwisePullProducerWorker)
    worker.current_layer = 7
    worker._pd_dispatched_layers = {1}
    worker.tp_size = 2
    worker.tp_rank = 1
    worker.pp_size = len(producer_layers)
    worker.kv_send_layer_thread = MagicMock()
    worker._routes_by_topology = {}
    metadata = LayerwisePullProducerMetadata()
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
    worker = LayerwisePullProducerWorker.__new__(LayerwisePullProducerWorker)
    worker.tp_rank, worker.tp_size = 0, 1
    worker.pp_rank, worker.pp_size = 0, 1
    worker._layer_order = (0, 1)
    worker._routes_by_topology = {}
    worker.kv_send_layer_thread = MagicMock()
    metadata = LayerwisePullProducerMetadata()
    metadata.producer_pp_layers = ((0, 1),)
    first = LayerwisePullProducerReqMeta(
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
    request = LayerwisePullProducerReqMeta(
        local_block_ids=[[1]],
        remote_tp_size=1,
    )
    assert not hasattr(request, "transfer_backend")


def test_startup_handshake_collects_real_worker_endpoints_and_layer_ownership():
    connector = LayerwisePullConnector.__new__(LayerwisePullConnector)
    connector.is_producer = False
    connector.is_consumer = True
    connector.connector_scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2, tensor_parallel_size=1))
    )
    handshake = {
        (0, 0): LayerwisePullHandshakeMetadata((0, 1, 2), "host-a", 4000),
        (1, 0): LayerwisePullHandshakeMetadata((3, 4, 5, 6), "host-b", 5000),
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
        connector.set_xfer_handshake_metadata_pp_aware({**handshake, (1, 0): LayerwisePullHandshakeMetadata((2, 3))})


def test_consumer_handshake_requires_ready_listener():
    connector = LayerwisePullConnector.__new__(LayerwisePullConnector)
    connector.is_producer = False
    reader = SimpleNamespace(
        ready_event=threading.Event(),
        startup_error=None,
        _host="worker-host",
        side_channel_port=4000,
        tp_rank=1,
    )
    connector.connector_worker = SimpleNamespace(layer_layouts={5: (), 6: ()}, _read_thread=reader)
    with pytest.raises(RuntimeError, match="listener must be ready"):
        connector.get_handshake_metadata()
    reader.ready_event.set()
    assert connector.get_handshake_metadata() == LayerwisePullHandshakeMetadata((5, 6), "worker-host", 4001)


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
    worker = LayerwisePullProducerWorker.__new__(LayerwisePullProducerWorker)
    worker.pp_rank, worker.pp_size = 1, 2
    worker.tp_rank, worker.tp_size = 0, 1
    worker._layer_order = (4, 5)
    worker._routes_by_topology = {}
    worker.kv_send_layer_thread = MagicMock()
    metadata = LayerwisePullProducerMetadata()
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
    reader = LayerwisePullReadThread(
        tp_rank=0,
        side_channel_port=1,
        backend=MagicMock(),
        state=ConsumerReadState(tp_size=1, layer_layouts=dict.fromkeys(local_layers, ()), dest_blocks_by_req={}),
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


@pytest.mark.parametrize("mixed_batch,cancel_before_failure", [(False, False), (True, False), (False, True)])
def test_failed_pp_read_cannot_write_after_request_cleanup(mixed_batch, cancel_before_failure):
    # Exercise the actual failure handler, worker completion/cleanup and late
    # messages from another PP source. Only the payload backend is mocked.
    backend = MagicMock()
    backend.read.side_effect = RuntimeError("P0 read failed")
    state = ConsumerReadState(
        tp_size=1,
        layer_layouts={layer: (_component(f"layer.{layer}", 0, 2000 + 100 * layer, 16),) for layer in (0, 1)},
        dest_blocks_by_req={"request": [[0]], "healthy": [[1]]},
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread.get_ip", return_value="127.0.0.1"
    ):
        reader = LayerwisePullReadThread(0, port, backend, state)
    worker = LayerwisePullConsumerWorker.__new__(LayerwisePullConsumerWorker)
    worker._pending_recv_req_ids = set()
    worker._deferred_cleanup_req_ids = set()
    worker.tp_size = 1
    worker._read_thread = reader
    worker._terminal_ext_ids = set()
    worker._dest_blocks_by_req = state.dest_blocks_by_req
    worker._invalid_block_ids = set()
    worker.request_map = {"request": "request123456789", "healthy": "healthy123456789"}
    worker._pending_recv_req_ids = set(worker.request_map.values())
    ctx = zmq.Context()
    dealers = []
    decoder = msgspec.msgpack.Decoder(type=tuple)
    reader.start()
    try:
        assert reader.ready_event.wait(timeout=2)
        assert reader.startup_error is None
        for pp in (0, 1):
            dealer = ctx.socket(zmq.DEALER)
            dealers.append(dealer)
            dealer.setsockopt(zmq.RCVTIMEO, 2000)
            dealer.connect(f"tcp://127.0.0.1:{port}")
            layout = {pp: {"components": [asdict(_component(f"layer.{pp}", 0, 1000, 16))]}}
            dealer.send(
                msgspec.msgpack.encode((LAYOUT_META, f"P{pp}", msgspec.msgpack.encode(layout), ((0,), (1,)), 1, pp, 0))
            )
            assert dealer.recv_multipart() == [b"", b"ACK"]
        if cancel_before_failure:
            assert worker.get_finished({"request123456789"}) == (set(), set())
            assert "request" in worker._dest_blocks_by_req
        dealers[0].send(
            msgspec.msgpack.encode((READ_READY_BATCH, 0, "", [("request", [[0]], [0])], ["request"], 0, 1, 0))
        )
        reply = decoder.decode(dealers[0].recv_multipart()[-1])
        assert reply == (READ_FAILED, 0, "P0 read failed", 0)
        assert worker.get_finished() == (set(), {"request123456789"})
        assert worker.get_block_ids_with_load_errors() == {0}
        assert worker.get_finished({"request123456789"}) == (set(), set())
        assert "request" not in state.dest_blocks_by_req
        assert "request123456789" not in worker._pending_recv_req_ids
        assert worker._deferred_cleanup_req_ids == set()
        # Even if an address mapping is present again, a late source must not
        # write to it. Cleanup must not remove the failure tombstone.
        state.dest_blocks_by_req["request"] = [[2]]
        backend.read.reset_mock(side_effect=True)
        late_requests = [("request", [[0]], [0])]
        terminal_ids = ["request"]
        if mixed_batch:
            late_requests.append(("healthy", [[1]], [0]))
            terminal_ids.append("healthy")
        for pp in (1, 0):
            dealers[pp].send(msgspec.msgpack.encode((READ_READY_BATCH, pp, "", late_requests, terminal_ids, 0, 1, 1)))
            # ACK releases P's slot even when every request was cancelled.
            assert decoder.decode(dealers[pp].recv_multipart()[-1]) == (READ_DONE, pp, 1)
        if mixed_batch:
            assert backend.read.call_count == 2
            for call, pp in zip(backend.read.call_args_list, (1, 0), strict=True):
                assert call.args == (f"P{pp}", [2000 + 100 * pp + 16], [1016], [16])
            assert worker.get_finished() == (set(), {"healthy123456789"})
        else:
            backend.read.assert_not_called()
            assert worker.get_finished() == (set(), set())
        assert reader.get_and_clear_failed() == set()
        assert "request" not in reader._done_contributors
        assert "request" not in reader._expected_sources
    finally:
        reader.stop(timeout=2)
        for dealer in dealers:
            dealer.close(linger=0)
        ctx.destroy(linger=0)
        assert not reader.is_alive()


@pytest.mark.parametrize("empty_blocks", [False, True])
def test_split_stage_sends_terminal_markers_and_waits_for_all_destinations(sending_thread, empty_blocks):
    thread, _ = sending_thread
    # Exercise the send implementation synchronously while its run loop is idle.
    thread._state.layer_layouts = {layer: (_component(f"layer.{layer}", 0, 1000, 16),) for layer in (0, 1)}
    thread._state.block_sizes = (16,)
    request = LayerwisePullProducerReqMeta(
        local_block_ids=[[] if empty_blocks else [1]],
        remote_tp_size=1,
        chunk_finish=True,
        chunk_start_blocks=[0],
        layer_endpoints={0: ("decode-a", 4000), 1: ("decode-b", 5000)},
        terminal_layers=frozenset({0, 1}),
    )
    paths = ["tcp://decode-a:4000", "tcp://decode-b:5000"]
    dealers = {path: MagicMock() for path in paths}
    thread._layout_meta_sent_paths = set(paths)
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread.make_zmq_path",
            side_effect=lambda protocol, host, port: f"{protocol}://{host}:{port}",
        ),
        patch.object(thread, "_ensure_dealer", side_effect=dealers.__getitem__),
    ):
        thread.track_requests({"request"}, set(request.layer_endpoints.values()))
        for layer in (0, 1):
            thread.mark_layer_pending(layer)
            thread._process_send_task(SendTask({"request": request}, layer_idx=layer), msgspec.msgpack.Encoder())
    for path, dealer in dealers.items():
        message = msgspec.msgpack.decode(dealer.send.call_args.args[0])
        assert message[4] == ["request"]
        assert bool(message[3]) is not empty_blocks
    assert thread.get_and_clear_finished_requests({"request"}) == set()
    # The later layer's ACK may arrive first; source blocks must stay held.
    thread._complete_reader((1, paths[1]))
    assert thread.get_and_clear_finished_requests(set()) == set()
    thread._complete_reader((0, paths[0]))
    assert thread.get_and_clear_finished_requests(set()) == {"request"}
    thread._complete_reader((0, paths[0]))
    assert thread.get_and_clear_finished_requests(set()) == set()


def test_pp_pull_over_real_control_sockets_with_chunked_slot_reuse():
    # Only the data-transfer backend/NPU events are mocked. Both control
    # threads, their wire messages, and the slot gates are real.
    # P1 feeds both D stages, while D0 reads from both P stages.
    producer_layers = ((0, 1, 2), (3, 4, 5))
    decode_layers = ((0, 1, 2, 3), (4, 5))
    readers = []
    workers = []
    endpoints = []
    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread.get_ip", return_value="127.0.0.1"),
        patch("vllm.distributed.get_world_group", return_value=SimpleNamespace(local_rank=0)),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread.torch"),
    ):
        try:
            for layers in decode_layers:
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                reader = LayerwisePullReadThread(
                    tp_rank=0,
                    side_channel_port=port,
                    backend=MagicMock(),
                    state=ConsumerReadState(
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
                worker = LayerwisePullProducerWorker.__new__(LayerwisePullProducerWorker)
                worker.pp_rank, worker.pp_size = pp, len(producer_layers)
                worker.tp_rank, worker.tp_size = 0, 1
                worker._layer_order = layers
                worker._routes_by_topology = {}
                worker.layer_storage_slots = dict.fromkeys(layers, (0,))
                worker.reused_storage_slots = frozenset({0})
                worker.kv_send_layer_thread = LayerwisePullSendingThread(
                    ready_event=threading.Event(),
                    state=ProducerSendState(
                        last_layer_idx=layers[-1],
                        layer_layouts={layer: (_component(f"layer.{layer}", 0, 1000, 16),) for layer in layers},
                        p_session=f"producer-{pp}",
                        block_sizes=(16,),
                        layer_storage_slots=worker.layer_storage_slots,
                        num_blocks=2,
                        pp_rank=pp,
                    ),
                )
                workers.append(worker)
                worker.kv_send_layer_thread.start()
                assert worker.kv_send_layer_thread.ready_event.wait(timeout=2)
                assert worker.kv_send_layer_thread.startup_error is None
            for chunk in (0, 1):
                for worker in workers:
                    meta = LayerwisePullProducerMetadata()
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
                        sender.mark_layer_pending(layer)
                        sender.enqueue(SendTask(meta.requests, layer_idx=layer, layer_name=f"layer.{layer}"))
                        # Never hang the test if routing or a completion marker
                        # is broken. Check the real gate before entering waiter.
                        assert sender.get_storage_send_event(0).wait(timeout=3)
                        worker.wait_for_slot_release(layer)
                if chunk == 0:
                    assert all(reader.get_and_clear_done() == set() for reader in readers)
            for reader, layers in zip(readers, decode_layers, strict=True):
                assert reader.get_and_clear_done() == {"request"}
                assert reader.get_and_clear_failed() == set()
                assert reader.backend.read.call_count == 2 * len(layers)
        finally:
            for worker in workers:
                worker.kv_send_layer_thread.stop(timeout=2)
                assert not worker.kv_send_layer_thread.is_alive()
            for reader in readers:
                reader.stop(timeout=2)
                assert not reader.is_alive()
