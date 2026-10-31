from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call, patch

import pytest
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import make_embedding_key

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    worker as worker_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.async_store_get import (
    AscendECMooncakeConnectorMetadata,
    AsyncStoreGetExecutor,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import AscendProducerMemoryPool
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_backend import PreparedStorePut
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import AscendMooncakeEmbeddingStoreClient
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import (
    AscendECMooncakeWorker,
    _resolve_ascend_config,
)


def _make_push(transfer_id: str, nbytes: int = 10) -> MagicMock:
    push = MagicMock()
    push.spec.transfer_id = transfer_id
    push.spec.mm_hash = f"hash-{transfer_id}"
    push.source_tensor = MagicMock(nbytes=nbytes)
    return push


def _make_writable_shard(
    session: str,
    destination: int,
    nbytes: int = 10,
) -> dict[str, object]:
    return {
        "ready": True,
        "cached": False,
        "cancelled": False,
        "write": True,
        "nbytes": nbytes,
        "dst_session": session,
        "dst_ptr": destination,
    }


def _make_lifecycle_worker() -> AscendECMooncakeWorker:
    worker = object.__new__(AscendECMooncakeWorker)
    worker.is_producer = True
    worker._store_config = ("namespace", 4, 1024, 256)
    worker._output_store = None
    worker._producer_memory = MagicMock()
    worker._producer_memory.release_after.side_effect = partial(
        AscendProducerMemoryPool.release_after, worker._producer_memory
    )
    worker._transfer = MagicMock()
    worker._buffer_device = "npu"
    worker._async_store_get = AsyncStoreGetExecutor(enabled=False, device=worker._buffer_device)
    worker._shutdown = False
    worker._dispatcher = None
    worker._control_server = None
    worker._producer_pushes = MagicMock()
    worker._producer_pushes.cancel_requests.return_value = []
    worker._control_executor = MagicMock()
    worker._io_executor = MagicMock()
    worker._shard_pool = None
    worker._control_client = MagicMock()
    worker._consumer_memory = MagicMock()
    worker._flush_pending_pushes = MagicMock()
    return worker


def test_p2p_sends_whole_tensors_once_per_session():
    worker = object.__new__(AscendECMooncakeWorker)
    worker._transfer = MagicMock()
    worker._producer_pushes = MagicMock()
    first = MagicMock()
    first.spec.transfer_id = "first"
    second = MagicMock()
    second.spec.transfer_id = "second"
    ready = [
        (first, {"dst_session": "session-a", "dst_ptr": 1000}),
        (second, {"dst_session": "session-a", "dst_ptr": 2000}),
        (second, {"dst_session": "session-b", "dst_ptr": 3000}),
    ]
    first_tensor, second_tensor = MagicMock(nbytes=10), MagicMock(nbytes=7)
    first_tensor.data_ptr.return_value = 100
    second_tensor.data_ptr.return_value = 200
    staged = SimpleNamespace(tensors=[first_tensor, second_tensor])
    tracked_future = MagicMock()

    def run_fanout(tasks, on_submit):
        assert len(tasks) == 2
        on_submit(1, tracked_future)
        return [task() for task in tasks]

    worker._run_fanout = MagicMock(side_effect=run_fanout)

    worker._write_staged_sources([first, second], ready, staged, [], [])

    assert worker._transfer.write.call_args_list == [
        call("session-a", [100, 200], [1000, 2000], [10, 7]),
        call("session-b", [200], [3000], [7]),
    ]
    worker._producer_pushes.track_shard_futures.assert_called_once_with(
        [second],
        [tracked_future],
    )


def _make_vllm_config(producer=False):
    vllm_config = MagicMock()

    parallel_config = vllm_config.parallel_config
    parallel_config.tensor_parallel_size = 1
    parallel_config.pipeline_parallel_size = 1
    parallel_config.data_parallel_size = 1
    parallel_config.data_parallel_index = 0

    ec_config = vllm_config.ec_transfer_config
    ec_config.is_ec_producer = producer
    ec_config.is_ec_consumer = not producer
    ec_config.is_encode_only = producer
    vllm_config.use_v2_model_runner = True
    vllm_config.lora_config = None
    vllm_config.model_config.multimodal_config = None
    ec_config.ec_buffer_device = "cuda"
    ec_config.ec_buffer_size = 1024
    ec_config.ec_ip = "127.0.0.1"
    ec_config.ec_port = 14579
    ec_config.ec_connector_extra_config = {}
    ec_config.get_from_extra_config.side_effect = lambda key, default: ec_config.ec_connector_extra_config.get(
        key, default
    )

    return vllm_config


def test_resolve_ascend_config_maps_upstream_defaults():
    config = _resolve_ascend_config(_make_vllm_config())
    assert config.protocol == "ascend"
    assert config.buffer_device == "npu"


def test_consumed_async_hit_binds_an_earlier_push_without_repeating_spec():
    worker = object.__new__(AscendECMooncakeWorker)
    worker._async_store_get = MagicMock(enabled=True)
    worker._async_store_get.consume.return_value = {
        "deferred-hit",
        "current-hit",
        "miss",
    }
    worker._bind_push_source = MagicMock()
    store = MagicMock(namespace="namespace")
    encoder_cache = {
        "deferred-hit": MagicMock(),
        "current-hit": MagicMock(),
    }
    metadata = AscendECMooncakeConnectorMetadata(
        pushes=[SimpleNamespace(mm_hash="current-hit")],
    )

    with patch.object(worker_module.ECMooncakeWorker, "start_save_caches"):
        worker._start_save_caches_with_async_store_get(
            store,
            metadata,
            encoder_cache,
        )

    worker._bind_push_source.assert_called_once_with(
        encoder_cache["deferred-hit"],
        "deferred-hit",
    )
    store.resolve_inputs.assert_not_called()
    store.store_client.get_payload_callback.assert_not_called()
    worker._async_store_get.dispatch.assert_called_once_with(
        store,
        metadata.store_get_requests,
        on_payloads=ANY,
    )


@pytest.mark.parametrize("candidate_state", ["absent", "cached", "consumed"])
def test_async_step_without_sync_reads_progresses_while_get_lock_is_held(candidate_state):
    worker = _make_lifecycle_worker()
    worker._async_store_get = MagicMock(enabled=True)
    worker._async_store_get.consume.return_value = {"image"} if candidate_state == "consumed" else set()
    worker._bind_deferred_store_hits = MagicMock()
    store = MagicMock(namespace="namespace")
    worker._output_store = store
    client = store.store_client
    client._get_io_lock = RLock()
    previous_callback = object()
    client.on_get_payloads = previous_callback
    client.get_payload_callback.side_effect = lambda callback: AscendMooncakeEmbeddingStoreClient.get_payload_callback(
        client, callback
    )
    metadata = AscendECMooncakeConnectorMetadata(
        store_candidates={} if candidate_state == "absent" else {"image": MagicMock()},
        store_get_requests={"next-image": MagicMock()},
    )
    cache = {"image": MagicMock()} if candidate_state == "cached" else {}

    with patch.object(worker_module.ECMooncakeWorker, "start_save_caches") as upstream, ThreadPoolExecutor(1) as model:
        # Model work must complete before the lock held by the background GET is released.
        # Always release on failure so a regression cannot strand the test thread.
        client._get_io_lock.acquire()
        try:
            step = model.submit(worker._start_save_caches_with_async_store_get, store, metadata, cache)
            step.result(timeout=3)
        finally:
            client._get_io_lock.release()

    upstream.assert_called_once_with(replace(metadata, store_candidates={}), cache)
    worker._bind_deferred_store_hits.assert_called_once()
    client.get_payload_callback.assert_not_called()
    store.resolve_inputs.assert_not_called()
    assert client.on_get_payloads is previous_callback
    store.set_step_candidates.assert_called_once_with(set(metadata.store_candidates))
    worker._async_store_get.dispatch.assert_called_once_with(store, metadata.store_get_requests, on_payloads=ANY)


def test_async_lookup_suppression_keeps_backend_visible_to_prepared_writer():
    worker = _make_lifecycle_worker()
    worker._async_store_get = MagicMock(enabled=True)
    worker._async_store_get.consume.return_value = set()
    worker._bind_deferred_store_hits = MagicMock()
    worker._run_fanout = MagicMock()
    store = MagicMock(namespace="namespace")
    worker._output_store = store
    metadata = AscendECMooncakeConnectorMetadata(store_candidates={"image": MagicMock()})
    entered, written = Event(), Event()

    def upstream_step(instance, current, encoder_cache, **kwargs):
        assert instance._output_store is store
        assert current.store_candidates == {}
        assert current.pushes is metadata.pushes
        entered.set()
        assert written.wait(timeout=3)

    with (
        patch.object(worker_module.ECMooncakeWorker, "start_save_caches", new=upstream_step),
        ThreadPoolExecutor(1) as model,
    ):
        step = model.submit(worker._start_save_caches_with_async_store_get, store, metadata, {})
        try:
            assert entered.wait(timeout=3)
            puts = [MagicMock()]
            assert worker._write_staged_sources([], [], SimpleNamespace(tensors=[]), puts, [])
            store.submit_prepared_batch.assert_called_once_with(puts)
        finally:
            written.set()
        step.result(timeout=3)
    assert worker._output_store is store
    assert "image" in metadata.store_candidates


def test_async_step_sync_bypass_keeps_scoped_payload_callback():
    worker = _make_lifecycle_worker()
    worker._async_store_get = MagicMock(enabled=True)
    worker._async_store_get.consume.return_value = set()
    worker._bind_deferred_store_hits = MagicMock()
    worker._bind_push_source = MagicMock()
    worker._send_loaded_payloads = MagicMock(return_value=[])
    store = MagicMock(namespace="namespace")
    worker._output_store = store
    client = store.store_client
    client._get_io_lock = RLock()
    previous_callback = object()
    client.on_get_payloads = previous_callback
    client.get_payload_callback.side_effect = lambda callback: AscendMooncakeEmbeddingStoreClient.get_payload_callback(
        client, callback
    )
    spec, payload, target = MagicMock(), MagicMock(), MagicMock()
    cache: dict[str, MagicMock] = {}
    metadata = AscendECMooncakeConnectorMetadata(store_candidates={"bypass": spec})
    pool_key = make_embedding_key(store.namespace, "bypass")

    def resolve(expected, encoder_cache, device):
        assert expected == {"bypass": spec}
        assert device == worker._buffer_device
        client.on_get_payloads({pool_key: payload})
        encoder_cache["bypass"] = target
        return {"bypass"}

    store.resolve_inputs.side_effect = resolve
    with patch.object(worker_module.ECMooncakeWorker, "start_save_caches"):
        worker._start_save_caches_with_async_store_get(store, metadata, cache)

    worker._send_loaded_payloads.assert_called_once_with({pool_key: "bypass"}, {pool_key: payload})
    worker._bind_push_source.assert_called_once_with(target, "bypass")
    assert client.on_get_payloads is previous_callback
    store.resolve_inputs.assert_called_once()
    worker._async_store_get.dispatch.assert_called_once_with(store, metadata.store_get_requests, on_payloads=ANY)


@pytest.mark.parametrize(
    "protocol,buffer_device,error",
    [("rdma", "cuda", "mooncake_protocol='ascend'")]
    + [("ascend", device, "ec_buffer_device='npu'") for device in ("cpu", "npu:abc")],
)
def test_resolve_ascend_config_rejects_incompatible_protocol_or_device(protocol, buffer_device, error):
    config = _make_vllm_config()
    config.ec_transfer_config.ec_connector_extra_config = {"mooncake_protocol": protocol}
    upstream_config = MagicMock(protocol=protocol, buffer_device=buffer_device)
    with (
        patch.object(worker_module.MooncakeECConfig, "from_vllm_config", return_value=upstream_config),
        pytest.raises(ValueError, match=error),
    ):
        _resolve_ascend_config(config)


def test_bind_push_source_uses_npu_event():
    worker = object.__new__(AscendECMooncakeWorker)
    worker._producer_pushes = MagicMock()
    tensor = MagicMock()
    tensor.device.type = "npu"
    stream = MagicMock()
    event = MagicMock()

    with (
        patch.object(worker_module.torch.npu, "current_stream", return_value=stream) as current_stream,
        patch.object(worker_module.torch.npu, "Event", return_value=event) as event_class,
    ):
        worker._bind_push_source(tensor, "image-hash")

    event_class.assert_called_once_with()
    current_stream.assert_called_once_with(tensor.device)
    event.record.assert_called_once_with(stream)
    worker._producer_pushes.bind_source.assert_called_once_with("image-hash", tensor, event)


@pytest.mark.parametrize("store_first", [False, True])
def test_ec_notification_does_not_wait_for_put_but_memory_does(store_first):
    worker = _make_lifecycle_worker()
    worker._output_store = MagicMock()
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._run_fanout = MagicMock(side_effect=lambda tasks, _: [task() for task in tasks])
    worker._use_store_independent_te = False
    push = _make_push("first")
    shard = _make_writable_shard("session", 1000)
    worker._producer_pushes.resolve_reservations.return_value = [shard]
    ticket = MagicMock(identifier=push.spec.mm_hash, tensor=push.source_tensor)
    worker._output_store.prepare_saves.return_value = {push.spec.mm_hash: ticket}
    pending: Future[None] = Future()
    worker._output_store.submit_prepared_batch.return_value = pending
    payload = MagicMock(nbytes=10)
    payload.data_ptr.return_value = 500
    header = MagicMock(nbytes=24)
    header.data_ptr.return_value = 476
    staged = SimpleNamespace(tensors=[payload], headers=[header])
    worker._producer_memory.stage_with_headers.return_value = staged
    if store_first:
        pending.set_result(None)
    worker._push_batch([push])
    worker._notify_completions.assert_called_once_with([(push, shard)])
    worker._producer_pushes.complete.assert_called_once_with([push])
    worker._transfer.write.assert_called_once_with("session", [500], [1000], [10])
    [prepared] = worker._output_store.submit_prepared_batch.call_args.args[0]
    assert prepared.addresses == [476, 500]
    assert prepared.owners == [staged]
    if not store_first:
        worker._producer_memory.release.assert_not_called()
        pending.set_result(None)
    worker._producer_memory.release.assert_called_once_with(staged)


@pytest.mark.parametrize("topology", ["shared", "independent", "fabric"])
def test_all_store_topologies_bind_fixed_pool_before_services(topology):
    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import store_backend

    worker = _make_lifecycle_worker()
    client = MagicMock()
    backend = MagicMock()
    sequence = MagicMock()
    sequence.attach_mock(worker._transfer.ensure_initialized, "initialize")
    sequence.attach_mock(worker._producer_memory.ensure_prepared, "prepare")
    sequence.attach_mock(client.bind_staging_buffer, "bind")
    retained = len(worker_module._PROCESS_LIFETIME_EC_OWNERS)
    with (
        patch.dict(worker_module.os.environ, {"ASCEND_ENABLE_USE_FABRIC_MEM": str(int(topology == "fabric"))}),
        patch.object(store_backend, "use_store_independent_te", return_value=topology == "independent"),
        patch.object(store_backend, "ensure_mooncake_store_imported"),
        patch.object(worker_module.torch, "device", return_value=SimpleNamespace(type="npu")),
        patch.object(store_backend, "create_ascend_mooncake_embedding_store_client", return_value=client) as factory,
        patch.object(worker_module, "AscendDeferredStoreBackend", return_value=backend),
        patch.object(worker_module.ECMooncakeWorker, "start_services") as start,
    ):
        sequence.attach_mock(factory, "factory")
        sequence.attach_mock(start, "services")
        try:
            worker.start_services()
            calls = [c[0] for c in sequence.mock_calls]
            expected = {
                "shared": ["factory", "prepare", "bind", "services"],
                "independent": ["prepare", "factory", "bind", "services"],
                "fabric": ["prepare", "factory", "bind", "services"],
            }
            assert calls == expected[topology]
            factory.assert_called_once_with(worker._producer_memory, worker._transfer, read_buffer_bytes=256)
            assert worker._output_store is backend
        finally:
            del worker_module._PROCESS_LIFETIME_EC_OWNERS[retained:]


def test_reservation_callback_uses_actual_receive_pool_device():
    import threading

    worker = _make_lifecycle_worker()
    worker._control_thread = threading.local()
    actual_device = SimpleNamespace(type="npu", index=1)
    worker._consumer_memory.tensor.device = actual_device
    with (
        patch.object(worker_module.torch.npu, "set_device") as bind,
        patch.object(worker_module.ECMooncakeWorker, "_reserve_push_destination", return_value={}),
    ):
        worker._reserve_push_destination({})
        worker._reserve_push_destination({})
    bind.assert_called_once_with(actual_device)


def test_store_shutdown_joins_memory_reclamation_callbacks_before_close():
    worker = _make_lifecycle_worker()
    client = MagicMock()
    client.exists.return_value = False
    backend = worker_module.AscendDeferredStoreBackend(client, "namespace", max_pending_items=1, max_pending_bytes=128)
    worker._output_store = backend
    put_started, put_gate, reclaim_started, reclaim_gate = Event(), Event(), Event(), Event()

    def put(*_):
        put_started.set()
        assert put_gate.wait(timeout=5)

    def reclaim(*_):
        reclaim_started.set()
        assert reclaim_gate.wait(timeout=5)

    client.put_prepared_tensors.side_effect = lambda items: (put(), [0] * len(items))[1]
    client.batch_exists.return_value = [False]
    worker._producer_memory.release = MagicMock(side_effect=reclaim)
    tensor = MagicMock()
    tensor.untyped_storage.return_value.nbytes.return_value = 128
    backend._step_candidates = {"first"}
    backend.save_output("first", tensor)
    ticket = backend.prepare_saves(["first"]).get("first")
    store_put = PreparedStorePut(ticket, [500], [10], [tensor])
    future = backend.submit_prepared_batch([store_put])
    retained = worker_module._PROCESS_LIFETIME_EC_OWNERS
    before = len(retained)
    try:
        assert put_started.wait(timeout=1)
        worker._release_after_store(MagicMock(), [future])
        with ThreadPoolExecutor(max_workers=1) as executor:
            closing = executor.submit(worker.close)
            try:
                put_gate.set()
                assert reclaim_started.wait(timeout=1)
                assert future.done()
                assert not closing.done()
                client.close.assert_not_called()
            finally:
                put_gate.set()
                reclaim_gate.set()
            closing.result(timeout=1)
        client.close.assert_called_once_with()
        worker._producer_memory.release.assert_called_once()
        worker._producer_memory.close.assert_not_called()
        worker._transfer.close.assert_not_called()
        assert any(worker._producer_memory in owners for _, owners in retained[before:])
    finally:
        put_gate.set()
        reclaim_gate.set()
        backend.shutdown()
        del retained[before:]


def test_unknown_ec_write_retains_staging_and_never_notifies_success():
    worker = _make_lifecycle_worker()
    worker._output_store = None
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._run_fanout = MagicMock(side_effect=RuntimeError("native status unknown"))
    push = _make_push("first")
    worker._producer_pushes.resolve_reservations.return_value = [_make_writable_shard("session", 1000)]
    payload = MagicMock(nbytes=10)
    staged = SimpleNamespace(tensors=[payload], headers=[None])
    worker._producer_memory.stage_wait.return_value = staged
    retained = len(worker_module._PROCESS_LIFETIME_EC_OWNERS)
    try:
        worker._push_batch([push])
        worker._producer_memory.release.assert_not_called()
        worker._producer_memory.poison.assert_called_once()
        worker._notify_completions.assert_not_called()
        worker._producer_pushes.fail.assert_called_once()
    finally:
        del worker_module._PROCESS_LIFETIME_EC_OWNERS[retained:]


def test_direct_get_submission_failure_restores_waiters():
    import torch
    from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import ECMooncakePushSpec

    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import AscendProducerPushManager

    manager = AscendProducerPushManager()
    spec = ECMooncakePushSpec("hash", 4, (1,), "float32", "consumer", "transfer")
    record, _ = manager.reserve(spec, Future)
    executor = MagicMock()
    executor.submit.side_effect = RuntimeError("executor stopped")
    with pytest.raises(RuntimeError, match="executor stopped"):
        manager.submit_loaded_batch({"hash": torch.zeros(1)}, executor, MagicMock())
    assert record.source_tensor is None
    assert record.batch_future is None
    assert "transfer" in manager._source_waiters["hash"]


def test_direct_get_batch_writes_multiple_payloads_in_one_native_call():
    import torch
    from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import ECMooncakePushSpec

    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import AscendProducerPushManager

    worker = _make_lifecycle_worker()
    worker._output_store = None
    manager = AscendProducerPushManager()
    worker._producer_pushes = manager
    worker._notify_completions = MagicMock()
    worker._run_fanout = lambda tasks, on_submit: [task() for task in tasks]
    payloads = {
        "first": torch.arange(4, dtype=torch.float32),
        "second": torch.arange(6, dtype=torch.float32),
    }
    reservations, records = [], []
    for index, (mm_hash, payload) in enumerate(payloads.items()):
        spec = ECMooncakePushSpec(
            mm_hash,
            payload.nbytes,
            tuple(payload.shape),
            "float32",
            "consumer",
            f"transfer-{index}",
            "request",
        )
        reservation: Future[list[dict[str, object]]] = Future()
        record, _ = manager.reserve(spec, lambda reservation=reservation: reservation)
        records.append(record)
        reservations.append(reservation)

    with ThreadPoolExecutor(1) as executor:
        future = manager.submit_loaded_batch(
            payloads,
            executor,
            partial(worker._push_batch, loaded_payloads=payloads),
        )
        assert future is not None
        assert not future.done()
        assert all(record.batch_future is future for record in records)
        assert all(record.source_tensor is payloads[record.spec.mm_hash] for record in records)
        assert manager.cancel_requests({"request"}) == []
        # Cache binding and normal dispatch must not replace or resend loaded payloads.
        manager.bind_source("first", torch.zeros_like(payloads["first"]), None)
        duplicate_executor = MagicMock()
        manager.submit_batches(duplicate_executor, worker._push_batch)
        duplicate_executor.submit.assert_not_called()
        for index, (reservation, payload) in enumerate(zip(reservations, payloads.values(), strict=True)):
            reservation.set_result([_make_writable_shard("peer", 1000 + index * 100, payload.nbytes)])
        future.result(timeout=3)

    worker._transfer.write.assert_called_once_with(
        "peer",
        [payload.data_ptr() for payload in payloads.values()],
        [1000, 1100],
        [payload.nbytes for payload in payloads.values()],
    )
    worker._producer_memory.stage_wait.assert_not_called()
    worker._producer_memory.stage_with_headers.assert_not_called()
    worker._notify_completions.assert_called_once()
    assert all(record.source_tensor is None for record in records)
    manager.poll()
    assert not manager.pending


def test_shutdown_drains_direct_get_tasks_before_returning():
    worker = _make_lifecycle_worker()
    worker._output_store = None
    worker._retain_native_owners = MagicMock()
    started, release, draining = Event(), Event(), Event()

    def send():
        started.set()
        assert release.wait(3)

    with ThreadPoolExecutor(1) as direct, ThreadPoolExecutor(1) as closing_executor:
        worker._get_executor = direct
        send_future = direct.submit(send)
        assert started.wait(3)
        original_shutdown = direct.shutdown

        def shutdown(*args, **kwargs):
            draining.set()
            return original_shutdown(*args, **kwargs)

        with patch.object(direct, "shutdown", side_effect=shutdown):
            closing = closing_executor.submit(worker.close)
            try:
                assert draining.wait(3)
                assert not closing.done()
            finally:
                release.set()
            closing.result(timeout=3)
        assert send_future.done()


@pytest.mark.parametrize("outcome", ["hit", "safe_failure", "fatal_failure"])
def test_async_get_scopes_payload_callback_and_restores_it(outcome):
    executor = AsyncStoreGetExecutor(enabled=False, device="npu:3")
    client = MagicMock()
    client._get_io_lock = RLock()
    previous = object()
    client.on_get_payloads = previous
    client.get_payload_callback.side_effect = lambda callback: AscendMooncakeEmbeddingStoreClient.get_payload_callback(
        client, callback
    )
    store = MagicMock(namespace="namespace", store_client=client)
    key = make_embedding_key(store.namespace, "image")
    payload, target, spec = MagicMock(), MagicMock(), MagicMock()
    miss_key = make_embedding_key(store.namespace, "miss")
    client.batch_exists.return_value = [True, False]
    send = MagicMock(return_value=[])

    def load(expected, device):
        assert expected == {key: spec}
        assert device == "npu:3"
        if outcome == "safe_failure":
            raise OSError("confirmed safe failure")
        if outcome == "fatal_failure":
            raise RuntimeError("unknown completion")
        client.on_get_payloads({key: payload})
        return {key: target}

    client.load_tensors_ready.side_effect = load
    expected = {"image": spec, "miss": MagicMock()}
    if outcome == "fatal_failure":
        with pytest.raises(RuntimeError, match="unknown completion"):
            executor._load_batch(store, expected, on_payloads=send)
    else:
        result = executor._load_batch(store, expected, on_payloads=send)
    assert client.on_get_payloads is previous
    client.batch_exists.assert_called_once_with([key, miss_key])
    client.load_tensors_ready.assert_called_once_with({key: spec}, "npu:3")
    if outcome == "hit":
        assert result.failures == set()
        assert result.tensors == {"image": target, "miss": None}
        send.assert_called_once_with({key: payload})
    else:
        if outcome == "safe_failure":
            assert result.failures == set(expected)
        send.assert_not_called()


def test_async_get_poll_surfaces_late_reader_poison_without_pending_get():
    executor = AsyncStoreGetExecutor(enabled=False, device="npu:3")
    executor._store_client = MagicMock()
    executor._store_client._check_healthy.side_effect = RuntimeError("late native reader failure")
    with pytest.raises(RuntimeError, match="late native reader failure"):
        executor.poll()


@pytest.mark.parametrize("fatal_get", [False, True])
def test_shutdown_keeps_direct_send_dependencies_alive_until_async_get_settles(fatal_get):
    worker = _make_lifecycle_worker()
    worker._retain_native_owners = MagicMock()
    worker._async_store_get = MagicMock()
    worker._get_executor = MagicMock()
    order = []

    def finish_get():
        order.append("get")
        worker._control_executor.shutdown.assert_not_called()
        worker._get_executor.shutdown.assert_not_called()
        worker._control_client.close.assert_not_called()
        if fatal_get:
            raise RuntimeError("fatal GET")

    worker._async_store_get.close.side_effect = finish_get
    worker._get_executor.shutdown.side_effect = lambda **kwargs: order.append("send")
    if fatal_get:
        with pytest.raises(RuntimeError, match="fatal GET"):
            worker.close()
    else:
        worker.close()
    assert order == ["get", "send"]
    worker._control_client.close.assert_called_once()
