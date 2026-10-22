from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    worker as worker_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    _resolve_bounce_arena_size,
    _TransferFragmentPlan,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_backend import PreparedStorePut
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import (
    AscendECMooncakeWorker,
    _AcquiredTransferWave,
    _resolve_ascend_config,
)


def _make_bounce_config(
    *,
    max_num_seqs: int = 128,
    max_num_active_seqs: int | None = None,
    extra_config: dict[str, object] | None = None,
) -> SimpleNamespace:
    resolved_extra_config = {} if extra_config is None else extra_config
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_active_seqs=max_num_active_seqs,
        ),
        ec_transfer_config=SimpleNamespace(ec_connector_extra_config=resolved_extra_config),
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
    worker._transfer = MagicMock()
    worker._buffer_device = "npu"
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


def test_write_transfer_wave_batches_fragments_once_per_session():
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
    acquired = _AcquiredTransferWave(
        fragments=[
            _TransferFragmentPlan(0, 100, 0, 10),
            _TransferFragmentPlan(1, 200, 0, 3),
            _TransferFragmentPlan(1, 300, 3, 7),
        ],
    )
    tracked_future = MagicMock()

    def run_fanout(tasks, on_submit):
        assert len(tasks) == 2
        on_submit(1, tracked_future)
        return [task() for task in tasks]

    worker._run_fanout = MagicMock(side_effect=run_fanout)

    worker._write_transfer_wave([first, second], ready, acquired)

    assert worker._transfer.write.call_args_list == [
        call("session-a", [100, 200, 300], [1000, 2000, 2003], [10, 3, 7]),
        call("session-b", [200, 300], [3000, 3003], [3, 7]),
    ]
    worker._producer_pushes.track_shard_futures.assert_called_once_with(
        [second],
        [tracked_future],
    )


def test_resolve_ascend_config_maps_upstream_defaults():
    vllm_config = MagicMock()

    parallel_config = vllm_config.parallel_config
    parallel_config.tensor_parallel_size = 1
    parallel_config.pipeline_parallel_size = 1
    parallel_config.data_parallel_size = 1
    parallel_config.data_parallel_index = 0

    ec_config = vllm_config.ec_transfer_config
    ec_config.is_ec_producer = False
    ec_config.is_ec_consumer = True
    ec_config.ec_buffer_device = "cuda"
    ec_config.ec_buffer_size = 1024
    ec_config.ec_ip = "127.0.0.1"
    ec_config.ec_port = 14579
    ec_config.ec_connector_extra_config = {}
    ec_config.get_from_extra_config.side_effect = lambda key, default: ec_config.ec_connector_extra_config.get(
        key, default
    )

    config = _resolve_ascend_config(vllm_config)

    assert config.protocol == "ascend"
    assert config.buffer_device == "npu"


def test_resolve_ascend_config_rejects_non_ascend_protocols():
    vllm_config = MagicMock()

    vllm_config.ec_transfer_config.ec_connector_extra_config = {"mooncake_protocol": "rdma"}
    upstream_config = MagicMock(
        protocol="rdma",
        buffer_device="cuda",
    )

    with (
        patch.object(
            worker_module.MooncakeECConfig,
            "from_vllm_config",
            return_value=upstream_config,
        ),
        pytest.raises(ValueError, match="mooncake_protocol='ascend'"),
    ):
        _resolve_ascend_config(vllm_config)


@pytest.mark.parametrize("buffer_device", ["cpu", "npu:abc"])
def test_resolve_ascend_config_rejects_non_npu_buffer_devices(buffer_device):
    vllm_config = MagicMock()

    vllm_config.ec_transfer_config.ec_connector_extra_config = {"mooncake_protocol": "ascend"}
    upstream_config = MagicMock(
        protocol="ascend",
        buffer_device=buffer_device,
    )

    with (
        patch.object(
            worker_module.MooncakeECConfig,
            "from_vllm_config",
            return_value=upstream_config,
        ),
        pytest.raises(ValueError, match="ec_buffer_device='npu'"),
    ):
        _resolve_ascend_config(vllm_config)


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


@pytest.mark.parametrize("cross_encoder_cache", [False, True])
def test_worker_initializes_ascend_data_plane_without_factory_hooks(
    cross_encoder_cache: bool,
):
    vllm_config = MagicMock()
    vllm_config.ec_transfer_config.is_ec_producer = True
    vllm_config.ec_transfer_config.ec_connector_extra_config = {}
    config = MagicMock(
        protocol="ascend",
        buffer_device="npu",
        pool_size=1024,
        cross_encoder_cache=cross_encoder_cache,
    )
    transfer = MagicMock()
    consumer = MagicMock()
    producer = MagicMock()
    reservations = MagicMock()
    control_client = MagicMock()
    push_manager = MagicMock()

    with (
        patch.dict(
            worker_module.os.environ,
            {
                "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
                "ASCEND_GLOBAL_RESOURCE_CONFIG": "{}",
            },
            clear=True,
        ),
        patch.object(worker_module, "_resolve_ascend_config", return_value=config),
        patch.object(worker_module, "_resolve_bounce_arena_size", return_value=4096),
        patch.object(
            worker_module,
            "configure_non_fabric_transport",
        ) as configure_transport,
        patch.object(worker_module, "ensure_mooncake_available") as ensure_available,
        patch.object(worker_module, "get_ip", return_value="127.0.0.1"),
        patch.object(worker_module, "build_embedding_namespace", return_value="namespace"),
        patch.object(worker_module.torch.npu, "current_device", return_value=3),
        patch.object(worker_module, "AscendMooncakeTransfer", return_value=transfer) as transfer_class,
        patch.object(worker_module, "AscendConsumerMemoryPool", return_value=consumer) as consumer_class,
        patch.object(worker_module, "AscendProducerMemoryPool", return_value=producer) as producer_class,
        patch.object(worker_module, "ConsumerReservationManager", return_value=reservations) as reservations_class,
        patch.object(worker_module, "ControlClient", return_value=control_client),
        patch.object(worker_module, "ProducerPushManager", return_value=push_manager),
        patch.object(worker_module, "ThreadPoolExecutor"),
        patch.object(worker_module.threading.Thread, "start"),
    ):
        lifecycle = MagicMock()
        lifecycle.attach_mock(configure_transport, "configure_transport")
        lifecycle.attach_mock(ensure_available, "ensure_mooncake")
        lifecycle.attach_mock(transfer_class, "create_transfer")
        worker = AscendECMooncakeWorker(vllm_config)

    assert lifecycle.mock_calls[:3] == [
        call.configure_transport(),
        call.ensure_mooncake(),
        call.create_transfer("127.0.0.1", 3),
    ]
    ensure_available.assert_called_once_with()
    transfer_class.assert_called_once_with("127.0.0.1", 3)
    consumer_class.assert_called_once()
    producer_class.assert_called_once()
    reservations_class.assert_called_once_with(
        consumer,
        worker_module._RESERVATION_TTL_SECONDS,
        worker_module._MAX_CANCELLED_TRANSFER_IDS,
    )
    assert worker._buffer_device == "npu"
    assert worker._transfer is transfer
    assert worker._consumer_memory is consumer
    assert worker._producer_memory is producer
    assert worker._reservations is reservations
    assert (worker._store_config is not None) == cross_encoder_cache
    assert worker._output_store is None


@pytest.mark.parametrize(
    ("max_num_seqs", "max_num_active_seqs", "expected_mib"),
    [
        (5, None, 10),
        (256, None, 256),
        (128, 16, 32),
        (128, 128, 256),
        (72, 36, 72),
        (5, 16, 10),
    ],
)
def test_resolve_default_bounce_arena_size(
    max_num_seqs: int,
    max_num_active_seqs: int | None,
    expected_mib: int,
):
    config = _make_bounce_config(
        max_num_seqs=max_num_seqs,
        max_num_active_seqs=max_num_active_seqs,
    )

    result = _resolve_bounce_arena_size(config)

    assert result == expected_mib * 1024 * 1024


def test_resolve_default_bounce_arena_size_without_active_field():
    config = _make_bounce_config(max_num_seqs=5)
    del config.scheduler_config.max_num_active_seqs

    assert _resolve_bounce_arena_size(config) == 10 * 1024 * 1024


@pytest.mark.parametrize(
    ("configured_size", "expected_size"),
    [(2 * 1024 * 1024 + 1, 4 * 1024 * 1024), (0, 0)],
    ids=("align-up", "disabled"),
)
def test_resolve_explicit_bounce_arena_size(configured_size, expected_size):
    config = _make_bounce_config(
        max_num_active_seqs=1,
        extra_config={
            "ascend_mooncake_bounce_arena_size": configured_size,
        },
    )

    assert _resolve_bounce_arena_size(config) == expected_size


@pytest.mark.parametrize(
    "value",
    [
        True,
        -1,
        2 * 1024 * 1024 - 1,
        2.0,
    ],
)
def test_resolve_explicit_bounce_arena_size_rejects_invalid_values(
    value: object,
):
    config = _make_bounce_config(
        extra_config={
            "ascend_mooncake_bounce_arena_size": value,
        },
    )

    with pytest.raises(
        ValueError,
        match="ascend_mooncake_bounce_arena_size",
    ):
        _resolve_bounce_arena_size(config)


@pytest.mark.parametrize("store_first", [False, True])
def test_ec_notification_does_not_wait_for_put_but_memory_does(store_first):
    worker = _make_lifecycle_worker()
    worker._deferred_store = MagicMock()
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._run_fanout = MagicMock(side_effect=lambda tasks, _: [task() for task in tasks])
    worker._use_store_independent_te = False
    push = _make_push("first")
    shard = _make_writable_shard("session", 1000)
    worker._producer_pushes.resolve_reservations.return_value = [shard]
    ticket = MagicMock(identifier=push.spec.mm_hash, tensor=push.source_tensor)
    worker._deferred_store.prepare_save.return_value = ticket
    pending = Future()
    worker._deferred_store.submit_prepared.return_value = pending
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
    prepared = worker._deferred_store.submit_prepared.call_args.args[0]
    assert prepared.addresses == [476, 500]
    assert prepared.owners == [staged]
    if not store_first:
        worker._producer_memory.release.assert_not_called()
        pending.set_result(None)
    worker._producer_memory.release.assert_called_once_with(staged)
    worker._transfer.acquire_registration_ranges.assert_not_called()


@pytest.mark.parametrize(
    "error", [None, worker_module.EmbeddingStoreOperationError("rejected"), RuntimeError("unknown")]
)
def test_all_put_readers_must_finish_before_release(error):
    worker = _make_lifecycle_worker()
    worker._deferred_store = MagicMock()
    a, b = Future(), Future()
    acquired = _AcquiredTransferWave([], store_futures=[a, b])
    staged = MagicMock()
    retained = len(worker_module._PROCESS_LIFETIME_EC_OWNERS)
    try:
        worker._release_after_store(acquired, staged)
        a.set_result(None)
        worker._producer_memory.release.assert_not_called()
        if error is None:
            b.set_result(None)
        else:
            b.set_exception(error)
        if type(error) is RuntimeError:
            worker._producer_memory.release.assert_not_called()
            worker._producer_memory.poison.assert_called_once()
        else:
            worker._producer_memory.release.assert_called_once_with(staged)
    finally:
        del worker_module._PROCESS_LIFETIME_EC_OWNERS[retained:]


def test_plain_p2p_uses_waiting_staging_and_never_dynamic_fallback():
    worker = _make_lifecycle_worker()
    worker._deferred_store = None
    push = _make_push("plain")
    ready = [(push, _make_writable_shard("session", 1000))]
    worker._resolve_writable_pushes = MagicMock(return_value=(ready, [push]))
    worker._push_staged_sources = MagicMock()
    worker._notify_completions = MagicMock()
    worker._push_batch([push])
    worker._producer_memory.stage_wait.assert_called_once_with([push.source_tensor])
    worker._producer_memory.stage.assert_not_called()
    worker._transfer.acquire_sources.assert_not_called()


@pytest.mark.parametrize("topology", ["shared", "independent", "fabric"])
def test_all_store_topologies_bind_fixed_pool_before_services(topology):
    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import store_setup

    worker = _make_lifecycle_worker()
    client = MagicMock()
    backend = MagicMock()
    sequence = MagicMock()
    sequence.attach_mock(worker._producer_memory.ensure_prepared, "prepare")
    sequence.attach_mock(client.bind_staging_buffer, "bind")
    retained = len(worker_module._PROCESS_LIFETIME_EC_OWNERS)
    with (
        patch.dict(worker_module.os.environ, {"ASCEND_ENABLE_USE_FABRIC_MEM": str(int(topology == "fabric"))}),
        patch.object(store_setup, "use_store_independent_te", return_value=topology == "independent"),
        patch.object(store_setup, "ensure_mooncake_store_imported"),
        patch.object(worker_module.torch, "device", return_value=SimpleNamespace(type="npu")),
        patch.object(store_setup, "create_ascend_mooncake_embedding_store_client", return_value=client) as factory,
        patch.object(worker_module, "AscendDeferredStoreBackend", return_value=backend),
        patch.object(worker_module.ECMooncakeWorker, "start_services") as start,
    ):
        sequence.attach_mock(factory, "factory")
        sequence.attach_mock(start, "services")
        try:
            worker.start_services()
            calls = [c[0] for c in sequence.mock_calls]
            assert calls == (
                ["factory", "prepare", "bind", "services"]
                if topology == "shared"
                else ["prepare", "factory", "bind", "services"]
            )
            factory.assert_called_once_with(worker._producer_memory, worker._transfer, read_buffer_bytes=256)
            assert worker._deferred_store is backend
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
    worker._deferred_store = backend
    worker._output_store = backend
    worker._retain_producer_slab = True
    put_started, put_gate, reclaim_started, reclaim_gate = Event(), Event(), Event(), Event()

    def put(*_):
        put_started.set()
        assert put_gate.wait(timeout=5)

    def reclaim(*_):
        reclaim_started.set()
        assert reclaim_gate.wait(timeout=5)

    client.put_prepared_tensor.side_effect = put
    worker._producer_memory.release = MagicMock(side_effect=reclaim)
    tensor = MagicMock()
    tensor.untyped_storage.return_value.nbytes.return_value = 128
    backend._step_candidates = {"first"}
    backend.save_output("first", tensor)
    ticket = backend.prepare_save("first")
    store_put = PreparedStorePut(ticket, [500], [10], [tensor])
    future = backend.submit_prepared(store_put)
    acquired = _AcquiredTransferWave([], [store_put], [future])
    retained = worker_module._PROCESS_LIFETIME_EC_OWNERS
    before = len(retained)
    try:
        assert put_started.wait(timeout=1)
        worker._release_after_store(acquired, MagicMock())
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
        assert any(owner is worker._producer_memory for _, owner in retained[before:])
    finally:
        put_gate.set()
        reclaim_gate.set()
        backend.shutdown()
        del retained[before:]


def test_unknown_ec_write_retains_staging_and_never_notifies_success():
    worker = _make_lifecycle_worker()
    worker._deferred_store = None
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
