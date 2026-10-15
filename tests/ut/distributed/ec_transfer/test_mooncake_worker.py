from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock, call, patch

import pytest
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding import (
    backend as store_backend_module,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    store_client as store_client_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    worker as worker_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    _BounceLease,
    _flatten_transfer_wave,
    _resolve_bounce_arena_size,
    _TransferFragmentPlan,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_backend import (
    PreparedStorePut,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import (
    AscendECMooncakeWorker,
    _AcquiredTransferWave,
    _resolve_ascend_config,
)


def _make_bounce_config(
    *,
    max_num_seqs: int = 128,
    extra_config: dict[str, object] | None = None,
) -> MagicMock:
    config = MagicMock()
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.ec_transfer_config.ec_connector_extra_config = {} if extra_config is None else extra_config
    return config


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


def test_start_services_prepares_ascend_store_before_upstream_startup():
    worker = _make_lifecycle_worker()
    store_client = MagicMock()
    output_store = MagicMock()
    device = MagicMock()

    def start_upstream():
        assert worker._output_store is output_store

    with (
        patch.dict(
            worker_module.os.environ,
            {"ASCEND_ENABLE_USE_FABRIC_MEM": "1"},
        ),
        patch.object(worker_module.torch, "device", return_value=device),
        patch.object(store_client_module, "ensure_mooncake_store_imported") as import_store,
        patch.object(
            store_client_module,
            "create_ascend_mooncake_embedding_store_client",
            return_value=store_client,
        ) as create_store_client,
        patch.object(
            store_backend_module,
            "MooncakeEmbeddingStoreBackend",
            return_value=output_store,
        ) as create_backend,
        patch.object(
            worker_module.ECMooncakeWorker,
            "start_services",
            side_effect=start_upstream,
        ) as start_upstream_mock,
    ):
        lifecycle = MagicMock()
        lifecycle.attach_mock(import_store, "import_store")
        lifecycle.attach_mock(worker._producer_memory.ensure_prepared, "prepare")
        lifecycle.attach_mock(create_store_client, "client")
        lifecycle.attach_mock(create_backend, "backend")
        lifecycle.attach_mock(start_upstream_mock, "upstream")
        worker.start_services()
        worker.start_services()

    assert lifecycle.mock_calls == [
        call.import_store(),
        call.prepare(device),
        call.client(
            worker._producer_memory.bounce_arena,
            worker._transfer,
            read_buffer_bytes=256,
        ),
        call.backend(
            store_client,
            "namespace",
            max_pending_items=4,
            max_pending_bytes=1024,
        ),
        call.upstream(),
        call.upstream(),
    ]
    assert worker._output_store is output_store
    worker._producer_memory.close.assert_not_called()


def test_start_services_shared_roce_sets_up_store_before_registering_staging():
    worker = _make_lifecycle_worker()
    store_client = MagicMock()
    output_store = MagicMock()
    device = MagicMock()

    with (
        patch.dict(
            worker_module.os.environ,
            {"ASCEND_ENABLE_USE_FABRIC_MEM": "0"},
        ),
        patch.object(worker_module.torch, "device", return_value=device),
        patch.object(store_client_module, "ensure_mooncake_store_imported") as import_store,
        patch.object(
            store_client_module,
            "create_ascend_mooncake_embedding_store_client",
            return_value=store_client,
        ) as create_store_client,
        patch.object(
            worker_module,
            "AscendDeferredStoreBackend",
            return_value=output_store,
        ) as create_backend,
        patch.object(worker_module.ECMooncakeWorker, "start_services") as start_upstream,
    ):
        lifecycle = MagicMock()
        lifecycle.attach_mock(import_store, "import_store")
        lifecycle.attach_mock(create_store_client, "client")
        lifecycle.attach_mock(worker._producer_memory.ensure_prepared, "prepare")
        lifecycle.attach_mock(create_backend, "backend")
        lifecycle.attach_mock(start_upstream, "upstream")
        worker.start_services()

    assert lifecycle.mock_calls == [
        call.import_store(),
        call.client(
            worker._producer_memory.bounce_arena,
            worker._transfer,
            read_buffer_bytes=256,
        ),
        call.prepare(device),
        call.backend(
            store_client,
            "namespace",
            max_pending_items=4,
            max_pending_bytes=1024,
        ),
        call.upstream(),
    ]


@pytest.mark.parametrize(
    ("failure_stage", "expected_cleanup"),
    [
        ("factory", [call.producer()]),
        ("backend", [call.client(), call.producer()]),
        ("upstream", [call.store(), call.producer()]),
    ],
)
def test_start_services_cleans_up_after_failure(failure_stage, expected_cleanup):
    worker = _make_lifecycle_worker()
    store_client = MagicMock()
    output_store = MagicMock()
    error = RuntimeError(f"{failure_stage} failed")
    cleanup = MagicMock()
    cleanup.attach_mock(store_client.close, "client")
    cleanup.attach_mock(output_store.shutdown, "store")
    cleanup.attach_mock(worker._producer_memory.close, "producer")

    with (
        patch.dict(
            worker_module.os.environ,
            {"ASCEND_ENABLE_USE_FABRIC_MEM": "1"},
        ),
        patch.object(worker_module.torch, "device"),
        patch.object(
            store_client_module,
            "create_ascend_mooncake_embedding_store_client",
            return_value=store_client,
            side_effect=error if failure_stage == "factory" else None,
        ) as create_store_client,
        patch.object(
            store_backend_module,
            "MooncakeEmbeddingStoreBackend",
            return_value=output_store,
            side_effect=error if failure_stage == "backend" else None,
        ) as create_backend,
        patch.object(
            worker_module.ECMooncakeWorker,
            "start_services",
            side_effect=error if failure_stage == "upstream" else None,
        ),
        pytest.raises(RuntimeError, match=f"{failure_stage} failed"),
    ):
        worker.start_services()

    assert cleanup.mock_calls == expected_cleanup
    if failure_stage == "factory":
        create_backend.assert_not_called()
    else:
        create_store_client.assert_called_once()
    assert worker._output_store is None


def test_close_shuts_down_store_before_upstream_resources():
    worker = _make_lifecycle_worker()
    output_store = MagicMock()
    worker._output_store = output_store
    events = []
    output_store.shutdown.side_effect = lambda: events.append("store")

    worker._producer_memory.close.side_effect = lambda: events.append("producer")
    worker.close()

    assert events == ["store", "producer"]
    worker._io_executor.shutdown.assert_called_once_with(wait=True)
    worker._transfer.close.assert_called_once_with()


def test_close_preserves_shared_memory_when_store_shutdown_fails():
    worker = _make_lifecycle_worker()
    output_store = MagicMock()
    output_store.shutdown.side_effect = RuntimeError("shutdown failed")
    worker._output_store = output_store

    with pytest.raises(RuntimeError, match="shutdown failed"):
        worker.close()

    worker._producer_memory.close.assert_not_called()


def test_close_keeps_dual_registered_staging_until_process_exit():
    worker = _make_lifecycle_worker()
    worker._retain_producer_slab = True
    worker._output_store = MagicMock()
    retained = worker_module._PROCESS_LIFETIME_EC_OWNERS
    before = len(retained)

    try:
        worker.close()

        assert worker._output_store is None
        assert retained[before:] == [
            (worker._transfer, worker._producer_memory)
        ]
        worker._producer_memory.close.assert_not_called()
        worker._transfer.close.assert_not_called()
    finally:
        del retained[before:]


def test_start_services_global_config_dual_registers_long_lived_staging():
    worker = _make_lifecycle_worker()
    store_client = MagicMock()
    output_store = MagicMock()
    device = MagicMock()
    staging = worker._producer_memory.tensor

    with (
        patch.dict(
            worker_module.os.environ,
            {
                "ASCEND_ENABLE_USE_FABRIC_MEM": "0",
                "ASCEND_GLOBAL_RESOURCE_CONFIG": "{}",
            },
            clear=True,
        ),
        patch.object(worker_module.torch, "device", return_value=device),
        patch.object(store_client_module, "ensure_mooncake_store_imported") as import_store,
        patch.object(
            store_client_module,
            "create_ascend_mooncake_embedding_store_client",
            return_value=store_client,
        ) as create_store_client,
        patch.object(
            worker_module,
            "AscendDeferredStoreBackend",
            return_value=output_store,
        ) as create_backend,
        patch.object(worker_module.ECMooncakeWorker, "start_services") as start_upstream,
    ):
        lifecycle = MagicMock()
        lifecycle.attach_mock(import_store, "import_store")
        lifecycle.attach_mock(create_store_client, "client")
        lifecycle.attach_mock(worker._producer_memory.ensure_prepared, "prepare")
        lifecycle.attach_mock(store_client.register_staging_buffer, "register")
        lifecycle.attach_mock(create_backend, "backend")
        lifecycle.attach_mock(start_upstream, "upstream")
        worker.start_services()

    assert lifecycle.mock_calls == [
        call.import_store(),
        call.client(
            worker._producer_memory.bounce_arena,
            worker._transfer,
            read_buffer_bytes=256,
        ),
        call.prepare(device),
        call.register(staging),
        call.backend(
            store_client,
            "namespace",
            max_pending_items=4,
            max_pending_bytes=1024,
        ),
        call.upstream(),
    ]
    assert worker._use_store_independent_te is True
    assert worker._retain_producer_slab is True
    worker._transfer.close.assert_not_called()
    assert worker._output_store is output_store
    assert worker._shutdown is False


def test_flatten_transfer_wave_preserves_unified_fragment_order():
    direct = MagicMock(prefix_nbytes=0, direct_address=100, direct_nbytes=10)
    mixed = MagicMock(prefix_nbytes=3, direct_address=200, direct_nbytes=7)
    bounced = MagicMock(prefix_nbytes=5, direct_address=None, direct_nbytes=0)
    wave = MagicMock()
    wave.sources = (
        MagicMock(source=direct, bounce_offset=None),
        MagicMock(source=mixed, bounce_offset=0),
        MagicMock(source=bounced, bounce_offset=3),
    )

    assert _flatten_transfer_wave(wave, 1000) == [
        _TransferFragmentPlan(0, 100, 0, 10),
        _TransferFragmentPlan(1, 1000, 0, 3),
        _TransferFragmentPlan(1, 200, 3, 7),
        _TransferFragmentPlan(2, 1003, 0, 5),
    ]


def test_acquire_transfer_wave_bypasses_bounce_for_direct_sources():
    worker = object.__new__(AscendECMooncakeWorker)
    producer_memory = MagicMock()
    producer_memory.bounce_arena.acquire.return_value = None
    transfer = MagicMock()
    transfer.acquire_registration_ranges.return_value = [64]
    worker._producer_memory = producer_memory
    worker._transfer = transfer

    source = MagicMock(prefix_nbytes=0, direct_address=100, direct_nbytes=10)
    registration_ranges = (MagicMock(),)
    wave = MagicMock(
        bounce_nbytes=0,
        sources=(MagicMock(source=source, bounce_offset=None),),
        registration_ranges=registration_ranges,
    )

    acquired = worker._acquire_transfer_wave(wave)

    producer_memory.bounce_arena.acquire.assert_called_once_with(0)
    producer_memory.bounce_arena.copy.assert_not_called()
    transfer.acquire_registration_ranges.assert_called_once_with(registration_ranges)
    assert acquired == _AcquiredTransferWave(
        fragments=[_TransferFragmentPlan(0, 100, 0, 10)],
        registration_addresses=[64],
        bounce_lease=None,
    )


def test_acquire_transfer_wave_returns_fragments_and_resources():
    worker = object.__new__(AscendECMooncakeWorker)
    producer_memory = MagicMock()
    lease = _BounceLease(offset=256, nbytes=3, allocated_nbytes=256)
    producer_memory.bounce_arena.acquire.return_value = lease
    producer_memory.bounce_arena.copy.return_value = 1000
    transfer = MagicMock()
    transfer.acquire_registration_ranges.return_value = [128]
    worker._producer_memory = producer_memory
    worker._transfer = transfer

    owner = MagicMock()
    source = MagicMock(
        owner=owner,
        prefix_nbytes=3,
        direct_address=200,
        direct_nbytes=7,
    )
    registration_ranges = (MagicMock(),)
    wave = MagicMock(
        bounce_nbytes=3,
        sources=(MagicMock(source=source, bounce_offset=0),),
        registration_ranges=registration_ranges,
    )

    acquired = worker._acquire_transfer_wave(wave)

    producer_memory.bounce_arena.acquire.assert_called_once_with(3)
    producer_memory.bounce_arena.copy.assert_called_once_with(
        lease,
        [(owner, 0, 3)],
    )
    transfer.acquire_registration_ranges.assert_called_once_with(registration_ranges)
    assert acquired == _AcquiredTransferWave(
        fragments=[
            _TransferFragmentPlan(0, 1000, 0, 3),
            _TransferFragmentPlan(0, 200, 3, 7),
        ],
        registration_addresses=[128],
        bounce_lease=lease,
    )


def test_joint_wave_copies_prefix_once_and_reuses_registration_and_addresses():
    worker = object.__new__(AscendECMooncakeWorker)
    producer_memory = MagicMock()
    lease = _BounceLease(offset=256, nbytes=56, allocated_nbytes=256)
    producer_memory.bounce_arena.acquire.return_value = lease
    producer_memory.bounce_arena.copy.return_value = 1000
    transfer = MagicMock()
    transfer.acquire_registration_ranges.return_value = [128]
    worker._producer_memory = producer_memory
    worker._transfer = transfer
    header = MagicMock()
    deferred = MagicMock()
    deferred.store_client.prepare_put_header.return_value = header
    worker._deferred_store = deferred
    tensor = MagicMock()
    ticket = MagicMock(tensor=tensor)
    source = MagicMock(
        owner=tensor,
        prefix_nbytes=32,
        direct_address=200,
        direct_nbytes=96,
    )
    registration_ranges = (MagicMock(),)
    wave = MagicMock(
        bounce_nbytes=56,
        sources=(
            MagicMock(
                source=source,
                store_header_offset=0,
                bounce_offset=24,
            ),
        ),
        registration_ranges=registration_ranges,
    )

    acquired = worker._acquire_transfer_wave(wave, [ticket])

    producer_memory.bounce_arena.copy.assert_called_once_with(
        lease,
        [(header, 0, 24), (tensor, 24, 32)],
    )
    transfer.acquire_registration_ranges.assert_called_once_with(registration_ranges)
    assert acquired.fragments == [
        _TransferFragmentPlan(0, 1024, 0, 32),
        _TransferFragmentPlan(0, 200, 32, 96),
    ]
    assert acquired.store_puts[0].addresses == [1000, 1024, 200]
    assert acquired.store_puts[0].sizes == [24, 32, 96]


def test_acquire_transfer_wave_releases_bounce_on_registration_failure():
    worker = object.__new__(AscendECMooncakeWorker)
    producer_memory = MagicMock()
    lease = _BounceLease(offset=256, nbytes=3, allocated_nbytes=256)
    producer_memory.bounce_arena.acquire.return_value = lease
    producer_memory.bounce_arena.copy.return_value = 1000
    transfer = MagicMock()
    transfer.acquire_registration_ranges.side_effect = RuntimeError("registration failed")
    worker._producer_memory = producer_memory
    worker._transfer = transfer

    source = MagicMock(
        owner=MagicMock(),
        prefix_nbytes=3,
        direct_address=None,
        direct_nbytes=0,
    )
    wave = MagicMock(
        bounce_nbytes=3,
        sources=(MagicMock(source=source, bounce_offset=0),),
        registration_ranges=(MagicMock(),),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        worker._acquire_transfer_wave(wave)

    transfer.release_registration_ranges.assert_not_called()
    producer_memory.bounce_arena.release.assert_called_once_with(lease)


def test_release_transfer_wave_retains_bounce_when_registration_release_raises():
    worker = object.__new__(AscendECMooncakeWorker)
    producer_memory = MagicMock()
    transfer = MagicMock()
    transfer.release_registration_ranges.side_effect = RuntimeError("release failed")
    worker._producer_memory = producer_memory
    worker._transfer = transfer
    lease = _BounceLease(offset=256, nbytes=3, allocated_nbytes=256)
    acquired = _AcquiredTransferWave([], [128], lease)

    with pytest.raises(RuntimeError, match="release raised an exception"):
        worker._release_transfer_wave(acquired)

    transfer.release_registration_ranges.assert_called_once_with([128])
    producer_memory.bounce_arena.release.assert_not_called()


def test_release_transfer_wave_rejects_unconfirmed_unregistration():
    worker = object.__new__(AscendECMooncakeWorker)
    worker._producer_memory = MagicMock()
    worker._transfer = MagicMock()
    worker._transfer.release_registration_ranges.return_value = False
    acquired = _AcquiredTransferWave([], [128], None)

    with pytest.raises(
        worker_module._RegistrationReleaseError,
        match="could not be confirmed",
    ):
        worker._release_transfer_wave(acquired)

    worker._producer_memory.bounce_arena.release.assert_not_called()


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
        registration_addresses=[],
        bounce_lease=None,
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


def test_joint_wave_settles_store_future_after_ec_failure_and_poisons_owners():
    worker = object.__new__(AscendECMooncakeWorker)
    worker._transfer = MagicMock()
    worker._producer_pushes = MagicMock()
    worker._run_fanout = MagicMock(side_effect=RuntimeError("EC failed"))
    deferred = MagicMock()
    store_future: Future[None] = Future()
    store_future.set_result(None)
    deferred.submit_prepared.return_value = store_future
    worker._deferred_store = deferred
    push = _make_push("first")
    owner = MagicMock()
    ticket = MagicMock(tensor=push.source_tensor)
    acquired = _AcquiredTransferWave(
        fragments=[_TransferFragmentPlan(0, 100, 0, 10)],
        registration_addresses=[128],
        bounce_lease=MagicMock(),
        store_puts=[
            PreparedStorePut(
                ticket=ticket,
                addresses=[10, 20],
                sizes=[24, 10],
                owners=[owner],
            )
        ],
    )

    reusable = worker._write_transfer_wave(
        [push],
        [(push, _make_writable_shard("session-a", 1000))],
        acquired,
    )

    assert reusable is False
    deferred.submit_prepared.assert_called_once()
    deferred.store_client.retain_fallback_owners.assert_called_once()
    retained = deferred.store_client.retain_fallback_owners.call_args.args[0]
    assert owner in retained
    assert push.source_tensor in retained


def test_push_batch_preserves_staging_first_path():
    worker = object.__new__(AscendECMooncakeWorker)
    push = _make_push("first")
    shard = _make_writable_shard("session-a", 1000)
    staged_tensor = MagicMock()
    staged_tensor.data_ptr.return_value = 500
    staged = MagicMock(tensors=[staged_tensor])
    worker._producer_memory = MagicMock()
    worker._producer_memory.stage.return_value = staged
    worker._producer_pushes = MagicMock()
    worker._producer_pushes.resolve_reservations.return_value = [shard]
    worker._transfer = MagicMock()
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._run_fanout = MagicMock(side_effect=lambda tasks, _on_submit: [task() for task in tasks])
    worker._deferred_store = MagicMock()

    with patch.object(worker_module, "_plan_transfer_waves") as plan_waves:
        worker._push_batch([push])

    worker._producer_memory.stage.assert_called_once_with([push.source_tensor])
    plan_waves.assert_not_called()
    worker._transfer.write.assert_called_once_with(
        "session-a",
        [500],
        [1000],
        [10],
    )
    worker._producer_memory.release.assert_called_once_with(staged)
    worker._deferred_store.submit_original.assert_called_once_with("hash-first")
    worker._producer_pushes.begin_notifying.assert_called_once_with([push])
    worker._notify_completions.assert_called_once_with([(push, shard)])
    worker._producer_pushes.complete.assert_called_once_with([push])


def test_independent_store_uses_header_and_same_staged_payload():
    worker = object.__new__(AscendECMooncakeWorker)
    push = _make_push("first")
    staged_tensor = MagicMock(nbytes=10)
    staged_tensor.data_ptr.return_value = 500
    staged = MagicMock(tensors=[staged_tensor])
    ticket = MagicMock(identifier="hash-first", tensor=push.source_tensor)
    lease = MagicMock()
    bounce_arena = MagicMock()
    bounce_arena.acquire.return_value = lease
    bounce_arena.copy.return_value = 9000
    worker._producer_memory = MagicMock()
    worker._producer_memory.bounce_arena = bounce_arena
    worker._deferred_store = MagicMock()
    header = worker._deferred_store.store_client.prepare_put_header.return_value

    acquired = worker._acquire_staged_wave(
        [push],
        staged,
        {"hash-first": ticket},
    )

    bounce_arena.acquire.assert_called_once_with(24)
    bounce_arena.copy.assert_called_once_with(
        lease,
        [(header, 0, 24)],
    )
    assert len(acquired.store_puts) == 1
    assert acquired.fragments == [
        _TransferFragmentPlan(0, 500, 0, 10),
    ]
    store_put = acquired.store_puts[0]
    assert store_put.addresses == [9000, 500]
    assert store_put.sizes == [24, 10]
    assert staged_tensor in store_put.owners


def test_push_batch_independent_store_releases_joint_staging_after_completion():
    worker = object.__new__(AscendECMooncakeWorker)
    worker._use_store_independent_te = True
    push = _make_push("first")
    shard = _make_writable_shard("session-a", 1000)
    staged = MagicMock()
    acquired = _AcquiredTransferWave([], [], MagicMock(), [])
    worker._producer_memory = MagicMock()
    worker._producer_pushes = MagicMock()
    worker._producer_pushes.resolve_reservations.return_value = [shard]
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._deferred_store = MagicMock()
    worker._push_store_independent_staging = MagicMock(
        return_value=(staged, acquired, [])
    )
    worker._release_transfer_wave = MagicMock()
    lifecycle = MagicMock()
    lifecycle.attach_mock(worker._release_transfer_wave, "wave")
    lifecycle.attach_mock(worker._producer_memory.release, "payload")
    lifecycle.attach_mock(worker._producer_pushes.begin_notifying, "notify")

    worker._push_batch([push])

    assert lifecycle.mock_calls == [
        call.wave(acquired),
        call.payload(staged),
        call.notify([push]),
    ]
    worker._notify_completions.assert_called_once_with([(push, shard)])
    worker._producer_pushes.complete.assert_called_once_with([push])
    worker._deferred_store.submit_original.assert_not_called()


def test_independent_store_rejection_runs_ec_without_store_put():
    worker = object.__new__(AscendECMooncakeWorker)
    push = _make_push("first")
    ready = [(push, _make_writable_shard("session-a", 1000))]
    staged = MagicMock()
    worker._deferred_store = MagicMock()
    worker._deferred_store.prepare_fallback.return_value = None
    worker._producer_memory = MagicMock()
    worker._producer_memory.stage_wait.return_value = staged
    acquired = _AcquiredTransferWave([], [], None, [])
    worker._acquire_staged_wave = MagicMock(return_value=acquired)
    worker._write_transfer_wave = MagicMock(return_value=True)

    result_staged, result_acquired, prepared = (
        worker._push_store_independent_staging(
            [push],
            ready,
            [push],
        )
    )

    assert result_staged is staged
    assert result_acquired is acquired
    assert prepared == []
    worker._acquire_staged_wave.assert_called_once_with([push], staged, {})
    worker._write_transfer_wave.assert_called_once_with(
        [push],
        ready,
        acquired,
    )


def test_independent_store_poisons_staging_when_wave_is_unconfirmed():
    worker = object.__new__(AscendECMooncakeWorker)
    push = _make_push("first")
    staged = MagicMock()
    acquired = _AcquiredTransferWave([], [], None, [])
    worker._deferred_store = MagicMock()
    worker._deferred_store.prepare_fallback.return_value = None
    worker._producer_memory = MagicMock()
    worker._producer_memory.stage_wait.return_value = staged
    worker._acquire_staged_wave = MagicMock(return_value=acquired)
    worker._write_transfer_wave = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="completion is unconfirmed"):
        worker._push_store_independent_staging([push], [], [push])

    worker._producer_memory.poison.assert_called_once_with()
    worker._producer_memory.release.assert_not_called()


def test_push_batch_rejects_staging_failure_when_fallback_is_disabled():
    worker = object.__new__(AscendECMooncakeWorker)
    push = _make_push("first")
    shard = _make_writable_shard("session-a", 1000)
    worker._producer_memory = MagicMock()
    worker._producer_memory.stage.return_value = None
    worker._producer_pushes = MagicMock()
    worker._producer_pushes.resolve_reservations.return_value = [shard]
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._bounce_arena_size = 0

    with patch.object(worker_module, "_plan_transfer_waves") as plan_waves:
        worker._push_batch([push])

    plan_waves.assert_not_called()
    worker._producer_pushes.begin_notifying.assert_not_called()
    worker._notify_completions.assert_not_called()
    worker._producer_pushes.complete.assert_not_called()
    worker._producer_pushes.settle_all.assert_called_once_with([push])
    worker._abandon_pushes.assert_called_once_with([push])
    failure = worker._producer_pushes.fail.call_args.args[1]
    assert isinstance(failure, RuntimeError)
    assert "direct/bounce fallback is disabled" in str(failure)


def test_push_batch_runs_and_releases_fallback_waves_in_order():
    worker = object.__new__(AscendECMooncakeWorker)
    first = _make_push("first")
    second = _make_push("second")
    first_shard = _make_writable_shard("session-a", 1000)
    second_shard = _make_writable_shard("session-a", 2000)
    worker._producer_memory = MagicMock()
    worker._producer_memory.stage.return_value = None
    worker._producer_pushes = MagicMock()
    worker._producer_pushes.resolve_reservations.side_effect = [
        [first_shard],
        [second_shard],
    ]
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._bounce_arena_size = 4096
    first_wave = MagicMock(sources=(MagicMock(),))
    second_wave = MagicMock(sources=(MagicMock(),))
    first_acquired = MagicMock()
    second_acquired = MagicMock()
    worker._acquire_transfer_wave = MagicMock(side_effect=[first_acquired, second_acquired])
    worker._write_transfer_wave = MagicMock()
    worker._release_transfer_wave = MagicMock()
    ready = [(first, first_shard), (second, second_shard)]

    with patch.object(
        worker_module,
        "_plan_transfer_waves",
        return_value=[first_wave, second_wave],
    ) as plan_waves:
        worker._push_batch([first, second])

    plan_waves.assert_called_once_with(
        [first.source_tensor, second.source_tensor],
        4096,
    )
    assert worker._acquire_transfer_wave.call_args_list == [
        call(first_wave),
        call(second_wave),
    ]
    assert worker._write_transfer_wave.call_args_list == [
        call([first], ready, first_acquired),
        call([second], ready, second_acquired),
    ]
    assert worker._release_transfer_wave.call_args_list == [
        call(first_acquired),
        call(second_acquired),
    ]
    worker._producer_pushes.begin_notifying.assert_called_once_with([first, second])
    worker._notify_completions.assert_called_once_with(ready)
    worker._producer_pushes.complete.assert_called_once_with([first, second])


def test_push_batch_later_wave_failure_releases_without_notifying():
    worker = object.__new__(AscendECMooncakeWorker)
    first = _make_push("first")
    second = _make_push("second")
    first_shard = _make_writable_shard("session-a", 1000)
    second_shard = _make_writable_shard("session-a", 2000)
    worker._producer_memory = MagicMock()
    worker._producer_memory.stage.return_value = None
    worker._producer_pushes = MagicMock()
    worker._producer_pushes.resolve_reservations.side_effect = [
        [first_shard],
        [second_shard],
    ]
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._bounce_arena_size = 4096
    first_wave = MagicMock(sources=(MagicMock(),))
    second_wave = MagicMock(sources=(MagicMock(),))
    first_acquired = MagicMock()
    second_acquired = MagicMock()
    worker._acquire_transfer_wave = MagicMock(side_effect=[first_acquired, second_acquired])
    worker._write_transfer_wave = MagicMock(side_effect=[None, RuntimeError("write failed")])
    worker._release_transfer_wave = MagicMock()

    with patch.object(
        worker_module,
        "_plan_transfer_waves",
        return_value=[first_wave, second_wave],
    ):
        worker._push_batch([first, second])

    assert worker._release_transfer_wave.call_args_list == [
        call(first_acquired),
    ]
    worker._producer_pushes.begin_notifying.assert_not_called()
    worker._notify_completions.assert_not_called()
    worker._producer_pushes.complete.assert_not_called()
    worker._producer_pushes.settle_all.assert_called_once_with([first, second])
    worker._abandon_pushes.assert_called_once_with([first, second])
    failure = worker._producer_pushes.fail.call_args.args[1]
    assert isinstance(failure, RuntimeError)
    assert str(failure) == "write failed"


def test_push_batch_waits_for_delayed_session_and_retains_failed_wave():
    worker = object.__new__(AscendECMooncakeWorker)
    push = _make_push("first")
    first_shard = _make_writable_shard("session-a", 1000)
    delayed_shard = _make_writable_shard("session-b", 2000)
    worker._producer_memory = MagicMock()
    worker._producer_memory.stage.return_value = None
    worker._producer_pushes = MagicMock()
    worker._producer_pushes.resolve_reservations.return_value = [
        first_shard,
        delayed_shard,
    ]
    worker._validate_push_source = MagicMock()
    worker._notify_completions = MagicMock()
    worker._abandon_pushes = MagicMock()
    worker._bounce_arena_size = 4096
    wave = MagicMock(sources=(MagicMock(),))
    acquired = _AcquiredTransferWave(
        fragments=[_TransferFragmentPlan(0, 100, 0, 10)],
        registration_addresses=[128],
        bounce_lease=MagicMock(),
    )
    worker._acquire_transfer_wave = MagicMock(return_value=acquired)
    worker._release_transfer_wave = MagicMock()
    worker._transfer = MagicMock()
    failed = Event()
    delayed = Event()
    finish_delayed = Event()

    def write(session, _sources, _destinations, _lengths):
        if session == "session-a":
            failed.set()
            raise RuntimeError("write failed")
        delayed.set()
        assert finish_delayed.wait(timeout=5)

    worker._transfer.write.side_effect = write

    with (
        ThreadPoolExecutor(max_workers=1) as shard_executor,
        ThreadPoolExecutor(max_workers=1) as batch_executor,
        patch.object(
            worker_module,
            "_plan_transfer_waves",
            return_value=[wave],
        ),
        patch.object(
            worker,
            "_shard_executor",
            return_value=shard_executor,
        ),
    ):
        batch_future = batch_executor.submit(worker._push_batch, [push])
        try:
            assert failed.wait(timeout=1)
            assert delayed.wait(timeout=1)
            assert not batch_future.done()
            worker._release_transfer_wave.assert_not_called()
            worker._producer_pushes.begin_notifying.assert_not_called()
        finally:
            finish_delayed.set()
        batch_future.result(timeout=1)

    worker._release_transfer_wave.assert_not_called()
    worker._producer_pushes.begin_notifying.assert_not_called()
    worker._notify_completions.assert_not_called()
    worker._producer_pushes.fail.assert_called_once()


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
            "configure_non_fabric_roce",
        ) as configure_roce,
        patch.object(worker_module, "ensure_mooncake_available") as ensure_available,
        patch.object(worker_module, "get_ip", return_value="127.0.0.1"),
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
        lifecycle.attach_mock(configure_roce, "configure_roce")
        lifecycle.attach_mock(ensure_available, "ensure_mooncake")
        lifecycle.attach_mock(transfer_class, "create_transfer")
        worker = AscendECMooncakeWorker(vllm_config)

    assert lifecycle.mock_calls[:3] == [
        call.configure_roce(),
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
    assert worker._store_config is None
    assert worker._output_store is None


@pytest.mark.parametrize(
    ("max_num_seqs", "expected_mib"),
    [
        (5, 10),
        (128, 256),
        (256, 256),
        (1024, 256),
    ],
)
def test_resolve_default_bounce_arena_size(
    max_num_seqs: int,
    expected_mib: int,
):
    config = _make_bounce_config(
        max_num_seqs=max_num_seqs,
    )

    result = _resolve_bounce_arena_size(config)

    assert result == expected_mib * 1024 * 1024


def test_resolve_explicit_bounce_arena_size_align_up():
    mib = 1024 * 1024
    config = _make_bounce_config(
        extra_config={
            "ascend_mooncake_bounce_arena_size": 2 * mib + 1,
        },
    )

    assert _resolve_bounce_arena_size(config) == 4 * mib


def test_resolve_zero_bounce_arena_size_disables_fallback():
    config = _make_bounce_config(
        extra_config={
            "ascend_mooncake_bounce_arena_size": 0,
        },
    )

    assert _resolve_bounce_arena_size(config) == 0


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        -1,
        2 * 1024 * 1024 - 1,
        2.0,
        "2097152",
        None,
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
