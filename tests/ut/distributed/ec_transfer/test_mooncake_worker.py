from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    worker as worker_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    _BounceLease,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import (
    AscendECMooncakeWorker,
    _AcquiredTransferWave,
    _flatten_transfer_wave,
    _resolve_ascend_config,
    _resolve_bounce_arena_size,
    _TransferFragmentPlan,
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


def test_flatten_transfer_wave_allows_zero_bounce_wave():
    direct = MagicMock(prefix_nbytes=0, direct_address=100, direct_nbytes=10)
    wave = MagicMock()
    wave.sources = (MagicMock(source=direct, bounce_offset=None),)

    assert _flatten_transfer_wave(wave, None) == [
        _TransferFragmentPlan(0, 100, 0, 10),
    ]


def test_acquire_transfer_wave_bypasses_bounce_for_direct_sources():
    worker = object.__new__(AscendECMooncakeWorker)
    producer_memory = MagicMock()
    producer_memory.acquire_bounce.return_value = None
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

    producer_memory.acquire_bounce.assert_called_once_with(0)
    producer_memory.copy_to_bounce.assert_not_called()
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
    producer_memory.acquire_bounce.return_value = lease
    producer_memory.copy_to_bounce.return_value = 1000
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

    producer_memory.acquire_bounce.assert_called_once_with(3)
    producer_memory.copy_to_bounce.assert_called_once_with(
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


def test_acquire_transfer_wave_releases_bounce_on_registration_failure():
    worker = object.__new__(AscendECMooncakeWorker)
    producer_memory = MagicMock()
    lease = _BounceLease(offset=256, nbytes=3, allocated_nbytes=256)
    producer_memory.acquire_bounce.return_value = lease
    producer_memory.copy_to_bounce.return_value = 1000
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
    producer_memory.release_bounce.assert_called_once_with(lease)


def test_release_transfer_wave_releases_bounce_when_registration_release_raises():
    worker = object.__new__(AscendECMooncakeWorker)
    producer_memory = MagicMock()
    transfer = MagicMock()
    transfer.release_registration_ranges.side_effect = RuntimeError("release failed")
    worker._producer_memory = producer_memory
    worker._transfer = transfer
    lease = _BounceLease(offset=256, nbytes=3, allocated_nbytes=256)
    acquired = _AcquiredTransferWave([], [128], lease)

    with pytest.raises(RuntimeError, match="release failed"):
        worker._release_transfer_wave(acquired)

    transfer.release_registration_ranges.assert_called_once_with([128])
    producer_memory.release_bounce.assert_called_once_with(lease)


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
    worker._producer_pushes.begin_notifying.assert_called_once_with([push])
    worker._notify_completions.assert_called_once_with([(push, shard)])
    worker._producer_pushes.complete.assert_called_once_with([push])


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
        call(second_acquired),
    ]
    worker._producer_pushes.begin_notifying.assert_not_called()
    worker._notify_completions.assert_not_called()
    worker._producer_pushes.complete.assert_not_called()
    worker._producer_pushes.settle_all.assert_called_once_with([first, second])
    worker._abandon_pushes.assert_called_once_with([first, second])
    failure = worker._producer_pushes.fail.call_args.args[1]
    assert isinstance(failure, RuntimeError)
    assert str(failure) == "write failed"


def test_push_batch_waits_for_delayed_session_before_releasing_failed_wave():
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

    worker._release_transfer_wave.assert_called_once_with(acquired)
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


def test_worker_initializes_ascend_data_plane_without_factory_hooks():
    vllm_config = MagicMock()
    vllm_config.ec_transfer_config.is_ec_producer = True
    vllm_config.ec_transfer_config.ec_connector_extra_config = {}
    config = MagicMock(
        protocol="ascend",
        buffer_device="npu",
        pool_size=1024,
    )
    transfer = MagicMock()
    consumer = MagicMock()
    producer = MagicMock()
    reservations = MagicMock()
    control_client = MagicMock()
    push_manager = MagicMock()

    with (
        patch.object(worker_module, "_resolve_ascend_config", return_value=config),
        patch.object(worker_module, "_resolve_bounce_arena_size", return_value=4096),
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
        worker = AscendECMooncakeWorker(vllm_config)

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
