from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    worker as worker_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    AscendConsumerMemoryPool,
    AscendContiguousAllocator,
    AscendProducerAllocator,
    AscendProducerMemoryPool,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import (
    AscendECMooncakeWorker,
    _resolve_bounce_arena_size,
)


def _make_bounce_config(
    *,
    max_num_seqs: int = 128,
    extra_config: dict[str, object] | None = None,
) -> MagicMock:
    config = MagicMock()
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.ec_transfer_config.ec_connector_extra_config = (
        {} if extra_config is None else extra_config
    )
    return config


def test_make_config_maps_upstream_defaults_to_ascend():
    worker = object.__new__(AscendECMooncakeWorker)
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

    config = worker._make_config(vllm_config)

    assert config.protocol == "ascend"
    assert config.buffer_device == "npu"


def test_make_config_rejects_non_ascend_protocols():
    worker = object.__new__(AscendECMooncakeWorker)
    vllm_config = MagicMock()

    vllm_config.ec_transfer_config.ec_connector_extra_config = {"mooncake_protocol": "rdma"}
    upstream_config = MagicMock(
        protocol="rdma",
        buffer_device="cuda",
    )

    with (
        patch.object(
            worker_module.ECMooncakeWorker,
            "_make_config",
            return_value=upstream_config,
        ),
        pytest.raises(ValueError, match="mooncake_protocol='ascend'"),
    ):
        worker._make_config(vllm_config)


@pytest.mark.parametrize("buffer_device", ["cpu", "npu:abc"])
def test_make_config_rejects_non_npu_buffer_devices(buffer_device):
    worker = object.__new__(AscendECMooncakeWorker)
    vllm_config = MagicMock()

    vllm_config.ec_transfer_config.ec_connector_extra_config = {"mooncake_protocol": "ascend"}
    upstream_config = MagicMock(
        protocol="ascend",
        buffer_device=buffer_device,
    )

    with (
        patch.object(
            worker_module.ECMooncakeWorker,
            "_make_config",
            return_value=upstream_config,
        ),
        pytest.raises(ValueError, match="ec_buffer_device='npu'"),
    ):
        worker._make_config(vllm_config)


def test_record_source_ready_event_uses_npu_event():
    worker = object.__new__(AscendECMooncakeWorker)
    tensor = MagicMock()
    tensor.device.type = "npu"
    stream = MagicMock()
    event = MagicMock()

    with (
        patch.object(worker_module.torch.npu, "current_stream", return_value=stream) as current_stream,
        patch.object(worker_module.torch.npu, "Event", return_value=event) as event_class,
    ):
        result = worker._record_source_ready_event(tensor)

    assert result is event
    event_class.assert_called_once_with()
    current_stream.assert_called_once_with(tensor.device)
    event.record.assert_called_once_with(stream)


def test_make_memory_pools_use_ascend_allocator():
    worker = object.__new__(AscendECMooncakeWorker)
    worker._bounce_arena_size = 2 * 1024 * 1024
    transfer = MagicMock()
    capacity = 1024

    consumer = worker._make_consumer_memory(capacity, transfer)
    producer = worker._make_producer_memory(capacity, transfer)

    assert isinstance(consumer, AscendConsumerMemoryPool)
    assert isinstance(producer, AscendProducerMemoryPool)
    assert isinstance(consumer._allocator, AscendContiguousAllocator)
    assert isinstance(producer._allocator, AscendProducerAllocator)
    assert producer._allocator.staging_capacity == capacity
    assert producer._allocator.bounce_capacity == worker._bounce_arena_size
    assert consumer._allocator is not producer._allocator


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


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
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
