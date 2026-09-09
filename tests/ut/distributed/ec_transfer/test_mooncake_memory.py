from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    memory as memory_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    AscendConsumerMemoryPool,
    AscendContiguousAllocator,
    AscendProducerMemoryPool,
)


def test_allocate_tensor_returns_2_mib_aligned_tensor():
    alignment = 2 * 1024 * 1024
    capacity = 4096
    device = torch.device("cpu")

    backing = torch.empty(capacity + alignment, dtype=torch.uint8, device=device)
    raw_tensor = backing.narrow(
        0,
        1,
        capacity + alignment - 1,
    )
    assert raw_tensor.data_ptr() % alignment != 0

    allocator = AscendContiguousAllocator(capacity)

    with patch.object(memory_module.torch, "empty", return_value=raw_tensor) as empty:
        tensor = allocator._allocate_tensor(device)

    expected_offset = (-raw_tensor.data_ptr()) % alignment

    empty.assert_called_once_with(
        capacity + alignment - 1,
        dtype=torch.uint8,
        device=device,
    )
    assert tensor.data_ptr() == raw_tensor.data_ptr() + expected_offset
    assert tensor.nbytes == capacity
    assert tensor.data_ptr() % alignment == 0
    assert tensor.dtype == torch.uint8


def test_consumer_records_npu_release_event():
    tensor = MagicMock()
    tensor.device.type = "npu"

    allocator = MagicMock()
    allocator.tensor = tensor

    pool = object.__new__(AscendConsumerMemoryPool)
    pool._allocator = allocator

    event = MagicMock()
    stream = MagicMock()

    with (
        patch.object(memory_module.torch.npu, "Event", return_value=event) as event_class,
        patch.object(memory_module.torch.npu, "current_stream", return_value=stream) as current_stream,
    ):
        result = pool._record_release_event()

    event_class.assert_called_once_with()
    current_stream.assert_called_once_with(tensor.device)
    event.record.assert_called_once_with(stream)
    assert result is event


def test_producer_copies_to_staging_on_npu_stream():
    pool = MagicMock()
    pool.device.type = "npu"

    source = MagicMock()
    destination = MagicMock()

    producer = object.__new__(AscendProducerMemoryPool)
    producer._local = SimpleNamespace(stream=None)

    stream = MagicMock()
    stream_context = MagicMock()

    with (
        patch.object(memory_module.torch.npu, "Stream", return_value=stream) as stream_class,
        patch.object(memory_module.torch.npu, "stream", return_value=stream_context) as use_stream,
    ):
        result = producer._copy_to_staging(pool, [destination], [source])

    stream_class.assert_called_once_with(device=pool.device)
    use_stream.assert_called_once_with(stream)
    destination.copy_.assert_called_once_with(
        source,
        non_blocking=True,
    )
    stream.synchronize.assert_called_once_with()
    assert producer._local.stream is stream
    assert result is None
