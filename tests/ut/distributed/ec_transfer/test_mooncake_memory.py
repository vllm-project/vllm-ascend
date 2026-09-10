from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import (
    memory as memory_module,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    AscendConsumerMemoryPool,
    AscendContiguousAllocator,
    AscendProducerAllocator,
    AscendProducerMemoryPool,
)

_MIB = 1024 * 1024


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


def test_producer_allocator_layout():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )

    assert allocator.staging_capacity == 3 * _MIB
    assert allocator.bounce_offset == 4 * _MIB
    assert allocator.padding == 1 * _MIB
    assert allocator.bounce_capacity == 2 * _MIB
    assert allocator.registered_capacity == 6 * _MIB
    assert allocator.raw_allocation_size == 8 * _MIB - 1
    assert allocator.bounce_tensor is None


def test_producer_allocator_prepares_partitioned_slab():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.return_value = 0

    allocator.prepare(torch.device("cpu"), transfer)

    tensor = allocator.tensor
    bounce = allocator.bounce_tensor
    assert tensor is not None
    assert bounce is not None

    assert tensor.nbytes == 6 * _MIB
    assert allocator._free == [(0, 3 * _MIB)]

    assert bounce.data_ptr() == tensor.data_ptr() + 4 * _MIB
    assert bounce.nbytes == 2 * _MIB

    transfer.register_memory.assert_called_once_with(tensor)


def test_producer_allocator_does_not_expose_bounce_to_staging():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.return_value = 0
    allocator.prepare(torch.device("cpu"), transfer)

    assert allocator.allocate(3 * _MIB) == (0, 3 * _MIB)
    assert allocator.allocate(1) is None


def test_producer_allocator_prepare_is_idempotent():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.return_value = 0

    allocator.prepare(torch.device("cpu"), transfer)
    region = allocator.allocate(256)
    free_after_allocate = list(allocator._free)

    allocator.prepare(torch.device("cpu"), transfer)

    assert region == (0, 256)
    assert allocator._free == free_after_allocate
    transfer.register_memory.assert_called_once()


def test_producer_allocator_registration_failure_is_fatal():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.return_value = 7

    with pytest.raises(
        RuntimeError,
        match=r"staging=.*padding=.*bounce=.*registered=.*allocation=",
    ):
        allocator.prepare(torch.device("cpu"), transfer)

    assert allocator.tensor is None
