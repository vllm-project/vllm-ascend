import time
from concurrent.futures import ThreadPoolExecutor
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
    _BounceLease,
    _BounceLeaseManager,
)

_MIB = 1024 * 1024


def _wait_for_waiters(manager: _BounceLeaseManager, count: int) -> None:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with manager._condition:
            if len(manager._waiters) == count:
                return
        time.sleep(0.001)
    raise AssertionError(f"expected {count} bounce waiters")


def test_prepare_registers_2_mib_aligned_tensor():
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
    transfer = MagicMock()
    transfer.register_memory.return_value = 0

    with patch.object(memory_module.torch, "empty", return_value=raw_tensor) as empty:
        allocator.prepare(device, transfer)

    expected_offset = (-raw_tensor.data_ptr()) % alignment
    tensor = allocator.tensor
    assert tensor is not None

    empty.assert_called_once_with(
        capacity + alignment - 1,
        dtype=torch.uint8,
        device=device,
    )
    assert tensor.data_ptr() == raw_tensor.data_ptr() + expected_offset
    assert tensor.nbytes == capacity
    assert tensor.data_ptr() % alignment == 0
    assert tensor.dtype == torch.uint8
    transfer.register_memory.assert_called_once_with(tensor)


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

    allocator = MagicMock()
    allocator.tensor = pool
    allocator.allocate.return_value = (0, 256)
    allocator.view.return_value = destination
    producer = object.__new__(AscendProducerMemoryPool)
    producer._allocator = allocator
    producer._transfer = MagicMock()
    producer._lock = memory_module.threading.Lock()
    producer._local = SimpleNamespace(stream=None)
    producer._free_regions = MagicMock()

    stream = MagicMock()
    stream_context = MagicMock()

    with (
        patch.object(memory_module.torch.npu, "Stream", return_value=stream) as stream_class,
        patch.object(memory_module.torch.npu, "stream", return_value=stream_context) as use_stream,
    ):
        result = producer.stage([source])

    stream_class.assert_called_once_with(device=pool.device)
    use_stream.assert_called_once_with(stream)
    destination.copy_.assert_called_once_with(
        source,
        non_blocking=True,
    )
    stream.synchronize.assert_called_once_with()
    assert producer._local.stream is stream
    assert result is not None
    assert result.tensors == [destination]
    assert result.regions == [(0, 256)]


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


def test_producer_allocator_zero_bounce_has_no_arena_or_padding():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=0,
    )
    transfer = MagicMock()
    transfer.register_memory.return_value = 0

    allocator.prepare(torch.device("cpu"), transfer)

    assert allocator.bounce_offset == 3 * _MIB
    assert allocator.padding == 0
    assert allocator.registered_capacity == 3 * _MIB
    assert allocator.tensor is not None
    assert allocator.tensor.nbytes == 3 * _MIB
    assert allocator.bounce_tensor is None
    assert allocator._free == [(0, 3 * _MIB)]
    transfer.register_memory.assert_called_once_with(allocator.tensor)


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


def test_producer_allocator_retries_with_bounce_only_slab():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.side_effect = [7, 0]

    allocator.prepare(torch.device("cpu"), transfer)

    tensor = allocator.tensor
    bounce = allocator.bounce_tensor
    assert tensor is not None
    assert bounce is not None
    assert allocator.configured_staging_capacity == 3 * _MIB
    assert allocator.staging_capacity == 0
    assert allocator.fallback_only
    assert allocator.bounce_offset == 0
    assert allocator.padding == 0
    assert allocator.registered_capacity == 2 * _MIB
    assert allocator.raw_allocation_size == 4 * _MIB - 1
    assert allocator._free == []
    assert allocator.allocate(1) is None
    assert tensor.nbytes == 2 * _MIB
    assert bounce.data_ptr() == tensor.data_ptr()
    assert bounce.nbytes == 2 * _MIB
    assert [call.args[0].nbytes for call in transfer.register_memory.call_args_list] == [
        6 * _MIB,
        2 * _MIB,
    ]


def test_producer_allocator_retries_bounce_only_after_allocation_failure():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.return_value = 0
    real_empty = torch.empty
    attempts = 0

    def allocate(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise torch.OutOfMemoryError("full producer slab")
        return real_empty(*args, **kwargs)

    with patch.object(memory_module.torch, "empty", side_effect=allocate):
        allocator.prepare(torch.device("cpu"), transfer)

    assert attempts == 2
    assert allocator.fallback_only
    assert allocator.staging_capacity == 0
    assert allocator.tensor is not None
    assert allocator.tensor.nbytes == 2 * _MIB
    assert allocator.bounce_tensor is not None
    transfer.register_memory.assert_called_once_with(allocator.tensor)


def test_producer_pool_returns_none_after_bounce_only_degradation():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.side_effect = [7, 0]
    pool = AscendProducerMemoryPool(3 * _MIB, transfer, allocator)
    source = torch.empty(1, dtype=torch.uint8)

    assert pool.stage([source]) is None
    assert allocator.fallback_only
    assert allocator.bounce_tensor is not None
    assert allocator._free == []


def test_producer_allocator_bounce_only_failure_is_fatal():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.return_value = 7

    with pytest.raises(
        RuntimeError,
        match=(
            r"configured_staging=.*effective_staging=0.*padding=0.*"
            r"bounce=.*registered=.*allocation="
        ),
    ):
        allocator.prepare(torch.device("cpu"), transfer)

    assert allocator.tensor is None
    assert allocator.fallback_only
    assert allocator.staging_capacity == 0
    assert transfer.register_memory.call_count == 2


def test_producer_allocator_zero_bounce_does_not_retry_failure():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=0,
    )
    transfer = MagicMock()
    transfer.register_memory.return_value = 7

    with pytest.raises(
        RuntimeError,
        match=r"configured_staging=.*effective_staging=.*bounce=0",
    ):
        allocator.prepare(torch.device("cpu"), transfer)

    assert allocator.tensor is None
    assert not allocator.fallback_only
    assert allocator.staging_capacity == 3 * _MIB
    transfer.register_memory.assert_called_once()


def test_producer_allocator_closes_bounce_only_slab():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    transfer = MagicMock()
    transfer.register_memory.side_effect = [7, 0]
    transfer.unregister_memory.return_value = True
    allocator.prepare(torch.device("cpu"), transfer)
    tensor = allocator.tensor
    assert tensor is not None

    assert allocator.close(transfer)
    transfer.unregister_memory.assert_called_once_with(tensor)
    assert allocator.tensor is None
    assert allocator._free == []


def test_bounce_lease_manager_zero_size_bypasses_queue():
    manager = _BounceLeaseManager(0)

    assert manager.acquire(0) is None


def test_copy_to_bounce_packs_bytes_within_lease():
    allocator = AscendProducerAllocator(staging_capacity=0, bounce_capacity=16)
    allocator.tensor = torch.zeros(16, dtype=torch.uint8)
    pool = AscendProducerMemoryPool(0, MagicMock(), allocator)
    lease = _BounceLease(offset=4, nbytes=5, allocated_nbytes=8)
    source_a = torch.tensor([1, 2, 3], dtype=torch.uint8)
    source_b = torch.tensor([4, 5, 6], dtype=torch.uint8)

    with (
        patch("vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory.torch.npu.Stream") as stream_cls,
        patch("vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory.torch.npu.stream") as stream_context,
    ):
        stream = stream_cls.return_value
        address = pool.copy_to_bounce(
            lease,
            [(source_a, 0, 2), (source_b, 2, 3)],
        )

    bounce = pool.bounce_tensor
    assert bounce is not None
    assert address == bounce.data_ptr() + lease.offset
    assert bounce.tolist() == [0, 0, 0, 0, 1, 2, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0]
    stream_cls.assert_called_once_with(device=bounce.device)
    stream_context.assert_called_once_with(stream)
    stream.synchronize.assert_called_once_with()


def test_bounce_lease_manager_aligns_one_whole_wave_lease():
    manager = _BounceLeaseManager(2048, alignment=256)

    lease = manager.acquire(800)

    assert lease is not None
    assert lease.offset == 0
    assert lease.nbytes == 800
    assert lease.allocated_nbytes == 1024

    manager.release(lease)
    assert manager._regions._free == [(0, 2048)]


def test_bounce_lease_manager_rejects_oversized_lease():
    manager = _BounceLeaseManager(2 * _MIB)

    with pytest.raises(ValueError, match="arena capacity"):
        manager.acquire(2 * _MIB + 1)


def test_bounce_lease_manager_waiters_do_not_bypass_fifo_head():
    manager = _BounceLeaseManager(8, alignment=1)
    held = manager.acquire(6)
    assert held is not None

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(manager.acquire, 4)
        _wait_for_waiters(manager, 1)
        second = executor.submit(manager.acquire, 2)
        _wait_for_waiters(manager, 2)

        try:
            assert not first.done()
            assert not second.done()
        finally:
            manager.release(held)

        first_lease = first.result(timeout=1)
        second_lease = second.result(timeout=1)

    assert first_lease is not None
    assert second_lease is not None
    assert first_lease.offset == 0
    assert second_lease.offset == 4

    manager.release(first_lease)
    manager.release(second_lease)


def test_bounce_lease_manager_released_batch_tail_rejoins_fifo():
    manager = _BounceLeaseManager(8, alignment=1)
    first_wave = manager.acquire(8)
    assert first_wave is not None

    with ThreadPoolExecutor(max_workers=2) as executor:
        other_batch = executor.submit(manager.acquire, 8)
        _wait_for_waiters(manager, 1)

        manager.release(first_wave)
        next_wave = executor.submit(manager.acquire, 8)

        other_lease = other_batch.result(timeout=1)
        assert other_lease is not None
        _wait_for_waiters(manager, 1)
        assert not next_wave.done()

        manager.release(other_lease)
        next_lease = next_wave.result(timeout=1)

    assert next_lease is not None
    manager.release(next_lease)


def test_bounce_lease_manager_coalesces_released_variable_size_leases():
    manager = _BounceLeaseManager(1024, alignment=1)
    first = manager.acquire(256)
    second = manager.acquire(256)
    tail = manager.acquire(512)
    assert first is not None
    assert second is not None
    assert tail is not None

    manager.release(second)
    manager.release(first)

    combined = manager.acquire(512)
    assert combined is not None
    assert combined.offset == 0

    manager.release(combined)
    manager.release(tail)


def test_producer_pool_shares_one_bounce_arena():
    allocator = AscendProducerAllocator(
        staging_capacity=3 * _MIB,
        bounce_capacity=2 * _MIB,
    )
    pool = AscendProducerMemoryPool(
        3 * _MIB,
        MagicMock(),
        allocator,
    )

    assert pool.bounce_tensor is None

    first = pool.acquire_bounce(300)
    second = pool.acquire_bounce(500)

    assert first is not None
    assert second is not None
    assert first.offset == 0
    assert second.offset == 512

    pool.release_bounce(first)
    pool.release_bounce(second)
