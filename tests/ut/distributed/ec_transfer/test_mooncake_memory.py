import threading
from concurrent.futures import ThreadPoolExecutor
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


def _make_pool(capacity=1024):
    allocator = AscendProducerAllocator(capacity, 0)
    allocator.tensor = torch.empty(capacity, dtype=torch.uint8)
    pool = AscendProducerMemoryPool(capacity, MagicMock(), allocator)
    pool._copy_staged = MagicMock(side_effect=lambda sources, targets: [d.copy_(s) for s, d in zip(sources, targets)])
    return pool


def test_legacy_bounce_bytes_are_available_to_staging():
    allocator = AscendProducerAllocator(256, 512)
    transfer = MagicMock()
    transfer.register_memory.return_value = 0
    pool = AscendProducerMemoryPool(256, transfer, allocator)
    assert pool.capacity == 768
    lease = pool.reserve_bytes([600], torch.device("cpu"))
    assert lease is not None
    assert lease.tensors[0].nbytes == 600
    pool.release(lease)
    assert allocator._free == [(0, 768)]


def test_headers_and_payload_are_one_atomic_allocation():
    pool = _make_pool(512)
    source = torch.arange(256, dtype=torch.uint8)
    header = torch.full((24,), 7, dtype=torch.uint8)
    staged = pool.stage_with_headers([source], [header])
    assert staged.regions == [(0, 512)]
    assert torch.equal(staged.tensors[0], source)
    assert torch.equal(staged.headers[0], header)
    assert staged.tensors[0].data_ptr() == staged.headers[0].data_ptr() + 24
    pool.release(staged)
    assert pool._allocator._free == [(0, 512)]


def test_waiting_stage_only_reuses_completed_lease():
    pool = _make_pool(256)
    source = torch.zeros(256, dtype=torch.uint8)
    first = pool.stage_wait([source])
    with ThreadPoolExecutor(1) as executor:
        waiting = executor.submit(pool.stage_wait, [source])
        try:
            assert not waiting.done()
        finally:
            pool.release(first)
        second = waiting.result(timeout=2)
    assert second.regions == [(0, 256)]
    pool.release(second)


def test_poison_wakes_waiters_without_recycling_live_memory():
    pool = _make_pool(256)
    source = torch.zeros(256, dtype=torch.uint8)
    first = pool.stage_wait([source])
    with ThreadPoolExecutor(1) as executor:
        waiting = executor.submit(pool.stage_wait, [source])
        pool.poison()
        with pytest.raises(RuntimeError, match="poisoned"):
            waiting.result(timeout=2)
    pool.release(first)
    assert pool._allocator._free == []


def test_oversized_batch_returns_without_waiting_or_partial_allocations():
    pool = _make_pool(256)
    assert pool.stage_wait([torch.zeros(257, dtype=torch.uint8)]) is None
    assert pool._allocator._free == [(0, 256)]
    pool._copy_staged.assert_not_called()


def _cache_get(pool):
    lease = pool.reserve_bytes([152], torch.device("cpu"))
    buffer = lease.tensors[0]
    payload = buffer[24:].view(torch.float32)
    payload.fill_(3)
    output = payload.clone()
    pool.cache_loaded(output, payload, buffer[:24], lease.regions[0])
    return output, payload


def test_get_to_p2p_transfers_lease_without_allocating_or_copying():
    pool = _make_pool(256)
    output, payload = _cache_get(pool)
    staged = pool.stage_wait([output])
    assert staged.tensors[0].data_ptr() == payload.data_ptr()
    assert pool._loaded == {}
    pool._copy_staged.assert_called_once_with([], [])
    pool.release(staged)
    assert pool._allocator._free == [(0, 256)]
    assert torch.all(output == 3)


def test_pressure_reclaims_completed_get_copy_without_changing_cache_tensor():
    pool = _make_pool(256)
    output, _ = _cache_get(pool)
    next_lease = pool.reserve_bytes([256], torch.device("cpu"))
    assert next_lease is not None
    next_lease.tensors[0].zero_()
    assert torch.all(output == 3)
    pool.release(next_lease)
    staged = pool.stage_wait([output])
    assert torch.equal(staged.tensors[0], output)
    pool.release(staged)


def test_cache_eviction_returns_optional_get_lease():
    import gc

    pool = _make_pool(256)
    output, _ = _cache_get(pool)
    del output
    gc.collect()
    assert pool._loaded == {}
    assert pool._allocator._free == [(0, 256)]


def test_borrowed_get_lease_cannot_be_reclaimed_during_wait():
    pool = _make_pool(512)
    output, payload = _cache_get(pool)
    active = pool.reserve_bytes([256], torch.device("cpu"))
    other = torch.ones(64, dtype=torch.uint8)
    entered = threading.Event()
    original = pool._space_available.wait

    def wait():
        entered.set()
        return original()

    with patch.object(pool._space_available, "wait", side_effect=wait), ThreadPoolExecutor(1) as executor:
        future = executor.submit(pool.stage_wait, [output, other])
        try:
            assert entered.wait(2)
            assert pool.reserve_bytes([256], torch.device("cpu")) is None
        finally:
            pool.release(active)
        staged = future.result(timeout=2)
    assert staged.tensors[0].data_ptr() == payload.data_ptr()
    pool.release(staged)
    assert pool._allocator._free == [(0, 512)]


def test_get_header_overhead_does_not_reject_payload_batch_that_fits():
    pool = _make_pool(512)
    lease = pool.reserve_bytes([280], torch.device("cpu"))
    buffer = lease.tensors[0]
    payload = buffer[24:]
    output = payload.clone()
    pool.cache_loaded(output, payload, buffer[:24], lease.regions[0])
    other = torch.zeros(256, dtype=torch.uint8)
    staged = pool.stage_wait([output, other])
    assert staged is not None
    assert len(staged.regions) == 2
    pool.release(staged)
    assert pool._allocator._free == [(0, 512)]


def test_copy_exception_retains_sources_and_lease_without_reuse():
    pool = _make_pool(256)
    source = torch.zeros(128, dtype=torch.uint8)
    pool._copy_staged.side_effect = RuntimeError("unknown copy completion")
    with pytest.raises(RuntimeError, match="unknown copy"):
        pool.stage_wait([source])
    assert pool._poisoned
    assert pool._allocator._free == []
    retained = pool._transfer.retain_owners.call_args.args[0]
    assert retained[0] is pool
    assert retained[-1] is source


def test_staging_copy_synchronizes_its_own_stream_before_return():
    pool = _make_pool(256)
    source, destination = MagicMock(), MagicMock()
    stream = MagicMock()
    ordered = MagicMock()
    ordered.attach_mock(destination.copy_, "copy")
    ordered.attach_mock(stream.synchronize, "synchronize")
    with patch.object(memory_module.torch.npu, "Stream", return_value=stream):
        AscendProducerMemoryPool._copy_staged(pool, [source], [destination])
    assert [call[0] for call in ordered.mock_calls] == ["copy", "synchronize"]
    destination.copy_.assert_called_once_with(source, non_blocking=True)
