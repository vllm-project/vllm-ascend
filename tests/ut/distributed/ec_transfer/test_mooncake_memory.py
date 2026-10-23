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
    AscendProducerMemoryPool,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
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
    allocator = AscendContiguousAllocator(capacity)
    allocator.tensor = torch.empty(capacity, dtype=torch.uint8)
    pool = AscendProducerMemoryPool(capacity, MagicMock(), allocator)
    pool._copy_staged = MagicMock(side_effect=lambda sources, targets: [d.copy_(s) for s, d in zip(sources, targets)])
    return pool


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
    with (
        patch.object(memory_module.torch.npu, "Stream", return_value=stream),
        patch.object(memory_module.torch.npu, "stream") as stream_context,
    ):
        AscendProducerMemoryPool._copy_staged(pool, [source], [destination])
    assert [call[0] for call in ordered.mock_calls] == ["copy", "synchronize"]
    destination.copy_.assert_called_once_with(source, non_blocking=True)
    stream_context.assert_called_once_with(stream)


def test_transfer_lazily_reuses_process_ascend_engine():
    transfer = AscendMooncakeTransfer("producer-host", 3)
    engine = MagicMock()

    with (
        patch(
            "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer.global_te.get_transfer_engine",
            return_value=engine,
        ) as get_transfer_engine,
        patch("vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer.torch.npu.set_device") as set_device,
    ):
        assert transfer._ensure_engine() is engine
        assert transfer._ensure_engine() is engine

    set_device.assert_called_once_with(3)
    get_transfer_engine.assert_called_once_with(
        "producer-host",
        device_name=None,
    )


def test_fixed_registrations_never_deregister_on_pool_or_transfer_close():
    transfer = AscendMooncakeTransfer("host", 0)
    engine = MagicMock()
    engine.batch_register_memory.return_value = 0
    tensor = MagicMock(nbytes=128)
    tensor.data_ptr.return_value = 2097152
    transfer._engine = engine
    assert transfer.register_memory(tensor) == 0
    assert not transfer.unregister_memory(tensor)
    transfer.close()
    engine.unregister_memory.assert_not_called()
    engine.batch_unregister_memory.assert_not_called()
    engine.close.assert_not_called()
    assert any(owner is tensor for owner in transfer._retained_owners)
