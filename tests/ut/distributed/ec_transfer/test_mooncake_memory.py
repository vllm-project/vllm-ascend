from concurrent.futures import Future, ThreadPoolExecutor
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


@pytest.mark.parametrize(
    "done_before_registration,error_first,outcome",
    [
        (False, False, "success"),
        (True, False, "success"),
        (False, True, "safe"),
        (False, True, "unsafe"),
        (True, True, "unsafe"),
        (False, False, "cancel"),
    ],
)
def test_release_after_waits_for_all_readers_and_applies_caller_error_policy(
    done_before_registration, error_first, outcome
):
    pool = _make_pool(256)
    lease = pool.reserve_bytes([64], torch.device("cpu"))
    first: Future[None] = Future()
    second: Future[None] = Future()
    retain = MagicMock(side_effect=pool.poison)

    def finish(reader):
        if reader is second and outcome == "cancel":
            reader.cancel()
        elif reader is second and outcome in ("safe", "unsafe"):
            reader.set_exception(OSError("reader failed"))
        else:
            reader.set_result(None)

    order = (second, first) if error_first else (first, second)
    if done_before_registration:
        for reader in order:
            finish(reader)
    pool.release_after(lease, [first, second], retain=retain, safe_errors=(OSError,) if outcome == "safe" else ())
    if not done_before_registration:
        finish(order[0])
        assert pool._allocator._free == []
        finish(order[1])
    if outcome in ("unsafe", "cancel"):
        retain.assert_called_once_with()
        assert pool._poisoned
        assert pool._allocator._free == []
    else:
        retain.assert_not_called()
        assert pool._allocator._free == [(0, 256)]


def test_release_after_concurrent_callbacks_release_once():
    pool = _make_pool(256)
    lease = pool.reserve_bytes([64], torch.device("cpu"))
    readers: list[Future[None]] = [Future() for _ in range(16)]
    retain = MagicMock()
    with patch.object(pool, "release", wraps=pool.release) as release:
        pool.release_after(lease, readers, retain=retain)
        with ThreadPoolExecutor(4) as executor:
            list(executor.map(lambda future: future.set_result(None), readers))
        release.assert_called_once_with(lease)
    retain.assert_not_called()


def test_release_after_failed_reclamation_retains_instead_of_recycling():
    pool = _make_pool(256)
    lease = pool.reserve_bytes([64], torch.device("cpu"))
    reader: Future[None] = Future()
    retain = MagicMock(side_effect=pool.poison)
    pool.release_after(lease, [reader], retain=retain)
    with patch.object(pool, "release", side_effect=RuntimeError("unknown reclamation")):
        reader.set_result(None)
    retain.assert_called_once_with()
    assert pool._allocator._free == []
