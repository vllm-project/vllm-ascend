# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCaches, CanonicalKVCacheTensor, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native import cpu_npu as module
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import (
    NPUOffloadingWorker,
)


def test_npu_worker_reuses_upstream_worker_protocol() -> None:
    assert issubclass(NPUOffloadingWorker, CPUOffloadingWorker)


class DeviceView:
    """Expose NPU device metadata around a real CPU allocation; perform no I/O."""

    def __init__(self, tensor):
        self.tensor = tensor
        self.device = SimpleNamespace(type="npu")

    def __getattr__(self, name):
        return getattr(self.tensor, name)


@pytest.fixture
def dma(monkeypatch):
    monkeypatch.setattr(module, "is_pin_memory_available", lambda: False)
    swap = MagicMock()
    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", swap)
    torch.npu.Stream.side_effect = lambda: MagicMock()
    torch.npu.Event.side_effect = lambda **kwargs: MagicMock(
        query=MagicMock(return_value=False), elapsed_time=MagicMock(return_value=2.5)
    )
    return swap


def make_handler(npu_to_cpu):
    npu = DeviceView(torch.zeros(6, 8, dtype=torch.int8))
    cpu = torch.zeros(4, 16, dtype=torch.int8)
    ref = CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=8)
    handler = module.SingleDirectionNPUOffloadingHandler([npu], [cpu], 2, [[ref], [ref]], npu_to_cpu)
    return handler, npu, cpu


@pytest.mark.parametrize("is_store", [False, True])
def test_partial_chunk_descriptors_and_completion_lifecycle(dma, is_store):
    handler, npu, cpu = make_handler(is_store)
    gpu_spec = GPULoadStoreSpec([2, 0], [0, 2], [0, 1])
    cpu_spec = CPULoadStoreSpec([1, 2])
    source, destination = (gpu_spec, cpu_spec) if is_store else (cpu_spec, gpu_spec)
    assert handler.transfer_async(7, source, destination)
    src, dst, sizes, direction = dma.call_args.args
    gpu_pointers = [npu.data_ptr() + 16, npu.data_ptr()]
    cpu_pointers = [cpu.data_ptr() + 24, cpu.data_ptr() + 32]
    assert src.tolist() == (gpu_pointers if is_store else cpu_pointers)
    assert dst.tolist() == (cpu_pointers if is_store else gpu_pointers)
    assert sizes.tolist() == [8, 8]
    assert direction == (1 if is_store else 0)
    assert handler.get_finished() == []
    transfer = handler._transfers[0]
    assert transfer.stream.wait_stream.call_count == int(is_store)
    handler.wait({7, 999})
    transfer.end_event.synchronize.assert_called_once_with()
    transfer.end_event.query.return_value = True
    results = handler.get_finished()
    assert [(r.job_id, r.success, r.transfer_size) for r in results] == [(7, True, 16)]
    assert results[0].transfer_time == pytest.approx(0.0025)
    assert not handler._transfer_events
    assert len(handler._buffer_pool) == len(handler._stream_pool) == 1
    assert len(handler._event_pool) == 2
    # Reuse the returned objects without allocating a stream or event.
    assert handler.transfer_async(8, source, destination)
    assert torch.npu.Stream.call_count == 1
    assert torch.npu.Event.call_count == 2
    handler.shutdown()
    assert not handler.src_tensors and not handler.dst_tensors
    assert not handler._transfers and not handler._buffer_pool and not handler._event_pool


def test_pending_jobs_wait_on_prior_event_and_grow_descriptor_buffer(dma):
    handler, _, _ = make_handler(True)
    handler.transfer_async(1, GPULoadStoreSpec([0], [0, 1], [0, 0]), CPULoadStoreSpec([0]))
    first = handler._transfers[0]
    handler.transfer_async(2, GPULoadStoreSpec([1], [0, 1], [0, 0]), CPULoadStoreSpec([1]))
    handler._transfers[1].stream.wait_event.assert_called_once_with(first.end_event)
    for transfer in handler._transfers:
        transfer.end_event.query.return_value = True
    assert [r.job_id for r in handler.get_finished()] == [1, 2]
    handler.transfer_async(3, GPULoadStoreSpec([0, 1, 2], [0, 3], [0, 0]), CPULoadStoreSpec([0, 1]))
    assert handler._transfers[0].batch_src.numel() == 3
    assert dma.call_args.args[2].tolist() == [8, 8, 8]
    handler.shutdown()
    handler.shutdown()


def test_empty_transfer_completes_without_device_copy(dma):
    handler, _, _ = make_handler(True)
    assert handler.transfer_async(1, GPULoadStoreSpec([], [0, 0], [0, 0]), CPULoadStoreSpec([]))
    dma.assert_not_called()
    handler._transfers[0].end_event.query.return_value = True
    result = handler.get_finished()[0]
    assert result.transfer_size == 0 and result.success
    handler.shutdown()


def test_worker_allocates_mirrors_and_keeps_handlers_independent(dma):
    tensor = DeviceView(torch.zeros(3, 8, dtype=torch.int8))
    caches = CanonicalKVCaches([CanonicalKVCacheTensor(tensor, 8)], [[CanonicalKVCacheRef(0, 8)]])
    worker = NPUOffloadingWorker(caches, 2, 4)
    assert worker._store_handler.dst_tensors[0].shape == (4, 16)
    assert worker._load_handler.src_tensors[0] is worker._store_handler.dst_tensors[0]
    assert torch.count_nonzero(worker._load_handler.src_tensors[0]).item() == 0
    worker._store_handler.shutdown()
    assert worker._load_handler.src_tensors and worker._load_handler.dst_tensors
    worker._load_handler.shutdown()


@pytest.mark.parametrize(
    "tensor,page_size",
    [(torch.zeros(3, 8), 8), (torch.zeros(24, dtype=torch.int8), 8), (torch.zeros(3, 8, dtype=torch.int8), 9)],
)
def test_worker_rejects_invalid_canonical_payload(dma, tensor, page_size):
    caches = CanonicalKVCaches([CanonicalKVCacheTensor(DeviceView(tensor), page_size)], [])
    with pytest.raises(ValueError, match="Canonical NPU KV cache"):
        NPUOffloadingWorker(caches, 2, 4)
    dma.assert_not_called()
