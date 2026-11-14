# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ctypes
import mmap
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_offload.base import CanonicalKVCacheRef, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native import cpu_npu, dram_transfer


@pytest.fixture
def registration(monkeypatch):
    memory = Mock()
    factory = Mock(return_value=memory)
    capabilities = Mock(return_value=0)
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(kv_offload_dram_capabilities=capabilities))
    monkeypatch.setattr(torch.classes, "_C_ascend", SimpleNamespace(KVOffloadHostMemory=factory))
    return capabilities, factory, memory


@pytest.mark.parametrize(
    ("capabilities", "backend", "mapped"),
    [
        (0, "auto", None),
        (0, "memcpy", None),
        (1, "auto", False),
        (1, "memcpy", False),
        (2, "auto", True),
        (3, "auto", True),
        (3, "memcpy", False),
        (3, "sdma", True),
    ],
)
def test_registration_capabilities(registration, capabilities, backend, mapped):
    capability_op, factory, memory = registration
    capability_op.return_value = capabilities
    region = SimpleNamespace(_base=object())
    result = dram_transfer.register_dram(region, backend)
    if mapped is None:
        assert result is None
        factory.assert_not_called()
    else:
        factory.assert_called_once_with(region._base, mapped)
        assert result.use_sdma is mapped
        result.close()
        memory.close.assert_called_once()


def test_old_extension_keeps_memcpy(monkeypatch):
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace())
    assert dram_transfer.register_dram(object(), "auto") is None


def test_explicit_sdma_requires_capability(registration):
    with pytest.raises(RuntimeError, match="requires an A3 build"):
        dram_transfer.register_dram(object(), "sdma")


def test_auto_registration_failure_retries_pinned(registration):
    capabilities, factory, _ = registration
    capabilities.return_value = 3
    factory.side_effect = [RuntimeError("mapping failed"), Mock()]
    region = SimpleNamespace(_base=object())
    result = dram_transfer.register_dram(region, "auto")
    assert not result.use_sdma
    assert [call.args[1] for call in factory.call_args_list] == [True, False]


def test_auto_pinning_failure_retains_original_path(registration):
    capabilities, factory, _ = registration
    capabilities.return_value = 1
    factory.side_effect = RuntimeError("pinning failed")
    assert dram_transfer.register_dram(SimpleNamespace(_base=object()), "auto") is None


def test_explicit_sdma_registration_error_is_not_hidden(registration):
    capabilities, factory, _ = registration
    capabilities.return_value = 3
    factory.side_effect = RuntimeError("mapping failed")
    with pytest.raises(RuntimeError, match="mapping failed"):
        dram_transfer.register_dram(SimpleNamespace(_base=object()), "sdma")
    assert factory.call_count == 1


def test_invalid_backend_is_rejected(registration):
    with pytest.raises(ValueError, match="dram_transfer_backend"):
        dram_transfer.register_dram(object(), "sdam")


def test_shutdown_drains_both_directions_before_unregister():
    calls = []
    worker = cpu_npu.NPUOffloadingWorker.__new__(cpu_npu.NPUOffloadingWorker)
    worker._store_handler = SimpleNamespace(shutdown=lambda: calls.append("store"))
    worker._load_handler = SimpleNamespace(shutdown=lambda: calls.append("load"))
    worker._dram = SimpleNamespace(close=lambda: calls.append("unregister"))
    worker._mmap_region = SimpleNamespace(cleanup=lambda: calls.append("unmap"))
    worker.shutdown()
    worker.shutdown()
    assert calls == ["store", "load", "unregister", "unmap", "store", "load"]


def test_failed_drain_keeps_registered_mapping():
    worker = cpu_npu.NPUOffloadingWorker.__new__(cpu_npu.NPUOffloadingWorker)
    worker._store_handler = SimpleNamespace(shutdown=Mock(side_effect=RuntimeError("DMA failed")))
    worker._load_handler = Mock()
    worker._dram = Mock()
    worker._mmap_region = Mock()
    with pytest.raises(RuntimeError, match="DMA failed"):
        worker.shutdown()
    worker._dram.close.assert_not_called()
    worker._mmap_region.cleanup.assert_not_called()


def test_initialization_failure_unregisters_and_releases_all_mmap_views(monkeypatch):
    calls = []

    class Region:
        def __init__(self):
            self.mapping = mmap.mmap(-1, mmap.PAGESIZE)
            self._base = torch.frombuffer(self.mapping, dtype=torch.int8)
            self.views = []

        def create_next_worker_view(self, page_size):
            view = self._base[: 3 * page_size].view(3, page_size)
            self.views.append(view)
            return view

        def cleanup(self):
            self.views.clear()
            self._base = None
            self.mapping.close()
            calls.append("unmap")

    region = Region()

    class Handler:
        def __init__(self, **kwargs):
            if not kwargs["npu_to_cpu"]:
                raise RuntimeError("second handler failed")
            self.tensors = list(kwargs["cpu_tensors"])

        def shutdown(self):
            self.tensors.clear()
            calls.append("drain")

    monkeypatch.setattr(cpu_npu, "SingleDirectionNPUOffloadingHandler", Handler)
    monkeypatch.setattr(
        cpu_npu, "register_dram", lambda *args: SimpleNamespace(close=lambda: calls.append("unregister"))
    )
    caches = SimpleNamespace(
        tensors=[SimpleNamespace(page_size_bytes=4, tensor=torch.empty((6, 4), dtype=torch.int8))],
        group_data_refs=[],
    )
    with pytest.raises(RuntimeError, match="second handler failed"):
        cpu_npu.NPUOffloadingWorker(caches, 2, 3, mmap_region=region)
    assert calls == ["drain", "unregister", "unmap"]


@pytest.mark.parametrize("npu_to_cpu", [False, True])
@pytest.mark.parametrize("use_sdma", [False, True])
def test_scattered_subblock_transfers_and_descriptor_lifetime(monkeypatch, npu_to_cpu, use_sdma):
    class DeviceTensor:
        device = SimpleNamespace(type="npu")

        def __init__(self, tensor):
            self.tensor = tensor

        def __getattr__(self, name):
            return getattr(self.tensor, name)

    class Event:
        done = False

        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            pass

        def query(self):
            return self.done

        def synchronize(self):
            self.done = True

        def elapsed_time(self, other):
            return 1.0

    stream = SimpleNamespace(wait_stream=Mock(), wait_event=Mock())
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Stream=lambda: stream, Event=Event, stream=lambda _: nullcontext(), current_stream=lambda: stream
        ),
        raising=False,
    )
    monkeypatch.setattr(cpu_npu, "is_pin_memory_available", lambda: False)

    def copy(src, dst, sizes, direction):
        for source, destination, size in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(destination, source, size)

    legacy_copy = Mock(side_effect=copy)
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(swap_blocks_batch=legacy_copy))
    batches = []

    class Batch:
        released = False

        def __init__(self, *args):
            self.args = args

        def launch(self):
            copy(*self.args)

        def release(self):
            self.released = True

    def prepare(*args):
        if not use_sdma:
            return None
        batch = Batch(*args)
        batches.append(batch)
        return batch

    device = torch.arange(24, dtype=torch.int8).view(6, 4)
    host = torch.arange(24, 48, dtype=torch.int8).view(3, 8)
    expected = device.clone() if npu_to_cpu else host.clone()
    handler = cpu_npu.SingleDirectionNPUOffloadingHandler(
        [DeviceTensor(device)],
        [host],
        2,
        [[CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=4)]],
        npu_to_cpu=npu_to_cpu,
        dram=SimpleNamespace(prepare=prepare),
    )
    gpu_spec = GPULoadStoreSpec([1, 4, 2], [3], [1])
    cpu_spec = CPULoadStoreSpec([2, 0])
    src, dst = (gpu_spec, cpu_spec) if npu_to_cpu else (cpu_spec, gpu_spec)
    assert handler.transfer_async(7, src, dst)
    if npu_to_cpu:
        assert torch.equal(host[2, 4:], expected[1])
        assert torch.equal(host[0], expected[[4, 2]].flatten())
    else:
        assert torch.equal(device[1], expected[2, 4:])
        assert torch.equal(device[[4, 2]].flatten(), expected[0])
    assert handler.get_finished() == []
    assert legacy_copy.call_count == (0 if use_sdma else 1)
    if use_sdma:
        assert not batches[0].released
    handler.wait({7})
    result = handler.get_finished()
    assert result[0].transfer_size == 12
    if use_sdma:
        assert batches[0].released
    handler.shutdown()
