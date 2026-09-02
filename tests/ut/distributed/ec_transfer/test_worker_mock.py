# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import vllm.distributed.ec_transfer.ec_connector.cpu.worker as upstream_worker_mod
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import (
    ECCPUTransferDirection,
    Transfer,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker.descriptor_buffers import (
    DescriptorBufferPool,
)

import vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker as worker_mod
from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker import (
    AscendECCPUWorker,
)


class FakeStream:
    def __init__(self, order=None):
        self.order = order
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1
        if self.order is not None:
            self.order.append("stream_synchronize")

    def wait_stream(self, stream):
        del stream


class FakeEvent:
    def __init__(self, *, completed=False, order=None):
        self.completed = completed
        self.order = order
        self.recorded_stream = None
        self.synchronize_calls = 0

    def record(self, stream):
        self.recorded_stream = stream

    def query(self):
        return self.completed

    def synchronize(self):
        self.synchronize_calls += 1
        if self.order is not None:
            self.order.append("event_synchronize")

    def elapsed_time(self, end_event):
        del end_event
        return 0.0


class FakeNPU:
    def __init__(self, order=None):
        self.events = []
        self.enable_timing = []
        self.order = order
        self.synchronize_calls = 0

    def Event(self, *, enable_timing=False):
        event = FakeEvent()
        self.events.append(event)
        self.enable_timing.append(enable_timing)
        return event

    def synchronize(self):
        self.synchronize_calls += 1
        if self.order is not None:
            self.order.append("npu_synchronize")


class FakePlatform:
    device_type = "cpu"

    def __init__(self):
        self.compute_stream = FakeStream()
        self.created_streams = []

    def Stream(self):
        stream = FakeStream()
        self.created_streams.append(stream)
        return stream

    def stream(self, stream):
        del stream
        return nullcontext()

    def current_stream(self):
        return self.compute_stream


class FakeRegion:
    def __init__(self):
        self.blocks = torch.empty((8, 16), dtype=torch.int8)
        self.block_size_bytes = 16
        self.num_blocks = 8
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


def raises(message):
    def raise_error(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(message)

    return raise_error


def make_worker():
    worker = AscendECCPUWorker.__new__(AscendECCPUWorker)
    worker._region = FakeRegion()
    worker._buf_pool = DescriptorBufferPool()
    worker._event_pool = []
    worker._mmap_pinned = True
    return worker


def patch_init_dependencies(
    monkeypatch,
    *,
    supported=True,
    register_error=None,
):
    region = FakeRegion()
    platform = FakePlatform()
    npu = FakeNPU()
    register_calls = []
    pcp_group = SimpleNamespace(rank_in_group=0)

    monkeypatch.setattr(upstream_worker_mod, "create_ec_shared_region", lambda cfg: region)
    monkeypatch.setattr(upstream_worker_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(upstream_worker_mod, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(worker_mod, "_supports_eccpu_offload", lambda: supported)
    monkeypatch.setattr(worker_mod.torch, "npu", npu, raising=False)
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    monkeypatch.setattr(upstream_worker_mod, "current_platform", platform)

    def register(blocks):
        register_calls.append(blocks)
        if register_error is not None:
            raise register_error

    monkeypatch.setattr(worker_mod, "_register_pinned_host_mmap", register)
    return region, platform, npu, register_calls


def test_init_registers_mmap_and_uses_upstream_lifecycle(monkeypatch):
    region, platform, _, register_calls = patch_init_dependencies(monkeypatch)
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    worker = AscendECCPUWorker(config)

    assert worker._region is region
    assert register_calls == [region.blocks]
    assert worker._mmap_pinned is True
    assert worker._is_save_rank is True
    assert worker._inflight_saves == deque()
    assert worker._inflight_loads == deque()
    assert platform.created_streams == []
    assert region.cleanup_calls == 0


def test_init_rejects_unsupported_runtime_and_cleans_region(monkeypatch):
    region, _, _, register_calls = patch_init_dependencies(monkeypatch, supported=False)
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="aclrtMemcpyBatchAsync"):
        AscendECCPUWorker(config)

    assert register_calls == []
    assert region.cleanup_calls == 1


def test_init_registration_failure_cleans_region(monkeypatch):
    region, _, _, register_calls = patch_init_dependencies(
        monkeypatch, register_error=RuntimeError("registration failed")
    )
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="registration failed"):
        AscendECCPUWorker(config)

    assert register_calls == [region.blocks]
    assert region.cleanup_calls == 1


def test_acquire_event_uses_npu_event_and_recycles(monkeypatch):
    worker = make_worker()
    npu = FakeNPU()
    monkeypatch.setattr(worker_mod.torch, "npu", npu, raising=False)

    created = worker._acquire_event()
    worker._event_pool.append(created)
    recycled = worker._acquire_event()

    assert recycled is created
    assert npu.enable_timing == [True]


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (ECCPUTransferDirection.HOST_TO_DEVICE, worker_mod._DIRECTION_H2D),
        (ECCPUTransferDirection.DEVICE_TO_HOST, worker_mod._DIRECTION_D2H),
    ],
)
def test_submit_transfer_maps_cann_direction_without_releasing(monkeypatch, direction, expected):
    worker = make_worker()
    bufs = worker._buf_pool.acquire(2)
    calls = []
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: calls.append(args))

    worker._submit_transfer(bufs, 1, direction)

    src, dst, sizes, cann_direction = calls[0]
    assert src.data_ptr() == bufs.src_ptrs.data_ptr()
    assert dst.data_ptr() == bufs.dst_ptrs.data_ptr()
    assert sizes.data_ptr() == bufs.sizes.data_ptr()
    assert cann_direction == expected
    assert worker._buf_pool._pool == []


def test_submit_failure_synchronizes_and_releases_descriptors(monkeypatch):
    worker = make_worker()
    platform = FakePlatform()
    bufs = worker._buf_pool.acquire(1)
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", raises("copy failed"))

    with pytest.raises(RuntimeError, match="copy failed"):
        worker._submit_transfer(
            bufs,
            1,
            ECCPUTransferDirection.DEVICE_TO_HOST,
        )

    assert platform.compute_stream.synchronize_calls == 1
    assert worker._buf_pool._pool == [bufs]


def test_flush_failure_clears_upstream_save_batch_state(monkeypatch):
    worker = make_worker()
    platform = FakePlatform()
    bufs = worker._buf_pool.acquire(1)
    worker._save_bufs = bufs
    worker._save_stream = FakeStream()
    worker._save_count = 1
    worker._save_bytes = 16
    worker._save_mm_hashes = ["hash"]
    worker._inflight_saves = deque()
    worker._acquire_event = lambda: FakeEvent()
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    monkeypatch.setattr(upstream_worker_mod, "current_platform", platform)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", raises("copy failed"))

    with pytest.raises(RuntimeError, match="copy failed"):
        worker.flush_saves()

    assert worker._save_bufs is None
    assert worker._save_stream is None
    assert worker._save_count == 0
    assert worker._save_bytes == 0
    assert worker._save_mm_hashes == []
    assert worker._buf_pool._pool == [bufs]


def test_upstream_completion_recycles_npu_transfer_resources():
    worker = make_worker()
    worker._stream_pool = []
    bufs = worker._buf_pool.acquire(1)
    stream = FakeStream()
    start_event = FakeEvent()
    end_event = FakeEvent(completed=True)
    inflight = deque(
        [
            Transfer(
                start_event=start_event,
                end_event=end_event,
                completions=["hash"],
                bufs=bufs,
                stream=stream,
                num_bytes=16,
            )
        ]
    )

    completed = worker._collect_finished(inflight, "save")

    assert completed == ["hash"]
    assert inflight == deque()
    assert worker._buf_pool._pool == [bufs]
    assert worker._stream_pool == [stream]
    assert worker._event_pool == [start_event, end_event]


def test_shutdown_waits_for_transfer_then_unregisters_and_cleans_up(monkeypatch):
    worker = make_worker()
    order: list[str] = []
    monkeypatch.setattr(worker_mod.torch, "npu", FakeNPU(order), raising=False)
    worker._dtype = torch.float32
    worker._is_save_rank = True
    worker._save_bufs = None
    worker._save_stream = None
    worker._save_count = 0
    worker._save_bytes = 0
    worker._save_mm_hashes = []
    worker._stream_pool = []
    worker._inflight_loads = deque()
    bufs = worker._buf_pool.acquire(1)
    end_event = FakeEvent(order=order)
    worker._inflight_saves = deque(
        [
            Transfer(
                start_event=FakeEvent(),
                end_event=end_event,
                completions=["hash"],
                bufs=bufs,
                stream=FakeStream(),
                num_bytes=16,
            )
        ]
    )
    worker._region.cleanup = lambda: order.append("cleanup")
    monkeypatch.setattr(
        worker_mod,
        "_unregister_pinned_host_mmap",
        lambda blocks: order.append("unregister"),
    )

    worker.shutdown()

    assert order == [
        "event_synchronize",
        "npu_synchronize",
        "unregister",
        "cleanup",
    ]
    assert worker._mmap_pinned is False


def test_unregister_failure_preserves_registration_for_retry(monkeypatch):
    worker = make_worker()
    fake_npu = FakeNPU()
    monkeypatch.setattr(worker_mod.torch, "npu", fake_npu, raising=False)
    calls = 0

    def fail_once(blocks):
        del blocks
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("unregister failed")

    monkeypatch.setattr(worker_mod, "_unregister_pinned_host_mmap", fail_once)

    with pytest.raises(RuntimeError, match="unregister failed"):
        worker._shutdown_transfer_backend()

    assert worker._mmap_pinned is True
    worker._shutdown_transfer_backend()
    assert calls == 2
    assert fake_npu.synchronize_calls == 2
    assert worker._mmap_pinned is False
