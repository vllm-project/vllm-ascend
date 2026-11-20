# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

import ctypes
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm.distributed.ec_transfer.ec_connector.cpu.worker as upstream_worker_mod
from vllm.distributed.ec_transfer.ec_connector.cpu.common import ECCPUConnectorMetadata
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
from vllm_ascend.distributed.ec_transfer.ec_connector.tiering.connector import AscendECTieringConnector


class FakeStream:
    def __init__(self, order=None):
        self.order = order
        self.synchronize_calls = 0
        self.waited_events = []
        self.waited_streams = []

    def synchronize(self):
        self.synchronize_calls += 1
        if self.order is not None:
            self.order.append("stream_synchronize")

    def wait_stream(self, stream):
        self.waited_streams.append(stream)

    def wait_event(self, event):
        self.waited_events.append(event)


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


def test_submit_failure_propagates_without_releasing_descriptors(monkeypatch):
    worker = make_worker()
    bufs = worker._buf_pool.acquire(1)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", raises("copy failed"))

    with pytest.raises(RuntimeError, match="copy failed"):
        worker._submit_transfer(
            bufs,
            1,
            ECCPUTransferDirection.DEVICE_TO_HOST,
        )

    assert worker._buf_pool._pool == []


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


def test_ascend_roundtrip_retains_buffers_and_acks_until_events_complete(monkeypatch):
    """Exercise inherited copy/stream plumbing using CPU pointers, not NPU DMA."""
    region, platform, npu, _ = patch_init_dependencies(monkeypatch)
    worker = AscendECCPUWorker(SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float32)))
    registered_streams = []
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: registered_streams.append(stream))
    directions = []

    def copy_batch(src, dst, sizes, direction):
        directions.append(direction)
        for source, target, size in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(target, source, size)

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", copy_batch)
    region.blocks.fill_(42)
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    metadata = ECCPUConnectorMetadata()
    metadata.saves = {"image": [2, 4]}
    worker.save_caches({"image": source}, "image", metadata)
    save_stream = worker._save_stream
    worker.flush_saves()
    save = worker._inflight_saves[0]
    assert save_stream.waited_streams == [platform.compute_stream]
    assert registered_streams == [save_stream]
    assert save.start_event.recorded_stream is save_stream
    assert save.end_event.recorded_stream is save_stream
    assert worker.build_connector_worker_meta() is None
    assert worker._buf_pool._pool == []
    save.end_event.completed = True
    assert worker.build_connector_worker_meta().completed_saves == ["image"]
    # Only logical tensor bytes were saved; FS must still store both full blocks.
    assert torch.all(region.blocks[4, 8:] == 42)

    metadata = ECCPUConnectorMetadata()
    metadata.loads = {"image": (17, [2, 4], (2, 3))}
    loaded = {}
    worker.start_load_caches(loaded, metadata)
    load = worker._inflight_loads[0]
    assert directions == [worker_mod._DIRECTION_D2H, worker_mod._DIRECTION_H2D]
    assert load.num_bytes == 32
    assert torch.equal(loaded["image"], source)
    assert registered_streams[-1] is platform.compute_stream
    assert platform.compute_stream.waited_events == []
    assert platform.compute_stream.waited_streams == [load.stream]
    assert load.start_event.recorded_stream is load.stream
    assert load.end_event.recorded_stream is load.stream
    assert worker.build_connector_worker_meta() is None
    assert worker._buf_pool._pool == []
    assert load.stream not in worker._stream_pool
    load.end_event.completed = True
    report = worker.build_connector_worker_meta()
    assert report.completed_loads == [17]
    assert worker.build_connector_worker_meta() is None
    assert worker._buf_pool._pool == [load.bufs]
    assert npu.synchronize_calls == 0
    assert all(event.synchronize_calls == 0 for event in npu.events)
    monkeypatch.setattr(worker_mod, "_unregister_pinned_host_mmap", lambda blocks: None)
    worker.shutdown()
    assert region.cleanup_calls == 1


@pytest.mark.parametrize("tp_rank,pcp_rank", [(1, 0), (0, 1), (1, 1)])
def test_non_save_ranks_still_report_every_load(monkeypatch, tp_rank, pcp_rank):
    _, _, _, _ = patch_init_dependencies(monkeypatch)
    monkeypatch.setattr(upstream_worker_mod, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(upstream_worker_mod, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=pcp_rank))
    worker = AscendECCPUWorker(SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float32)))
    metadata = ECCPUConnectorMetadata()
    metadata.saves = {"image": [0]}
    worker.save_caches({}, "image", metadata)
    assert worker._save_count == 0
    assert worker._is_save_rank is False
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: None)
    metadata.loads = {"image": (31, [0], (1, 4))}
    worker.start_load_caches({}, metadata)
    assert worker.build_connector_worker_meta() is None
    worker._inflight_loads[0].end_event.completed = True
    assert worker.build_connector_worker_meta().completed_loads == [31]
    assert worker.build_connector_worker_meta() is None
    monkeypatch.setattr(worker_mod, "_unregister_pinned_host_mmap", lambda blocks: None)
    worker.shutdown()


@pytest.mark.parametrize("tp,pcp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_promoted_entry_stays_pinned_until_all_ascend_ranks_complete(monkeypatch, tp, pcp):
    """Fake FS completion -> real primary/metadata -> mocked Ascend DMA/acks."""
    import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as cpu_scheduler_mod
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
    from vllm.distributed.ec_transfer.ec_connector.tiering.filesystem import ECReadResult
    from vllm.distributed.ec_transfer.ec_connector.tiering.manager import ECTieringManager

    region, _, _, _ = patch_init_dependencies(monkeypatch)
    region.num_blocks = 2
    region.blocks = region.blocks[:2]
    monkeypatch.setattr(cpu_scheduler_mod, "create_ec_shared_region", lambda cfg: region)
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(dtype=torch.float32),
        ec_transfer_config=SimpleNamespace(
            is_ec_producer=True,
            is_ec_consumer=True,
            ec_connector_extra_config={},
            mm_encoder_output_widths={"image": 3},
            get_from_extra_config=lambda key, default: default,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, prefill_context_parallel_size=pcp, distributed_executor_backend="mp"
        ),
    )
    connector = AscendECTieringConnector(config, ECConnectorRole.SCHEDULER)
    scheduler = connector.connector_scheduler
    secondary = Mock()
    secondary.contains.return_value = True
    secondary.submit_read.return_value = True
    secondary.poll_completed_reads.return_value = []
    secondary.poll_completed_writes.return_value = []
    scheduler._tiering_manager = ECTieringManager(scheduler._primary, secondary, "a" * 64)
    feature = SimpleNamespace(
        identifier="image",
        modality="image",
        mm_position=SimpleNamespace(offset=0, length=2, get_num_embeds=lambda: 2),
    )
    request = SimpleNamespace(request_id="r", mm_features=[feature])
    connector.on_new_request(request)
    assert not connector.ensure_cache_available(request, 0)
    secondary.submit_read.assert_called_once()
    assert scheduler._primary.alloc("other", 2) is None
    key, destinations = secondary.submit_read.call_args.args
    assert sum(map(len, destinations)) == 32
    source = torch.arange(8, dtype=torch.float32)
    offset = 0
    payload = source.numpy().tobytes()
    for destination in destinations:
        destination[:] = payload[offset : offset + len(destination)]
        offset += len(destination)
    secondary.poll_completed_reads.return_value = [ECReadResult(key, True, file_size_bytes=32)]
    assert connector.ensure_cache_available(request, 0)
    secondary.poll_completed_reads.return_value = []
    assert connector.has_cache_item("image")
    connector.update_state_after_alloc(request, 0)
    metadata = connector.build_connector_meta(None)
    assert metadata.saves == {}
    assert metadata.loads["image"][2] == (2, 3)
    assert scheduler._primary.alloc("other", 2) is None
    reports = []
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)

    def copy_batch(src, dst, sizes, direction):
        assert direction == worker_mod._DIRECTION_H2D
        for source_ptr, target, size in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(target, source_ptr, size)

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", copy_batch)
    monkeypatch.setattr(worker_mod, "_unregister_pinned_host_mmap", lambda blocks: None)
    workers = []
    for _ in range(tp * pcp):
        worker = AscendECCPUWorker(config)
        workers.append(worker)
        loaded = {}
        worker.start_load_caches(loaded, metadata)
        assert torch.equal(loaded["image"], source[:6].reshape(2, 3))
        assert worker.build_connector_worker_meta() is None
        worker._inflight_loads[0].end_event.completed = True
        reports.append(worker.build_connector_worker_meta())
    for report in reports[:-1]:
        connector.update_connector_output(SimpleNamespace(ec_connector_worker_meta=report))
        assert scheduler._primary.alloc("other", 2) is None
    output = SimpleNamespace(ec_connector_worker_meta=reports[-1])
    connector.update_connector_output(output)
    connector.update_connector_output(output)  # Stale completed transfer must not double-release.
    assert scheduler._primary.alloc("other", 2) is not None
    for worker in workers:
        worker.shutdown()
    connector.shutdown()
