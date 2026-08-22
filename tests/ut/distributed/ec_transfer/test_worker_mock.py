# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.worker.descriptor_buffers import (
    DescriptorBufferPool,
)

import vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker as worker_mod
from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker import (
    AscendECCPUWorker,
)


class FakeStream:
    def __init__(self):
        self.synchronize_calls = 0
        self.waited_streams = []

    def synchronize(self):
        self.synchronize_calls += 1

    def wait_stream(self, stream):
        self.waited_streams.append(stream)


class FakeEvent:
    def __init__(self, *, record_error=False, query_error=False):
        self.completed = False
        self.recorded_stream = None
        self.record_error = record_error
        self.query_error = query_error

    def record(self, stream):
        self.recorded_stream = stream
        if self.record_error:
            raise RuntimeError("event record failed")

    def query(self):
        if self.query_error:
            raise RuntimeError("event query failed")
        return self.completed


class FakeNPU:
    def __init__(self, *, record_error=False):
        self.events = []
        self.synchronize_calls = 0
        self.record_error = record_error

    def Event(self):
        event = FakeEvent(record_error=self.record_error)
        self.events.append(event)
        return event

    def synchronize(self):
        self.synchronize_calls += 1
        for event in self.events:
            event.completed = True


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


def make_worker(dtype=torch.float32):
    worker = AscendECCPUWorker.__new__(AscendECCPUWorker)
    worker._region = FakeRegion()
    worker._dtype = dtype
    worker._is_save_rank = True
    worker._buf_pool = DescriptorBufferPool()
    worker._save_bufs = None
    worker._save_count = 0
    worker._inflight_descriptor_bufs = []
    worker._load_stream = FakeStream()
    worker._mmap_pinned = True
    return worker


def patch_npu_events(monkeypatch, *, record_error=False):
    npu = FakeNPU(record_error=record_error)
    monkeypatch.setattr(worker_mod.torch, "npu", npu, raising=False)
    return npu


def make_worker_env(monkeypatch, dtype=torch.float32, *, record_error=False):
    worker = make_worker(dtype=dtype)
    platform = FakePlatform()
    npu = patch_npu_events(monkeypatch, record_error=record_error)
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    return worker, platform, npu


def patch_copy_capture(monkeypatch):
    calls = []

    def capture_copy(src_ptrs, dst_ptrs, sizes, direction):
        calls.append(
            (
                src_ptrs.clone(),
                dst_ptrs.clone(),
                sizes.clone(),
                direction,
            )
        )

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", capture_copy)
    return calls


def raises(message):
    def raise_error(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(message)

    return raise_error


def save_one_block(worker, source=None):
    source = torch.arange(16, dtype=torch.int8) if source is None else source
    metadata = ECCPUConnectorMetadata(saves={"hash": [0]})
    worker.save_caches({"hash": source}, "hash", metadata)
    return source, metadata


def patch_init_dependencies(
    monkeypatch,
    *,
    supported=True,
    register_error=None,
):
    region = FakeRegion()
    platform = FakePlatform()
    register_calls = []
    pcp_group = SimpleNamespace(rank_in_group=0)

    monkeypatch.setattr(worker_mod, "create_ascend_ec_shared_region", lambda cfg: region)
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(worker_mod, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(worker_mod, "_supports_eccpu_offload", lambda: supported)
    monkeypatch.setattr(worker_mod, "current_platform", platform)

    def register(blocks):
        register_calls.append(blocks)
        if register_error is not None:
            raise register_error

    monkeypatch.setattr(worker_mod, "_register_pinned_host_mmap", register)
    return region, platform, register_calls


@pytest.mark.parametrize(
    ("block_ids", "expected"),
    [
        ([], []),
        ([4], [(0, 4, 1)]),
        ([3, 4, 5], [(0, 3, 3)]),
        ([3, 4, 7, 8, 2], [(0, 3, 2), (2, 7, 2), (4, 2, 1)]),
    ],
)
def test_iter_contiguous_block_runs(block_ids, expected):
    assert list(worker_mod._iter_contiguous_block_runs(block_ids)) == expected


def test_init_registers_mmap_after_creating_load_stream(monkeypatch):
    region, platform, register_calls = patch_init_dependencies(monkeypatch)
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    worker = AscendECCPUWorker(config)

    assert worker._region is region
    assert worker._load_stream is platform.created_streams[0]
    assert register_calls == [region.blocks]
    assert worker._mmap_pinned is True
    assert worker._is_save_rank is True
    assert region.cleanup_calls == 0


def test_init_rejects_unsupported_runtime_and_cleans_region(monkeypatch):
    region, platform, register_calls = patch_init_dependencies(monkeypatch, supported=False)
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="aclrtMemcpyBatchAsync"):
        AscendECCPUWorker(config)

    assert platform.created_streams == []
    assert register_calls == []
    assert region.cleanup_calls == 1


def test_init_registration_failure_cleans_region(monkeypatch):
    region, platform, register_calls = patch_init_dependencies(
        monkeypatch, register_error=RuntimeError("registration failed")
    )
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="registration failed"):
        AscendECCPUWorker(config)

    assert len(platform.created_streams) == 1
    assert register_calls == [region.blocks]
    assert region.cleanup_calls == 1


def test_save_and_flush_builds_d2h_descriptors(monkeypatch):
    worker, platform, npu = make_worker_env(monkeypatch, dtype=torch.int8)
    calls = patch_copy_capture(monkeypatch)

    src = torch.arange(20, dtype=torch.int8)
    metadata = ECCPUConnectorMetadata(saves={"hash": [3, 1]})
    worker.save_caches({"hash": src}, "hash", metadata)
    worker.flush_saves()

    src_ptrs, dst_ptrs, sizes, direction = calls[0]
    assert src_ptrs.tolist() == [src.data_ptr(), src.data_ptr() + 16]
    assert dst_ptrs.tolist() == [
        worker._region.blocks.data_ptr() + 3 * 16,
        worker._region.blocks.data_ptr() + 16,
    ]
    assert sizes.tolist() == [16, 4]
    assert direction == worker_mod._DIRECTION_D2H
    assert worker._save_count == 0
    assert worker._save_bufs is None
    assert len(worker._inflight_descriptor_bufs) == 1
    assert npu.events[0].recorded_stream is platform.compute_stream
    assert platform.compute_stream.synchronize_calls == 0


def test_descriptor_reuse_waits_for_event_completion(monkeypatch):
    worker, _, npu = make_worker_env(monkeypatch, dtype=torch.int8)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: None)

    source, metadata = save_one_block(worker)
    first_bufs = worker._save_bufs
    worker.flush_saves()

    worker.save_caches({"hash": source}, "hash", metadata)
    second_bufs = worker._save_bufs
    assert second_bufs is not first_bufs
    assert npu.events[0].completed is False
    assert len(worker._buf_pool._pool) == 0
    worker.flush_saves()

    npu.events[0].completed = True
    worker.save_caches({"hash": source}, "hash", metadata)

    assert worker._save_bufs is first_bufs
    assert worker._inflight_descriptor_bufs == [(npu.events[1], second_bufs)]


def test_event_record_failure_falls_back_to_synchronous_d2h_release(
    monkeypatch,
):
    worker, platform, npu = make_worker_env(monkeypatch, dtype=torch.int8, record_error=True)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: None)

    save_one_block(worker)
    worker.flush_saves()

    assert platform.compute_stream.synchronize_calls == 1
    assert len(worker._buf_pool._pool) == 1
    assert worker._inflight_descriptor_bufs == []
    assert worker._save_bufs is None
    assert worker._save_count == 0
    assert npu.events[0].recorded_stream is platform.compute_stream


def test_d2h_event_record_and_sync_failure_propagates_sync_error(monkeypatch):
    worker, platform, _ = make_worker_env(monkeypatch, dtype=torch.int8, record_error=True)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: None)

    save_one_block(worker)
    original_bufs = worker._save_bufs

    monkeypatch.setattr(
        platform.compute_stream,
        "synchronize",
        raises("stream synchronize failed"),
    )
    with pytest.raises(RuntimeError, match="stream synchronize failed") as exc_info:
        worker.flush_saves()

    assert isinstance(exc_info.value.__context__, RuntimeError)
    assert str(exc_info.value.__context__) == "event record failed"
    assert worker._save_bufs is None
    assert worker._save_count == 0
    assert original_bufs not in worker._buf_pool._pool


def test_d2h_sync_failure_is_propagated_without_latching_worker(monkeypatch):
    worker, platform, _ = make_worker_env(monkeypatch, dtype=torch.int8)

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", raises("copy failed"))
    source, _ = save_one_block(worker)
    original_bufs = worker._save_bufs

    monkeypatch.setattr(
        platform.compute_stream,
        "synchronize",
        raises("stream synchronize failed"),
    )
    with pytest.raises(RuntimeError, match="stream synchronize failed"):
        worker.flush_saves()

    assert worker._save_bufs is None
    assert worker._save_count == 0
    assert len(worker._buf_pool._pool) == 0
    assert original_bufs not in worker._buf_pool._pool

    worker.save_caches(
        {"hash": source},
        "hash",
        ECCPUConnectorMetadata(saves={"hash": [0]}),
    )
    assert worker._save_bufs is not None


def test_reclaim_preserves_unprocessed_descriptors_when_query_fails():
    worker = make_worker()
    first_bufs = worker._buf_pool.acquire(1)
    second_bufs = worker._buf_pool.acquire(1)
    completed_event = FakeEvent()
    completed_event.completed = True
    failing_event = FakeEvent(query_error=True)
    worker._inflight_descriptor_bufs = [
        (completed_event, first_bufs),
        (failing_event, second_bufs),
    ]

    with pytest.raises(RuntimeError, match="event query failed"):
        worker._reclaim_completed_descriptor_bufs()

    assert worker._buf_pool._pool == [first_bufs]
    assert worker._inflight_descriptor_bufs == [(failing_event, second_bufs)]

    failing_event.query_error = False
    failing_event.completed = True
    worker._reclaim_completed_descriptor_bufs()
    assert len(worker._buf_pool._pool) == 2
    assert worker._inflight_descriptor_bufs == []


def test_save_coalesces_contiguous_blocks_into_one_descriptor(monkeypatch):
    worker, _, _ = make_worker_env(monkeypatch, dtype=torch.int8)
    calls = patch_copy_capture(monkeypatch)

    src = torch.arange(40, dtype=torch.int8)
    metadata = ECCPUConnectorMetadata(saves={"hash": [2, 3, 4]})
    worker.save_caches({"hash": src}, "hash", metadata)
    worker.flush_saves()

    src_ptrs, dst_ptrs, sizes, direction = calls[0]
    assert src_ptrs.tolist() == [src.data_ptr()]
    assert dst_ptrs.tolist() == [worker._region.blocks.data_ptr() + 2 * 16]
    assert sizes.tolist() == [40]
    assert direction == worker_mod._DIRECTION_D2H


def test_save_descriptor_buffer_capacity_uses_block_count():
    worker = make_worker(dtype=torch.int8)
    metadata = ECCPUConnectorMetadata(saves={"first": [0, 1, 4], "second": [6, 7]})

    worker.save_caches({"first": torch.arange(48, dtype=torch.int8)}, "first", metadata)

    assert worker._save_bufs is not None
    assert worker._save_bufs.src_ptrs.numel() == 5
    assert worker._save_count == 2

    worker.save_caches({"second": torch.arange(32, dtype=torch.int8)}, "second", metadata)

    assert worker._save_bufs.src_ptrs.numel() == 5
    assert worker._save_count == 3


@pytest.mark.parametrize(
    ("is_save_rank", "cache_hash", "metadata"),
    [
        (True, "missing", ECCPUConnectorMetadata()),
        (False, "hash", ECCPUConnectorMetadata(saves={"hash": [0]})),
    ],
    ids=["not-allocated", "non-save-rank"],
)
def test_save_early_exits_without_reading_encoder_cache(monkeypatch, is_save_rank, cache_hash, metadata):
    worker = make_worker(dtype=torch.int8)
    worker._is_save_rank = is_save_rank
    calls: list[object] = []
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", calls.append)

    worker.save_caches({}, cache_hash, metadata)
    worker.flush_saves()

    assert calls == []
    assert worker._save_count == 0
    assert worker._save_bufs is None


def test_save_rejects_non_contiguous_encoder_cache():
    worker = make_worker(dtype=torch.int8)
    source = torch.arange(32, dtype=torch.int8).reshape(4, 8).t()
    metadata = ECCPUConnectorMetadata(saves={"hash": [0, 1]})

    with pytest.raises(RuntimeError, match="Non-contiguous"):
        worker.save_caches({"hash": source}, "hash", metadata)


@pytest.mark.parametrize("block_ids", [[0], [0, 1, 2]])
def test_save_rejects_mismatched_block_count(block_ids):
    worker = make_worker(dtype=torch.int8)
    source = torch.arange(20, dtype=torch.int8)
    metadata = ECCPUConnectorMetadata(saves={"hash": block_ids})

    with pytest.raises(AssertionError, match="block count mismatch"):
        worker.save_caches({"hash": source}, "hash", metadata)


def test_flush_releases_descriptors_when_dma_submission_fails(monkeypatch):
    worker, platform, _ = make_worker_env(monkeypatch, dtype=torch.int8)
    save_one_block(worker)

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", raises("D2H submission failed"))

    with pytest.raises(RuntimeError, match="D2H submission failed"):
        worker.flush_saves()

    assert worker._save_count == 0
    assert worker._save_bufs is None
    assert len(worker._buf_pool._pool) == 1
    assert platform.compute_stream.synchronize_calls == 1


def test_load_builds_h2d_descriptors_and_waits(monkeypatch):
    worker, platform, npu = make_worker_env(monkeypatch)
    calls = patch_copy_capture(monkeypatch)

    existing = torch.zeros((1, 4), dtype=torch.float32)
    encoder_cache = {"cached": existing}
    metadata = ECCPUConnectorMetadata(loads={"loaded": [7, 2], "cached": [1]})
    worker.start_load_caches(encoder_cache, metadata)

    loaded = encoder_cache["loaded"]
    src_ptrs, dst_ptrs, sizes, direction = calls[0]
    assert loaded.shape == (2, 4)
    assert loaded.dtype == torch.float32
    assert encoder_cache["cached"] is existing
    assert src_ptrs.tolist() == [
        worker._region.blocks.data_ptr() + 7 * 16,
        worker._region.blocks.data_ptr() + 2 * 16,
    ]
    assert dst_ptrs.tolist() == [loaded.data_ptr(), loaded.data_ptr() + 16]
    assert sizes.tolist() == [16, 16]
    assert direction == worker_mod._DIRECTION_H2D
    assert platform.compute_stream.waited_streams == [worker._load_stream]
    assert len(worker._inflight_descriptor_bufs) == 1
    assert npu.events[0].recorded_stream is worker._load_stream
    assert worker._load_stream.synchronize_calls == 0


def test_load_coalesces_contiguous_blocks_into_one_descriptor(monkeypatch):
    worker, platform, _ = make_worker_env(monkeypatch)
    calls = patch_copy_capture(monkeypatch)

    encoder_cache: dict[str, torch.Tensor] = {}
    metadata = ECCPUConnectorMetadata(loads={"loaded": [2, 3, 4]})
    worker.start_load_caches(encoder_cache, metadata)

    loaded = encoder_cache["loaded"]
    src_ptrs, dst_ptrs, sizes, direction = calls[0]
    assert loaded.shape == (3, 4)
    assert src_ptrs.tolist() == [worker._region.blocks.data_ptr() + 2 * 16]
    assert dst_ptrs.tolist() == [loaded.data_ptr()]
    assert sizes.tolist() == [3 * 16]
    assert direction == worker_mod._DIRECTION_H2D
    assert platform.compute_stream.waited_streams == [worker._load_stream]


def test_load_coalesces_each_cache_item_without_changing_output_offsets(
    monkeypatch,
):
    worker, _, _ = make_worker_env(monkeypatch)
    calls = patch_copy_capture(monkeypatch)

    encoder_cache: dict[str, torch.Tensor] = {}
    metadata = ECCPUConnectorMetadata(loads={"first": [0, 1], "second": [4, 5]})
    worker.start_load_caches(encoder_cache, metadata)

    first = encoder_cache["first"]
    second = encoder_cache["second"]
    src_ptrs, dst_ptrs, sizes, direction = calls[0]
    assert first.shape == second.shape == (2, 4)
    assert src_ptrs.tolist() == [
        worker._region.blocks.data_ptr(),
        worker._region.blocks.data_ptr() + 4 * 16,
    ]
    assert dst_ptrs.tolist() == [first.data_ptr(), second.data_ptr()]
    assert sizes.tolist() == [2 * 16, 2 * 16]
    assert direction == worker_mod._DIRECTION_H2D


def test_load_descriptor_buffer_capacity_uses_block_count(monkeypatch):
    worker, _, _ = make_worker_env(monkeypatch)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: None)
    encoder_cache: dict[str, torch.Tensor] = {}
    metadata = ECCPUConnectorMetadata(loads={"first": [0, 1, 4], "second": [6, 7]})

    worker.start_load_caches(encoder_cache, metadata)

    assert encoder_cache["first"].shape == (3, 4)
    assert encoder_cache["second"].shape == (2, 4)
    assert len(worker._buf_pool._pool) == 0
    assert len(worker._inflight_descriptor_bufs) == 1
    assert worker._inflight_descriptor_bufs[0][1].src_ptrs.numel() == 5


def test_load_is_noop_when_all_hashes_are_already_cached(monkeypatch):
    worker = make_worker()
    platform = FakePlatform()
    monkeypatch.setattr(worker_mod, "current_platform", platform)
    calls: list[object] = []
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", calls.append)
    existing = torch.zeros((1, 4), dtype=torch.float32)
    encoder_cache = {"cached": existing}
    metadata = ECCPUConnectorMetadata(loads={"cached": [1]})

    worker.start_load_caches(encoder_cache, metadata)

    assert encoder_cache == {"cached": existing}
    assert calls == []
    assert platform.compute_stream.waited_streams == []


def test_load_releases_descriptors_when_dma_submission_fails(monkeypatch):
    worker, platform, _ = make_worker_env(monkeypatch)

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", raises("H2D submission failed"))
    encoder_cache: dict[str, torch.Tensor] = {}
    metadata = ECCPUConnectorMetadata(loads={"hash": [1]})

    with pytest.raises(RuntimeError, match="H2D submission failed"):
        worker.start_load_caches(encoder_cache, metadata)

    assert encoder_cache == {}
    assert len(worker._buf_pool._pool) == 1
    assert worker._load_stream.synchronize_calls == 1
    assert platform.compute_stream.waited_streams == []


def test_h2d_sync_failure_is_propagated_without_latching_worker(monkeypatch):
    worker, _, _ = make_worker_env(monkeypatch)
    copy_calls = 0

    def fail_copy(*args):
        nonlocal copy_calls
        del args
        copy_calls += 1
        raise RuntimeError("H2D submission failed")

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", fail_copy)
    monkeypatch.setattr(
        worker._load_stream,
        "synchronize",
        raises("load stream synchronize failed"),
    )
    metadata = ECCPUConnectorMetadata(loads={"hash": [1]})

    with pytest.raises(RuntimeError, match="load stream synchronize failed"):
        worker.start_load_caches({}, metadata)

    assert copy_calls == 1
    with pytest.raises(RuntimeError, match="load stream synchronize failed"):
        worker.start_load_caches({}, metadata)

    assert copy_calls == 2


def test_event_record_failure_falls_back_to_synchronous_h2d_release(
    monkeypatch,
):
    worker, platform, npu = make_worker_env(monkeypatch, record_error=True)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: None)

    encoder_cache: dict[str, torch.Tensor] = {}
    worker.start_load_caches(encoder_cache, ECCPUConnectorMetadata(loads={"hash": [1]}))

    assert worker._load_stream.synchronize_calls == 1
    assert len(worker._buf_pool._pool) == 1
    assert worker._inflight_descriptor_bufs == []
    assert npu.events[0].recorded_stream is worker._load_stream
    assert encoder_cache["hash"].shape == (1, 4)
    assert platform.compute_stream.waited_streams == [worker._load_stream]


def test_h2d_event_record_and_sync_failure_propagates_sync_error(monkeypatch):
    worker, platform, _ = make_worker_env(monkeypatch, record_error=True)
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: None)

    monkeypatch.setattr(
        worker._load_stream,
        "synchronize",
        raises("load stream synchronize failed"),
    )
    with pytest.raises(RuntimeError, match="load stream synchronize failed") as exc_info:
        worker.start_load_caches({}, ECCPUConnectorMetadata(loads={"hash": [1]}))

    assert isinstance(exc_info.value.__context__, RuntimeError)
    assert str(exc_info.value.__context__) == "event record failed"
    assert len(worker._buf_pool._pool) == 0
    assert worker._inflight_descriptor_bufs == []
    assert platform.compute_stream.waited_streams == []


@pytest.mark.parametrize("block_ids", [[], [1, 1], [-1], [8]])
def test_invalid_block_ids_are_rejected(block_ids):
    worker = make_worker()
    metadata = ECCPUConnectorMetadata(loads={"hash": block_ids})

    with pytest.raises(RuntimeError):
        worker.start_load_caches({}, metadata)


def test_shutdown_synchronizes_releases_and_unregisters_before_cleanup(
    monkeypatch,
):
    worker, _, npu = make_worker_env(monkeypatch, dtype=torch.int8)
    order: list[str | tuple[str, object]] = []
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", lambda *args: None)
    worker._region.cleanup = lambda: order.append("cleanup")

    def synchronize():
        npu.synchronize_calls += 1
        order.append("synchronize")

    npu.synchronize = synchronize
    monkeypatch.setattr(
        worker_mod,
        "_unregister_pinned_host_mmap",
        lambda blocks: order.append(("unregister", blocks.data_ptr())),
    )

    save_one_block(worker)
    worker.flush_saves()
    assert len(worker._buf_pool._pool) == 0

    worker.shutdown()

    assert npu.synchronize_calls == 1
    assert worker._inflight_descriptor_bufs == []
    assert len(worker._buf_pool._pool) == 1
    assert order == [
        "synchronize",
        ("unregister", worker._region.blocks.data_ptr()),
        "cleanup",
    ]
    assert worker._mmap_pinned is False


def test_shutdown_preserves_mmap_when_device_synchronize_fails(monkeypatch):
    worker = make_worker()
    order: list[str | tuple[str, object]] = []
    worker._region.cleanup = lambda: order.append("cleanup")
    monkeypatch.setattr(
        worker_mod,
        "_unregister_pinned_host_mmap",
        lambda blocks: order.append(("unregister", blocks.data_ptr())),
    )

    monkeypatch.setattr(
        worker_mod.torch,
        "npu",
        SimpleNamespace(synchronize=raises("stream failed")),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        worker.shutdown()

    assert order == []
    assert worker._mmap_pinned is True


def test_shutdown_unregister_failure_preserves_mmap_and_allows_retry(monkeypatch):
    worker = make_worker()
    order = []
    worker._save_bufs = worker._buf_pool.acquire(1)
    worker._save_count = 1
    worker._region.cleanup = lambda: order.append("cleanup")
    monkeypatch.setattr(
        worker_mod.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: order.append("synchronize")),
        raising=False,
    )
    unregister_calls = 0

    def fail_once(blocks):
        del blocks
        nonlocal unregister_calls
        unregister_calls += 1
        order.append("unregister")
        if unregister_calls == 1:
            raise RuntimeError("unregister failed")

    monkeypatch.setattr(worker_mod, "_unregister_pinned_host_mmap", fail_once)

    with pytest.raises(RuntimeError, match="unregister failed"):
        worker.shutdown()

    assert worker._save_bufs is None
    assert worker._save_count == 0
    assert worker._inflight_descriptor_bufs == []
    assert len(worker._buf_pool._pool) == 1
    assert worker._mmap_pinned is True
    assert order == ["synchronize", "unregister"]

    worker.shutdown()
    assert unregister_calls == 2
    assert len(worker._buf_pool._pool) == 1
    assert worker._mmap_pinned is False
    assert order == [
        "synchronize",
        "unregister",
        "synchronize",
        "unregister",
        "cleanup",
    ]
