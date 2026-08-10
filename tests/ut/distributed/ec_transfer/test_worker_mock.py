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
    worker._load_stream = FakeStream()
    worker._mmap_pinned = True
    return worker


def patch_init_dependencies(
    monkeypatch,
    *,
    device_type=worker_mod.AscendDeviceType.A2,
    supported=True,
    register_error=None,
):
    region = FakeRegion()
    platform = FakePlatform()
    register_calls = []
    pcp_group = SimpleNamespace(rank_in_group=0)

    monkeypatch.setattr(worker_mod, "create_ec_shared_region", lambda cfg: region)
    monkeypatch.setattr(worker_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(worker_mod, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(worker_mod, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(worker_mod, "_supports_eccpu_offload", lambda: supported)
    monkeypatch.setattr(worker_mod, "current_platform", platform)

    def register(blocks):
        register_calls.append(blocks)
        if register_error is not None:
            raise register_error

    monkeypatch.setattr(worker_mod, "_register_pinned_host_mmap", register)
    return region, platform, register_calls


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


def test_init_rejects_unsupported_device_and_cleans_region(monkeypatch):
    region, platform, register_calls = patch_init_dependencies(monkeypatch, device_type=worker_mod.AscendDeviceType.A5)
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))

    with pytest.raises(RuntimeError, match="supports only A2/A3"):
        AscendECCPUWorker(config)

    assert platform.created_streams == []
    assert register_calls == []
    assert region.cleanup_calls == 1


def test_init_rejects_missing_cann_capability_and_cleans_region(monkeypatch):
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
    worker = make_worker(dtype=torch.int8)
    platform = FakePlatform()
    monkeypatch.setattr(worker_mod, "current_platform", platform)

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


def test_save_is_noop_without_allocated_metadata(monkeypatch):
    worker = make_worker(dtype=torch.int8)
    calls: list[object] = []
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", calls.append)

    worker.save_caches({}, "missing", ECCPUConnectorMetadata())
    worker.flush_saves()

    assert calls == []
    assert worker._save_count == 0
    assert worker._save_bufs is None


def test_non_save_rank_does_not_read_encoder_cache(monkeypatch):
    worker = make_worker(dtype=torch.int8)
    worker._is_save_rank = False
    calls: list[object] = []
    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", calls.append)
    metadata = ECCPUConnectorMetadata(saves={"hash": [0]})

    worker.save_caches({}, "hash", metadata)
    worker.flush_saves()

    assert calls == []
    assert worker._save_count == 0


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
    worker = make_worker(dtype=torch.int8)
    source = torch.arange(16, dtype=torch.int8)
    metadata = ECCPUConnectorMetadata(saves={"hash": [0]})
    worker.save_caches({"hash": source}, "hash", metadata)

    def fail_copy(*args):
        raise RuntimeError("D2H submission failed")

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", fail_copy)

    with pytest.raises(RuntimeError, match="D2H submission failed"):
        worker.flush_saves()

    assert worker._save_count == 0
    assert worker._save_bufs is None
    assert len(worker._buf_pool._pool) == 1


def test_load_builds_h2d_descriptors_and_waits(monkeypatch):
    worker = make_worker(dtype=torch.float32)
    platform = FakePlatform()
    monkeypatch.setattr(worker_mod, "current_platform", platform)

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


def test_load_is_noop_when_all_hashes_are_already_cached(monkeypatch):
    worker = make_worker(dtype=torch.float32)
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
    worker = make_worker(dtype=torch.float32)
    platform = FakePlatform()
    monkeypatch.setattr(worker_mod, "current_platform", platform)

    def fail_copy(*args):
        raise RuntimeError("H2D submission failed")

    monkeypatch.setattr(worker_mod, "_swap_blocks_batch", fail_copy)
    encoder_cache: dict[str, torch.Tensor] = {}
    metadata = ECCPUConnectorMetadata(loads={"hash": [1]})

    with pytest.raises(RuntimeError, match="H2D submission failed"):
        worker.start_load_caches(encoder_cache, metadata)

    assert encoder_cache == {}
    assert len(worker._buf_pool._pool) == 1
    assert platform.compute_stream.waited_streams == []


@pytest.mark.parametrize("block_ids", [[], [1, 1], [-1], [8]])
def test_invalid_block_ids_are_rejected(block_ids):
    worker = make_worker()
    metadata = ECCPUConnectorMetadata(loads={"hash": block_ids})

    with pytest.raises(RuntimeError):
        worker.start_load_caches({}, metadata)


def test_shutdown_unregisters_before_mmap_cleanup(monkeypatch):
    worker = make_worker()
    order: list[str | tuple[str, int]] = []
    worker._region.cleanup = lambda: order.append("cleanup")
    monkeypatch.setattr(
        worker_mod.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: order.append("synchronize")),
        raising=False,
    )
    monkeypatch.setattr(
        worker_mod,
        "_unregister_pinned_host_mmap",
        lambda blocks: order.append(("unregister", blocks.data_ptr())),
    )

    worker.shutdown()

    assert order == [
        "synchronize",
        ("unregister", worker._region.blocks.data_ptr()),
        "cleanup",
    ]
    assert worker._mmap_pinned is False


def test_shutdown_preserves_mmap_when_device_synchronize_fails(monkeypatch):
    worker = make_worker()
    order: list[str | tuple[str, int]] = []
    worker._region.cleanup = lambda: order.append("cleanup")
    monkeypatch.setattr(
        worker_mod,
        "_unregister_pinned_host_mmap",
        lambda blocks: order.append(("unregister", blocks.data_ptr())),
    )

    def fail_synchronize():
        raise RuntimeError("stream failed")

    monkeypatch.setattr(
        worker_mod.torch,
        "npu",
        SimpleNamespace(synchronize=fail_synchronize),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="stream failed"):
        worker.shutdown()

    assert order == []
    assert worker._mmap_pinned is True


def test_shutdown_preserves_mmap_when_unregister_fails(monkeypatch):
    worker = make_worker()
    order = []
    worker._region.cleanup = lambda: order.append("cleanup")
    monkeypatch.setattr(
        worker_mod.torch,
        "npu",
        SimpleNamespace(synchronize=lambda: order.append("synchronize")),
        raising=False,
    )

    def fail_unregister(blocks):
        del blocks
        order.append("unregister")
        raise RuntimeError("unregister failed")

    monkeypatch.setattr(worker_mod, "_unregister_pinned_host_mmap", fail_unregister)

    with pytest.raises(RuntimeError, match="unregister failed"):
        worker.shutdown()

    assert order == ["synchronize", "unregister"]
    assert worker._mmap_pinned is True
