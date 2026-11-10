# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Shared host registration lifecycle; ACL operations are mocked on CPU."""

import gc
import os
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
import torch
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu as cpu_npu_mod
import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu as npu_mod


@pytest.fixture
def pin_ops(monkeypatch):
    ops = SimpleNamespace(
        supports_eccpu_offload=Mock(return_value=True),
        register_pinned_host_mmap=Mock(),
        unregister_pinned_host_mmap=Mock(),
    )
    monkeypatch.setattr(torch.ops, "_C_ascend", ops)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)
    yield ops
    assert not getattr(cpu_npu_mod, "_PINNED_MMAP_REGISTRATIONS", ())


def _worker(region):
    worker = cpu_npu_mod.NPUOffloadingWorker.__new__(cpu_npu_mod.NPUOffloadingWorker)
    worker._mmap_region = region
    worker._mmap_pinned = False
    worker._pinned_tensors = []
    worker._pinned_registration = None
    return worker


@pytest.mark.parametrize("slot_only", [False, True])
def test_registers_real_mmap_ranges_without_copy(pin_ops, slot_only):
    # Four workers, four chunks; slot registration must cover exactly rank 2.
    page_size = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    region = SharedOffloadRegion(
        engine_id=f"pin-ut-{uuid4().hex}",
        num_chunks=4,
        rank=2,
        kv_bytes_per_chunk=4 * page_size,
        cpu_page_size=page_size,
    )
    worker = _worker(region)
    use_slots = slot_only and hasattr(region, "iter_worker_slot_ranges")
    tensors = []
    try:
        worker._pin_shared_region(True, slot_only)
        tensors = [call.args[0] for call in pin_ops.register_pinned_host_mmap.call_args_list]
        assert worker._mmap_pinned
        expected_offsets = [row * 4 * page_size + 2 * page_size for row in range(4)] if use_slots else [0]
        assert [t.data_ptr() - region.base_tensor.data_ptr() for t in tensors] == expected_offsets
        assert [t.numel() for t in tensors] == ([page_size] * 4 if use_slots else [16 * page_size])
        # Registration uses mmap views: writes remain visible in the same region.
        tensors[0][0] = 37
        assert region.base_tensor[expected_offsets[0]] == 37
        pointers = [t.data_ptr() for t in tensors]
        tensors.clear()
        pin_ops.register_pinned_host_mmap.reset_mock()
        worker._unpin_shared_region()
        assert [c.args[0].data_ptr() for c in pin_ops.unregister_pinned_host_mmap.call_args_list] == pointers[::-1]
        pin_ops.unregister_pinned_host_mmap.reset_mock()
        assert not worker._mmap_pinned
        assert not worker._pinned_tensors
        worker._unpin_shared_region()
        torch.npu.synchronize.assert_called_once()
    finally:
        tensors.clear()
        worker._unpin_shared_region()
        pin_ops.register_pinned_host_mmap.reset_mock()
        pin_ops.unregister_pinned_host_mmap.reset_mock()
        region.cleanup()


@pytest.mark.parametrize("region,enabled", [(None, True), (object(), False)])
def test_disabled_or_non_shared_worker_does_not_register(pin_ops, region, enabled):
    worker = _worker(region)
    worker._pin_shared_region(enabled, True)
    pin_ops.supports_eccpu_offload.assert_not_called()
    pin_ops.register_pinned_host_mmap.assert_not_called()


@pytest.mark.parametrize("support", [False, AttributeError("old extension"), RuntimeError("unavailable")])
def test_missing_capability_keeps_pageable_mmap(pin_ops, support):
    if isinstance(support, Exception):
        pin_ops.supports_eccpu_offload.side_effect = support
    else:
        pin_ops.supports_eccpu_offload.return_value = support
    worker = _worker(object())
    worker._pin_shared_region(True, False)
    assert not worker._mmap_pinned
    pin_ops.register_pinned_host_mmap.assert_not_called()


@pytest.mark.parametrize("missing_ranges", [False, True])
def test_slot_api_failure_falls_back_to_whole_region(pin_ops, missing_ranges):
    region = SimpleNamespace(base_tensor=torch.empty(32, dtype=torch.int8), rank=0)
    if not missing_ranges:
        region.iter_worker_slot_ranges = Mock(side_effect=RuntimeError("bad ranges"))
    worker = _worker(region)
    worker._pin_shared_region(True, True)
    assert pin_ops.register_pinned_host_mmap.call_args.args[0] is region.base_tensor
    worker._unpin_shared_region()


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_partial_registration_rolls_back_in_reverse_order(pin_ops, failure_index):
    region = SimpleNamespace(
        base_tensor=torch.empty(24, dtype=torch.int8),
        rank=0,
        iter_worker_slot_ranges=lambda: [(0, 8), (8, 8), (16, 8)],
    )
    pin_ops.register_pinned_host_mmap.side_effect = [None] * failure_index + [RuntimeError("registration limit")]
    worker = _worker(region)
    worker._pin_shared_region(True, True)
    registered = pin_ops.register_pinned_host_mmap.call_args_list[:failure_index]
    assert [c.args[0].data_ptr() for c in pin_ops.unregister_pinned_host_mmap.call_args_list] == [
        c.args[0].data_ptr() for c in reversed(registered)
    ]
    assert not worker._mmap_pinned
    assert not worker._pinned_tensors


@pytest.mark.parametrize("failure", ["synchronize", "unregister"])
def test_shutdown_retains_mapping_until_unpin_can_succeed(pin_ops, failure):
    region = SimpleNamespace(base_tensor=torch.empty(8, dtype=torch.int8), cleanup=Mock())
    worker = _worker(region)
    worker._store_handler = SimpleNamespace(shutdown=Mock())
    worker._load_handler = SimpleNamespace(shutdown=Mock())
    worker._pin_shared_region(True, False)
    failing_call = torch.npu.synchronize if failure == "synchronize" else pin_ops.unregister_pinned_host_mmap
    failing_call.side_effect = RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        worker.shutdown()
    region.cleanup.assert_not_called()
    assert worker._mmap_pinned
    assert len(worker._pinned_tensors) == 1
    failing_call.side_effect = None
    worker.shutdown()
    region.cleanup.assert_called_once()
    assert worker._mmap_region is None
    assert not worker._pinned_tensors


@pytest.mark.parametrize("handler_failure", [False, True])
def test_worker_constructor_and_shutdown_release_in_order(monkeypatch, pin_ops, handler_failure):
    events = []
    region = SimpleNamespace(
        base_tensor=torch.empty(8, dtype=torch.int8),
        create_next_worker_view=lambda size: torch.empty((2, size), dtype=torch.int8),
        cleanup=lambda: events.append("cleanup"),
    )
    pin_ops.register_pinned_host_mmap.side_effect = lambda tensor: events.append("pin")
    pin_ops.unregister_pinned_host_mmap.side_effect = lambda tensor: events.append("unpin")
    torch.npu.synchronize.side_effect = lambda: events.append("sync")

    def handler(**kwargs):
        if not kwargs["npu_to_cpu"] and handler_failure:
            raise RuntimeError("load handler failed")
        return SimpleNamespace(shutdown=lambda: events.append("handler"))

    monkeypatch.setattr(cpu_npu_mod, "SingleDirectionNPUOffloadingHandler", handler)
    monkeypatch.setattr(cpu_npu_mod, "is_pin_memory_available", lambda: False)
    kv_caches = SimpleNamespace(
        tensors=[SimpleNamespace(tensor=torch.empty((4, 4), dtype=torch.int8), page_size_bytes=4)],
        group_data_refs=[],
    )
    kwargs: dict[str, Any] = dict(kv_caches=kv_caches, blocks_per_chunk=2, num_cpu_blocks=2, mmap_region=region)
    if handler_failure:
        with pytest.raises(RuntimeError, match="load handler failed"):
            cpu_npu_mod.NPUOffloadingWorker(**kwargs)
        assert events == ["pin", "handler", "sync", "unpin", "cleanup"]
    else:
        worker = cpu_npu_mod.NPUOffloadingWorker(**kwargs)
        worker.shutdown()
        assert events == ["pin", "handler", "handler", "sync", "unpin", "cleanup"]


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize(
    "extra,pin_enabled",
    [
        ({}, True),
        ({"pin_shared_mmap": False}, False),
        ({"pin_shared_mmap": True}, True),
        ({"pin_shared_mmap": "false"}, False),
        ({"pin_shared_mmap": "0"}, False),
        ({"pin_shared_mmap": "NO"}, False),
        ({"pin_shared_mmap": "true"}, True),
    ],
)
def test_tiering_chooses_pinning_scope_from_layout(monkeypatch, replicated, extra, pin_enabled):
    monkeypatch.setattr(npu_mod.cpu_spec, "_all_workers_barrier", Mock())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 3), raising=False)
    region = object()
    worker_factory = Mock(return_value=object())
    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", Mock(return_value=region))
    monkeypatch.setattr(npu_mod, "NPUOffloadingWorker", worker_factory)
    spec = npu_mod.NPUTieringOffloadingSpec.__new__(npu_mod.NPUTieringOffloadingSpec)
    spec.config = SimpleNamespace(parallel=SimpleNamespace(world_size=2, rank=1))
    spec.extra_config = extra
    spec.replicated_layout = replicated
    spec._engine_id = "pin-options"
    spec.cpu_page_size_per_worker = 4096
    spec.kv_bytes_per_chunk = 8192
    spec.num_chunks = 2
    spec.blocks_per_chunk = 2
    spec.create_worker(object())
    kwargs = worker_factory.call_args.kwargs
    assert kwargs["pin_shared_mmap"] is pin_enabled
    assert kwargs["worker_slot_only"] is (not replicated)
    assert kwargs["mmap_region"] is region


@pytest.mark.parametrize("direction", ["store", "load"])
def test_handler_shutdown_failure_still_unpins_and_closes_mapping(pin_ops, direction):
    region = SimpleNamespace(base_tensor=torch.empty(8, dtype=torch.int8), cleanup=Mock())
    worker = _worker(region)
    worker._store_handler = SimpleNamespace(shutdown=Mock())
    worker._load_handler = SimpleNamespace(shutdown=Mock())
    worker._pin_shared_region(True, False)
    getattr(worker, f"_{direction}_handler").shutdown.side_effect = RuntimeError("handler shutdown")
    with pytest.raises(RuntimeError, match="handler shutdown"):
        worker.shutdown()
    worker._store_handler.shutdown.assert_called_once()
    worker._load_handler.shutdown.assert_called_once()
    pin_ops.unregister_pinned_host_mmap.assert_called_once()
    region.cleanup.assert_called_once()
    assert not worker._pinned_tensors


def test_unpin_retry_does_not_unregister_released_ranges_twice(pin_ops):
    region = SimpleNamespace(
        base_tensor=torch.empty(24, dtype=torch.int8),
        rank=0,
        iter_worker_slot_ranges=lambda: [(0, 8), (8, 8), (16, 8)],
    )
    worker = _worker(region)
    worker._pin_shared_region(True, True)
    pointers = [tensor.data_ptr() for tensor in worker._pinned_tensors]
    pin_ops.unregister_pinned_host_mmap.side_effect = [None, RuntimeError("retry")]
    with pytest.raises(RuntimeError, match="retry"):
        worker._unpin_shared_region()
    assert len(worker._pinned_tensors) == 2
    pin_ops.unregister_pinned_host_mmap.side_effect = None
    pin_ops.unregister_pinned_host_mmap.reset_mock()
    worker._unpin_shared_region()
    assert [c.args[0].data_ptr() for c in pin_ops.unregister_pinned_host_mmap.call_args_list] == pointers[1::-1]
    assert not worker._mmap_pinned


@pytest.mark.parametrize("construction_failure", ["registration", "handler"])
@pytest.mark.parametrize("release_failure", ["synchronize", "unregister"])
def test_failed_constructor_keeps_registered_mapping_alive(monkeypatch, pin_ops, construction_failure, release_failure):
    has_slots = hasattr(SharedOffloadRegion, "iter_worker_slot_ranges")
    if construction_failure == "registration" and not has_slots:
        pytest.skip("This vLLM version registers only the whole region, not multiple slots")
    # The simulated ACL table holds addresses only, just like the native table.
    # Mock call histories or a retained exception would hide the lifetime bug.
    registered: set[int] = set()
    unregister_attempts: list[int] = []
    synchronize_attempts: list[bool] = []
    worker_refs: list[weakref.ReferenceType] = []
    release_fails = True

    def register(tensor):
        if construction_failure == "registration" and registered:
            raise RuntimeError("registration failed")
        registered.add(tensor.data_ptr())

    def unregister(tensor):
        unregister_attempts.append(tensor.data_ptr())
        if release_fails and release_failure == "unregister":
            raise RuntimeError("release failed")
        registered.remove(tensor.data_ptr())

    def synchronize():
        synchronize_attempts.append(True)
        if release_fails and release_failure == "synchronize":
            raise RuntimeError("release failed")

    def fail_handler(**kwargs):
        raise RuntimeError("handler failed")

    original_pin = cpu_npu_mod.NPUOffloadingWorker._pin_shared_region

    def track_worker(worker, *args):
        worker_refs.append(weakref.ref(worker))
        return original_pin(worker, *args)

    monkeypatch.setattr(pin_ops, "register_pinned_host_mmap", register)
    monkeypatch.setattr(pin_ops, "unregister_pinned_host_mmap", unregister)
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    monkeypatch.setattr(cpu_npu_mod, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(cpu_npu_mod, "SingleDirectionNPUOffloadingHandler", fail_handler)
    # Pytest logging handlers retain exc_info even after caplog.clear().
    monkeypatch.setattr(cpu_npu_mod.logger, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(cpu_npu_mod.NPUOffloadingWorker, "_pin_shared_region", track_worker)

    def construct():
        page_size = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
        region = SharedOffloadRegion(
            engine_id=f"pin-constructor-ut-{uuid4().hex}",
            num_chunks=3,
            rank=0,
            kv_bytes_per_chunk=2 * page_size,
            cpu_page_size=page_size,
        )
        region_ref = weakref.ref(region)
        mapping_ref = weakref.ref(region.mmap_obj)
        path, fd = region.mmap_path, region.fd
        caches = SimpleNamespace(
            tensors=[SimpleNamespace(tensor=torch.empty((4, 4), dtype=torch.int8), page_size_bytes=4)],
            group_data_refs=[],
        )
        # Catch inside this function so no traceback or local owns the worker
        # or its region when the caller forces garbage collection.
        try:
            cpu_npu_mod.NPUOffloadingWorker(
                kv_caches=caches,
                blocks_per_chunk=2,
                num_cpu_blocks=3,
                mmap_region=region,
                worker_slot_only=True,
            )
        except RuntimeError as error:
            assert str(error) == "release failed"
        else:
            pytest.fail("constructor unexpectedly succeeded")
        return region_ref, mapping_ref, path, fd

    region_ref, mapping_ref, path, fd = construct()
    gc.collect()
    try:
        assert worker_refs and all(ref() is None for ref in worker_refs)
        assert len(registered) == (1 if construction_failure == "registration" or not has_slots else 3)
        expected_attempts = 2 if construction_failure == "registration" else 1
        assert len(synchronize_attempts) == expected_attempts
        assert len(unregister_attempts) == (expected_attempts if release_failure == "unregister" else 0)
        assert region_ref() is not None
        mapping = mapping_ref()
        assert mapping is not None and not mapping.closed
        mapping[0] = 73
        assert mapping[0] == 73
        del mapping
    finally:
        # Recover the orphan via its independent registration owner, then
        # release the real mapping so this test leaves no process-global pins.
        release_fails = False
        owners = [
            owner
            for owner in getattr(cpu_npu_mod, "_PINNED_MMAP_REGISTRATIONS", ())
            if owner.mmap_region is region_ref()
        ]
        for owner in owners:
            owner.unregister()
            owner.mmap_region.cleanup()
        if owners:
            del owner
        owners.clear()
        if region_ref() is None and fd is not None and registered:
            # Only the unfixed implementation loses the owner before cleanup.
            os.close(fd)
            registered.clear()
        Path(path).unlink(missing_ok=True)
    assert not registered
    gc.collect()
    assert region_ref() is None
    assert mapping_ref() is None
