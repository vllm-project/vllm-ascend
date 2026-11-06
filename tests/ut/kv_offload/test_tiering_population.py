# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Tiering population contracts and concurrent real shared-mmap coverage."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock, local
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import torch
from vllm.v1.kv_offload.cpu import shared_offload_region as region_mod

import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu as npu_mod


def _spec(replicated, worker_index, engine_id="population-test"):
    spec = npu_mod.NPUTieringOffloadingSpec.__new__(npu_mod.NPUTieringOffloadingSpec)
    spec.config = SimpleNamespace(parallel=SimpleNamespace(world_size=4, rank=worker_index))
    spec.extra_config = {}
    spec.replicated_layout = replicated
    spec._engine_id = engine_id
    spec.cpu_page_size_per_worker = 4096
    spec.kv_bytes_per_chunk = 16384
    spec.num_chunks = 8
    spec.blocks_per_chunk = 2
    spec._manager = None
    spec._scheduler_mmap = None
    spec.eviction_policy = "lru"
    spec.cache_policy_module_path = None
    spec.kv_events_config = SimpleNamespace(enable_kv_cache_events=False)
    spec.secondary_tier_configs = []
    if hasattr(npu_mod.cpu_spec, "CPUOffloadingInfo"):
        spec.cpu_info = None
    if hasattr(npu_mod.tiering_spec, "DEFAULT_TIER_POLL_INTERVAL_S"):
        spec.tier_poll_interval_s = 0
    return spec


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("worker_index", [0, 1, 3])
@pytest.mark.parametrize("segment_constant", ["REGION_SEGMENTS", "_REGION_SEGMENTS", None])
def test_tiering_matches_upstream_population_api(monkeypatch, replicated, worker_index, segment_constant):
    for name in ("REGION_SEGMENTS", "_REGION_SEGMENTS"):
        monkeypatch.delattr(npu_mod.cpu_spec, name, raising=False)
    if segment_constant is not None:
        monkeypatch.setattr(npu_mod.cpu_spec, segment_constant, 4, raising=False)
    region = object()
    events = []
    region_factory = Mock(side_effect=lambda **kwargs: events.append("region") or region)
    worker_factory = Mock(side_effect=lambda **kwargs: events.append("worker") or object())
    monkeypatch.setattr(npu_mod.cpu_spec, "_all_workers_barrier", lambda: events.append("barrier"))
    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", region_factory)
    monkeypatch.setattr(npu_mod, "NPUOffloadingWorker", worker_factory)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 9), raising=False)
    spec = _spec(replicated, worker_index)
    spec.create_worker(object())
    kwargs = region_factory.call_args.kwargs
    assert "barrier" not in kwargs
    assert events == ["region", "barrier", "worker"]
    assert kwargs["rank"] == (0 if replicated else 1)
    if segment_constant is None:
        # The release version accepts the barrier, but not the new keywords.
        assert "num_segments" not in kwargs
        assert "populate_shard" not in kwargs
    else:
        assert kwargs["num_segments"] == 4
        assert kwargs["populate_shard"] == ((worker_index, 4) if replicated else None)
    assert worker_factory.call_args.kwargs["mmap_region"] is region
    assert worker_factory.call_args.kwargs["worker_slot_only"] is (not replicated)


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("scheduler_first", [False, True])
def test_parallel_population_and_scheduler_share_same_region(monkeypatch, replicated, scheduler_first):
    # Force small segments to exercise the real multi-file mapping with KiBs.
    page_size = region_mod.SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    engine_id = f"populate-ut-{uuid4().hex}"
    context = local()
    gate = Barrier(4, timeout=10)
    lock = Lock()
    populations = defaultdict(list)
    barrier_calls = Counter()
    regions = []
    manager = None
    segment_constant = next(
        (name for name in ("REGION_SEGMENTS", "_REGION_SEGMENTS") if hasattr(npu_mod.cpu_spec, name)),
        None,
    )
    if segment_constant is not None:
        monkeypatch.setattr(npu_mod.cpu_spec, segment_constant, 4)
        monkeypatch.setattr(region_mod, "_MIN_SEGMENT_BYTES", 8 * page_size)
    real_region = region_mod.SharedOffloadRegion
    get_populate = region_mod._get_populate_write_fn

    def tracked_populate(mapping):
        populate = get_populate(mapping)

        def run(mapping, offset, length):
            populate(mapping, offset, length)
            with lock:
                populations[getattr(context, "worker_index", "scheduler")].append((offset, length))

        return run

    def barrier():
        with lock:
            barrier_calls[context.worker_index] += 1
        gate.wait()

    def create_region(**kwargs):
        region = real_region(**kwargs)
        with lock:
            regions.append(region)
        return region

    def create_worker(**kwargs):
        # All workers must finish their population before any starts pinning.
        assert all(populations[index] for index in range(4))
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(region_mod, "_get_populate_write_fn", tracked_populate)
    monkeypatch.setattr(npu_mod.cpu_spec, "_all_workers_barrier", barrier)
    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", create_region)
    monkeypatch.setattr(npu_mod, "NPUOffloadingWorker", create_worker)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: context.worker_index + 8), raising=False)

    def make_spec(index):
        spec = _spec(replicated, index, engine_id)
        spec.cpu_page_size_per_worker = (4 if replicated else 1) * page_size
        spec.kv_bytes_per_chunk = 4 * page_size
        return spec

    def run_worker(index):
        context.worker_index = index
        return make_spec(index).create_worker(object())

    scheduler = make_spec(0)
    try:
        if scheduler_first:
            manager = scheduler.get_manager()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_worker, index) for index in range(4)]
            workers = [future.result(timeout=20) for future in futures]
        assert barrier_calls == {index: 1 for index in range(4)}

        # Only bulk worker population is counted here, excluding the scheduler.
        pages = Counter()
        for index in range(4):
            for offset, length in populations[index]:
                pages.update(range(offset // page_size, (offset + length) // page_size))
            if replicated and segment_constant is not None:
                assert populations[index] == [(index * 8 * page_size, 8 * page_size)]
            elif not replicated:
                assert {
                    page
                    for offset, length in populations[index]
                    for page in range(offset // page_size, (offset + length) // page_size)
                } == set(range(index, 32, 4))
        expected_visits = 4 if replicated and segment_constant is None else 1
        assert pages == {page: expected_visits for page in range(32)}

        # Keep paths so a scheduler constructed after workers can join.
        for region in regions:
            segments = getattr(region, "_segments", [(region.mmap_path, 0, region.total_size_bytes)])
            assert len(segments) == (4 if segment_constant is not None else 1)
            assert all(Path(path).exists() for path, _, _ in segments)
        workers[0].mmap_region.base_tensor[-1] = 91
        if not scheduler_first:
            manager = scheduler.get_manager()
        assert scheduler.get_manager() is manager
        assert len(regions) == 5
        assert int(scheduler._scheduler_mmap.base_tensor[-1]) == 91
        scheduler._scheduler_mmap.base_tensor[0] = 73
        assert all(int(worker.mmap_region.base_tensor[0]) == 73 for worker in workers)
    finally:
        if manager is not None:
            manager.shutdown()
        for region in regions:
            region.cleanup()
    assert not list(Path("/dev/shm").glob(f"vllm_offload_{engine_id}.mmap*"))


@pytest.mark.parametrize("failure", ["mapping", "barrier", "both"])
def test_worker_population_failure_releases_peers_and_mapping(monkeypatch, failure):
    region = SimpleNamespace(cleanup=Mock())
    region_factory = Mock(return_value=region)
    barrier = Mock()
    if failure in ("mapping", "both"):
        region_factory.side_effect = ValueError("mapping failure")
    if failure in ("barrier", "both"):
        barrier.side_effect = RuntimeError("barrier failure")
    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", region_factory)
    monkeypatch.setattr(npu_mod.cpu_spec, "_all_workers_barrier", barrier)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    error = RuntimeError if failure == "barrier" else ValueError
    with pytest.raises(error, match="failure"):
        _spec(True, 0).create_worker(object())
    barrier.assert_called_once()
    assert region.cleanup.call_count == (1 if failure == "barrier" else 0)


@pytest.mark.parametrize("failure", ["primary", "secondary", "manager"])
def test_scheduler_initialization_failure_cleans_constructed_tiers(monkeypatch, failure):
    region = SimpleNamespace(cleanup=Mock())
    primary = SimpleNamespace(
        get_kv_memoryview=lambda: memoryview(bytearray(8)), shutdown=Mock(side_effect=region.cleanup)
    )
    secondary = SimpleNamespace(tier_type="test", shutdown=Mock())
    region_factory = Mock(return_value=region)
    primary_factory = Mock(return_value=primary)
    secondary_factory = Mock(return_value=secondary)
    manager_factory = Mock()
    if failure == "primary":
        primary_factory.side_effect = RuntimeError("primary failure")
    elif failure == "secondary":
        secondary_factory.side_effect = [secondary, RuntimeError("secondary failure")]
    else:
        manager_factory.side_effect = RuntimeError("manager failure")
    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", region_factory)
    monkeypatch.setattr(npu_mod.tiering_spec, "CPUPrimaryTierOffloadingManager", primary_factory)
    monkeypatch.setattr(npu_mod.tiering_spec.SecondaryTierFactory, "create_secondary_tier", secondary_factory)
    monkeypatch.setattr(npu_mod.tiering_spec, "TieringOffloadingManager", manager_factory)
    spec = _spec(True, 0)
    spec.secondary_tier_configs = [{}, {}] if failure == "secondary" else [{}]
    with pytest.raises(RuntimeError, match=f"{failure} failure"):
        spec.get_manager()
    region.cleanup.assert_called_once()
    assert spec._scheduler_mmap is None
    assert spec._manager is None
    if failure != "primary":
        primary.shutdown.assert_called_once()
        secondary.shutdown.assert_called_once()


def test_tiering_still_rejects_store_threshold_before_mapping(monkeypatch):
    region_factory = Mock()
    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", region_factory)
    spec = _spec(True, 0)
    spec.extra_config["store_threshold"] = 2
    with pytest.raises(ValueError, match="store_threshold"):
        spec.get_manager()
    region_factory.assert_not_called()
