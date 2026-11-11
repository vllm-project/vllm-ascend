# SPDX-License-Identifier: Apache-2.0
"""Ascend-owned lifecycle tests and real-Core differential CPU replay.

The native store is replaced with byte-copy IO; Core's recovery algorithms,
cache specs and physical tensor descriptors are real, not mocked.
"""

import ctypes
import importlib
import queue
import random
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

from tests.ut.distributed.ascend_store.test_recovery_scheduler import _make_bare_scheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.recovery import coordinator, data, protocol, worker

CORE = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store"


def groups():
    return [
        KVCacheGroupSpec(
            ["attention"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
        ),
        KVCacheGroupSpec(
            ["mamba"], MambaSpec(block_size=16, shapes=((2,),), dtypes=(torch.float32,), mamba_cache_mode="align")
        ),
    ]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("eagle", [False, True])
def test_same_group_hits_as_upstream(seed, eagle):
    ref = importlib.import_module(CORE + ".coordinator")
    hashes = [BlockHash(bytes([i + 1]) * 8) for i in range(32)]
    rng = random.Random(seed)
    exists = {(g, bytes(h)) for g in range(2) for h in hashes if rng.random() > 0.2}
    results = []
    for module, cls in [(coordinator, coordinator.StoreCoordinator), (ref, ref.MooncakeStoreCoordinator)]:
        coord = cls(groups(), 16, 4, use_eagle=eagle, enable_partial_hash_hits=True)
        pool = module.ExternalCachedBlockPool(4, exists)
        results.append([coord.find_longest_cache_hit(hashes, end, pool) for end in range(4, 129, 4)])
    assert results[0] == results[1]


class ByteStore:
    def __init__(self):
        self.values = {}

    def batch_is_exist(self, keys):
        return [int(key in self.values) for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, *args, **kwargs):
        for key, ptrs, lens in zip(keys, addresses, sizes):
            self.values[key] = b"".join(ctypes.string_at(ptr, length) for ptr, length in zip(ptrs, lens))
        return [0] * len(keys)


def replay(implementation, seed):
    if implementation == "ascend":
        d, w, c = data, worker, coordinator.StoreCoordinator
    else:
        d = importlib.import_module(CORE + ".data")
        w = importlib.import_module(CORE + ".worker")
        c = importlib.import_module(CORE + ".coordinator").MooncakeStoreCoordinator
    coord = c(groups(), 16, 4, enable_partial_hash_hits=True)
    rng = torch.Generator().manual_seed(seed)
    sources = [torch.randn(4, 16, 1, generator=rng), torch.randn(4, 2, generator=rng)]
    targets = [torch.full_like(source, -999) for source in sources]
    dbs = []
    for group, source in enumerate(sources):
        db = d.ChunkedTokenDatabase(d.KeyMetadata("model", 0, 0, 0, 0, group_id=group), 16, 4)
        db.store_layout.register_kv_caches([source], 4)
        dbs.append(db)
    store = ByteStore()
    send = w.KVCacheStoreSendingThread(
        store=store,
        coord=coord,
        token_databases=dbs,
        block_size=16,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=Mock(),
    )
    hashes = [BlockHash(bytes([i + 1]) * 8) for i in range(5)]
    req = d.ReqMeta(
        req_id="seed",
        token_len_chunk=0,
        block_ids=([1], [2]),
        block_hashes=hashes,
        can_save=True,
        num_prompt_tokens=12,
        completed_token_len=12,
        boundary_puts=[(0, 1, 8), (1, 3, 8)],
        store_job_id=7,
    )
    req.current_event = Mock()
    send.add_request(req)
    send._handle_request(send.request_queue.get())
    req.current_event.synchronize.assert_called_once()
    cls = w.StoreWorker if implementation == "ascend" else w.MooncakeStoreWorker
    instance = object.__new__(cls)
    instance._capacity_only = False
    instance.coord, instance.hash_block_size = coord, 4
    instance._kv_cache_groups = groups()
    instance.token_dbs, instance.store = dbs, store
    instance._lookup_key_prefixes = [db.store_layout.lookup_key_prefixes([(0, 0, 0, 0)]) for db in dbs]
    instance._record_kv_connector_operation = Mock()
    result = instance.lookup(13, hashes)
    assert result.hit_length == 8
    # Restore into different physical blocks; a Mamba value is the entire
    # checkpoint, not 8/16 of that checkpoint. Padding is not treated as tokens.
    for group, target in enumerate(targets):
        db = dbs[group]
        key = db.key_for(hashes[result.hit_length // 4 - 1])
        destination = d.RankLocalStoreLayout(db.metadata, 16, 4)
        destination.register_kv_caches([target], 4)
        ptrs, lengths = destination.prepare_value_for_block(0)
        cursor = 0
        for ptr, length in zip(ptrs, lengths):
            ctypes.memmove(ptr, store.values[key][cursor : cursor + length], length)
            cursor += length
        assert cursor == len(store.values[key])
        torch.testing.assert_close(target[0], sources[group][1 if group == 0 else 3])
        assert torch.all(target[1:] == -999)
    # A deterministic CPU continuation consuming both recovered groups. This
    # proves state replay, not NPU/model-generation acceptance.
    output = targets[0][0, :8].sum() + targets[1][0].sum()
    return (
        result.hit_length,
        tuple((b.group_id, b.num_tokens) for b in result.tail_key_boundaries),
        store.values,
        output,
    )


@pytest.mark.parametrize("seed", [0, 42, 1024])
def test_same_seed_target_boundary_payload_and_replay_output(seed):
    own, upstream = replay("ascend", seed), replay("upstream", seed)
    assert own[:3] == upstream[:3]
    torch.testing.assert_close(own[3], upstream[3], rtol=0, atol=0)


def completion(scheduler, job, rank, *, failed=False):
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=data.StoreWorkerMetadata(
                completed_saves={job: 1},
                completed_save_ranks={job: {rank}},
                failed_saves={job} if failed else set(),
            )
        )
    )


def test_async_rank_order_duplicate_failure_and_release():
    scheduler = _make_bare_scheduler()
    scheduler._num_workers = 2
    pool = scheduler._gpu_block_pool
    pool.touch([pool.blocks[7], pool.blocks[8]])
    scheduler._pinned_saves = {10: ([7], 2), 11: ([8], 2)}
    completion(scheduler, 11, 1, failed=True)
    completion(scheduler, 10, 0)
    completion(scheduler, 11, 1)  # duplicate must not impersonate another rank
    assert pool.blocks[7].ref_cnt == pool.blocks[8].ref_cnt == 1
    completion(scheduler, 11, 0)
    assert pool.blocks[8].ref_cnt == 0 and pool.blocks[7].ref_cnt == 1
    completion(scheduler, 10, 1)
    completion(scheduler, 10, 1)  # late duplicate is harmless
    assert not scheduler.has_pending_push_work()
    assert pool.blocks[7].ref_cnt == 0


def test_reset_close_never_free_inflight_dma():
    scheduler = _make_bare_scheduler()
    scheduler.client = Mock(futures={})
    pool = scheduler._gpu_block_pool
    pool.touch([pool.blocks[7]])
    scheduler._pinned_saves = {10: ([7], 1)}
    assert not scheduler.reset_store()
    with pytest.raises(RuntimeError, match="drain"):
        scheduler.close()
    assert pool.blocks[7].ref_cnt == 1
    completion(scheduler, 10, 0, failed=True)
    assert scheduler.reset_store()
    scheduler.close()
    assert pool.blocks[7].ref_cnt == 0
    scheduler.client.close.assert_called_once()


def test_worker_close_timeout_keeps_registration_alive():
    instance = object.__new__(worker.StoreWorker)
    instance.store = Mock()
    instance.kv_send_thread = SimpleNamespace(request_queue=SimpleNamespace(unfinished_tasks=1), name="send")
    instance.kv_recv_threads = []
    with (
        patch.object(worker.time, "monotonic", side_effect=[0, 6]),
        pytest.raises(RuntimeError, match="IO in flight"),
    ):
        instance.close()
    instance.store.close.assert_not_called()


def test_metadata_rank_aggregation_wire_and_isolation():
    a = data.StoreWorkerMetadata({1: 1}, {1: {0}}, {1})
    a.aggregate(data.StoreWorkerMetadata({1: 1}, {1: {0}}))
    a.aggregate(data.StoreWorkerMetadata({1: 1}, {1: {1}}))
    assert a.completed_save_ranks == {1: {0, 1}} and a.failed_saves == {1}
    ref_data = importlib.import_module(CORE + ".data")
    ref_protocol = importlib.import_module(CORE + ".protocol")
    own = data.StoreLookupResult(8, (data.TailKeyBoundary(1, 8),))
    ref = ref_data.MooncakeLookupResult(8, (ref_data.TailKeyBoundary(1, 8),))
    assert protocol.encode_lookup_response(own) == ref_protocol.encode_lookup_response(ref)
    assert protocol.decode_lookup_response(ref_protocol.encode_lookup_response(ref)) == own
    for module in (coordinator, worker):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "kv_connector.v1.mooncake.store" not in source
    inherited = [base.__module__ for base in worker.StoreWorker.__mro__]
    assert not any(CORE in module for module in inherited)


def test_transfer_thread_stops_on_sentinel():
    thread = object.__new__(worker.KVTransferThread)
    thread.ready_event = threading.Event()
    thread.request_queue = queue.Queue()
    thread.request_queue.put(None)
    thread.run()
    assert thread.request_queue.unfinished_tasks == 0


@pytest.mark.parametrize("cancelled,failed", [(True, False), (False, True), (False, False)])
def test_cancel_failure_and_success_all_ack_after_job_stops(cancelled, failed):
    coord = coordinator.StoreCoordinator(groups(), 16, 4, enable_partial_hash_hits=True)
    send = worker.KVCacheStoreSendingThread(
        store=ByteStore(),
        coord=coord,
        token_databases=[],
        block_size=16,
        tp_rank=0,
        group_put_steps=[1, 1],
        kv_role="kv_both",
        ready_event=threading.Event(),
        replicate_config=Mock(),
    )
    req = data.ReqMeta(
        req_id="cancel",
        token_len_chunk=0,
        block_ids=([], []),
        block_hashes=[],
        can_save=True,
        store_job_id=99,
    )
    send.add_request(req)
    if cancelled:
        send.delete_finished_stored_request(req.req_id)
    with patch.object(send, "_maybe_offload_boundary_states", return_value=not failed) as put:
        send._handle_request(send.request_queue.get())
    assert put.call_count == int(not cancelled)
    completed, failures = send.take_save_outcomes()
    assert completed == {99: 1}
    assert failures == ({99} if failed else set())
    assert send.request_queue.unfinished_tasks == 0
    assert send.take_save_outcomes() == ({}, set())


def test_resolved_geometry_shared_and_producer_does_not_drop_checkpoint_twice():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.recovery.geometry import StoreGeometry
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.recovery.scheduler import StoreScheduler

    original = groups()
    original[1].is_eagle_group = True
    transfer = SimpleNamespace(
        is_kv_producer=True, is_kv_consumer=False, kv_role="kv_producer", kv_connector_extra_config={}
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, world_size=1),
        kv_transfer_config=transfer,
        kv_events_config=None,
        speculative_config=SimpleNamespace(use_eagle_block_drop=lambda: True),
    )
    cache = SimpleNamespace(
        prefix_cacheable_groups=original,
        prefix_cacheable_group_ids=(0, 1),
        cache_hit_alignment_tokens=4,
        prefix_cache_retention_interval=None,
    )
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.recovery.geometry.resolve_kv_cache_block_sizes",
        return_value=(16, 4),
    ) as resolve:
        geometry = StoreGeometry.resolve(config, cache)
    assert resolve.call_count == 1
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.recovery.scheduler.LookupKeyClient"):
        scheduler = StoreScheduler(config, cache, geometry)
    assert scheduler.geometry is geometry
    assert scheduler._store_coord.hash_block_size == geometry.hash_block_size
    assert not scheduler._store_coord.eagle_group_ids
    assert not scheduler._store_coord.use_eagle
    assert original[1].is_eagle_group  # Never mutate the Core plan.
