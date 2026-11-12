# SPDX-License-Identifier: Apache-2.0
"""Hash hits reuse the existing coordinator, scheduler and byte IO paths."""

import ctypes
import importlib
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import KVConnectorBlockState
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import coordinator, metadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.in_flight_save_blocks import InFlightSaveBlockTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

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
    upstream = importlib.import_module(CORE + ".coordinator")
    hashes = [BlockHash(bytes([i + 1]) * 8) for i in range(32)]
    rng = random.Random(seed)
    exists = {(g, bytes(h)) for g in range(2) for h in hashes if rng.random() > 0.2}
    results = []
    for module, cls in [
        (coordinator, coordinator.AscendStoreCoordinator),
        (upstream, upstream.MooncakeStoreCoordinator),
    ]:
        coord = cls(groups(), 16, 4, use_eagle=eagle, enable_partial_hash_hits=True)
        pool = module.ExternalCachedBlockPool(4, exists)
        results.append([coord.find_longest_cache_hit(hashes, end, pool)[1] for end in range(4, 129, 4)])
    assert results[0] == results[1]


class ByteStore:
    def __init__(self):
        self.values = {}

    def exists(self, keys):
        return [int(key in self.values) for key in keys]

    def put(self, keys, addresses, sizes):
        for key, ptrs, lengths in zip(keys, addresses, sizes, strict=True):
            self.values[key] = b"".join(ctypes.string_at(p, n) for p, n in zip(ptrs, lengths, strict=True))
        return [True] * len(keys)


def bare_worker(coord, db, store):
    worker = object.__new__(KVPoolWorker)
    worker.cache_coordinator, worker.token_database, worker.m_store = coord, db, store
    worker.num_kv_cache_groups, worker.hash_block_size = 2, 4
    worker._expand_lookup_key_variants = lambda key, group, all_ranks: [key]
    return worker


@pytest.mark.parametrize("seed", [0, 42, 1024])
def test_boundary_save_lookup_and_physical_page_replay(seed):
    coord = coordinator.AscendStoreCoordinator(groups(), 16, 4, enable_partial_hash_hits=True)
    rng = torch.Generator().manual_seed(seed)
    sources = [torch.randn(4, 16, 1, generator=rng), torch.randn(4, 2, generator=rng)]
    db = metadata.ChunkedTokenDatabase(
        [metadata.KeyMetadata("model", 0, 0, 0, g) for g in range(2)], [16, 16], None, hash_block_size=4
    )
    db.cache_coordinator = coord
    db.set_group_buffers(
        {g: [source.data_ptr()] for g, source in enumerate(sources)},
        {g: [source[0].numel() * 4] for g, source in enumerate(sources)},
        {g: [source.stride(0) * 4] for g, source in enumerate(sources)},
    )
    store = ByteStore()
    send = KVCacheStoreSendingThread(store, db, [16, 16], tp_rank=0)
    hashes = [BlockHash(bytes([i + 1]) * 8) for i in range(5)]
    req = metadata.ReqMeta(
        "seed",
        token_len_chunk=0,
        can_save=True,
        block_hashes=hashes,
        kv_cache_group_ids=[],
        boundary_puts=[(0, 1, 8), (1, 3, 8)],
        event_id=7,
    )
    req.current_event = Mock()
    send.add_save_batch([req])
    send._handle_request(send.request_queue.get())
    req.current_event.synchronize.assert_called_once()
    assert send.take_save_outcomes() == ({7: 1}, set())
    result = bare_worker(coord, db, store)._lookup_with_coordinator(13, hashes, [0, 1], True)
    assert result.hit_length == 8
    assert result.tail_key_boundaries == ((0, 8), (1, 8))
    # Attention is a whole physical page, and Mamba is an entire checkpoint,
    # even though the recoverable hash boundary is inside that page.
    output = 0
    for group, source in enumerate(sources):
        key = db._get_key_prefix(group) + metadata.block_hash_to_str(hashes[1])
        target = torch.full_like(source, -999)
        payload = store.values[key]
        ctypes.memmove(target.data_ptr(), payload, len(payload))
        torch.testing.assert_close(target[0], source[1 if group == 0 else 3])
        assert torch.all(target[1:] == -999)
        output += target[0].sum()
    torch.testing.assert_close(output, sources[0][1].sum() + sources[1][3].sum(), rtol=0, atol=0)


def test_core_boundary_offload_is_not_positional_recurrent_table():
    scheduler = object.__new__(KVPoolScheduler)
    scheduler.enable_partial_hash_hits = True
    scheduler.cache_coordinator = coordinator.AscendStoreCoordinator(groups(), 16, 4, enable_partial_hash_hits=True)
    scheduler.can_put = True
    scheduler._request_trackers = {"r": SimpleNamespace(prefill_end_tokens=12, num_prompt_tokens=12, token_len=12)}
    scheduler._unfinished_requests = {"r": (SimpleNamespace(block_hashes=[b"a", b"b", b"c"]), None)}
    scheduler._finished_partial_tail_metas = {}
    scheduler._partial_tail_puts = lambda req: []
    tables = {"r": ([1], [2])}
    state = KVConnectorBlockState(
        req_ids={"r"}, resolve_block_ids=tables.__getitem__, boundary_state_offloads={"r": [(1, 9, 8)]}
    )
    meta = metadata.AscendConnectorMetadata(preempted_req_ids=set())
    scheduler._prepare_boundary_saves(meta, SimpleNamespace(kv_connector_block_state=state))
    assert meta.requests[0].boundary_puts == [(1, 9, 8)]
    assert meta.requests[0].block_ids_by_group == [[1], [2]]
    tracker = InFlightSaveBlockTracker(1)
    pool = SimpleNamespace(blocks=list(range(20)), touch=Mock(), free_blocks=Mock())
    tracker.retain(meta.requests[0], {1: 1}, pool, retain_all=True)
    pool.touch.assert_called_once_with([9, 1])


def test_failed_save_acknowledges_without_advancing_saved_offset():
    db = Mock(cache_coordinator=None, group_block_len={0: [64]})
    send = KVCacheStoreSendingThread(Mock(), db, [16], tp_rank=0)
    req = metadata.ReqMeta("failed", token_len_chunk=16, can_save=True, event_id=3)
    send._handle_stored_request = Mock(return_value=False)
    send.add_save_batch([req])
    send._handle_request(send.request_queue.get())
    assert send.take_save_outcomes() == ({3: 1}, {3})
    assert send.get_saved_offset("failed") == 0


def test_close_cancels_queued_save_jobs_with_failed_completion():
    db = Mock(cache_coordinator=None, group_block_len={0: [64]})
    send = KVCacheStoreSendingThread(Mock(), db, [16], tp_rank=0)
    requests = [metadata.ReqMeta(str(i), can_save=True, event_id=i) for i in range(2)]
    fence = send.add_save_batch(requests)
    send._handle_cancelled_request(send.request_queue.get())
    assert fence.done.is_set()
    assert send.take_save_outcomes() == ({0: 1, 1: 1}, {0, 1})
    assert send.request_queue.unfinished_tasks == 0
    assert all(send.get_saved_offset(req.req_id) == 0 for req in requests)
    send.stop()
    rejected = metadata.ReqMeta("late", can_save=True, event_id=2)
    assert send.add_save_batch([rejected]).done.is_set()
    assert send.take_save_outcomes() == ({2: 1}, {2})
