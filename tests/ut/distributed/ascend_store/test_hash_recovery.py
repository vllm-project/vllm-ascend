# SPDX-License-Identifier: Apache-2.0
"""Hash hits reuse the existing coordinator, scheduler and byte IO paths."""

import ctypes
import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

try:
    importlib.import_module("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator")
except ModuleNotFoundError:
    CORE = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store"
else:
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


@pytest.mark.parametrize("eagle", [False, True])
def test_same_group_hits_as_upstream(eagle):
    upstream = importlib.import_module(CORE + ".coordinator")
    hashes = [BlockHash(bytes([i + 1]) * 8) for i in range(32)]
    # Start with a shared prefix, then leave different holes in each group.
    exists = {(g, bytes(h)) for g in range(2) for i, h in enumerate(hashes) if i not in (12 + g, 21 + g)}
    results = []
    for module, cls in [
        (coordinator, coordinator.AscendStoreCoordinator),
        (upstream, upstream.MooncakeStoreCoordinator),
    ]:
        geometry = (
            {"group_block_sizes": [16, 16], "group_cache_families": ["default", "default"]}
            if cls is coordinator.AscendStoreCoordinator
            else {}
        )
        coord = cls(groups(), 16, 4, use_eagle=eagle, enable_partial_hash_hits=True, **geometry)
        pool = module.ExternalCachedBlockPool(4, exists)
        results.append([coord.find_longest_cache_hit(hashes, end, pool)[1] for end in range(4, 129, 4)])
    assert results[0] == results[1]


def test_partial_full_hit_falls_back_to_recoverable_hbm_boundary():
    coord = coordinator.AscendStoreCoordinator(
        groups(),
        16,
        4,
        group_block_sizes=[16, 16],
        group_cache_families=["default", "default"],
        enable_partial_hash_hits=True,
    )
    hashes = [BlockHash(bytes([i + 1]) * 8) for i in range(7)]
    pool = coordinator.ExternalCachedBlockPool(4, {(g, bytes(hashes[-1])) for g in range(2)})

    # The first page is local. The store can only supply the checkpoint at 28.
    assert coord.find_partial_hit_tokens(hashes, 29, pool, hbm_hit_tokens=0) == 0
    assert coord.find_partial_hit_tokens(hashes, 29, pool, hbm_hit_tokens=16) == 28
    assert coord.find_partial_hit_tokens(hashes, 28, pool, hbm_hit_tokens=16) == 16


class ByteStore:
    requires_exists_before_put = True

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


@pytest.mark.parametrize("store_job_id", [None, 99])
def test_boundary_save_lookup_and_physical_page_replay(store_job_id):
    coord = coordinator.AscendStoreCoordinator(
        groups(),
        16,
        4,
        group_block_sizes=[16, 16],
        group_cache_families=["default", "default"],
        enable_partial_hash_hits=True,
    )
    rng = torch.Generator().manual_seed(42)
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
        boundary_puts=[metadata.BoundaryPut(0, 1, 8), metadata.BoundaryPut(1, 3, 8)],
        store_job_id=store_job_id,
    )
    tracker = InFlightSaveBlockTracker(1)
    pool = SimpleNamespace(blocks=list(range(4)), touch=Mock(), free_blocks=Mock())
    tracker.retain(req, pool)
    assert req.store_job_id == store_job_id
    req.current_event = Mock()
    send.add_save_batch([req])
    assert req.store_job_id is not None and req.store_job_id != req.event_id
    send._handle_request(send.request_queue.get())
    req.current_event.synchronize.assert_called_once()
    completed = send.get_completed_events()
    assert completed == {req.event_id: 1}
    tracker.complete(completed, pool)
    pool.free_blocks.assert_called_once_with([3, 1])
    assert not tracker.has_pending_work()
    result = bare_worker(coord, db, store)._lookup_with_coordinator(13, hashes, [0, 1], True)
    assert result.hit_length == 8
    assert result.tail_key_boundaries == ((0, 8), (1, 8))
    # The worker's token-count API remains an int, even in hash mode.
    assert bare_worker(coord, db, store).lookup(13, hashes, [0, 1]) == 8
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


@pytest.mark.parametrize("tail_key_boundary, tail_hash_index", [(None, 4), (20, 4), (24, 5)])
def test_partial_chunks_do_not_fold_hashes_into_page_keys(tail_key_boundary, tail_hash_index):
    db = metadata.ChunkedTokenDatabase([metadata.KeyMetadata("model", 0, 0, 0, 0)], [16], None, hash_block_size=4)
    db.cache_coordinator = Mock(cacheable_group_ids=[0])
    hashes = [BlockHash(bytes([i + 1]) * 8) for i in range(6)]
    with patch.object(metadata, "get_block_hashes", side_effect=AssertionError("unused page hash folding")):
        chunks = list(db._iter_token_chunks(20, hashes, tail_key_boundary=tail_key_boundary))
    assert chunks == [(0, 16, hashes[3], None), (16, 32, hashes[tail_hash_index], None)]


@pytest.mark.parametrize(
    "completed, mamba_tails, publish, expected",
    [
        (40, [40], False, []),
        (75, [68], True, [(0, 17, 68), (0, 18, 72)]),
        (75, [40, 68], True, [(0, 17, 68), (0, 18, 72)]),
        (75, [68, 68], False, [(0, 17, 68), (0, 18, 72)]),
        (68, [68], False, []),
        (75, [], True, [(0, 17, 68), (0, 18, 72)]),
        (75, [], False, []),
    ],
)
def test_partial_tail_non_mamba_puts_follow_upstream_prompt_checkpoint(completed, mamba_tails, publish, expected):
    # Match upstream's shared-prefix junction cases: a 75-token prompt has an
    # EAGLE checkpoint at 68, with attention proof through token 72.
    scheduler = object.__new__(KVPoolScheduler)
    scheduler.cache_coordinator = SimpleNamespace(
        mamba_group_ids=[1],
        hash_block_size=4,
        lcm_block_size=16,
        use_eagle=True,
        eagle_proof_margin_by_group={0: 4},
    )
    scheduler.grouped_block_size = [4, 16]
    scheduler.cacheable_group_ids = [0, 1]
    req_meta = metadata.ReqMeta(
        "tail",
        token_len_chunk=0,
        block_ids_by_group=[list(range(1, 20)), [50, 51, 52, 53, 54]],
        block_hashes=[bytes([i]) for i in range(18)],
        num_prompt_tokens=75,
        completed_token_len=completed,
        boundary_puts=[metadata.BoundaryPut(1, 60 + i, n) for i, n in enumerate(mamba_tails)],
        publish_partial_tail=publish,
    )
    assert scheduler._partial_tail_non_mamba_puts(req_meta) == expected


@pytest.mark.parametrize("blocks, expected", [([1, 2, 0], []), ([1, 2], []), ([1, 2, 3], [(0, 3, 44)])])
def test_partial_tail_non_mamba_puts_follow_upstream_eagle_drop(blocks, expected):
    # Like upstream, EAGLE block drop determines the checkpoint even when the
    # attention group itself has no EAGLE proof margin.
    scheduler = object.__new__(KVPoolScheduler)
    scheduler.cache_coordinator = SimpleNamespace(
        mamba_group_ids=[1],
        hash_block_size=4,
        lcm_block_size=16,
        use_eagle=True,
        eagle_proof_margin_by_group={},
    )
    scheduler.grouped_block_size = [16, 16]
    scheduler.cacheable_group_ids = [0, 1]
    req_meta = metadata.ReqMeta(
        "tail",
        token_len_chunk=0,
        block_ids_by_group=[blocks, [50]],
        block_hashes=[bytes([i]) for i in range(12)],
        num_prompt_tokens=49,
        completed_token_len=49,
        boundary_puts=[metadata.BoundaryPut(1, 50, 44)],
    )
    assert scheduler._partial_tail_non_mamba_puts(req_meta) == expected


@pytest.fixture
def finishing_tail():
    scheduler = object.__new__(KVPoolScheduler)
    scheduler.can_put = True
    scheduler.enable_partial_hash_hits = True
    scheduler.cache_coordinator = SimpleNamespace(
        mamba_group_ids=[1],
        hash_block_size=4,
        lcm_block_size=16,
        use_eagle=True,
        eagle_proof_margin_by_group={},
    )
    scheduler.grouped_block_size = [16, 16]
    scheduler.cacheable_group_ids = [0, 1]
    scheduler.num_mamba_speculative_blocks_by_group = {1: 1}
    scheduler._request_trackers = {"tail": SimpleNamespace(prefill_end_tokens=65, num_prompt_tokens=65)}
    scheduler._finished_partial_tail_metas = {}
    scheduler._in_flight_save_blocks = InFlightSaveBlockTracker(2)
    scheduler._block_pool = SimpleNamespace(blocks=list(range(100)), touch=Mock(), free_blocks=Mock())
    request = SimpleNamespace(
        request_id="tail",
        num_prompt_tokens=49,
        num_computed_tokens=65,
        block_hashes=[bytes([i]) for i in range(16)],
    )
    return scheduler, request


@pytest.mark.parametrize("partial_hits", [False, True])
def test_finished_partial_tail_pins_exact_sources_through_next_step(finishing_tail, partial_hits):
    scheduler, request = finishing_tail
    scheduler.enable_partial_hash_hits = partial_hits
    tables = ([1, 2, 3, 4], [50])
    # Duplicate entries must not take duplicate source references.
    assert scheduler.register_finished_partial_tail(request, tables, [(1, 50, 44), (1, 50, 44)]) is False
    req_meta = scheduler._finished_partial_tail_metas["tail"]
    assert req_meta.num_prompt_tokens == 49
    assert req_meta.token_len_chunk == 0 and req_meta.can_save
    assert req_meta.boundary_puts == [(0, 3, 44), (1, 50, 44), (1, 50, 44)]
    scheduler._block_pool.touch.assert_called_once_with([3, 50])
    # Core may now clear the request's containers. The queued save has its own
    # snapshots and references and requires no current block table next step.
    tables[0].clear()
    request.block_hashes.clear()
    scheduler._request_trackers.clear()
    assert req_meta.block_ids_by_group == [[1, 2, 3, 4], [50]]
    assert len(req_meta.block_hashes) == 16
    meta = metadata.AscendConnectorMetadata(preempted_req_ids=set())
    state = SimpleNamespace(boundary_state_offloads={}, get_block_ids=Mock(side_effect=AssertionError("finished")))
    if state.boundary_state_offloads:
        scheduler._handle_boundary_state_offloads(state.boundary_state_offloads, meta)
    scheduler._apply_current_save_block_ids(meta, SimpleNamespace(kv_connector_block_state=state))
    for finished_meta in scheduler._finished_partial_tail_metas.values():
        meta.add_request(finished_meta)
    scheduler._finished_partial_tail_metas.clear()
    assert meta.requests == [req_meta]
    assert not scheduler._finished_partial_tail_metas
    scheduler.retain_blocks_for_save(req_meta)
    scheduler._block_pool.touch.assert_called_once_with([3, 50])
    tracker = scheduler._in_flight_save_blocks
    tracker.complete({req_meta.event_id: 1}, scheduler._block_pool)
    scheduler._block_pool.free_blocks.assert_not_called()
    tracker.complete({req_meta.event_id: 1}, scheduler._block_pool)
    scheduler._block_pool.free_blocks.assert_called_once_with([50, 3])
    assert not tracker.has_pending_work()


@pytest.mark.parametrize("offloads", [[], [(1, 50, 68)], [(0, 3, 44)], [(1, 0, 44)]])
def test_finished_partial_tail_rejects_unusable_handoff(finishing_tail, offloads):
    scheduler, request = finishing_tail
    assert scheduler.register_finished_partial_tail(request, ([1, 2, 3], [50]), offloads) is False
    assert not scheduler._finished_partial_tail_metas
    scheduler._block_pool.touch.assert_not_called()


def test_current_save_tables_are_refreshed_without_mamba_offloads():
    scheduler = object.__new__(KVPoolScheduler)
    req_meta = metadata.ReqMeta(req_id="dense", token_len_chunk=16, can_save=True)
    meta = metadata.AscendConnectorMetadata(preempted_req_ids=set())
    meta.add_request(req_meta)
    state = SimpleNamespace(get_block_ids=Mock(return_value=([7], [8])))
    scheduler._apply_current_save_block_ids(meta, SimpleNamespace(kv_connector_block_state=state))
    assert req_meta.block_ids_by_group == [[7], [8]]
    state.get_block_ids.assert_called_once_with("dense")


def test_no_save_jobs_need_no_current_block_state():
    scheduler = object.__new__(KVPoolScheduler)
    meta = metadata.AscendConnectorMetadata(preempted_req_ids=set())
    scheduler._apply_current_save_block_ids(meta, SimpleNamespace(kv_connector_block_state=None))


def test_finished_partial_tail_requires_common_boundary(finishing_tail):
    scheduler, request = finishing_tail
    with pytest.raises(ValueError, match="must share a boundary"):
        scheduler.register_finished_partial_tail(request, ([1, 2, 3], [50]), [(1, 50, 44), (1, 51, 40)])
    scheduler._block_pool.touch.assert_not_called()


@pytest.mark.parametrize("partial_hits", [False, True])
def test_core_boundary_offload_is_not_positional_recurrent_table(partial_hits):
    scheduler = object.__new__(KVPoolScheduler)
    scheduler.enable_partial_hash_hits = partial_hits
    scheduler.cache_coordinator = coordinator.AscendStoreCoordinator(
        groups(),
        16,
        4,
        group_block_sizes=[16, 16],
        group_cache_families=["default", "default"],
        enable_partial_hash_hits=partial_hits,
    )
    scheduler.can_put = True
    scheduler._request_trackers = {"r": SimpleNamespace(prefill_end_tokens=12, num_prompt_tokens=12, token_len=12)}
    scheduler._unfinished_requests = {"r": (SimpleNamespace(block_hashes=[b"a", b"b", b"c"]), None)}
    scheduler._finished_partial_tail_metas = {}
    scheduler._partial_tail_non_mamba_puts = lambda req: []
    tables = {"r": ([1], [2])}
    state = KVConnectorBlockState(
        req_ids={"r"}, resolve_block_ids=tables.__getitem__, boundary_state_offloads={"r": [(1, 9, 8)]}
    )
    meta = metadata.AscendConnectorMetadata(preempted_req_ids=set())
    if state.boundary_state_offloads:
        scheduler._handle_boundary_state_offloads(state.boundary_state_offloads, meta)
    scheduler._apply_current_save_block_ids(meta, SimpleNamespace(kv_connector_block_state=state))
    for finished_meta in scheduler._finished_partial_tail_metas.values():
        meta.add_request(finished_meta)
    scheduler._finished_partial_tail_metas.clear()
    assert meta.requests[0].boundary_puts == [(1, 9, 8)]
    assert meta.requests[0].block_ids_by_group == [[1], [2]]
    tracker = InFlightSaveBlockTracker(1)
    pool = SimpleNamespace(blocks=list(range(20)), touch=Mock(), free_blocks=Mock())
    tracker.retain(meta.requests[0], pool)
    pool.touch.assert_called_once_with([9])


def test_failed_save_acknowledges_without_advancing_saved_offset():
    db = Mock(cache_coordinator=None, group_block_len={0: [64]})
    send = KVCacheStoreSendingThread(Mock(), db, [16], tp_rank=0)
    req = metadata.ReqMeta("failed", token_len_chunk=16, can_save=True, event_id=3)
    send._handle_stored_request = Mock(return_value=False)
    send.add_save_batch([req])
    send._handle_request(send.request_queue.get())
    assert send.get_completed_events() == {3: 1}
    assert send.get_saved_offset("failed") == 0


@pytest.mark.parametrize("partial_hits", [False, True])
def test_aligned_boundary_put_uses_exact_state_without_positional_mamba_save(partial_hits):
    coord = coordinator.AscendStoreCoordinator(
        groups(),
        16,
        4,
        group_block_sizes=[16, 16],
        group_cache_families=["default", "default"],
        enable_partial_hash_hits=partial_hits,
    )
    sources = [
        torch.arange(64, dtype=torch.float32).reshape(4, 16, 1),
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
    ]
    db = metadata.ChunkedTokenDatabase(
        [metadata.KeyMetadata("model", 0, 0, 0, g) for g in range(2)],
        [16, 16],
        None,
        hash_block_size=4,
    )
    db.cache_coordinator = coord
    db.set_group_buffers(
        {g: [source.data_ptr()] for g, source in enumerate(sources)},
        {g: [source[0].numel() * 4] for g, source in enumerate(sources)},
        {g: [source.stride(0) * 4] for g, source in enumerate(sources)},
    )
    store = ByteStore()
    send = KVCacheStoreSendingThread(store, db, [16, 16], tp_rank=0)
    hashes = [BlockHash(bytes([i + 1]) * 8) for i in range(4)]
    req = metadata.ReqMeta(
        "aligned",
        token_len_chunk=16,
        can_save=True,
        block_hashes=hashes,
        block_ids_by_group=[[1], [2]],
        kv_cache_group_ids=[0, 1],
        num_prompt_tokens=16,
        boundary_puts=[metadata.BoundaryPut(1, 3, 16)],
    )
    assert send._handle_stored_request(req) is True
    state_key = db._get_key_prefix(1) + hashes[-1].hex()
    assert store.values[state_key] == sources[1][3].numpy().tobytes()
    # Whole-block mode retains positional saves for requests without a boundary
    # handoff. This request's explicit handoff still selected block 3 above.
    assert coord.store_mask(16, 16)[1] == [not partial_hits]


@pytest.mark.parametrize(
    "requires_exists, exists, expected_keys",
    [
        (True, [1, 0], ["g1-0202020202020202"]),
        (True, [1, 1], []),
        (False, [], ["g0-0101010101010101", "g1-0202020202020202"]),
    ],
)
def test_boundary_put_reuses_backend_exists_policy_and_deduplicates(requires_exists, exists, expected_keys):
    send = object.__new__(KVCacheStoreSendingThread)
    send.token_database = SimpleNamespace(hash_block_size=4, _get_key_prefix=lambda group: f"g{group}-")
    send.m_store = SimpleNamespace(
        requires_exists_before_put=requires_exists,
        exists=Mock(return_value=exists),
        put=Mock(return_value=True),
    )
    send._get_block_size = Mock(return_value=16)
    send._prepare_value = Mock(side_effect=lambda *args, block_id, **kwargs: ([block_id * 100], [16], block_id))
    req = metadata.ReqMeta(
        "boundary",
        block_hashes=[BlockHash(bytes([1]) * 8), BlockHash(bytes([2]) * 8)],
        boundary_puts=[metadata.BoundaryPut(0, 1, 4), metadata.BoundaryPut(1, 2, 8), metadata.BoundaryPut(0, 1, 4)],
    )
    assert send._store_boundary_puts(req) is True
    if expected_keys:
        actual_keys, addrs, sizes = send.m_store.put.call_args.args
        assert actual_keys == expected_keys
        assert addrs == ([[200]] if requires_exists else [[100], [200]])
        assert sizes == [[16]] * len(expected_keys)
    else:
        send.m_store.put.assert_not_called()
    if not requires_exists:
        send.m_store.exists.assert_not_called()
