# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact state ownership across core cleanup and asynchronous Store saves."""

import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import KVConnectorBlockState
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    BoundaryPut,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    StoreLookupResult,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler


def _scheduler(pool, *, fine_grained=False, use_eagle=False):
    config = make_config(kv_role="kv_producer", extra_config={"backend": "mooncake"})
    config.cache_config.prefix_match_unit = 4
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.speculative_config.use_eagle.return_value = use_eagle
    config.parallel_config.world_size = 2
    groups = [
        KVCacheGroupSpec(
            ["attention"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
        ),
        KVCacheGroupSpec(
            ["state"],
            MambaSpec(
                block_size=16,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_speculative_blocks=2 if use_eagle else 0,
            ),
        ),
    ]
    geometry: dict = {}
    if hasattr(KVCacheConfig, "cache_hit_alignment_tokens"):
        geometry = dict(hash_block_size=4, cache_hit_alignment_tokens=4 if fine_grained else 16)
    elif fine_grained:
        pytest.skip("fine-grained Store recovery requires core-resolved hash geometry")
    plan = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=groups, **geometry)
    # Only backend construction is replaced; configuration, capability gating,
    # group inference and the coordinator all run through production code.
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as backend:
        backend.import_module.return_value = MagicMock()
        scheduler = KVPoolScheduler(config, use_layerwise=False, kv_cache_config=plan)
    assert scheduler.cache_coordinator.enable_partial_hash_hits is fine_grained
    scheduler._block_pool = pool
    scheduler._request_trackers = {"owner": RequestTracker("owner", 12, num_prompt_tokens=13, prefill_end_tokens=13)}
    return scheduler


@pytest.mark.parametrize("fine_grained", [False, True])
@pytest.mark.parametrize("put_succeeds", [False, True])
def test_finished_state_is_saved_from_pinned_source_then_released_by_all_workers(put_succeeds, fine_grained):
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    free_before = pool.get_num_free_blocks()
    source, newer = pool.get_new_blocks(2)
    scheduler = _scheduler(pool, fine_grained=fine_grained)
    hashes = [bytes([i]) * 32 for i in range(1, 4)]
    request = SimpleNamespace(request_id="owner", num_prompt_tokens=13, num_computed_tokens=12, block_hashes=hashes)
    assert (
        scheduler.register_finished_partial_tail(
            request, ([newer.block_id], [newer.block_id]), [(1, source.block_id, 12)]
        )
        is False
    )
    assert source.ref_cnt == 2
    pool.free_blocks([source, newer])
    assert source.ref_cnt == 1
    assert scheduler.has_pending_push_work()
    output = SimpleNamespace(
        finished_req_ids={"owner"},
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        kv_connector_block_state=KVConnectorBlockState(
            req_ids=set(),
            resolve_block_ids=lambda _: (),
            boundary_state_offloads={},
        ),
    )
    meta = scheduler.build_connector_meta(output)
    assert "owner" not in scheduler._request_trackers
    assert len(meta.requests) == 1
    state_meta = meta.requests[0]
    expected_puts = [BoundaryPut(1, source.block_id, 12)]
    if fine_grained:
        expected_puts.append(BoundaryPut(0, newer.block_id, 12))
        assert newer.ref_cnt == 1
    assert state_meta.boundary_puts == expected_puts
    assert not scheduler.build_connector_meta(output).requests

    # Fake only the DMA backend and event. Addresses and byte lengths come
    # from the production token database, backed by actual CPU memory.
    storage = (ctypes.c_ubyte * (16 * 16))()
    for offset in range(16):
        storage[newer.block_id * 16 + offset] = 99

    def synchronize():
        for offset in range(16):
            storage[source.block_id * 16 + offset] = 37

    state_meta.current_event = SimpleNamespace(synchronize=synchronize)
    db = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0) for _ in range(2)], [16, 16], None, hash_block_size=4)
    db.set_group_buffers({0: [ctypes.addressof(storage)], 1: [ctypes.addressof(storage)]}, {0: [16], 1: [16]})
    saved = {}

    def put(keys, addrs, sizes):
        for key, addresses, lengths in zip(keys, addrs, sizes):
            saved[key] = b"".join(ctypes.string_at(addr, size) for addr, size in zip(addresses, lengths))
        return put_succeeds

    thread = KVCacheStoreSendingThread(
        SimpleNamespace(put=put, exists=lambda keys: [0] * len(keys)),
        db,
        [16, 16],
        tp_rank=0,
        group_uses_align_state=[False, True],
    )
    thread.add_stored_request(state_meta)
    thread.request_queue.put(state_meta)
    thread._handle_request(state_meta)
    completed = thread.get_completed_events()
    assert completed == {state_meta.event_id: 1}
    assert list(saved.values()) == ([bytes([37]) * 16, bytes([99]) * 16] if fine_grained else [bytes([37]) * 16])
    assert next(iter(saved)).endswith(hashes[2].hex())

    def complete():
        scheduler.update_connector_output(
            SimpleNamespace(
                finished_recving=None,
                kv_connector_worker_meta=AscendStoreKVConnectorWorkerMetadata(completed_events=completed),
            )
        )

    complete()
    assert source.ref_cnt == 1
    complete()
    assert source.ref_cnt == 0
    assert not scheduler.has_pending_push_work()
    assert pool.get_num_free_blocks() == free_before


def test_attention_boundary_keeps_physical_page_layout():
    storage = (ctypes.c_ubyte * 64)(*range(64))
    db = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [16], None, hash_block_size=4)
    db.set_group_buffers({0: [ctypes.addressof(storage)]}, {0: [16]})
    saved: list[bytes] = []

    def put(keys, addrs, sizes):
        saved.extend(
            ctypes.string_at(address, size)
            for addresses, lengths in zip(addrs, sizes)
            for address, size in zip(addresses, lengths)
        )
        return True

    thread = KVCacheStoreSendingThread(
        SimpleNamespace(put=put, exists=lambda keys: [0] * len(keys)),
        db,
        [16],
        tp_rank=0,
        group_uses_align_state=[False],
    )
    request = ReqMeta("owner", block_hashes=[b"a", b"b", b"c"], can_save=True, boundary_puts=[BoundaryPut(0, 2, 12)])
    assert thread._handle_stored_request(request)
    assert saved == [bytes(range(32, 48))]


@pytest.mark.skipif(
    not hasattr(KVCacheConfig, "cache_hit_alignment_tokens"),
    reason="fine-grained Store recovery requires core-resolved hash geometry",
)
@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize("ratio", [2, 4])
def test_partial_state_save_lookup_and_load_preserve_exact_tail_key(use_eagle, ratio):
    groups = [
        KVCacheGroupSpec(
            ["attention"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
        ),
        KVCacheGroupSpec(
            ["state"], MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align")
        ),
        KVCacheGroupSpec(
            ["indexer"],
            AscendMLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                tokens_per_state=ratio,
                model_version="glm5_next",
                indexes_kv_by_block_stride=True,
            ),
        ),
        KVCacheGroupSpec(
            ["private"],
            AscendIndexerKPoolTailSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=4,
                compress_ratio=4,
            ),
        ),
    ]
    coord = AscendStoreCoordinator(
        groups,
        16,
        4,
        [16, 16, 16, 4],
        ["default", "mamba", "default", "default"],
        use_eagle=use_eagle,
        enable_partial_hash_hits=True,
    )
    scheduler = _scheduler(None)
    scheduler.cache_coordinator = coord
    scheduler.cacheable_group_ids = [0, 1, 2]
    scheduler.grouped_block_size = [16, 16, 16, 4]
    scheduler.cache_transfer_granularity = 16
    hashes = [bytes([i]) * 32 for i in range(1, 8)]
    boundary = 24 if use_eagle else 28
    states = [BoundaryPut(1, 5, boundary)]
    puts = states + scheduler._attention_boundary_puts(states, [[2, 3], [0, 5], [2, 3], [6]], 29, hashes)
    assert puts == [*states, BoundaryPut(0, 3, 28), BoundaryPut(2, 3, 28)]
    storage = [(ctypes.c_ubyte * 256)(*[(group * 31 + i) % 256 for i in range(256)]) for group in range(4)]
    db = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0) for _ in groups], [16, 16, 16, 4], None, hash_block_size=4)
    db.cache_coordinator = coord
    db.set_group_buffers(
        {i: [ctypes.addressof(a)] for i, a in enumerate(storage)}, {0: [16], 1: [16], 2: [16 // ratio], 3: [4]}
    )
    values = {}

    def put(keys, addrs, sizes):
        for key, addresses, lengths in zip(keys, addrs, sizes):
            values[key] = b"".join(ctypes.string_at(addr, size) for addr, size in zip(addresses, lengths))
        return True

    def get(keys, addrs, sizes):
        for key, addresses, lengths in zip(keys, addrs, sizes):
            value = values[key]
            offset = 0
            for address, size in zip(addresses, lengths):
                ctypes.memmove(address, value[offset : offset + size], size)
                offset += size
        return [0] * len(keys)

    backend = SimpleNamespace(
        put=put,
        get=get,
        exists=lambda keys: [int(key in values) for key in keys],
        requires_exists_before_put=True,
    )
    send = KVCacheStoreSendingThread(
        backend, db, [16, 16, 16, 4], tp_rank=0, group_uses_align_state=[False, True, False, False]
    )
    # Save the complete prefix through the normal production path, too.
    # The backend starts empty: no test-injected ancestor keys may hide a
    # missing save or an incorrect physical block address.
    prefix = ReqMeta(
        "owner",
        token_len_chunk=16,
        num_prompt_tokens=29,
        block_hashes=hashes,
        block_ids_by_group=[[2, 3], [0, 5], [2, 3], [6]],
        kv_cache_group_ids=[0, 1, 2, 3],
        can_save=True,
        state_handoff_groups=(1,),
    )
    assert send._handle_stored_request(prefix)
    assert values == {
        db._get_key_prefix(0) + hashes[3].hex(): bytes(storage[0][32:48]),
        db._get_key_prefix(2) + hashes[3].hex(): bytes(storage[2][32 // ratio : 48 // ratio]),
    }
    saved = ReqMeta("owner", block_hashes=hashes, can_save=True, boundary_puts=puts)
    assert send._handle_stored_request(saved)
    queried = []

    def query(group_id, candidates, mask):
        queried.append(group_id)
        return [h for h in candidates if db._get_key_prefix(group_id) + h.hex() in values]

    result = coord.find_partial_hit(hashes, 29, query)
    assert result.hit_tokens == boundary
    assert result.tail_key_boundaries == {0: 28, 1: boundary, 2: 28}
    assert queried == [0, 1, 2]
    expected_attention = bytes(storage[0][48:64])
    expected_state = bytes(storage[1][80:96])
    expected_indexer = bytes(storage[2][48 // ratio : 64 // ratio])
    private_before = bytes(storage[3])
    load_spec = LoadSpec(0, boundary, True, token_len=boundary, tail_key_boundaries=result.tail_key_boundaries)
    restored = ReqMeta(
        "replay",
        block_hashes=hashes,
        load_spec=load_spec,
        block_ids_by_group=[[7, 8], [0, 9], [7, 8], [10]],
        kv_cache_group_ids=[0, 1, 2, 3],
        skip_null_blocks_by_group=[False, True, False, True],
    )
    receive = KVCacheStoreRecvingThread(backend, db, [16, 16, 16, 4], tp_rank=0)
    receive.request_queue.put(restored)
    receive._handle_request(restored)
    assert bytes(storage[0][112:128]) == bytes(storage[0][32:48])
    assert bytes(storage[0][128:144]) == expected_attention
    assert bytes(storage[1][144:160]) == expected_state
    assert bytes(storage[2][112 // ratio : 128 // ratio]) == bytes(storage[2][32 // ratio : 48 // ratio])
    assert bytes(storage[2][128 // ratio : 144 // ratio]) == expected_indexer
    assert bytes(storage[3]) == private_before
    # A longer/different recurrent state cannot substitute for this checkpoint.
    state_key = db._get_key_prefix(1) + hashes[boundary // 4 - 1].hex()
    del values[state_key]
    assert coord.find_partial_hit(hashes, 29, query).hit_tokens == 0


def test_lookup_rpc_preserves_tail_keys_and_dispatches_admin_commands(monkeypatch, tmp_path):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import ascend_store_connector, pool_scheduler

    endpoint = f"ipc://{tmp_path}/lookup.sock"
    monkeypatch.setattr(ascend_store_connector, "get_zmq_rpc_path_lookup", lambda _: endpoint)
    monkeypatch.setattr(pool_scheduler, "get_zmq_rpc_path_lookup", lambda _: endpoint)
    hashes = [bytes([i]) * 32 for i in range(1, 8)]
    calls = []

    def details(token_len, block_hashes, groups, hbm_hit_tokens):
        calls.append((token_len, block_hashes, groups, hbm_hit_tokens))
        return StoreLookupResult(24, {0: 28, 1: 24}, 28)

    worker = SimpleNamespace(
        lookup_details=details, lookup_scheduler=lambda *args, **kwargs: 16, reset_store=MagicMock()
    )
    server = ascend_store_connector.LookupKeyServer(worker, None)
    client = pool_scheduler.LookupKeyClient(None)
    try:
        result = client.lookup_details(29, hashes, [0, 1], hbm_hit_tokens=16)
        assert result == StoreLookupResult(24, {0: 28, 1: 24}, 28)
        assert calls == [(29, [h.hex() for h in hashes], [0, 1], 16)]
        assert client.lookup(29, hashes, [0, 1]) == 16
        assert client.reset()
        worker.reset_store.assert_called_once_with()
        assert client.lookup_details(29, hashes, [0, 1]) == result
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("command", ["lookup", "lookup_details", "reset", "unknown", "malformed"])
def test_lookup_rpc_errors_preserve_the_admin_channel(command, monkeypatch, tmp_path):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import RESP_ERR

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import ascend_store_connector, pool_scheduler
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LOOKUP_DETAILS_MSG

    endpoint = f"ipc://{tmp_path}/lookup.sock"
    monkeypatch.setattr(ascend_store_connector, "get_zmq_rpc_path_lookup", lambda _: endpoint)
    monkeypatch.setattr(pool_scheduler, "get_zmq_rpc_path_lookup", lambda _: endpoint)
    worker = SimpleNamespace(
        lookup_details=MagicMock(side_effect=RuntimeError("lookup failed")),
        lookup_scheduler=MagicMock(side_effect=RuntimeError("lookup failed")),
        reset_store=MagicMock(side_effect=RuntimeError("reset failed")),
    )
    server = ascend_store_connector.LookupKeyServer(worker, None)
    client = pool_scheduler.LookupKeyClient(None)
    try:
        if command == "lookup":
            assert client.lookup(16, [b"hash"], [0]) == 0
        elif command == "lookup_details":
            assert client.lookup_details(16, [b"hash"], [0]) == StoreLookupResult()
        elif command == "reset":
            assert not client.reset()
        else:
            client.socket.send(LOOKUP_DETAILS_MSG if command == "malformed" else b"unknown")
            assert client.socket.recv() == RESP_ERR
        assert not client.closed
        worker.lookup_details.side_effect = None
        worker.lookup_details.return_value = StoreLookupResult(12, {0: 16, 1: 12}, 16)
        assert client.lookup_details(17, [b"hash"], [0, 1]) == worker.lookup_details.return_value
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("ordinary_save", [False, True])
@pytest.mark.parametrize("fine_grained", [False, True])
@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize("put_succeeds", [False, True])
def test_live_boundary_handoff_uses_core_snapshot_and_pins_until_all_workers_finish(
    fine_grained,
    use_eagle,
    put_succeeds,
    ordinary_save,
):
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    free_before = pool.get_num_free_blocks()
    attention, checkpoint, scratch = pool.get_new_blocks(3)
    scheduler = _scheduler(pool, fine_grained=fine_grained, use_eagle=use_eagle)
    boundary = 12 if fine_grained else 16
    completed = boundary + (4 if use_eagle else 0)
    hashes = [bytes([i]) * 32 for i in range(1, completed // 4 + 1)]
    request = SimpleNamespace(num_prompt_tokens=completed + 1, block_hashes=hashes)
    scheduler._unfinished_requests["owner"] = (request, 0)
    scheduler._request_trackers["owner"] = RequestTracker(
        "owner",
        completed,
        num_prompt_tokens=completed + 1,
        prefill_end_tokens=completed + 1,
    )
    state = KVConnectorBlockState(
        req_ids={"owner"},
        resolve_block_ids=lambda _: ([attention.block_id], [scratch.block_id]),
        boundary_state_offloads={"owner": [(1, checkpoint.block_id, boundary)]},
    )
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        kv_connector_block_state=state,
    )
    if ordinary_save:
        meta = AscendConnectorMetadata(preempted_req_ids=set())
        meta.add_request(
            ReqMeta(
                "owner",
                token_len_chunk=completed // 16 * 16,
                num_prompt_tokens=completed + 1,
                block_hashes=hashes,
                block_ids_by_group=[[attention.block_id], [scratch.block_id]],
                kv_cache_group_ids=[0, 1],
                can_save=True,
            )
        )
        scheduler._add_boundary_handoffs(meta, output)
        assert meta.requests[0].state_handoff_groups == (1,)
    else:
        meta = scheduler.build_connector_meta(output)
    assert len(meta.requests) == 1
    job = meta.requests[0]
    assert BoundaryPut(1, checkpoint.block_id, boundary) in job.boundary_puts
    assert all(put.block_id != scratch.block_id for put in job.boundary_puts)
    assert checkpoint.ref_cnt == 2
    assert scratch.ref_cnt == 1
    assert (attention.ref_cnt == 2) is fine_grained

    # Core can release the request immediately after constructing this job.
    pool.free_blocks([attention, checkpoint, scratch])
    assert checkpoint.ref_cnt == 1
    assert scratch.ref_cnt == 0
    storage = (ctypes.c_ubyte * (16 * 16))()
    for i in range(16):
        storage[checkpoint.block_id * 16 + i] = 37
        storage[scratch.block_id * 16 + i] = 99
    db = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)] * 2, [16, 16], None, hash_block_size=4)
    db.set_group_buffers({0: [ctypes.addressof(storage)], 1: [ctypes.addressof(storage)]}, {0: [16], 1: [16]})
    saved = {}

    def put(keys, addrs, sizes):
        for key, addresses, lengths in zip(keys, addrs, sizes):
            saved[key] = b"".join(ctypes.string_at(addr, size) for addr, size in zip(addresses, lengths))
        return put_succeeds

    sender = KVCacheStoreSendingThread(
        SimpleNamespace(put=put, exists=lambda keys: [0] * len(keys)),
        db,
        [16, 16],
        tp_rank=0,
        group_uses_align_state=[False, True],
    )
    job.current_event = SimpleNamespace(synchronize=lambda: None)
    sender.add_stored_request(job)
    sender.request_queue.put(job)
    sender._handle_request(job)
    state_values = [value for key, value in saved.items() if "@group:1@" in key]
    assert state_values == [bytes([37]) * 16]
    completed_events = sender.get_completed_events()
    assert completed_events == {job.event_id: 1}
    reply = SimpleNamespace(
        finished_recving=None,
        kv_connector_worker_meta=AscendStoreKVConnectorWorkerMetadata(completed_events=completed_events),
    )
    scheduler.update_connector_output(reply)
    assert checkpoint.ref_cnt == 1
    scheduler.update_connector_output(reply)
    assert checkpoint.ref_cnt == 0
    assert pool.get_num_free_blocks() == free_before
