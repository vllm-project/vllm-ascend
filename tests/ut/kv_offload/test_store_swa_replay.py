# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real scheduler/Store/receive paths with CPU bytes and a simulated backend."""

import copy
import ctypes
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config import CacheConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole, SupportsHMA
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from tests.ut.kv_offload.utils import create_model_runner_output
from tests.ut.patch.platform.test_patch_swa_bounded_replay_legacy import (
    BLOCK_SIZE,
    WINDOW,
    _pd_request,
    enable_legacy_replay,  # noqa: F401
)
from tests.ut.patch.platform.test_patch_swa_bounded_replay_legacy import (
    replay_scheduler as _replay_scheduler_fixture,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendCircularBufferSpec,
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler, pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
    get_block_hashes,
    store_replay_enabled,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy import _initialize_replay_window

replay_scheduler = _replay_scheduler_fixture


class MemoryBackend:
    def __init__(self):
        self.objects = {}
        self.gets = []
        self.error = None

    def set_device(self):
        pass

    def register_buffer(self, *_args):
        pass

    def exists(self, keys):
        return [int(key in self.objects) for key in keys]

    def put(self, keys, addresses, sizes):
        for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
            self.objects[key] = tuple(ctypes.string_at(ptr, size) for ptr, size in zip(pointers, lengths, strict=True))
        return [0] * len(keys)

    def get(self, keys, addresses, sizes):
        self.gets.append(list(keys))
        if isinstance(self.error, Exception):
            raise self.error
        if self.error == "none":
            return None
        if self.error == "short":
            return []
        if self.error == "code":
            return [-1] * len(keys)
        for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
            for pointer, size, data in zip(pointers, lengths, self.objects[key], strict=True):
                assert len(data) == size
                ctypes.memmove(pointer, data, size)
        return [0] * len(keys)


class NoRemotePD(KVConnectorBase_V1, SupportsHMA):
    """Only the unavailable remote source is simulated; Store is real."""

    def start_load_kv(self, *_args, **_kwargs):
        pass

    def wait_for_layer_load(self, *_args):
        pass

    def save_kv_layer(self, *_args, **_kwargs):
        pass

    def wait_for_save(self):
        pass

    def get_num_new_matched_tokens(self, *_args):
        return 0, False

    def get_loaded_kv_cache_group_ids(self, _request):
        return (0, 1)

    def update_state_after_alloc(self, *_args):
        pass

    def on_new_request(self, *_args):
        pass

    def build_connector_meta(self, *_args):
        return None

    def update_connector_output(self, *_args):
        pass

    def has_pending_block_frees(self):
        return False

    def has_pending_push_work(self):
        return False

    def request_finished_all_groups(self, *_args):
        return False, None


@pytest.fixture(params=["kv_consumer", "kv_producer"], ids=["decode", "prefill"])
def store_runtime(replay_scheduler, monkeypatch, request):
    scheduler = replay_scheduler
    plan = scheduler.kv_cache_manager.kv_cache_config
    config = copy.copy(scheduler.vllm_config)
    config.model_config = SimpleNamespace(
        model="test/store-replay",
        max_model_len=256,
        use_mla=True,
        hf_text_config=SimpleNamespace(),
        hf_config=SimpleNamespace(),
        get_num_layers=lambda _parallel: 2,
        get_total_num_kv_heads=lambda: 1,
    )
    config.parallel_config = copy.copy(config.parallel_config)
    config.parallel_config.rank = 1  # Avoid starting the real ZMQ lookup service.
    config.speculative_config = None
    config.kv_events_config = None
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role=request.param,
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "MooncakeHybridConnector",
                    "kv_role": request.param,
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": 4, "tp_size": 4},
                        "decode": {"dp_size": 8, "tp_size": 2},
                    },
                },
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": request.param,
                    "kv_connector_extra_config": {
                        "backend": "mooncake",
                        "consumer_is_to_load": True,
                        "load_async": False,
                    },
                },
            ]
        },
    )

    def resolve_source(transfer):
        return AscendStoreConnector if transfer.kv_connector == "AscendStoreConnector" else NoRemotePD

    monkeypatch.setattr(KVConnectorFactory, "get_connector_class", resolve_source)
    backend = MemoryBackend()
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pool_worker.KVPoolWorker, "_align_kv_ptrs", lambda *_args: None)
    monkeypatch.setattr(KVCacheStoreRecvingThread, "start", lambda self: self.ready_event.set())
    monkeypatch.setattr(KVCacheStoreSendingThread, "start", lambda self: self.ready_event.set())
    with patch.object(
        pool_worker.importlib, "import_module", return_value=SimpleNamespace(MooncakeBackend=lambda *_a, **_k: backend)
    ):
        worker_multi = AscendMultiConnector(config, KVConnectorRole.WORKER, plan)
        worker = worker_multi._connectors[1]
    caches = {
        name: (torch.full((plan.num_blocks, group.kv_cache_spec.block_size, 1, 128), -777, dtype=torch.bfloat16),)
        for group in plan.kv_cache_groups
        for name in group.layer_names
    }
    worker.register_kv_caches(caches)
    pool = worker.connector_worker
    client = SimpleNamespace(lookup=lambda tokens, hashes, groups: pool.lookup_scheduler(tokens, hashes, groups))
    with patch.object(pool_scheduler, "LookupKeyClient", return_value=client):
        multi = AscendMultiConnector(config, KVConnectorRole.SCHEDULER, plan)
        store = multi._connectors[1]
    scheduler.connector = multi
    scheduler.recompute_kv_load_failures = True
    _initialize_replay_window(scheduler)
    assert store.connector_scheduler.load_async and pool.load_async
    assert store.get_loaded_kv_cache_group_ids(None) == (0,)
    yield SimpleNamespace(
        scheduler=scheduler,
        store=store,
        multi=multi,
        worker=worker,
        pool=pool,
        caches=caches,
        backend=backend,
        worker_multi=worker_multi,
    )
    pool.kv_recv_thread.executor.shutdown(wait=True)
    if pool.kv_send_thread is not None:
        pool.kv_send_thread.executor.shutdown(wait=True)


def seed_store(runtime, request, hit):
    pool = runtime.pool
    count = hit // BLOCK_SIZE
    ids = [list(range(1, count + 1)), list(range(1, count + 1))]
    runtime.caches["full"][0].fill_(42)
    sender = KVCacheStoreSendingThread(
        runtime.backend,
        pool.token_database,
        pool.grouped_block_size,
        0,
        1,
        1,
        "kv_producer",
        threading.Event(),
        pool.group_uses_align_state,
    )
    meta = ReqMeta(
        "seed",
        count * BLOCK_SIZE,
        block_ids_by_group=ids,
        block_hashes=request.block_hashes,
        can_save=True,
        current_event=SimpleNamespace(synchronize=lambda: None),
        kv_cache_group_ids=[0, 1],
        num_prompt_tokens=count * BLOCK_SIZE,
    )
    sender.add_stored_request("seed")
    sender.request_queue.put(meta)
    sender._handle_request(sender.request_queue.get_nowait())
    sender.executor.shutdown(wait=True)
    private_keys = {
        chunk.key.to_string()
        for chunk in pool.token_database.process_transfer_chunks(
            count * BLOCK_SIZE,
            request.block_hashes,
            kv_cache_group_id=1,
        )
    }
    assert not private_keys.intersection(runtime.backend.objects)
    runtime.caches["full"][0].fill_(-777)


def finish_receive(runtime, first):
    runtime.worker_multi.bind_connector_metadata(first.kv_connector_metadata)
    runtime.worker_multi.start_load_kv(None)
    thread = runtime.pool.kv_recv_thread
    thread._handle_request(thread.request_queue.get_nowait())
    result = runtime.worker_multi.get_transfer_results(set())
    output = create_model_runner_output([])
    output.kv_connector_output = KVConnectorOutput(
        finished_sending=result.finished_sending,
        finished_recving=result.finished_recving,
        failed_recving=result.failed_recving,
    )
    runtime.scheduler.update_from_output(first, output)
    return result


@pytest.mark.parametrize("hit", [16, 32, 95, 96])
def test_store_bytes_load_before_window_rebuild(store_runtime, hit):
    r = store_runtime
    req = _pd_request(r.scheduler, hit + 1)
    seed_store(r, req, hit)
    first = r.scheduler.schedule()
    adopted = hit // BLOCK_SIZE * BLOCK_SIZE
    if adopted <= WINDOW:
        assert first.scheduled_new_reqs[0].num_computed_tokens == 0
        assert r.pool.kv_recv_thread.request_queue.empty()
        return
    assert first.total_num_scheduled_tokens == 0
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert first.kv_connector_metadata.metadata[1].requests[0].can_save is False
    assert req.replay_start == 0
    assert r.multi.get_loaded_kv_cache_group_ids(req) == (0,)
    before = r.scheduler.schedule()
    assert before.total_num_scheduled_tokens == 0
    blocks = r.scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0][:]
    result = finish_receive(r, first)
    assert result.finished_recving == {req.request_id} and not result.failed_recving
    for block in blocks[: adopted // BLOCK_SIZE]:
        assert torch.all(r.caches["full"][0][block.block_id] == 42)
    assert torch.all(r.caches["swa"][0] == -777)
    replay = r.scheduler.schedule()
    data = replay.scheduled_new_reqs[0]
    assert data.num_computed_tokens == data.replay_start == adopted - WINDOW
    assert replay.num_scheduled_tokens[req.request_id] == BLOCK_SIZE
    assert r.scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0] == blocks
    r.scheduler.schedule()
    assert r.scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0] == blocks


@pytest.mark.parametrize("error", ["none", "short", "code", RuntimeError("backend disconnected")])
def test_failed_store_receive_retries_cold_without_loading_again(store_runtime, error):
    r = store_runtime
    req = _pd_request(r.scheduler, 97)
    seed_store(r, req, 96)
    r.backend.error = error
    first = r.scheduler.schedule()
    result = finish_receive(r, first)
    assert result.failed_recving == result.finished_recving == {req.request_id}
    assert r.worker.get_transfer_results(set()).failed_recving == set()
    retry = r.scheduler.schedule()
    assert retry.scheduled_new_reqs[0].num_computed_tokens == 0
    assert req.replay_start == 0
    assert req.request_id not in r.multi._requests_to_connector
    assert req.request_id in r.store.connector_scheduler._failed_load_requests
    assert len(r.backend.gets) == 1


def test_store_preemption_rebuilds_from_local_global_prefix(store_runtime):
    r = store_runtime
    req = _pd_request(r.scheduler, 97)
    seed_store(r, req, 96)
    first = r.scheduler.schedule()
    finish_receive(r, first)
    r.scheduler.schedule()
    r.scheduler.running.remove(req)
    req.num_in_flight_tokens = 0
    r.scheduler._preempt_request(req, time.monotonic())
    resumed = r.scheduler.schedule()
    assert req.replay_start == 96 - WINDOW
    assert req.request_id not in r.multi._requests_to_connector
    if r.scheduler.use_v2_model_runner:
        assert resumed.scheduled_new_reqs[0].replay_start == 96 - WINDOW
    else:
        assert resumed.scheduled_cached_reqs.replay_start == {req.request_id: 96 - WINDOW}


def test_store_load_preserves_a_shared_local_global_prefix(store_runtime):
    r = store_runtime
    req = _pd_request(r.scheduler, 97)
    seed_store(r, req, 96)
    warm = Request(
        "shared",
        list(range(97)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    manager = r.scheduler.kv_cache_manager
    assert manager.allocate_slots(warm, 32) is not None
    manager.cache_blocks(warm, 32)
    shared = manager.get_blocks(warm.request_id).blocks[0][:]
    for block in shared:
        r.caches["full"][0][block.block_id].fill_(123)
    first = r.scheduler.schedule()
    assert first.total_num_scheduled_tokens == 0
    assert manager.get_blocks(req.request_id).blocks[0][:2] == shared
    assert all(block.ref_cnt == 2 for block in shared)
    finish_receive(r, first)
    assert len(r.backend.gets[0]) == 4
    for block in shared:
        assert torch.all(r.caches["full"][0][block.block_id] == 123)
    before = [r.caches["full"][0][block.block_id].clone() for block in shared]
    r.scheduler.schedule()
    r.scheduler.schedule()
    for block, contents in zip(shared, before, strict=True):
        assert block.ref_cnt == 2
        assert torch.equal(r.caches["full"][0][block.block_id], contents)


def test_failure_policy_fail_does_not_start_replay(store_runtime):
    r = store_runtime
    r.scheduler.recompute_kv_load_failures = False
    req = _pd_request(r.scheduler, 97)
    seed_store(r, req, 96)
    r.backend.error = "code"
    first = r.scheduler.schedule()
    finish_receive(r, first)
    assert req.is_finished()
    assert req.replay_start == 0
    assert r.scheduler.schedule().total_num_scheduled_tokens == 0


def test_store_full_prompt_hit_aligns_the_actual_load(store_runtime):
    r = store_runtime
    req = _pd_request(r.scheduler, 96)
    seed_store(r, req, 96)
    first = r.scheduler.schedule()
    metadata = first.kv_connector_metadata.metadata[1]
    assert metadata.requests[0].load_spec.kvpool_cached_tokens == 80
    finish_receive(r, first)
    assert len(r.backend.gets[0]) == 5
    replay = r.scheduler.schedule()
    assert replay.scheduled_new_reqs[0].replay_start == 80 - WINDOW


def test_store_hit_works_with_local_prefix_caching_disabled(store_runtime):
    r = store_runtime
    r.scheduler.cache_config.enable_prefix_caching = False
    manager = r.scheduler.kv_cache_manager
    manager.enable_caching = False
    for single in manager.coordinator.single_type_managers:
        single.enable_caching = False
    req = _pd_request(r.scheduler, 97)
    seed_store(r, req, 96)
    first = r.scheduler.schedule()
    assert first.total_num_scheduled_tokens == 0
    finish_receive(r, first)
    replay = r.scheduler.schedule()
    assert replay.scheduled_new_reqs[0].replay_start == 96 - WINDOW


@pytest.mark.parametrize("wrapped", [False, True])
def test_store_masks_preserve_private_group_ids(store_runtime, wrapped):
    r = store_runtime
    groups = list(reversed(r.pool.kv_cache_config.kv_cache_groups))
    if wrapped:
        groups = [
            KVCacheGroupSpec(
                g.layer_names, UniformTypeKVCacheSpecs.from_specs({name: g.kv_cache_spec for name in g.layer_names})
            )
            for g in groups
        ]
    coord = AscendStoreCoordinator(groups, 16, 16, [16, 16], ["default", "default"])
    assert coord.prefix_cacheable_group_ids == (1,)
    assert coord.store_mask(96) == ([False] * 6, [True] * 6)
    req = _pd_request(r.scheduler, 97)
    pool = ExternalCachedBlockPool(16, {(1, h) for h in req.block_hashes})
    masks, hit = coord.find_longest_cache_hit(req.block_hashes, 96, pool)
    assert hit == 96 and masks == ([], [True] * 6)
    assert coord.load_mask(req.block_hashes, 96) == ([False] * 6, [True] * 6)
    private = AscendStoreCoordinator(groups[:1], 16, 16, [16], ["default"])
    assert private.find_longest_cache_hit(req.block_hashes, 96, pool) == (([],), 0)
    assert private.load_mask(req.block_hashes, 96) == ([False] * 6,)


@pytest.mark.parametrize("pcp,dcp,layerwise", [(2, 1, False), (1, 2, False), (1, 1, True)])
def test_unsupported_store_replay_layouts_fail_explicitly(store_runtime, pcp, dcp, layerwise):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(swa_bounded_replay=True),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp),
    )
    with pytest.raises(ValueError, match="PCP=1 and DCP=1"):
        store_replay_enabled(config, store_runtime.pool.kv_cache_config, use_layerwise=layerwise)


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressed_store_hashes_are_not_grouped_twice(store_runtime, ratio):
    r = store_runtime
    groups = r.pool.kv_cache_config.kv_cache_groups
    block_size = 16 * ratio
    coord = AscendStoreCoordinator(groups, block_size, 16, [16, 16], [f"c{ratio}", "default"])
    raw_hashes = [bytes([i % 251]) * 32 for i in range(2 * ratio)]
    hashes = get_block_hashes(raw_hashes, block_size, 16)
    pool = ExternalCachedBlockPool(16, {(0, h) for h in hashes})
    masks, hit = coord.find_longest_cache_hit(raw_hashes, 2 * block_size, pool)
    assert hit == 2 * block_size and masks == ([True, True], [])
    assert pool.hash_block_size == 16


@pytest.mark.parametrize("error", [None, "code"])
def test_cancel_during_receive_keeps_blocks_until_completion(store_runtime, error):
    r = store_runtime
    req = _pd_request(r.scheduler, 97)
    seed_store(r, req, 96)
    r.backend.error = error
    first = r.scheduler.schedule()
    blocks = r.scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0][:]
    r.scheduler.finish_requests([req.request_id], RequestStatus.FINISHED_ABORTED)
    assert req.request_id in r.scheduler.requests
    assert all(block.ref_cnt > 0 for block in blocks)
    finish_receive(r, first)
    assert req.request_id not in r.scheduler.requests
    assert all(block.ref_cnt == 0 for block in blocks)
    assert r.scheduler.schedule().total_num_scheduled_tokens == 0


def test_partial_c128_store_mode_is_not_silently_enabled(store_runtime):
    config = copy.copy(store_runtime.store._vllm_config)
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_consumer",
        kv_connector_extra_config={"hybrid_cache_c128_chunk": True},
    )
    with pytest.raises(ValueError, match="does not support hybrid_cache_c128_chunk"):
        store_replay_enabled(config, store_runtime.pool.kv_cache_config, use_layerwise=False)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("local_caching", [False, True])
def test_native_compressed_specs_preserve_logical_pages_and_physical_bytes(store_runtime, ratio, local_caching):
    config = copy.copy(store_runtime.store._vllm_config)
    config.model_config = copy.copy(config.model_config)
    config.model_config.max_model_len = 512
    config.cache_config = CacheConfig(block_size=128, enable_prefix_caching=local_caching, swa_bounded_replay=True)
    full = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        tokens_per_state=ratio,
        storage_block_size=128 // ratio,
        model_version="deepseek_v41",
    )
    swa = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
    )
    plan = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], full),
            KVCacheGroupSpec(["swa"], swa),
        ],
    )
    backend = MemoryBackend()
    with patch.object(
        pool_worker.importlib, "import_module", return_value=SimpleNamespace(MooncakeBackend=lambda *_a, **_k: backend)
    ):
        worker = AscendStoreConnector(config, KVConnectorRole.WORKER, plan)
    caches = {
        "full": (torch.full((64, 128 // ratio, 1, 8), -777, dtype=torch.bfloat16),),
        "swa": (torch.full((64, 128, 1, 8), -777, dtype=torch.bfloat16),),
    }
    worker.register_kv_caches(caches)
    pool = worker.connector_worker
    try:
        with patch.object(pool_scheduler, "LookupKeyClient"):
            scheduler = pool_scheduler.KVPoolScheduler(config, False, plan)
        assert scheduler.original_block_size == pool.original_block_size == [128, 128]
        assert scheduler.hash_block_size == pool.hash_block_size == 128
        assert scheduler.cache_transfer_granularity == pool.cache_transfer_granularity == 128
        init_none_hash(sha256)
        req = Request(
            "native",
            list(range(385)),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=get_request_block_hasher(128, sha256),
        )
        seed_store(SimpleNamespace(pool=pool, caches=caches, backend=backend), req, 384)
        assert pool.lookup_scheduler(384, req.block_hashes, [0, 1]) == 384
        caches["full"][0][8].fill_(123)
        metadata = AscendConnectorMetadata({"native"}, set())
        metadata.add_request(
            ReqMeta(
                "native",
                384,
                block_ids_by_group=[[8, 9, 10], [0, 0, 11]],
                block_hashes=req.block_hashes,
                kv_cache_group_ids=[0, 1],
                can_save=False,
                load_spec=LoadSpec(vllm_cached_tokens=128, kvpool_cached_tokens=384, can_load=True),
            )
        )
        worker.bind_connector_metadata(metadata)
        worker.start_load_kv(None)
        receiver = pool.kv_recv_thread
        receiver._handle_request(receiver.request_queue.get_nowait())
        result = worker.get_transfer_results(set())
        assert result.finished_recving == {"native"} and not result.failed_recving
        assert torch.all(caches["full"][0][8] == 123)
        assert torch.all(caches["full"][0][9:11] == 42)
        assert torch.all(caches["swa"][0] == -777)
        assert len(backend.gets[0]) == 2
    finally:
        pool.kv_recv_thread.executor.shutdown(wait=True)
        if pool.kv_send_thread is not None:
            pool.kv_send_thread.executor.shutdown(wait=True)


@pytest.mark.parametrize("bounded_replay", [False, True], ids=["swa-off", "swa-on"])
@pytest.mark.parametrize("kv_events", [False, True])
def test_small_private_rings_do_not_enter_store_hashing(store_runtime, bounded_replay, kv_events):
    """Exercise all 13 groups, including private rings smaller than hash=128."""
    config = copy.copy(store_runtime.store._vllm_config)
    config.model_config = copy.copy(config.model_config)
    config.model_config.max_model_len = 512
    config.cache_config = CacheConfig(block_size=128, prefix_match_unit=128, swa_bounded_replay=bounded_replay)
    config.kv_transfer_config = KVTransferConfig(kv_connector="AscendStoreConnector", kv_role="kv_producer")
    specs = [
        AscendCircularBufferSpec(block_size=size, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
        for size in [4, 128, 8, 128, 16, 128, 4, 8, 16, 32, 64, 128, 4]
    ]
    for group_id, ratio in [(1, 4), (5, 128)]:
        specs[group_id] = AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            tokens_per_state=ratio,
            storage_block_size=128 // ratio,
            model_version="deepseek_v41",
        )
    specs[3] = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=bounded_replay,
    )
    plan = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"group{index}"], spec) for index, spec in enumerate(specs)],
    )
    backend = MemoryBackend()
    with patch.object(
        pool_worker.importlib, "import_module", return_value=SimpleNamespace(MooncakeBackend=lambda *_a, **_k: backend)
    ):
        worker = AscendStoreConnector(config, KVConnectorRole.WORKER, plan)
    caches = {
        f"group{index}": (
            torch.full((32, getattr(spec, "storage_block_size", None) or spec.block_size, 1, 8), 42, dtype=spec.dtype),
        )
        for index, spec in enumerate(specs)
    }
    worker.register_kv_caches(caches)
    pool = worker.connector_worker
    sender = pool.kv_send_thread
    receiver = KVCacheStoreRecvingThread(
        backend,
        pool.token_database,
        pool.grouped_block_size,
        0,
        1,
        threading.Event(),
        set(),
        threading.Lock(),
        report_request_failures=True,
    )
    try:
        with patch.object(pool_scheduler, "LookupKeyClient"):
            scheduler = pool_scheduler.KVPoolScheduler(config, False, plan)
        expected_groups = (1, 5) if bounded_replay else (1, 3, 5)
        assert pool.cache_coordinator.prefix_cacheable_group_ids == expected_groups
        assert scheduler.hash_block_size == 128
        assert pool.hash_block_size == pool.cache_transfer_granularity == 128
        init_none_hash(sha256)
        req = Request(
            "rings",
            list(range(385)),
            SamplingParams(max_tokens=1),
            None,
            block_hasher=get_request_block_hasher(128, sha256),
        )
        sender.enable_kv_event = kv_events
        metadata = ReqMeta(
            "rings",
            384,
            block_ids_by_group=[list(range(1, 4)) if index in (1, 3, 5) else [1] for index in range(13)],
            block_hashes=req.block_hashes,
            kv_cache_group_ids=list(range(13)),
            num_prompt_tokens=384,
            original_block_size=pool.original_block_size,
        )
        sender.add_stored_request(req.request_id)
        sender.request_queue.put(metadata)
        sender._handle_request(sender.request_queue.get_nowait())
        assert sender.stored_requests[req.request_id] == 0
        assert sender.request_queue.unfinished_tasks == 0
        assert bool(sender.get_kv_events()) == kv_events
        assert pool.lookup_scheduler(384, req.block_hashes, list(range(13))) == 384

        masks = pool.token_database.store_mask(384, 384)
        expected_keys = set()
        for group_id in range(13):
            chunks = list(
                pool.token_database.process_transfer_chunks(384, req.block_hashes, kv_cache_group_id=group_id)
            )
            if group_id not in expected_groups:
                assert not chunks
            expected_keys.update(
                chunk.key.to_string()
                for chunk in chunks
                if pool.token_database.mask_allows_chunk(masks, group_id, chunk.raw_start)
            )
        assert set(backend.objects) == expected_keys and expected_keys
        for tensors in caches.values():
            tensors[0].fill_(-777)
        metadata.block_ids_by_group = [list(range(8, 11)) if index in (1, 3, 5) else [8] for index in range(13)]
        metadata.load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=384, can_load=True, token_len=384)
        load_masks = pool.token_database.load_mask(req.block_hashes, 384)
        expected_load_keys = {
            chunk.key.to_string()
            for group_id in expected_groups
            for chunk in pool.token_database.process_transfer_chunks(384, req.block_hashes, kv_cache_group_id=group_id)
            if pool.token_database.mask_allows_chunk(load_masks, group_id, chunk.raw_start)
        }
        # The synchronous planner and asynchronous receiver must select the same keys.
        sync_keys, *_ = pool._collect_sync_load_chunks(metadata, 384, list(range(13)))
        assert set(sync_keys) == expected_load_keys
        receiver.request_queue.put(metadata)
        receiver._handle_request(receiver.request_queue.get_nowait())
        assert receiver.get_and_clear_finished_requests() == {req.request_id}
        assert not receiver.get_and_clear_failed_requests()
        assert set(backend.gets[-1]) == expected_load_keys
        for group_id, tensors in enumerate(caches.values()):
            if group_id not in expected_groups:
                assert torch.all(tensors[0] == -777)
                continue
            for page_index in range(3):
                expected = 42 if load_masks[group_id][page_index] else -777
                assert torch.all(tensors[0][8 + page_index] == expected)
    finally:
        receiver.executor.shutdown(wait=True)
        if pool.kv_recv_thread is not None:
            pool.kv_recv_thread.executor.shutdown(wait=True)
        sender.executor.shutdown(wait=True)


def test_incompatible_cacheable_group_is_not_silently_skipped(store_runtime):
    database = copy.copy(store_runtime.pool.token_database)
    database.block_size = [4]
    database.hash_block_size = 128
    database.cache_coordinator = SimpleNamespace(prefix_cacheable_group_ids=(0,))
    with pytest.raises(AssertionError, match="block_size must be divisible by hash_block_size"):
        list(database.process_transfer_chunks(128, ["hash"], kv_cache_group_id=0))
