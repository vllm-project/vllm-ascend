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
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole, SupportsHMA
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from tests.ut.kv_offload.replay_utils import (
    BLOCK_SIZE,
    WINDOW,
    _initialize_replay_window,
    _pd_request,
)
from tests.ut.kv_offload.replay_utils import (
    replay_scheduler as _replay_scheduler_fixture,
)
from tests.ut.kv_offload.utils import create_model_runner_output
from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler, pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
)

replay_scheduler = _replay_scheduler_fixture


class MemoryBackend:
    requires_exists_before_put = True

    @staticmethod
    def create_scheduler_client(_parallel):
        return None

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
        return [True] * len(keys)

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
    client = SimpleNamespace(
        closed=False,
        lookup=lambda tokens, hashes, groups, **kwargs: pool.lookup_scheduler(tokens, hashes, groups, **kwargs),
    )
    with (
        patch.object(pool_scheduler, "LookupKeyClient", return_value=client),
        patch.object(
            pool_scheduler.importlib, "import_module", return_value=SimpleNamespace(MooncakeBackend=type(backend))
        ),
    ):
        multi = AscendMultiConnector(config, KVConnectorRole.SCHEDULER, plan)
        store = multi._connectors[1]
        store.connector_scheduler.client = client
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
        tp_size=1,
        put_step=1,
        kv_role="kv_producer",
        ready_event=threading.Event(),
        group_uses_align_state=pool.group_uses_align_state,
    )
    meta = ReqMeta(
        "seed",
        count * BLOCK_SIZE,
        block_ids_by_group=ids,
        block_hashes=request.block_hashes,
        can_save=True,
        current_event=SimpleNamespace(synchronize=lambda: None),
        kv_cache_group_ids=[0],
        num_prompt_tokens=count * BLOCK_SIZE,
    )
    sender.add_stored_request(meta)
    sender.request_queue.put(meta)
    sender._handle_request(sender.request_queue.get_nowait())
    assert runtime.backend.objects
    assert all("@group:0" in key for key in runtime.backend.objects)
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
