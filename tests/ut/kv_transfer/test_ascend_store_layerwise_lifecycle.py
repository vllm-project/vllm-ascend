# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-token lifecycle with real scheduling, connectors, threads and CPU bytes.

Only device operations, model computation and the external Memcache service are
simulated. Keys, hit lengths, masks, block IDs and transfer addresses are produced
by the same constructors and request entry points used in serving.
"""

import ctypes
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_model_runner_output, create_vllm_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    attention_transfer_window,
    record_attention_compute_start,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import get_layerwise_kv_cache_specs


class MemoryStore:
    """External Store leaf; validate production-generated addresses and leases."""

    def __init__(self):
        self.objects = {}
        self.complete = set()
        self.regions = []
        self.leases = set()
        self.copies = []
        self.lock = threading.Lock()

    def init(self, device_id, init_bm=True):
        return 0

    def register_buffer(self, address, size):
        self.regions.append((address, size))

    def batch_is_exist(self, keys):
        return [int(key in self.complete) for key in keys]

    def batch_alloc(self, keys, sizes, replicas, lease_ttl_ms):
        for key, size in zip(keys, sizes, strict=True):
            assert key not in self.objects
            self.objects[key] = np.zeros(size, dtype=np.uint8)
        return [self.objects[key].ctypes.data for key in keys]

    def batch_get_key_info(self, keys):
        return [
            SimpleNamespace(
                size=lambda key=key: self.objects[key].size if key in self.complete else 0,
                gva_list=lambda key=key: [self.objects[key].ctypes.data] if key in self.complete else [],
            )
            for key in keys
        ]

    def batch_add_lease(self, keys, lease_ttl_ms):
        assert all(key in self.complete for key in keys)
        self.leases.update(keys)
        return [0] * len(keys)

    def batch_remove_lease(self, keys):
        self.leases.difference_update(keys)
        return 0

    def batch_write_finish(self, keys, results):
        assert not any(results)
        self.complete.update(keys)
        return [0] * len(keys)

    def batch_copy(self, gvas, addresses, sizes, direction):
        with self.lock:
            for gva, address, size in zip(gvas, addresses, sizes, strict=True):
                assert any(start <= address < address + size <= start + length for start, length in self.regions)
                key = next(
                    key
                    for key, buf in self.objects.items()
                    if buf.ctypes.data <= gva < gva + size <= buf.ctypes.data + buf.size
                )
                if direction == 1:
                    assert key in self.complete and key in self.leases
                source, dest = (address, gva) if direction == 0 else (gva, address)
                ctypes.memmove(dest, source, size)
                self.copies.append((direction, address, size, key))
        return 0


def aligned_tensor(shape):
    alignment = 2 * 1024 * 1024
    size = int(np.prod(shape)) * 4
    backing = np.zeros(size + alignment, dtype=np.uint8)
    offset = -backing.ctypes.data % alignment
    return torch.from_numpy(backing[offset : offset + size]).view(torch.float32).view(shape)


def payload(tokens, end, layer, plane):
    # Independent of Store keys, masks, hashes and physical allocation.
    return (sum(tokens[:end]) + layer * 13 + plane * 7) % 97 + 1


def wait_until(predicate, pool):
    deadline = time.monotonic() + 5
    while not predicate():
        pool.kv_send_thread.raise_if_failed()
        pool.kv_recv_thread.raise_if_failed()
        assert time.monotonic() < deadline, "Transfer did not complete"
        time.sleep(0.001)


@pytest.fixture
def memory_store(monkeypatch):
    store = MemoryStore()
    monkeypatch.setitem(sys.modules, "memcache_hybrid", SimpleNamespace(DistributedObjectStore=lambda: store))
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_rank", lambda: 0)
    register_all_kvcache_specs(None)
    init_none_hash(sha256)
    return store


@pytest.mark.parametrize(
    "recurrent_type,reuse",
    [
        (MambaAttentionBackendEnum.MAMBA2, False),
        (MambaAttentionBackendEnum.GDN_ATTN, False),
        (MambaAttentionBackendEnum.LINEAR, False),
        (None, False),
        (None, True),
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_raw_tokens_hybrid_roundtrip(memory_store, recurrent_type, reuse, wrapped):
    full = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)
    recurrent = (
        MambaSpec(
            block_size=32,
            shapes=((8,), (8,)),
            dtypes=(torch.float32, torch.float32),
            mamba_type=recurrent_type,
            mamba_cache_mode="align",
        )
        if recurrent_type is not None
        else SlidingWindowSpec(block_size=32, num_kv_heads=1, head_size=8, dtype=torch.float32, sliding_window=32)
    )
    groups = [
        KVCacheGroupSpec(["model.layers.0.attn", "model.layers.2.attn"], full),
        KVCacheGroupSpec(["model.layers.1.attn", "model.layers.3.attn"], recurrent),
    ]
    plan = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=groups)
    # Scheduler merged specs and worker per-layer specs must describe the same layout.
    worker_plan = KVCacheConfig(
        num_blocks=plan.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                group.layer_names,
                UniformTypeKVCacheSpecs(
                    block_size=group.kv_cache_spec.block_size,
                    kv_cache_specs={name: group.kv_cache_spec for name in group.layer_names},
                )
                if wrapped
                else group.kv_cache_spec,
            )
            for group in groups
        ],
    )
    config = create_vllm_config(max_num_batched_tokens=32, block_size=16)
    config.model_config.hf_text_config.num_hidden_layers = 4
    config.model_config.model_arch_config.total_num_hidden_layers = 4
    config.model_config.max_model_len = 512
    config.scheduler_config.max_model_len = 512
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.num_gpu_blocks = plan.num_blocks
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"backend": "memcache", "use_layerwise": True},
    )
    if reuse:
        config.kv_transfer_config.kv_connector_extra_config.update(
            layerwise_num_shared_buffers=1,
            layerwise_prefetch_layers=3,
            layerwise_independent_layers=[],
        )
    block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
    worker = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, worker_plan)
    specs = get_layerwise_kv_cache_specs(worker_plan)
    caches = {
        name: tuple(aligned_tensor((plan.num_blocks, *shape)) for shape in spec.shapes)
        if isinstance(spec, MambaSpec)
        else tuple(aligned_tensor((plan.num_blocks, spec.block_size, 1, spec.head_size)) for _ in range(2))
        for name, spec in specs.items()
    }
    pool = worker.connector_worker
    if reuse:
        for slot in pool._layerwise_reuse_layout.buffer_slots:
            for layer in slot[1:]:
                caches[f"model.layers.{layer}.attn"] = caches[f"model.layers.{slot[0]}.attn"]
    worker.register_kv_caches(caches)
    release_waits = {layer: threading.Event() for layer in pool.prefetch_layer_map.values()}
    release_finishes = {layer: threading.Event() for layer in release_waits}
    if reuse:
        # Observe (and still execute) the production receiver-to-worker fence.
        wait_for_compute = pool.kv_recv_thread.compute_release_waiter

        def observe_release(layer):
            release_waits[layer].set()
            wait_for_compute(layer)
            release_finishes[layer].set()

        pool.kv_recv_thread.compute_release_waiter = observe_release
    scheduler = Scheduler(
        config,
        plan,
        StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_size,
    )
    assert isinstance(pool.m_store, MemcacheBackend)
    assert isinstance(pool.kv_send_thread, KVCacheStoreLayerSendingThread)
    assert isinstance(pool.kv_recv_thread, KVCacheStoreLayerRecvingThread)
    assert isinstance(pool.cache_coordinator, AscendStoreCoordinator)
    assert pool.grouped_block_size == [16, 32]
    assert pool.hash_block_size == hash_size == 16
    assert pool.cache_transfer_granularity == 32
    assert pool.kv_recv_thread.group_builders[1].group_id == 1
    assert pool.kv_send_thread.token_database is pool.token_database

    def run_request(req_id, tokens, expected_hit):
        request = Request(
            request_id=req_id,
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=2),
            pooling_params=None,
            block_hasher=get_request_block_hasher(hash_size, sha256),
        )
        scheduler.add_request(request)
        starts = []
        loaded = False

        def compute(layer, block_ids, start, end):
            name = f"model.layers.{layer}.attn"
            spec = specs[name]
            for index, bid in enumerate(block_ids):
                boundary = (index + 1) * spec.block_size
                if bid and start < boundary <= end:
                    for plane, cache in enumerate(caches[name]):
                        cache[bid].fill_(payload(request.all_token_ids, boundary, layer, plane))

        for _ in range(12):
            output = scheduler.schedule()
            meta = output.kv_connector_metadata
            worker.handle_preemptions(meta)
            worker.bind_connector_metadata(meta)
            scheduled = output.num_scheduled_tokens.get(req_id, 0)
            for event in release_waits.values():
                event.clear()
            for event in release_finishes.values():
                event.clear()
            worker.start_load_kv(SimpleNamespace() if scheduled else None)
            if scheduled:
                end = request.num_computed_tokens
                start = end - scheduled
                starts.append(start)
                blocks = scheduler.kv_cache_manager.get_blocks(req_id).get_block_ids()
                load_meta = next((item for item in meta.requests if item.load_spec and item.load_spec.can_load), None)
                if load_meta is not None and len(starts) == 1:
                    assert load_meta.load_spec.kvpool_cached_tokens == expected_hit
                    loaded = True
                for layer in range(4):
                    name = f"model.layers.{layer}.attn"
                    spec = specs[name]
                    group_id = layer % 2
                    worker.wait_for_layer_load(name)
                    # Validate restored bytes BEFORE simulating the next model operation.
                    if load_meta is not None:
                        stored_prefix = load_meta.load_spec.kvpool_store_skip_tokens or start
                        for index, bid in enumerate(blocks[group_id]):
                            if bid and (index + 1) * spec.block_size <= stored_prefix:
                                for plane, cache in enumerate(caches[name]):
                                    assert torch.all(
                                        cache[bid] == payload(tokens, (index + 1) * spec.block_size, layer, plane)
                                    )

                    if isinstance(spec, MambaSpec):
                        record_attention_compute_start()
                        # Let an incorrectly early PUT finish before the state
                        # update, so the warm request deterministically sees it.
                        if layer in pool._attention_saved_layers:
                            wait_until(pool.layer_save_finished_events[layer].is_set, pool)
                        # Conv/SSM updates happen inside recurrent attention.
                        compute(layer, blocks[group_id], start, end)
                    else:
                        # Full attention's cache scatter precedes the kernel.
                        compute(layer, blocks[group_id], start, end)
                        before = [cache.clone() for cache in caches[name]] if layer in release_waits else []
                        with attention_transfer_window():
                            assert layer in pool._attention_saved_layers
                            if layer in release_waits:
                                wait_until(release_waits[layer].is_set, pool)
                                assert not release_finishes[layer].wait(timeout=0.01)
                                wait_until(pool.layer_save_finished_events[layer].is_set, pool)
                                assert not pool._compute_recorded_events[layer].is_set()
                                for actual, expected in zip(caches[name], before, strict=True):
                                    assert torch.equal(actual, expected), "GET overwrote an active attention buffer"
                    worker.save_kv_layer(name, caches[name], None)
                assert pool.kv_send_thread.request_queue.unfinished_tasks == 0
            sending, recving = worker.get_finished(output.finished_req_ids)
            result = create_model_runner_output([request] if scheduled else [])
            if scheduled and request.num_computed_tokens < len(tokens):
                result.sampled_token_ids = [[]]
            result.kv_connector_output = KVConnectorOutput(
                finished_sending=sending,
                finished_recving=recving,
                kv_connector_worker_meta=worker.build_connector_worker_meta(),
            )
            scheduler.update_from_output(output, result)
            worker.clear_connector_metadata()
            if req_id not in scheduler.requests:
                terminal = scheduler.schedule()
                worker.bind_connector_metadata(terminal.kv_connector_metadata)
                worker.get_finished(terminal.finished_req_ids)
                worker.clear_connector_metadata()
                break
        assert starts[0] == expected_hit
        assert any(start >= len(tokens) for start in starts), "Decode was not exercised"
        assert loaded == bool(expected_hit)
        assert not scheduler.requests and not scheduler.running and not scheduler.waiting
        assert not scheduler.finished_req_ids and not scheduler.finished_recving_kv_req_ids
        block_pool = scheduler.kv_cache_manager.block_pool
        assert block_pool.get_num_free_blocks() == plan.num_blocks - 1
        assert all(block.ref_cnt == 0 for block in block_pool.blocks if not block.is_null)
        assert all(not manager.req_to_blocks for manager in scheduler.kv_cache_manager.coordinator.single_type_managers)
        assert not scheduler.connector.connector_scheduler.load_specs
        assert not memory_store.leases
        assert not pool.get_block_ids_with_load_errors()
        return expected_hit

    producer = list(range(128))
    assert run_request("cold", producer, 0) == 0
    assert memory_store.complete
    # Evict local APC so each next hit must come from the external store.
    assert scheduler.reset_prefix_cache()
    for values in caches.values():
        for cache in values:
            cache.fill_(-1)
    assert run_request("warm", producer + [128], 128) == 128
    assert scheduler.reset_prefix_cache()
    # Preserve main's complete-hit policy: reserve the final token for replay.
    assert run_request("complete-hit", producer, 127) == 127
    assert scheduler.reset_prefix_cache()
    assert run_request("boundary", producer[:-1], 96) == 96
    assert scheduler.reset_prefix_cache()
    forked = producer.copy()
    forked[64] += 1000
    assert run_request("shared-prefix", forked, 64) == 64
    assert scheduler.reset_prefix_cache()
    for key in list(memory_store.objects):
        if "@1@" in key:
            del memory_store.objects[key]
            memory_store.complete.discard(key)
    assert run_request("missing-state", list(range(160)), 0) == 0
