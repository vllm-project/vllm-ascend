# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-token lifecycle with real scheduling, connectors, threads and CPU bytes.

Only device operations, model computation and the external Mooncake service are
simulated. Keys, hit lengths, masks, block IDs and transfer addresses are produced
by the same constructors and request entry points used in serving.
"""

import ctypes
import json
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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import get_layerwise_kv_cache_specs


class MemoryStore:
    """Mooncake client leaf; copy real bytes through session/range APIs."""

    def __init__(self):
        self.objects = {}
        self.complete = set()
        self.open_reads = set()
        self.open_puts = set()
        self.regions = []
        self.copies = []
        self.lock = threading.Lock()

    def setup(self, **kwargs):
        return 0

    def register_buffer(self, address, size):
        self.regions.append((address, size))
        return 0

    def batch_is_exist(self, keys):
        return [int(key in self.complete) for key in keys]

    def batch_put_session_start(self, keys, sizes, config):
        for key, size in zip(keys, sizes, strict=True):
            assert key not in self.objects
            self.objects[key] = np.zeros(size, dtype=np.uint8)
        self.open_puts.update(keys)
        return [0] * len(keys)

    def batch_put_session_end(self, keys):
        assert set(keys) <= self.open_puts
        self.open_puts.difference_update(keys)
        self.complete.update(keys)
        return [0] * len(keys)

    def batch_put_session_revoke(self, keys):
        for key in keys:
            self.objects.pop(key, None)
        self.open_puts.difference_update(keys)
        self.complete.difference_update(keys)
        return [0] * len(keys)

    def batch_get_session_start(self, keys):
        assert set(keys) <= self.complete
        self.open_reads.update(keys)
        return [0] * len(keys)

    def batch_get_session_end(self, keys):
        self.open_reads.difference_update(keys)
        return 0

    def batch_put_from_multi_buffer_ranges(self, keys, buffers, sizes, offsets):
        return self.copy_ranges(keys, buffers, sizes, offsets, saving=True)

    def batch_get_into_multi_buffer_ranges(self, keys, buffers, sizes, offsets):
        return self.copy_ranges(keys, buffers, sizes, offsets, saving=False)

    def copy_ranges(self, keys, buffers, sizes, offsets, *, saving):
        with self.lock:
            for key, addresses, lengths, starts in zip(keys, buffers, sizes, offsets, strict=True):
                assert key in (self.open_puts if saving else self.open_reads)
                for address, size, offset in zip(addresses, lengths, starts, strict=True):
                    assert any(start <= address < address + size <= start + length for start, length in self.regions)
                    assert 0 <= offset < offset + size <= self.objects[key].size
                    remote = self.objects[key].ctypes.data + offset
                    source, dest = (address, remote) if saving else (remote, address)
                    ctypes.memmove(dest, source, size)
                    self.copies.append((saving, address, size, key))
        return [sum(row) for row in sizes]


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
def memory_store(monkeypatch, tmp_path):
    store = MemoryStore()
    monkeypatch.setitem(
        sys.modules,
        "mooncake.store",
        SimpleNamespace(MooncakeDistributedStore=lambda: store, ReplicateConfig=SimpleNamespace),
    )
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps({"protocol": "ascend", "metadata_server": "test", "master_server_address": "test"})
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0")
    monkeypatch.setenv("ASCEND_GLOBAL_RESOURCE_CONFIG", "{}")
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_rank", lambda: 0)
    register_all_kvcache_specs(None)
    init_none_hash(sha256)
    return store


@pytest.mark.parametrize(
    "recurrent_type",
    [MambaAttentionBackendEnum.MAMBA2, MambaAttentionBackendEnum.GDN_ATTN, MambaAttentionBackendEnum.LINEAR, None],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_raw_tokens_hybrid_roundtrip(memory_store, recurrent_type, wrapped):
    # The fixture installs the external client leaf before importing its backend.
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

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
    retention_interval = None
    plan = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        prefix_cache_retention_interval=retention_interval,
    )
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
        prefix_cache_retention_interval=retention_interval,
    )
    config = create_vllm_config(max_num_batched_tokens=32, block_size=16)
    config.model_config.hf_text_config.num_hidden_layers = 4
    config.model_config.model_arch_config.total_num_hidden_layers = 4
    config.model_config.max_model_len = 512
    config.scheduler_config.max_model_len = 512
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.prefix_cache_retention_interval = retention_interval
    config.cache_config.num_gpu_blocks = plan.num_blocks
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"backend": "mooncake", "use_layerwise": True},
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
    worker.register_kv_caches(caches)
    scheduler = Scheduler(
        config,
        plan,
        StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_size,
    )
    assert isinstance(pool.m_store, MooncakeBackend)
    assert isinstance(pool.kv_send_thread, KVCacheStoreLayerSendingThread)
    assert isinstance(pool.kv_recv_thread, KVCacheStoreLayerRecvingThread)
    assert isinstance(pool.cache_coordinator, AscendStoreCoordinator)
    assert pool.block_key_hybrid
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
                        with attention_transfer_window():
                            assert layer in pool._attention_saved_layers
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
        assert not memory_store.open_reads and not memory_store.open_puts
        assert not pool._put_started_keys
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
        if "@group:1@" in key:
            del memory_store.objects[key]
            memory_store.complete.discard(key)
    assert run_request("missing-state", list(range(160)), 0) == 0
