# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU request lifecycle; only device operations and the external store are fake."""

import ctypes
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import assert_scheduler_empty, create_model_runner_output, create_vllm_config
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec, is_prefix_cacheable
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import backend_map
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)


def aligned_buffer(size):
    alignment = 2 * 1024 * 1024
    data = np.zeros(size + alignment, dtype=np.uint8)
    offset = -data.ctypes.data % alignment
    return torch.from_numpy(data[offset : offset + size])


class MemoryBackend(Backend):
    """The Store leaf API copies real bytes using production-generated addresses."""

    payloads = {}
    requires_exists_before_put = False

    def __init__(self, parallel_config, **kwargs):
        self.regions = []
        self.loads = []
        self.lookups = []

    def set_device(self):
        pass

    def register_buffer(self, ptrs, lengths):
        self.regions.extend(zip(ptrs, lengths))

    def exists(self, keys):
        self.lookups.extend(keys)
        return [int(key in self.payloads) for key in keys]

    def check_address(self, addr, size):
        assert any(start <= addr and addr + size <= start + length for start, length in self.regions)

    def put(self, keys, addrs, sizes):
        for key, row, lengths in zip(keys, addrs, sizes, strict=True):
            for addr, size in zip(row, lengths, strict=True):
                self.check_address(addr, size)
            self.payloads[key] = [ctypes.string_at(addr, size) for addr, size in zip(row, lengths, strict=True)]
        return [0] * len(keys)

    def get(self, keys, addrs, sizes):
        for key, row, lengths in zip(keys, addrs, sizes, strict=True):
            for addr, size, data in zip(row, lengths, self.payloads[key], strict=True):
                self.check_address(addr, size)
                assert len(data) == size
                ctypes.memmove(addr, data, size)
            self.loads.append(key)
        return [0] * len(keys)


@pytest.fixture
def cache_layout():
    """Mixed C1/C2 token pages, private ring, and sliding-window cache."""
    specs = {
        "model.layers.0.long_kv_cache": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            tokens_per_state=2,
        ),
        "model.layers.1.long_kv_cache": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
        ),
    }
    ring = CircularBufferSpec(block_size=32, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32)
    swa = AscendSlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    plan = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=specs)),
            KVCacheGroupSpec(["model.layers.0.state_cache"], ring),
            KVCacheGroupSpec(["model.layers.0.swa_cache"], swa),
        ],
    )

    def allocate():
        caches = {}
        for group in plan.kv_cache_groups:
            for name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                rows = spec.block_size // spec.tokens_per_state
                shape = (plan.num_blocks, rows, 1, spec.head_size)
                raw = aligned_buffer(int(np.prod(shape)) * torch.empty((), dtype=spec.dtype).element_size())
                caches[name] = raw.view(spec.dtype).view(shape)
        return caches

    return plan, allocate


@pytest.fixture
def devices_and_store(monkeypatch):
    monkeypatch.setattr(MemoryBackend, "payloads", {})
    monkeypatch.setitem(backend_map, "cpu", {"path": __name__, "name": "MemoryBackend"})
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_rank", lambda: 0)
    register_all_kvcache_specs(None)
    init_none_hash(sha256)
    # Unix-domain socket paths have a short platform-dependent size limit.
    with TemporaryDirectory(dir="/tmp") as rpc_dir:
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", rpc_dir)
        yield


def cache_entries(plan, caches):
    for gid, group in enumerate(plan.kv_cache_groups):
        for name in group.layer_names:
            values = caches[name]
            for plane, tensor in enumerate(values if isinstance(values, tuple) else (values,)):
                yield gid, name, plane, tensor


def payload_value(tokens, end, name, plane):
    # Independent of block IDs, hashes, keys and transfer metadata.
    return (sum(tokens[:end]) + sum(name.encode()) + plane * 11) % 101 + 1


@pytest.mark.parametrize("prefix_unit", [None, 32, 128])
@pytest.mark.parametrize("load_async", [False, True])
def test_raw_sequence_lifecycle(cache_layout, devices_and_store, prefix_unit, load_async):
    plan, allocate = cache_layout
    config = create_vllm_config(
        max_num_batched_tokens=1024,
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            kv_connector_extra_config={"backend": "cpu", "load_async": load_async},
        ),
    )
    config.scheduler_config.max_num_batched_tokens = 256
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.prefix_match_unit = prefix_unit
    config.cache_config.num_gpu_blocks = plan.num_blocks
    block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
    worker = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, plan)
    caches = allocate()
    worker.register_kv_caches(caches)
    pool = worker.connector_worker
    assert isinstance(pool.m_store, MemoryBackend)
    assert type(worker.lookup_server.socket).__module__.startswith("zmq")
    assert not pool.m_store.payloads
    assert isinstance(pool.kv_send_thread, KVCacheStoreSendingThread)
    assert isinstance(pool.kv_recv_thread, KVCacheStoreRecvingThread) == load_async
    assert pool.hash_block_size == hash_size == (prefix_unit or 128)
    assert pool.cache_transfer_granularity == 128
    scheduler = Scheduler(
        config,
        plan,
        StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_size,
    )
    producer_tokens = list(range(512))

    def run_request(req_id, tokens, expected_hit, expected_local=0):
        request = Request(
            request_id=req_id,
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=2),
            pooling_params=None,
            block_hasher=get_request_block_hasher(hash_size, sha256),
        )
        scheduler.add_request(request)
        seen_load = False
        saw_decode = False
        compute_starts = []
        for _ in range(12):
            output = scheduler.schedule()
            meta = output.kv_connector_metadata
            worker.handle_preemptions(meta)
            worker.bind_connector_metadata(meta)
            # Mark private state at its newly allocated location before loading.
            block_ids = (
                scheduler.kv_cache_manager.get_blocks(req_id).get_block_ids() if req_id in scheduler.requests else None
            )
            if block_ids:
                for gid, _, _, tensor in cache_entries(plan, caches):
                    if not is_prefix_cacheable(plan.kv_cache_groups[gid].kv_cache_spec):
                        for bid in block_ids[gid]:
                            tensor[bid].fill_(-7)
            worker.start_load_kv(SimpleNamespace() if output.total_num_scheduled_tokens else None)
            if load_async and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                assert output.total_num_scheduled_tokens == 0
                deadline = time.monotonic() + 5
                while not pool.kv_recv_thread.finished_requests:
                    pool.kv_recv_thread.raise_if_failed()
                    assert time.monotonic() < deadline, "asynchronous receive did not finish"
                    time.sleep(0.001)
            for req_meta in meta.requests:
                if req_meta.load_spec is None or not req_meta.load_spec.can_load:
                    continue
                assert not seen_load
                seen_load = True
                assert req_meta.load_spec.kvpool_cached_tokens == expected_hit
                assert req_meta.load_spec.vllm_cached_tokens == expected_local
                for gid, name, plane, tensor in cache_entries(plan, caches):
                    for index, bid in enumerate(block_ids[gid]):
                        if not is_prefix_cacheable(plan.kv_cache_groups[gid].kv_cache_spec):
                            assert torch.all(tensor[bid] == -7)
                        elif bid and (index + 1) * 128 <= expected_hit:
                            assert torch.all(tensor[bid] == payload_value(tokens, (index + 1) * 128, name, plane))
            scheduled = output.num_scheduled_tokens.get(req_id, 0)
            if scheduled:
                end = request.num_computed_tokens
                start = end - scheduled
                compute_starts.append(start)
                saw_decode |= start >= len(tokens)
                # Model-compute leaf: write deterministic CPU bytes to allocated pages.
                for gid, name, plane, tensor in cache_entries(plan, caches):
                    if not is_prefix_cacheable(plan.kv_cache_groups[gid].kv_cache_spec):
                        continue
                    for index, bid in enumerate(block_ids[gid]):
                        if bid and start < (index + 1) * 128 <= end:
                            tensor[bid].fill_(payload_value(request.all_token_ids, (index + 1) * 128, name, plane))
            worker.wait_for_save()
            pool.wait_for_previous_save()
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
                # Consume the finished-ID notification and release Store references.
                output = scheduler.schedule()
                worker.bind_connector_metadata(output.kv_connector_metadata)
                worker.get_finished(output.finished_req_ids)
                worker.clear_connector_metadata()
                break
        assert seen_load == (expected_hit > 0)
        assert saw_decode
        assert compute_starts[0] == expected_hit
        assert_scheduler_empty(scheduler)
        assert not scheduler.connector.connector_scheduler.load_specs
        assert all(not manager.req_to_blocks for manager in scheduler.kv_cache_manager.coordinator.single_type_managers)
        assert not pool._invalid_block_ids

    try:
        run_request("producer", producer_tokens, 0)
        saved = dict(MemoryBackend.payloads)
        assert saved and not any("@group:1@" in key for key in saved)
        # Clear the real local prefix cache; the next request must use the Store.
        for case, (length, fork, hit) in enumerate(
            [
                (127, None, 0),
                (128, None, 0),
                (129, None, 128),
                (256, None, 128),
                (257, None, 256),
                (512, None, 384),
                (513, None, 512),
                (512, 127, 0),
                (512, 128, 128),
                (512, 255, 128),
                (512, 256, 256),
                (512, 511, 384),
            ]
        ):
            assert scheduler.reset_prefix_cache()
            MemoryBackend.payloads = dict(saved)
            pool.m_store.loads.clear()
            for _, _, _, tensor in cache_entries(plan, caches):
                tensor.fill_(-3)
            tokens = list(range(length))
            if fork is not None:
                tokens[fork] += 1000
            run_request(f"consumer-{case}", tokens, hit)
            assert bool(pool.m_store.loads) == bool(hit)
            assert not any("@group:1@" in key for key in MemoryBackend.payloads)
        assert scheduler.reset_prefix_cache()
        MemoryBackend.payloads = dict(saved)
        run_request("local-prefix", producer_tokens[:128], 0)
        run_request("mixed-local-remote", list(range(513)), 512, expected_local=128)
        assert scheduler.reset_prefix_cache()
        MemoryBackend.payloads = {key: value for key, value in saved.items() if "@group:2@" not in key}
        run_request("missing-swa", producer_tokens, 0)
        assert not any("@group:1@" in key for key in pool.m_store.lookups)
    finally:
        server = worker.lookup_server
        server.running = False
        client = scheduler.connector.connector_scheduler.client
        if client is not None:
            client.lookup(0, [], list(range(len(plan.kv_cache_groups))))
            server.thread.join(timeout=5)
            client.close()
            client.ctx.term()
        server.close()
        server.ctx.term()
