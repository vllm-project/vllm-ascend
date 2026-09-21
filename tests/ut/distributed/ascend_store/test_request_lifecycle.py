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
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput
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

    payloads: dict[str, list[bytes]] = {}

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


@pytest.fixture(params=[None, "mtp", "dspark"])
def speculative_method(request):
    return request.param


@pytest.fixture(params=["private-state", "full-attention", "full-and-swa"])
def cache_layout(request, speculative_method):
    """Dense attention or mixed C1/C2 + SWA, with optional private state."""
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

    if request.param != "private-state":
        plan.kv_cache_groups.pop(1)
    if request.param == "full-attention":
        plan.kv_cache_groups.pop()
        plan.kv_cache_groups[0].kv_cache_spec = FullAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        )
    if speculative_method:
        draft_names = [f"mtp.{i}.self_attn" for i in range(3 if speculative_method == "dspark" else 1)]
        if request.param == "full-attention":
            plan.kv_cache_groups[0].layer_names.extend(draft_names)
        else:
            plan.kv_cache_groups.append(KVCacheGroupSpec(draft_names, swa))

    def allocate():
        caches = {}
        for group in plan.kv_cache_groups:
            for name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                rows = spec.block_size // spec.tokens_per_state
                shape = (plan.num_blocks, rows, 1, spec.head_size)
                planes = 2 if type(spec) is FullAttentionSpec else 1
                raw = aligned_buffer(planes * int(np.prod(shape)) * torch.empty((), dtype=spec.dtype).element_size())
                tensors = raw.view(spec.dtype).view(planes, *shape)
                caches[name] = tuple(tensors) if planes == 2 else tensors[0]
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
@pytest.mark.parametrize("save_decode_cache", [False, True])
def test_raw_sequence_lifecycle(
    cache_layout, devices_and_store, prefix_unit, load_async, speculative_method, save_decode_cache
):
    plan, allocate = cache_layout
    if len(plan.kv_cache_groups) == 1 and prefix_unit == 32:
        pytest.skip("The upstream single-group scheduler always hashes whole 128-token blocks")
    has_private_state = any(not is_prefix_cacheable(group.kv_cache_spec) for group in plan.kv_cache_groups)
    config = create_vllm_config(
        max_num_batched_tokens=256,
        speculative_method=speculative_method,
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            kv_connector_extra_config={
                "backend": "cpu",
                "load_async": load_async,
                "save_decode_cache": save_decode_cache,
            },
        ),
    )
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
    assert pool.use_eagle == (speculative_method is not None)
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
            sampling_params=SamplingParams(max_tokens=10 if speculative_method else 2),
            pooling_params=None,
            block_hasher=get_request_block_hasher(hash_size, sha256),
        )
        scheduler.add_request(request)
        seen_load = False
        saw_decode = False
        compute_starts = []
        accepted_counts: list[int] = []
        for _ in range(12):
            output = scheduler.schedule()
            assert output.total_num_scheduled_tokens <= 256
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
                input_tokens = list(request.all_token_ids) + output.scheduled_spec_decode_tokens.get(req_id, [])
                for gid, name, plane, tensor in cache_entries(plan, caches):
                    if not is_prefix_cacheable(plan.kv_cache_groups[gid].kv_cache_spec):
                        continue
                    for index, bid in enumerate(block_ids[gid]):
                        if bid and start < (index + 1) * 128 <= end:
                            tensor[bid].fill_(payload_value(input_tokens, (index + 1) * 128, name, plane))
            worker.wait_for_save()
            pool.wait_for_previous_save()
            sending, recving = worker.get_finished(output.finished_req_ids)
            result = create_model_runner_output([request] if scheduled else [])
            if scheduled and request.num_computed_tokens < len(tokens):
                result.sampled_token_ids = [[]]
            drafts = output.scheduled_spec_decode_tokens.get(req_id, [])
            if drafts:
                accepted = (0, 1, 3)[min(len(accepted_counts), 2)]
                accepted = min(accepted, len(drafts))
                accepted_counts.append(accepted)
                result.sampled_token_ids = [drafts[:accepted] + [0]]
            result.kv_connector_output = KVConnectorOutput(
                finished_sending=sending,
                finished_recving=recving,
                kv_connector_worker_meta=worker.build_connector_worker_meta(),
            )
            computed_before_update = request.num_computed_tokens
            scheduler.update_from_output(output, result)
            if drafts:
                assert request.num_computed_tokens == computed_before_update - len(drafts) + accepted
            if speculative_method:
                scheduler.update_draft_token_ids(
                    DraftTokenIds([req_id], [[1000 + request.num_tokens + i for i in range(3)]])
                )
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
        if req_id == "producer":
            assert sum(start < len(tokens) for start in compute_starts) >= 2
        if speculative_method:
            assert accepted_counts[:3] == [0, 1, 3]
        assert compute_starts[0] == expected_hit
        assert_scheduler_empty(scheduler)
        assert not scheduler.connector.connector_scheduler.load_specs
        assert all(not manager.req_to_blocks for manager in scheduler.kv_cache_manager.coordinator.single_type_managers)
        assert not pool._invalid_block_ids
        return list(request.all_token_ids)

    try:
        run_request("producer", producer_tokens, 0)
        saved = dict(MemoryBackend.payloads)
        assert saved
        # Clear the real local prefix cache; the next request must use the Store.
        for case, (length, fork, hit) in enumerate(
            [
                (125, None, 0),
                (126, None, 0),
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
            if not has_private_state and not speculative_method and fork is None:
                # Legacy full hits still reserve only the final token.
                hit = {125: 0, 126: 0, 127: 0, 128: 127, 129: 128, 256: 255, 257: 256, 512: 511, 513: 512}[length]
            generated = run_request(f"consumer-{case}", tokens, hit)
            assert bool(pool.m_store.loads) == bool(hit)
            if speculative_method and length in (125, 126):
                assert scheduler.reset_prefix_cache()
                for _, _, _, tensor in cache_entries(plan, caches):
                    tensor.fill_(-3)
                # Reuse a page completed only after rejected drafts rolled back.
                run_request(f"decoded-prefix-{case}", generated, 128 if save_decode_cache else 0)
            if has_private_state:
                assert not any("@group:1@" in key for key in MemoryBackend.payloads)
        assert scheduler.reset_prefix_cache()
        MemoryBackend.payloads = dict(saved)
        seed_length = 256 if speculative_method else 128
        seed_hit = 128 if speculative_method else (0 if has_private_state else 127)
        run_request("local-prefix", producer_tokens[:seed_length], seed_hit)
        run_request("mixed-local-remote", list(range(513)), 512, expected_local=128)
        assert scheduler.reset_prefix_cache()
        missing_group = len(plan.kv_cache_groups) - 1
        MemoryBackend.payloads = {key: value for key, value in saved.items() if f"@group:{missing_group}@" not in key}
        run_request("missing-group", producer_tokens, 0)
        if has_private_state:
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
