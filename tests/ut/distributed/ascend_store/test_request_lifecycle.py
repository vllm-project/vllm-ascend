# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU request lifecycle; only device operations and the external store are fake."""

import ctypes
import threading
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
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
from vllm_ascend.utils import vllm_version_is


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
        return [True] * len(keys)

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
            plan.kv_cache_groups[0].is_eagle_group = True
        else:
            plan.kv_cache_groups.append(KVCacheGroupSpec(draft_names, swa, is_eagle_group=True))

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


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
@pytest.mark.parametrize("prefix_unit", [None, 128])
def test_private_only_layout_rejected(devices_and_store, role, prefix_unit):
    config = create_vllm_config(
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            kv_connector_extra_config={"backend": "cpu"},
        ),
    )
    config.cache_config.prefix_match_unit = prefix_unit
    spec = CircularBufferSpec(block_size=32, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32)
    plan = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.state_cache"], spec)],
    )
    with pytest.raises(AssertionError, match="AscendStore requires at least one prefix-cacheable KV cache group"):
        KVConnectorFactory.create_connector(config, role, plan)


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
    # Match the planner's role annotation so kv_both keeps the EAGLE safety drop.
    plan.kv_transfer_config = config.kv_transfer_config
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
    requires_load_wait = load_async or len(plan.kv_cache_groups) > 1
    assert pool.requires_load_wait == requires_load_wait
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
            if requires_load_wait and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                assert output.total_num_scheduled_tokens == 0
                if load_async:
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
            # Same fence the connector applies at the start of the next step:
            # drain the saves of requests whose blocks this output released.
            pool.handle_released_saves(meta.released_req_ids)
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
            output_tokens_before_update = request.num_output_tokens
            scheduler.update_from_output(output, result)
            if drafts:
                expected_computed = computed_before_update - len(drafts) + accepted
                if not vllm_version_is("0.32.0"):
                    # vLLM main trims ``num_computed_tokens`` to cover only the
                    # tokens actually kept when a request stops mid token-list.
                    appended = request.num_output_tokens - output_tokens_before_update
                    expected_computed -= (accepted + 1) - appended
                assert request.num_computed_tokens == expected_computed
            if speculative_method:
                scheduler.update_draft_token_ids(
                    DraftTokenIds([req_id], [[1000 + request.num_tokens + i for i in range(3)]])
                )
            worker.clear_connector_metadata()
            if req_id not in scheduler.requests:
                # Consume the finished-ID notification and release Store references.
                output = scheduler.schedule()
                # The connector fences this output like any other: drain the
                # finished request's final saves before its blocks are reused.
                worker.handle_preemptions(output.kv_connector_metadata)
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


@pytest.fixture
def bulk_isolation_runtime(cache_layout, devices_and_store):
    """Real scheduler/connector/cache pages; fake only model compute and Store I/O."""
    plan, allocate = cache_layout
    config = create_vllm_config(
        max_num_seqs=4,
        max_num_batched_tokens=512,
        kv_transfer_config=KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
            kv_connector_extra_config={"backend": "cpu", "load_async": False},
        ),
    )
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.num_gpu_blocks = plan.num_blocks
    plan.kv_transfer_config = config.kv_transfer_config
    block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
    worker = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, plan)
    caches = allocate()
    worker.register_kv_caches(caches)
    pool = worker.connector_worker
    scheduler = Scheduler(
        config, plan, StructuredOutputManager(config), block_size=block_size, hash_block_size=hash_size
    )
    runtime = SimpleNamespace(
        config=config,
        plan=plan,
        worker=worker,
        pool=pool,
        scheduler=scheduler,
        caches=caches,
        hash_size=hash_size,
        forwarded=[],
        computed_pages=[],
    )

    def add(req_id, tokens, max_tokens=1):
        request = Request(
            request_id=req_id,
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=max_tokens),
            pooling_params=None,
            block_hasher=get_request_block_hasher(hash_size, sha256),
        )
        scheduler.add_request(request)
        return request

    def prepare():
        output = scheduler.schedule()
        worker.handle_preemptions(output.kv_connector_metadata)
        worker.bind_connector_metadata(output.kv_connector_metadata)
        return output

    def compute(output):
        scheduled = [scheduler.requests[req_id] for req_id in output.num_scheduled_tokens]
        for request in scheduled:
            req_id = request.request_id
            start = request.num_computed_tokens - output.num_scheduled_tokens[req_id]
            end = request.num_computed_tokens
            runtime.forwarded.append(req_id)
            blocks = scheduler.kv_cache_manager.get_blocks(req_id).get_block_ids()
            for gid, name, plane, tensor in cache_entries(plan, caches):
                if not is_prefix_cacheable(plan.kv_cache_groups[gid].kv_cache_spec):
                    continue
                for index, bid in enumerate(blocks[gid]):
                    if bid and start < (index + 1) * 128 <= end:
                        value = payload_value(list(request.all_token_ids), (index + 1) * 128, name, plane)
                        tensor[bid].fill_(value)
                        runtime.computed_pages.append((req_id, tensor, bid, value))
        result = create_model_runner_output(scheduled)
        for index, request in enumerate(scheduled):
            if request.num_computed_tokens < request.num_prompt_tokens:
                result.sampled_token_ids[index] = []
        return result

    def finish(output, result, aggregator=None, peer_outputs=()):
        worker.wait_for_save()
        pool.handle_released_saves(output.kv_connector_metadata.released_req_ids)
        transfer = worker.get_transfer_results(output.finished_req_ids)
        result.kv_connector_output = KVConnectorOutput(
            finished_sending=transfer.finished_sending,
            finished_recving=transfer.finished_recving,
            failed_recving=transfer.failed_recving,
            kv_connector_worker_meta=worker.build_connector_worker_meta(),
        )
        if aggregator is not None:
            result = aggregator.aggregate([result, *peer_outputs])
        updates = scheduler.update_from_output(output, result)
        worker.clear_connector_metadata()
        return transfer, updates

    def step():
        output = prepare()
        if output.has_sync_kv_loads:
            worker.start_load_kv(SimpleNamespace())
        result = compute(output)
        if not output.has_sync_kv_loads:
            worker.start_load_kv(SimpleNamespace() if output.total_num_scheduled_tokens else None)
        transfer, updates = finish(output, result)
        return output, transfer, updates

    def run_until_finished(request):
        for _ in range(12):
            step()
            if request.request_id not in scheduler.requests:
                step()  # Consume the completed request's final released-ID notification.
                return
        pytest.fail(f"request did not finish: {request.request_id}")

    runtime.add = add
    runtime.prepare = prepare
    runtime.compute = compute
    runtime.finish = finish
    runtime.step = step
    runtime.run_until_finished = run_until_finished
    assert len(plan.kv_cache_groups) > 1
    assert pool.load_async is False
    assert pool.requires_load_wait is True
    assert pool.kv_recv_thread is None
    assert scheduler.connector.connector_scheduler.requires_load_wait is True
    assert worker.requires_mamba_state_copy_after_load is False
    try:
        yield runtime
    finally:
        pool.shutdown()
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


@pytest.mark.parametrize("speculative_method", [None], indirect=True)
@pytest.mark.parametrize("cache_layout", ["full-and-swa"], indirect=True)
@pytest.mark.parametrize("failure_result", ["partial", "missing", "short", "partial-with-local-prefix"])
def test_sync_bulk_load_failure_isolated_before_forward(bulk_isolation_runtime, monkeypatch, failure_result):
    runtime = bulk_isolation_runtime
    scheduler, worker, pool = runtime.scheduler, runtime.worker, runtime.pool
    tokens = list(range(257))
    producer = runtime.add("producer", tokens)
    runtime.run_until_finished(producer)
    assert pool.m_store.payloads
    assert scheduler.reset_prefix_cache()
    for _, _, _, tensor in cache_entries(runtime.plan, runtime.caches):
        tensor.fill_(-3)
    shared_local_prefix = failure_result == "partial-with-local-prefix"
    if shared_local_prefix:
        local_prefix = runtime.add("local-prefix", tokens[:129])
        runtime.run_until_finished(local_prefix)
    original_get = pool.m_store.get
    injected = []

    def fail_one_group(keys, addrs, sizes):
        if injected:
            return original_get(keys, addrs, sizes)
        failed = max(index for index, key in enumerate(keys) if "@group:1@" in key)
        assert any("@group:0@" in key for key in keys)
        for index, (key, row, lengths) in enumerate(zip(keys, addrs, sizes, strict=True)):
            if index != failed:
                assert original_get([key], [row], [lengths]) == [0]
        injected.append(keys[failed])
        results = [0] * len(keys)
        results[failed] = -3007
        if failure_result == "missing":
            return None
        if failure_result == "short":
            return results[:-1]
        return results

    monkeypatch.setattr(pool.m_store, "get", fail_one_group)
    bad = runtime.add("bad", tokens)
    healthy_tokens = (tokens[:128] if shared_local_prefix else []) + list(range(1000, 1129))
    healthy = runtime.add("healthy", healthy_tokens, max_tokens=2)
    output = runtime.prepare()
    assert bad.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert bad.request_id not in output.num_scheduled_tokens
    assert healthy.request_id in output.num_scheduled_tokens
    assert output.has_sync_kv_loads is False
    assert bad.request_id in output.kv_connector_metadata.loading_req_ids
    shared_pages = []
    if shared_local_prefix:
        bad_blocks = scheduler.kv_cache_manager.get_blocks(bad.request_id).get_block_ids()
        healthy_blocks = scheduler.kv_cache_manager.get_blocks(healthy.request_id).get_block_ids()
        assert bad_blocks[0][0] == healthy_blocks[0][0] != 0
        for gid, _, _, tensor in cache_entries(runtime.plan, runtime.caches):
            if gid == 0:
                block_id = healthy_blocks[0][0]
                shared_pages.append((tensor, block_id, tensor[block_id].clone()))
        assert shared_pages
        load_meta = next(meta for meta in output.kv_connector_metadata.requests if meta.req_id == bad.request_id)
        assert load_meta.load_spec.vllm_cached_tokens == 128
    result = runtime.compute(output)
    worker.start_load_kv(SimpleNamespace())
    assert injected
    assert not any(req_id == bad.request_id for req_id in runtime.forwarded)
    healthy_pages = [page for page in runtime.computed_pages if page[0] == healthy.request_id]
    assert healthy_pages
    assert all(torch.all(tensor[bid] == value) for _, tensor, bid, value in healthy_pages)
    assert all(torch.equal(tensor[bid], before) for tensor, bid, before in shared_pages)
    transfer, updates = runtime.finish(output, result)
    assert transfer.failed_recving == transfer.finished_recving == {bad.request_id}
    assert bad.status == RequestStatus.FINISHED_ERROR
    assert bad.request_id not in scheduler.requests
    assert healthy.request_id in scheduler.requests
    assert any(
        item.request_id == bad.request_id and not item.new_token_ids
        for batch in updates.values()
        for item in batch.outputs
    )
    runtime.run_until_finished(healthy)
    assert healthy.status != RequestStatus.FINISHED_ERROR
    third = runtime.add("after-failure", tokens)
    runtime.run_until_finished(third)
    assert third.status != RequestStatus.FINISHED_ERROR
    assert third.request_id in runtime.forwarded
    assert bad.request_id not in runtime.forwarded
    assert pool.kv_send_thread._fatal_error is None
    assert_scheduler_empty(scheduler)


@pytest.mark.parametrize("speculative_method", [None], indirect=True)
@pytest.mark.parametrize("cache_layout", ["full-and-swa"], indirect=True)
def test_sync_bulk_load_cancel_retains_blocks_until_receive_completion(bulk_isolation_runtime, monkeypatch):
    runtime = bulk_isolation_runtime
    scheduler, worker, pool = runtime.scheduler, runtime.worker, runtime.pool
    tokens = list(range(257))
    producer = runtime.add("producer", tokens)
    runtime.run_until_finished(producer)
    assert scheduler.reset_prefix_cache()
    request = runtime.add("cancel-during-load", tokens)
    output = runtime.prepare()
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert output.num_scheduled_tokens == {}
    result = runtime.compute(output)
    entered, release = threading.Event(), threading.Event()
    original_get = pool.m_store.get
    errors = []

    def held_get(keys, addrs, sizes):
        entered.set()
        assert release.wait(timeout=5), "test did not release the in-flight store operation"
        return original_get(keys, addrs, sizes)

    def start_worker_load():
        try:
            worker.start_load_kv(None)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(pool.m_store, "get", held_get)
    dispatch = threading.Thread(target=start_worker_load)
    dispatch.start()
    try:
        assert entered.wait(timeout=5)
        scheduler.finish_requests([request.request_id], RequestStatus.FINISHED_ABORTED)
        assert request.request_id in scheduler.requests
        assert request.status == RequestStatus.FINISHED_ABORTED
        assert any(
            manager.req_to_blocks.get(request.request_id)
            for manager in scheduler.kv_cache_manager.coordinator.single_type_managers
        )
    finally:
        release.set()
        dispatch.join(timeout=5)
    assert not dispatch.is_alive()
    assert not errors
    transfer, _ = runtime.finish(output, result)
    assert transfer.finished_recving == {request.request_id}
    assert not transfer.failed_recving
    assert request.request_id not in scheduler.requests
    assert request.request_id not in runtime.forwarded
    runtime.step()
    assert_scheduler_empty(scheduler)


@pytest.mark.parametrize("speculative_method", [None], indirect=True)
@pytest.mark.parametrize("cache_layout", ["full-and-swa"], indirect=True)
def test_sync_bulk_load_failure_waits_for_all_worker_completions(bulk_isolation_runtime, monkeypatch):
    runtime = bulk_isolation_runtime
    scheduler, worker, pool = runtime.scheduler, runtime.worker, runtime.pool
    tokens = list(range(257))
    producer = runtime.add("producer", tokens)
    runtime.run_until_finished(producer)
    assert scheduler.reset_prefix_cache()
    original_get = pool.m_store.get

    def fail_one_group(keys, addrs, sizes):
        result = original_get(keys, addrs, sizes)
        result[-1] = -3007
        return result

    monkeypatch.setattr(pool.m_store, "get", fail_one_group)
    bad = runtime.add("one-worker-failed", tokens)
    healthy = runtime.add("healthy", list(range(1000, 1129)), max_tokens=3)
    output = runtime.prepare()
    assert bad.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert bad.request_id not in output.num_scheduled_tokens
    result = runtime.compute(output)
    worker.start_load_kv(SimpleNamespace())
    aggregator = KVOutputAggregator(expected_finished_count=2)
    peer = create_model_runner_output([])
    peer.kv_connector_output = KVConnectorOutput()
    transfer, _ = runtime.finish(output, result, aggregator, [peer])
    assert transfer.failed_recving == transfer.finished_recving == {bad.request_id}
    assert not result.kv_connector_output.failed_recving
    assert not result.kv_connector_output.finished_recving
    assert bad.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert bad.request_id in scheduler.requests
    assert all(
        manager.req_to_blocks.get(bad.request_id)
        for manager in scheduler.kv_cache_manager.coordinator.single_type_managers
    )

    # The failing worker reports only once. A later completion from its peer
    # must carry the retained failure through the real output aggregator.
    output = runtime.prepare()
    assert bad.request_id not in output.num_scheduled_tokens
    result = runtime.compute(output)
    worker.start_load_kv(SimpleNamespace())
    peer = create_model_runner_output([])
    peer.kv_connector_output = KVConnectorOutput(finished_recving={bad.request_id})
    transfer, _ = runtime.finish(output, result, aggregator, [peer])
    assert not transfer.failed_recving
    assert not transfer.finished_recving
    assert result.kv_connector_output.failed_recving == {bad.request_id}
    assert result.kv_connector_output.finished_recving == {bad.request_id}
    assert bad.status == RequestStatus.FINISHED_ERROR
    assert bad.request_id not in scheduler.requests
    assert bad.request_id not in runtime.forwarded
    runtime.run_until_finished(healthy)
    assert healthy.status != RequestStatus.FINISHED_ERROR
    assert_scheduler_empty(scheduler)
