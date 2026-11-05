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
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes
from vllm.v1.core.sched.interface import PauseState
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


class BytePoolingSession:
    """Drive real scheduler/connector/transfer paths; only model compute is synthetic."""

    def __init__(self, layout="strided", load_async=False, use_v2=True, num_blocks=16):
        self.block_size = 128
        self.scale = 2 if layout == "segmented" else 1
        rows = self.block_size // self.scale
        padding = 0 if layout == "contiguous" else 7
        names = [f"model.layers.{layer}.self_attn.attn" for layer in range(2)]
        spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.uint8)
        self.plan = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(names, spec)],
        )
        config = create_vllm_config(
            max_num_batched_tokens=128,
            kv_transfer_config=KVTransferConfig(
                kv_connector="AscendStoreConnector",
                kv_role="kv_both",
                kv_load_failure_policy="recompute",
                kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
                kv_connector_extra_config={"backend": "cpu", "load_async": load_async},
            ),
        )
        assert config.use_v2_model_runner is use_v2
        config.scheduler_config.max_model_len = 1024
        config.cache_config.num_gpu_blocks = num_blocks
        self.worker = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, self.plan)
        self.caches = {}
        self.padding = []
        for name in names:
            raw = aligned_buffer(2 * num_blocks * self.scale * (rows + padding) * 8)
            backing = raw.view(2, num_blocks * self.scale, rows + padding, 1, 8)
            backing.fill_(253)
            self.caches[name] = tuple(backing[:, :, :rows])
            self.padding.append(backing[:, :, rows:])
        self.worker.register_kv_caches(self.caches)
        self.pool = self.worker.connector_worker
        self.backend = self.pool.m_store
        self.scheduler = Scheduler(
            config,
            self.plan,
            StructuredOutputManager(config),
            block_size=128,
            hash_block_size=128,
        )
        self.computed = []
        self.reported_errors = []

    def new_request(self, req_id, tokens):
        return Request(
            request_id=req_id,
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_block_hasher(128, sha256),
        )

    def block_ids(self, request):
        return self.scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()[0]

    def block_views(self, request):
        for index, block_id in enumerate(self.block_ids(request)):
            for _, name, plane, tensor in cache_entries(self.plan, self.caches):
                yield (
                    index,
                    name,
                    plane,
                    tensor[block_id * self.scale : (block_id + 1) * self.scale],
                )

    def expected_bytes(self, request, index, name, plane, shape):
        # Depends on token content and logical position, never on ReqMeta, keys or physical IDs.
        end = (index + 1) * self.block_size
        seed = sum(request.prompt_token_ids[:end]) + end + sum(name.encode()) + 17 * plane
        return ((torch.arange(int(np.prod(shape))).view(shape) + seed) % 251).to(torch.uint8)

    def assert_loaded(self, request, hit_tokens):
        for index, name, plane, view in self.block_views(request):
            if (index + 1) * self.block_size <= hit_tokens:
                assert torch.equal(view, self.expected_bytes(request, index, name, plane, view.shape))
        self.assert_padding()

    def assert_padding(self):
        for padding in self.padding:
            assert torch.all(padding == 253), "KV transfer overwrote a physical-page gap"

    def execute(self, output, request=None, before_compute=None):
        meta = output.kv_connector_metadata
        self.worker.handle_preemptions(meta)
        self.worker.bind_connector_metadata(meta)
        self.worker.start_load_kv(SimpleNamespace() if output.total_num_scheduled_tokens else None)
        sending, recving = self.worker.get_finished(output.finished_req_ids)
        if request is not None and request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            deadline = time.monotonic() + 5
            while request.request_id not in recving:
                self.pool.kv_recv_thread.raise_if_failed()
                assert time.monotonic() < deadline, "receive thread did not complete"
                time.sleep(0.001)
                done_sending, done_recving = self.worker.get_finished(output.finished_req_ids)
                sending |= done_sending
                recving |= done_recving
        errors = self.worker.get_block_ids_with_load_errors()
        self.reported_errors.append(errors)
        scheduled = output.num_scheduled_tokens.get(request.request_id, 0) if request else 0
        if before_compute is not None:
            before_compute(output)
        if scheduled:
            end = request.num_computed_tokens
            start = end - scheduled
            self.computed.append((request.request_id, start, end))
            for index, name, plane, view in self.block_views(request):
                if start < (index + 1) * self.block_size <= end:
                    view.copy_(self.expected_bytes(request, index, name, plane, view.shape))
        self.worker.wait_for_save()
        result = create_model_runner_output([request] if scheduled else [])
        if scheduled and request.num_computed_tokens < len(request.prompt_token_ids):
            result.sampled_token_ids = [[]]
        result.kv_connector_output = KVConnectorOutput(
            finished_sending=sending,
            finished_recving=recving,
            invalid_block_ids=errors,
            kv_connector_worker_meta=self.worker.build_connector_worker_meta(),
        )
        self.scheduler.update_from_output(output, result)
        self.worker.clear_connector_metadata()

    def run(self, request, before_compute=None, add_request=True):
        if add_request:
            self.scheduler.add_request(request)
        for _ in range(12):
            self.execute(self.scheduler.schedule(), request, before_compute)
            if request.request_id not in self.scheduler.requests:
                self.execute(self.scheduler.schedule())
                assert_scheduler_empty(self.scheduler)
                self.assert_padding()
                return
        pytest.fail("request did not finish after load/recompute")

    def clear_local_cache(self):
        assert self.scheduler.reset_prefix_cache()
        for _, _, _, tensor in cache_entries(self.plan, self.caches):
            tensor.fill_(252)

    def close(self):
        self.worker.shutdown()
        self.scheduler.connector.shutdown()


@pytest.fixture
def byte_pooling_session(devices_and_store, request, monkeypatch):
    def create(**kwargs):
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if kwargs.get("use_v2", True) else "0")
        session = BytePoolingSession(**kwargs)
        request.addfinalizer(session.close)
        return session

    return create


@pytest.mark.parametrize("layout", ["contiguous", "strided", "segmented"])
@pytest.mark.parametrize("load_async", [False, True])
def test_scheduler_worker_roundtrip_preserves_bytes_and_page_gaps(byte_pooling_session, layout, load_async):
    session = byte_pooling_session(layout=layout, load_async=load_async)
    producer = session.new_request("byte-producer", list(range(384)))
    producer_blocks = []
    session.run(producer, lambda output: producer_blocks.extend(session.block_ids(producer)))
    assert session.backend.payloads
    session.clear_local_cache()
    consumer = session.new_request("byte-consumer", list(range(385)))
    loaded = []

    def check_loaded(output):
        for meta in output.kv_connector_metadata.requests:
            if meta.load_spec is not None and meta.load_spec.can_load:
                session.assert_loaded(consumer, 384)
                loaded.extend(session.block_ids(consumer)[:3])

    session.run(consumer, check_loaded)
    assert len(loaded) == 3
    assert loaded != producer_blocks[-3:]
    assert session.backend.loads
    starts = [start for req_id, start, _ in session.computed if req_id == consumer.request_id]
    assert starts == [384]


@pytest.mark.parametrize("use_v2", [False, True])
def test_preempted_request_restores_bytes_into_new_blocks_before_appending(byte_pooling_session, use_v2):
    session = byte_pooling_session(use_v2=use_v2)
    request = session.new_request("byte-resume", list(range(384)))
    session.scheduler.add_request(request)
    session.execute(session.scheduler.schedule(), request)
    session.pool.handle_released_saves({request.request_id})
    previous_blocks = list(session.block_ids(request))
    assert session.scheduler.reset_prefix_cache(reset_running_requests=True)
    # Deliver the release notification before the next tick resumes the request.
    session.scheduler.set_pause_state(PauseState.PAUSED_ALL)
    session.execute(session.scheduler.schedule())
    session.scheduler.set_pause_state(PauseState.UNPAUSED)
    # Old addresses contain deliberately wrong bytes; a stale block table cannot pass.
    for _, _, _, tensor in cache_entries(session.plan, session.caches):
        tensor.fill_(252)
    restored = []

    def check_restored(output):
        for meta in output.kv_connector_metadata.requests:
            if meta.load_spec is not None and meta.load_spec.can_load:
                session.assert_loaded(request, 128)
                restored.extend(session.block_ids(request))

    session.run(request, check_restored, add_request=False)
    assert restored and restored[0] != previous_blocks[0]
    session.clear_local_cache()
    consumer = session.new_request("after-resume", list(range(385)))
    loaded = []

    def check_complete_prefix(output):
        for meta in output.kv_connector_metadata.requests:
            if meta.load_spec is not None and meta.load_spec.can_load:
                session.assert_loaded(consumer, 384)
                loaded.append(True)

    session.run(consumer, check_complete_prefix)
    assert loaded == [True]


@pytest.mark.parametrize("load_async", [False, True])
@pytest.mark.parametrize("failed_chunk", [0, 1])
def test_load_failure_reaches_scheduler_and_recomputes_failed_suffix(
    byte_pooling_session, monkeypatch, load_async, failed_chunk
):
    session = byte_pooling_session(load_async=load_async)
    session.run(session.new_request("failure-producer", list(range(384))))
    session.clear_local_cache()
    real_get = session.backend.get
    failed_blocks = []
    first_read = True

    def fail_one_chunk(keys, addresses, sizes):
        nonlocal first_read
        if not first_read:
            return real_get(keys, addresses, sizes)
        first_read = False
        results = []
        for index, (key, row, lengths) in enumerate(zip(keys, addresses, sizes, strict=True)):
            if index == failed_chunk:
                # Eviction after lookup: retry must recompute the unavailable suffix.
                session.backend.payloads.pop(key)
            results.append(1 if index == failed_chunk else real_get([key], [row], [lengths])[0])
        return results

    monkeypatch.setattr(session.backend, "get", fail_one_chunk)
    consumer = session.new_request("failure-consumer", list(range(385)))
    recovered_views = {}

    def capture_blocks(output):
        for index, name, plane, view in session.block_views(consumer):
            recovered_views[index, name, plane] = view
        for meta in output.kv_connector_metadata.requests:
            if meta.load_spec is not None and meta.load_spec.can_load:
                failed_blocks.append(session.block_ids(consumer)[failed_chunk])

    session.run(consumer, capture_blocks)
    assert failed_blocks
    assert {failed_blocks[0]} in session.reported_errors
    starts = [start for req_id, start, _ in session.computed if req_id == consumer.request_id]
    assert failed_chunk * 128 in starts, "scheduler did not recompute from the failed logical chunk"
    # Inspect real recomputed pages before another request reuses them.
    assert len(recovered_views) >= 3 * 2 * 2
    for (index, name, plane), view in recovered_views.items():
        if (index + 1) * 128 <= 384:
            assert torch.equal(view, session.expected_bytes(consumer, index, name, plane, view.shape))


def test_released_save_fence_protects_bytes_before_physical_block_reuse(byte_pooling_session, monkeypatch):
    session = byte_pooling_session(num_blocks=2)
    write_entered = threading.Event()
    release_write = threading.Event()
    fence_entered = threading.Event()
    forward_entered = threading.Event()
    failures = []
    real_put = session.backend.put

    def delayed_put(*args):
        write_entered.set()
        assert release_write.wait(5), "test did not release the paused store write"
        return real_put(*args)

    monkeypatch.setattr(session.backend, "put", delayed_put)
    producer = session.new_request("fence-producer", list(range(128)))
    session.scheduler.add_request(producer)
    first_output = session.scheduler.schedule()
    old_block = session.block_ids(producer)[0]
    session.execute(first_output, producer)
    consumer = session.new_request("reuse-writer", list(range(1000, 1128)))
    thread = None
    try:
        assert write_entered.wait(5)
        assert producer.request_id not in session.scheduler.requests
        assert session.scheduler.reset_prefix_cache()
        session.scheduler.add_request(consumer)
        output = session.scheduler.schedule()
        assert session.block_ids(consumer)[0] == old_block

        def reuse_block():
            try:
                fence_entered.set()
                session.execute(output, consumer, lambda output: forward_entered.set())
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=reuse_block, daemon=True)
        thread.start()
        assert fence_entered.wait(5)
        assert not forward_entered.wait(0.1), "reused block was overwritten while the old save was pending"
    finally:
        release_write.set()
        if thread is not None:
            thread.join(timeout=5)
    assert not thread.is_alive()
    assert not failures
    assert forward_entered.is_set()
    assert consumer.request_id not in session.scheduler.requests
    session.execute(session.scheduler.schedule())
    session.clear_local_cache()
    reader = session.new_request("fence-reader", list(range(128)))
    verified = []

    def check_old_payload(output):
        for meta in output.kv_connector_metadata.requests:
            if meta.load_spec is not None and meta.load_spec.can_load:
                session.assert_loaded(reader, 128)
                verified.append(True)

    session.run(reader, check_old_payload)
    assert verified == [True]


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
