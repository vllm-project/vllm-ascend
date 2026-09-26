"""Exercise the extracted classic path through vLLM Connector hooks."""

import sys
import tempfile
from threading import Event, Thread
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, SlidingWindowSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LoadSpec as LegacyLoadSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta as LegacyReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    RequestTracker as LegacyRequestTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import backend as v1_backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import factory as service_factory
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupCodec,
    LookupRequest,
    LookupResult,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    AscendStoreV1Metadata,
    LoadRequest,
    LoadRequestBatch,
    StoreRequest,
    StoreRequestBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import lookup as scheduler_lookup
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import service as scheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.layout import (
    SchedulerTransferLayout,
    resolve_scheduler_transfer_layout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.load import (
    DeferredLoadScheduling,
    ImmediateLoadScheduling,
    LoadCandidate,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.load import (
    LoadService as SchedulerLoadService,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.request_tracker import RequestTracker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.store import (
    StoreService as SchedulerStoreService,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import layout as worker_layout
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import resources as worker_resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import service as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.coordinator import (
    ChunkSelection,
    HybridKVTransferCoordinator,
    LookupChunkSelection,
    LookupObservation,
    UnitaryKVTransferCoordinator,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layout import (
    KVCacheGroupLayout,
    StridedKVPartitioner,
    TPPartitionSpec,
    WorkerTransferLayout,
    resolve_tp_partition,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.load import LoadResult
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.load import LoadService as WorkerLoadService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.load.async_executor import AsyncLoadExecutor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.load.executor import SynchronousLoadExecutor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.load.task import (
    ContiguousLoadTaskBuilder,
    LoadChunk,
    LoadTask,
    StridedLoadTaskBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.lookup import LookupService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.lookup.executor import LookupExecutor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.lookup.task import LookupTaskBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.store import StoreService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.store.executor import StoreExecutor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.store.task import (
    ContiguousStoreTaskBuilder,
    StoreTask,
    StridedStoreTaskBuilder,
)


def make_scheduler_store_service(
    *,
    discard_partial_chunks: bool = True,
    save_decode_cache: bool = False,
) -> SchedulerStoreService:
    return SchedulerStoreService(
        cache_transfer_granularity=4,
        discard_partial_chunks=discard_partial_chunks,
        save_decode_cache=save_decode_cache,
        enabled=True,
    )


def make_scheduler_load_service(*, deferred: bool = False) -> SchedulerLoadService:
    scheduling = DeferredLoadScheduling() if deferred else ImmediateLoadScheduling()
    return SchedulerLoadService(scheduling)


def configure_scheduler_transfer_boundary(
    service: scheduler.SchedulerService,
    *,
    discard_partial_chunks: bool = True,
) -> None:
    service._layout = SchedulerTransferLayout(4, 4, (0,), discard_partial_chunks)


def fixed_scheduler_transfer_layout(_vllm_config, _kv_cache_config) -> SchedulerTransferLayout:
    return SchedulerTransferLayout(4, 4, (0,), True)


def fixed_worker_transfer_layout(_vllm_config, _kv_cache_config) -> WorkerTransferLayout:
    return make_worker_transfer_layout()


def make_worker_transfer_layout(*, tp_mismatch: bool = False) -> WorkerTransferLayout:
    tp_partition = TPPartitionSpec(tp_mismatch, 2 if tp_mismatch else 1, 2 if tp_mismatch else 1)
    kv_group = KVCacheGroupLayout(0, 4, ("layers.0",), KeyMetadata("model", 0, 0, 0))
    return WorkerTransferLayout(0, 1, 1, 0, 1, 1, 1, 4, 4, tp_partition, (kv_group,), (0,))


def make_unitary_coordinator(
    *,
    group_id: int = 0,
    block_size: int = 4,
    max_model_len: int = 64,
    cache_transfer_granularity: int = 4,
) -> UnitaryKVTransferCoordinator:
    return UnitaryKVTransferCoordinator(group_id, block_size, max_model_len, cache_transfer_granularity)


def test_coordinators_preserve_original_group_ids() -> None:
    unitary = make_unitary_coordinator(group_id=3)

    assert unitary.select_load((b"a",), 4) == (ChunkSelection(3, None),)

    hybrid = HybridKVTransferCoordinator(
        (1, 3),
        [
            KVCacheGroupSpec(
                ["layers.0"],
                FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
            KVCacheGroupSpec(
                ["layers.1"],
                FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
        ],
        scheduler_block_size=8,
        hash_block_size=4,
        max_model_len=64,
    )
    block_hashes = tuple(bytes([index]) * 32 for index in range(4))
    observations = (
        LookupObservation(1, (4, 8, 12, 16), block_hashes, (True, True, True, True)),
        LookupObservation(3, (8, 16), (block_hashes[1], block_hashes[3]), (True, False)),
    )

    assert tuple(selection.group_id for selection in hybrid.select_lookup(16, 0)) == (1, 3)
    assert tuple(selection.group_id for selection in hybrid.select_store(16, 16)) == (1, 3)
    assert hybrid.resolve_lookup(block_hashes, 16, 0, observations) == 8


def configure_worker_factory(monkeypatch) -> None:
    cache_resources = SimpleNamespace(backend=SimpleNamespace(), token_database=SimpleNamespace())
    monkeypatch.setattr(service_factory, "resolve_worker_transfer_layout", fixed_worker_transfer_layout)
    monkeypatch.setattr(service_factory.WorkerCacheResources, "create", lambda *args: cache_resources)


def test_scheduler_transfer_layout_resolves_classic_boundaries() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        cache_config=SimpleNamespace(block_size=4, prefix_match_unit=2),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"discard_partial_chunks": False}),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layers.0"], kv_cache_spec=SimpleNamespace(block_size=4))],
        transfer_group_ids=(0,),
    )

    assert resolve_scheduler_transfer_layout(vllm_config, kv_cache_config) == SchedulerTransferLayout(8, 8, (0,), False)


def test_request_tracker_advances_every_block_group() -> None:
    tracker = RequestTracker("request", 4, [[1], [10]], [b"a"], 8)

    tracker.advance(4, ([2], [11]), [b"a", b"b"])

    assert tracker.block_ids_by_group == [[1, 2], [10, 11]]


@pytest.mark.parametrize(
    ("load_async", "tp_mismatch", "load_is_deferred", "executor_type", "task_builder_type"),
    [
        (False, False, False, SynchronousLoadExecutor, ContiguousLoadTaskBuilder),
        (True, False, True, AsyncLoadExecutor, ContiguousLoadTaskBuilder),
        (False, True, False, SynchronousLoadExecutor, StridedLoadTaskBuilder),
        (True, True, True, AsyncLoadExecutor, StridedLoadTaskBuilder),
    ],
)
def test_load_execution_mode_selects_scheduler_and_worker_components(
    monkeypatch, load_async, tp_mismatch, load_is_deferred, executor_type, task_builder_type
) -> None:
    def take_scheduler_load_service(layout, lookup_service, load_service, store_service):
        return load_service

    def take_worker_transfer_services(cache_resources, lookup_service, load_service, store_service):
        return load_service, store_service

    monkeypatch.setattr(service_factory, "SchedulerService", take_scheduler_load_service)
    monkeypatch.setattr(service_factory, "WorkerService", take_worker_transfer_services)
    monkeypatch.setattr(service_factory, "resolve_scheduler_transfer_layout", fixed_scheduler_transfer_layout)
    configure_worker_factory(monkeypatch)
    monkeypatch.setattr(
        service_factory,
        "resolve_worker_transfer_layout",
        lambda _vllm_config, _kv_cache_config: make_worker_transfer_layout(tp_mismatch=tp_mismatch),
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(max_model_len=64),
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer", kv_connector_extra_config={"load_async": load_async}),
    )

    scheduler_load_service = service_factory.build_scheduler_service(vllm_config, object(), "ipc:///lookup")
    worker_load_service, worker_store_service = service_factory.build_worker_service(
        vllm_config, SimpleNamespace(num_blocks=4)
    )

    assert scheduler_load_service.is_deferred is load_is_deferred
    assert type(worker_load_service._executor) is executor_type
    assert type(worker_load_service._task_builder) is task_builder_type
    expected_store_builder_type = StridedStoreTaskBuilder if tp_mismatch else ContiguousStoreTaskBuilder
    assert type(worker_store_service._task_builder) is expected_store_builder_type


def test_tp_mismatch_rejects_multiple_transfer_groups() -> None:
    layout = SimpleNamespace(
        tp_partition=TPPartitionSpec(True, 2, 2),
        transfer_group_ids=(0, 1),
        tp_rank=0,
    )
    cache_resources = SimpleNamespace(token_database=object())

    with pytest.raises(ValueError, match="TP mismatch requires one transferable KV cache group"):
        service_factory._build_strided_kv_partitioner(cache_resources, layout, SimpleNamespace(block_size=4))


@pytest.mark.parametrize(
    ("kv_role", "consumer_is_to_put", "store_enabled"),
    [
        ("kv_producer", False, True),
        ("kv_both", False, True),
        ("kv_consumer", False, False),
        ("kv_consumer", True, True),
    ],
)
def test_store_capability_selects_scheduler_and_worker_components(
    monkeypatch, kv_role, consumer_is_to_put, store_enabled
) -> None:
    def take_scheduler_store_service(layout, lookup_service, load_service, store_service):
        return store_service

    def take_worker_store_service(cache_resources, lookup_service, load_service, store_service):
        return store_service

    monkeypatch.setattr(service_factory, "SchedulerService", take_scheduler_store_service)
    monkeypatch.setattr(service_factory, "WorkerService", take_worker_store_service)
    monkeypatch.setattr(service_factory, "resolve_scheduler_transfer_layout", fixed_scheduler_transfer_layout)
    configure_worker_factory(monkeypatch)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(max_model_len=64),
        kv_transfer_config=SimpleNamespace(
            kv_role=kv_role,
            kv_connector_extra_config={"consumer_is_to_put": consumer_is_to_put},
        ),
    )

    scheduler_store_service = service_factory.build_scheduler_service(vllm_config, object(), "ipc:///lookup")
    worker_store_service = service_factory.build_worker_service(vllm_config, SimpleNamespace(num_blocks=4))

    assert scheduler_store_service.is_enabled is store_enabled
    assert isinstance(worker_store_service, StoreService) is store_enabled


@pytest.mark.parametrize(
    ("kv_role", "consumer_is_to_load", "lookup_enabled"),
    [
        ("kv_producer", False, True),
        ("kv_both", False, True),
        ("kv_consumer", False, False),
        ("kv_consumer", True, True),
    ],
)
def test_lookup_capability_selects_scheduler_component(
    monkeypatch, kv_role, consumer_is_to_load, lookup_enabled
) -> None:
    def take_scheduler_lookup_service(layout, lookup_service, load_service, store_service):
        return lookup_service

    monkeypatch.setattr(service_factory, "SchedulerService", take_scheduler_lookup_service)
    monkeypatch.setattr(service_factory, "resolve_scheduler_transfer_layout", fixed_scheduler_transfer_layout)
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role=kv_role,
            kv_connector_extra_config={"consumer_is_to_load": consumer_is_to_load},
        )
    )

    lookup_service = service_factory.build_scheduler_service(vllm_config, object(), "ipc:///lookup")

    assert lookup_service.enabled is lookup_enabled


def test_connector_adapts_scheduler_lookup_request() -> None:
    received = []

    def lookup(lookup_request):
        received.append(lookup_request)
        return scheduler_lookup.SchedulerLookupResult(4, False)

    instance = connector.AscendStoreV1Connector.__new__(connector.AscendStoreV1Connector)
    instance.scheduler = SimpleNamespace(lookup=lookup)
    block_hashes = [b"a", b"b"]
    request = SimpleNamespace(request_id="request", prompt_token_ids=[0] * 8, num_tokens=9, block_hashes=block_hashes)

    assert instance.get_num_new_matched_tokens(request, 3) == (4, False)
    assert received == [scheduler_lookup.SchedulerLookupRequest("request", 8, 9, block_hashes, 3)]
    assert received[0].block_hashes is block_hashes


def test_scheduler_lookup_preserves_full_hit_allocation() -> None:
    calls = []

    def lookup(request):
        calls.append(request)
        return LookupResult(12)

    lookup_service = scheduler_lookup.LookupService(
        "ipc:///unused/lookup",
        transfer_group_ids=(0,),
        cache_transfer_granularity=4,
        discard_partial_chunks=True,
        enabled=True,
    )
    lookup_service.client = SimpleNamespace(lookup=lookup)
    service = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    service._lookup_service = lookup_service
    service._load_service = make_scheduler_load_service()
    block_hashes = [b"a", b"b", b"c"]
    request = scheduler_lookup.SchedulerLookupRequest("request", 12, 12, block_hashes, 0)

    assert service.lookup(request) == scheduler_lookup.SchedulerLookupResult(11, False)
    assert calls == [LookupRequest(12, (0,), 0, tuple(block_hashes))]
    load_candidate = service._load_service._pending_candidates["request"]
    assert load_candidate is not None
    assert load_candidate.kv_pool_cached_tokens == 11


def test_lookup_protocol_round_trip_preserves_business_messages() -> None:
    codec = LookupCodec()
    request = LookupRequest(12, (1, 3), 4, (b"a", b"b"))
    result = LookupResult(8)

    assert codec.decode_request(codec.encode_request(request)) == request
    assert codec.decode_result(codec.encode_result(result)) == result


def test_disabled_scheduler_lookup_skips_rpc() -> None:
    lookup_service = scheduler_lookup.LookupService(
        "ipc:///unused/lookup",
        transfer_group_ids=(0,),
        cache_transfer_granularity=4,
        discard_partial_chunks=True,
        enabled=False,
    )
    request = scheduler_lookup.SchedulerLookupRequest("request", 12, 12, [b"a", b"b", b"c"], 0)

    assert lookup_service.lookup(request) is None
    assert lookup_service.client is None


def test_scheduler_publishes_async_load_after_allocation() -> None:
    service = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    configure_scheduler_transfer_boundary(service)
    service._lookup_service = SimpleNamespace(lookup=lambda request: 11)
    service._load_service = make_scheduler_load_service(deferred=True)
    service._store_service = make_scheduler_store_service()
    service.request_trackers = {}
    service.unfinished_requests = {}
    service.preempted_request_ids = set()
    block_hashes = [b"a", b"b", b"c"]
    lookup_request = scheduler_lookup.SchedulerLookupRequest("request", 12, 12, block_hashes, 0)
    request = SimpleNamespace(request_id="request", prompt_token_ids=[0] * 12, block_hashes=block_hashes)

    assert service.lookup(lookup_request) == scheduler_lookup.SchedulerLookupResult(11, True)
    service.update_state_after_alloc(request, ([1, 2, 3], [10, 11, 12]), 11)
    empty_cached = SimpleNamespace(req_ids=[], new_block_ids=[])
    output = SimpleNamespace(
        finished_req_ids=set(), preempted_req_ids=set(), scheduled_new_reqs=[], scheduled_cached_reqs=empty_cached
    )

    metadata = service.build_connector_meta(output)
    assert metadata.store.requests == ()
    assert metadata.load.requests == (
        LoadRequest("request", 12, ((1, 2, 3), (10, 11, 12)), tuple(block_hashes), 0, 11),
    )

    next_metadata = service.build_connector_meta(output)
    assert next_metadata.load.requests == ()


@pytest.mark.parametrize(
    ("target_tokens", "saved_tokens", "hash_count", "load_tokens", "can_load", "discard_partial_chunks"),
    [
        (8, 0, 3, None, False, True),
        (8, 8, 3, None, False, True),
        (8, 0, 3, 8, True, True),
        (8, 0, 3, 8, False, True),
        (12, 8, 2, None, False, True),
        (7, 0, 2, None, False, True),
        (7, 0, 2, None, False, False),
    ],
)
def test_classic_transfer_requests_match_legacy_operation(
    target_tokens: int,
    saved_tokens: int,
    hash_count: int,
    load_tokens: int | None,
    can_load: bool,
    discard_partial_chunks: bool,
) -> None:
    hashes = [bytes([index + 1]) for index in range(hash_count)]
    legacy_tracker = LegacyRequestTracker(
        "request",
        target_tokens,
        allocated_block_ids_by_group=[[1, 2, 3]],
        num_saved_tokens=saved_tokens,
        num_prompt_tokens=12,
    )
    tracker = RequestTracker("request", target_tokens, [[1, 2, 3]], hashes, 12)
    legacy_load = LegacyLoadSpec(0, load_tokens, can_load) if load_tokens is not None else None
    load_candidate = LoadCandidate(0, load_tokens) if load_tokens is not None and can_load else None

    legacy = LegacyReqMeta.from_request_tracker(
        legacy_tracker,
        4,
        load_spec=legacy_load,
        block_hashes=hashes,
        discard_partial_chunks=discard_partial_chunks,
        hash_block_size=4,
    )
    service = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    configure_scheduler_transfer_boundary(service, discard_partial_chunks=discard_partial_chunks)
    service._load_service = make_scheduler_load_service()
    service._store_service = make_scheduler_store_service(discard_partial_chunks=discard_partial_chunks)
    if saved_tokens:
        service._store_service._scheduled_tokens["request"] = saved_tokens
    load_request, store_request = service._schedule_request_transfer(tracker, load_candidate)

    assert service._store_service._scheduled_tokens.get("request", 0) == legacy_tracker.num_saved_tokens
    if legacy is None or (not legacy.can_save and legacy.load_spec is None):
        assert load_request is store_request is None
    elif legacy.load_spec is not None:
        assert load_request is not None and store_request is None
        assert load_request.transfer_end_token == legacy.token_len_chunk
        assert load_request.kv_pool_cached_tokens == legacy.load_spec.kvpool_cached_tokens
    else:
        assert store_request is not None and load_request is None
        assert store_request.store_end_token == legacy.token_len_chunk


def test_finished_request_keeps_unconsumed_load_candidate_like_legacy() -> None:
    service = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    load_candidate = LoadCandidate(0, 4)
    service._load_service = make_scheduler_load_service()
    service._load_service.record_candidate("request", load_candidate)
    service._store_service = make_scheduler_store_service()
    service.request_trackers = {"request": RequestTracker("request", 4, [[1]], [b"a"], 4)}
    service.unfinished_requests = {"request": SimpleNamespace()}
    service.preempted_request_ids = {"request"}
    output = SimpleNamespace(
        finished_req_ids={"request"},
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )

    service.build_connector_meta(output)

    assert service._load_service._pending_candidates["request"] is load_candidate
    assert "request" not in service.request_trackers
    assert "request" not in service.unfinished_requests


@pytest.mark.parametrize(
    ("branch", "expected_message"),
    [
        ("new", "scheduled as a new request"),
        ("preempted", "scheduled as a preempted cached request"),
        ("running_tracker", "not in _request_trackers, but it is scheduled to be cached"),
        ("running_request", "not in _unfinished_requests, but it is scheduled to be cached"),
    ],
)
def test_missing_scheduler_state_reports_legacy_error(branch: str, expected_message: str) -> None:
    service = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    service._load_service = make_scheduler_load_service()
    service._store_service = make_scheduler_store_service()
    service.request_trackers = {}
    service.unfinished_requests = {}
    service.preempted_request_ids = set()
    new_requests = []
    cached_requests = SimpleNamespace(req_ids=[], new_block_ids=[])
    if branch == "new":
        service._load_service.record_candidate("request", LoadCandidate(0, 4))
        new_requests = [SimpleNamespace(req_id="request", num_computed_tokens=0)]
    else:
        cached_requests = SimpleNamespace(req_ids=["request"], new_block_ids=[([1],)])
        if branch == "preempted":
            service.preempted_request_ids.add("request")
            service._load_service.record_candidate("request", LoadCandidate(0, 4))
        if branch == "running_tracker":
            service.unfinished_requests["request"] = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=4)
        if branch == "running_request":
            service.request_trackers["request"] = RequestTracker("request", 4, [[1]], [b"a"], 4)
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=new_requests,
        scheduled_cached_reqs=cached_requests,
        num_scheduled_tokens={"request": 4},
    )

    with pytest.raises(ValueError, match=expected_message):
        service.build_connector_meta(output)
    if branch in ("new", "preempted"):
        assert service._load_service.take_for_transfer("request") is None


@pytest.mark.parametrize("resume_load", [False, True])
def test_preempted_cached_request_matches_legacy(resume_load: bool) -> None:
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=4,
        num_prompt_tokens=12,
        prompt_token_ids=[0] * 12,
        block_hashes=[b"a", b"b", b"c"],
    )
    legacy = KVPoolScheduler.__new__(KVPoolScheduler)
    legacy.kv_role = "kv_producer"
    legacy.consumer_is_to_put = False
    legacy._request_trackers = {
        "request": LegacyRequestTracker("request", 8, allocated_block_ids_by_group=[[1, 2]], num_saved_tokens=8)
    }
    legacy._unfinished_requests = {"request": (request, [[1, 2]])}
    legacy._preempted_req_ids = set()
    legacy._loading_req_ids = set()
    legacy.load_specs = {}
    legacy.kv_cache_group_ids = [0]
    legacy.tp_mismatch = False
    legacy.layerwise_offload = False
    legacy.use_layerwise = False
    legacy.load_async = False
    legacy.use_hybrid = False
    legacy.num_speculative_blocks_by_group = {}
    legacy.save_decode_cache = False
    legacy.enable_kv_events = False
    legacy.cache_transfer_granularity = 4
    legacy._discard_partial_chunks = True
    legacy.original_block_size = 4
    legacy.grouped_block_size = [4]
    legacy.kv_cache_group_families = []
    legacy.hash_block_size = 4

    current = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    configure_scheduler_transfer_boundary(current)
    current.request_trackers = {"request": RequestTracker("request", 8, [[1, 2]], request.block_hashes, 12)}
    current.unfinished_requests = {"request": request}
    current.preempted_request_ids = set()
    current._load_service = make_scheduler_load_service()
    current._store_service = make_scheduler_store_service()
    current._store_service._scheduled_tokens["request"] = 8

    empty_cached = SimpleNamespace(req_ids=[], new_block_ids=[])
    preempt_step = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"request"},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=empty_cached,
        num_scheduled_tokens={},
    )
    legacy_preempted = legacy.build_connector_meta(preempt_step)
    current_preempted = current.build_connector_meta(preempt_step)
    assert legacy_preempted.requests == []
    assert current_preempted.load.requests == current_preempted.store.requests == ()
    assert legacy_preempted.preempted_req_ids == {"request"}
    assert "request" not in legacy._request_trackers
    assert "request" not in current.request_trackers

    if resume_load:
        legacy.load_specs["request"] = LegacyLoadSpec(4, 8, False)
        current._load_service.record_candidate("request", LoadCandidate(4, 8))
    allocated_blocks = SimpleNamespace(get_block_ids=lambda: [[3, 4]])
    external_tokens = 4 if resume_load else 0
    legacy.update_state_after_alloc(request, allocated_blocks, external_tokens)
    current.update_state_after_alloc(request, ([3, 4],), external_tokens)

    resume_step = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["request"], new_block_ids=[([3, 4],)]),
        num_scheduled_tokens={"request": 4},
    )
    legacy_meta = legacy.build_connector_meta(resume_step)
    current_meta = current.build_connector_meta(resume_step)
    legacy_request = legacy_meta.requests[0]
    if resume_load:
        assert len(current_meta.load.requests) == 1
        assert current_meta.store.requests == ()
        current_request = current_meta.load.requests[0]
        assert current_request.request_id == legacy_request.req_id
        assert current_request.block_ids_by_group == tuple(tuple(ids) for ids in legacy_request.block_ids_by_group)
        assert current_request.transfer_end_token == legacy_request.token_len_chunk
        assert current_request.kv_pool_cached_tokens == legacy_request.load_spec.kvpool_cached_tokens == 8
        assert legacy_request.load_spec.can_load
    else:
        assert current_meta.load.requests == ()
        assert len(current_meta.store.requests) == 1
        current_request = current_meta.store.requests[0]
        assert current_request.request_id == legacy_request.req_id
        assert current_request.block_ids_by_group == tuple(tuple(ids) for ids in legacy_request.block_ids_by_group)
        assert current_request.store_end_token == legacy_request.token_len_chunk
        assert legacy_request.can_save
    assert current.request_trackers["request"].request_token_len == legacy._request_trackers["request"].token_len
    current_saved_tokens = current._store_service._scheduled_tokens.get("request", 0)
    assert current_saved_tokens == legacy._request_trackers["request"].num_saved_tokens
    assert current_saved_tokens == (0 if resume_load else 8)
    assert "request" not in legacy.load_specs
    assert current._load_service.take_for_transfer("request") is None
    assert "request" not in legacy._preempted_req_ids
    assert "request" not in current.preempted_request_ids


def test_allocation_mismatch_raises_legacy_assertion() -> None:
    request = SimpleNamespace(request_id="request")
    blocks = SimpleNamespace(get_block_ids=lambda: [[1]])
    legacy = KVPoolScheduler.__new__(KVPoolScheduler)
    legacy.kv_cache_group_ids = [0]
    legacy._unfinished_requests = {}
    legacy.load_specs = {"request": LegacyLoadSpec(0, 8, False)}
    legacy.use_layerwise = False

    current = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    current.unfinished_requests = {}
    current._load_service = make_scheduler_load_service()
    current._load_service.record_candidate("request", LoadCandidate(0, 8))

    with pytest.raises(AssertionError):
        legacy.update_state_after_alloc(request, blocks, 4)
    with pytest.raises(AssertionError):
        current.update_state_after_alloc(request, ([1],), 4)


@pytest.mark.parametrize(
    ("computed_tokens", "save_decode_cache", "expected_tokens", "should_emit_meta"),
    [(4, False, 8, True), (12, False, 12, False), (12, True, 16, True)],
)
def test_running_cached_request_matches_legacy(
    computed_tokens: int, save_decode_cache: bool, expected_tokens: int, should_emit_meta: bool
) -> None:
    block_ids = list(range(1, computed_tokens // 4 + 1))
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=computed_tokens,
        num_prompt_tokens=12,
        prompt_token_ids=[0] * 12,
        all_token_ids=[0] * 16,
        block_hashes=[b"a", b"b", b"c", b"d"],
    )
    legacy = KVPoolScheduler.__new__(KVPoolScheduler)
    legacy.kv_role = "kv_producer"
    legacy.consumer_is_to_put = False
    legacy._request_trackers = {
        "request": LegacyRequestTracker(
            "request",
            computed_tokens,
            allocated_block_ids_by_group=[block_ids.copy()],
            num_saved_tokens=computed_tokens,
            num_prompt_tokens=12,
        )
    }
    legacy._unfinished_requests = {"request": (request, [block_ids.copy()])}
    legacy._preempted_req_ids = set()
    legacy._loading_req_ids = set()
    legacy.load_specs = {}
    legacy.kv_cache_group_ids = [0]
    legacy.tp_mismatch = False
    legacy.layerwise_offload = False
    legacy.use_hybrid = False
    legacy.num_speculative_blocks_by_group = {}
    legacy.save_decode_cache = save_decode_cache
    legacy.enable_kv_events = False
    legacy.cache_transfer_granularity = 4
    legacy._discard_partial_chunks = True
    legacy.original_block_size = 4
    legacy.grouped_block_size = [4]
    legacy.kv_cache_group_families = []
    legacy.hash_block_size = 4

    current = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    configure_scheduler_transfer_boundary(current)
    current.request_trackers = {
        "request": RequestTracker("request", computed_tokens, [block_ids.copy()], request.block_hashes, 12)
    }
    current.unfinished_requests = {"request": request}
    current.preempted_request_ids = set()
    current._load_service = make_scheduler_load_service()
    current._store_service = make_scheduler_store_service(save_decode_cache=save_decode_cache)
    current._store_service._scheduled_tokens["request"] = computed_tokens

    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["request"], new_block_ids=[([4],)]),
        num_scheduled_tokens={"request": 4},
    )
    legacy_meta = legacy.build_connector_meta(output)
    current_meta = current.build_connector_meta(output)

    assert current_meta.load.requests == ()
    assert len(current_meta.store.requests) == len(legacy_meta.requests) == int(should_emit_meta)
    legacy_tracker = legacy._request_trackers["request"]
    current_tracker = current.request_trackers["request"]
    assert current_tracker.request_token_len == legacy_tracker.token_len == expected_tokens
    assert current_tracker.block_ids_by_group == legacy_tracker.allocated_block_ids_by_group
    assert current._store_service._scheduled_tokens["request"] == legacy_tracker.num_saved_tokens
    if should_emit_meta:
        current_request = current_meta.store.requests[0]
        legacy_request = legacy_meta.requests[0]
        assert current_request.request_id == legacy_request.req_id
        assert current_request.block_ids_by_group == tuple(tuple(ids) for ids in legacy_request.block_ids_by_group)
        assert current_request.store_end_token == legacy_request.token_len_chunk == expected_tokens
        assert legacy_request.can_save
        assert legacy_request.load_spec is None


def test_running_cached_request_stores_completed_chunk_without_new_block() -> None:
    block_hashes = [b"a"]
    request = SimpleNamespace(
        request_id="request",
        num_computed_tokens=0,
        num_prompt_tokens=8,
        prompt_token_ids=[0] * 8,
        block_hashes=block_hashes,
    )
    service = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    configure_scheduler_transfer_boundary(service)
    service.request_trackers = {}
    service.unfinished_requests = {"request": request}
    service.preempted_request_ids = set()
    service._load_service = make_scheduler_load_service()
    service._store_service = make_scheduler_store_service()
    new_request_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=0, block_ids=([1],))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        num_scheduled_tokens={"request": 2},
    )
    cached_request_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["request"], new_block_ids=[None]),
        num_scheduled_tokens={"request": 2},
    )

    first_metadata = service.build_connector_meta(new_request_output)
    request.num_computed_tokens = 2
    second_metadata = service.build_connector_meta(cached_request_output)

    assert first_metadata.store.requests == ()
    assert service.request_trackers["request"].request_token_len == 4
    assert len(second_metadata.store.requests) == 1
    assert second_metadata.store.requests[0].store_end_token == 4
    assert second_metadata.store.requests[0].block_ids_by_group == ((1,),)


@pytest.mark.parametrize(
    ("save_decode_cache", "expected_token_len", "expected_store_count"), [(False, 5, 0), (True, 8, 1)]
)
def test_running_decode_without_new_block_follows_store_policy(
    save_decode_cache: bool, expected_token_len: int, expected_store_count: int
) -> None:
    block_hashes = [b"a", b"b"]
    request = SimpleNamespace(
        request_id="request", num_computed_tokens=5, num_prompt_tokens=4, block_hashes=block_hashes
    )
    service = scheduler.SchedulerService.__new__(scheduler.SchedulerService)
    configure_scheduler_transfer_boundary(service)
    service.request_trackers = {"request": RequestTracker("request", 5, [[1, 2]], block_hashes, 4)}
    service.unfinished_requests = {"request": request}
    service.preempted_request_ids = set()
    service._load_service = make_scheduler_load_service()
    service._store_service = make_scheduler_store_service(save_decode_cache=save_decode_cache)
    service._store_service._scheduled_tokens["request"] = 4
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["request"], new_block_ids=[None]),
        num_scheduled_tokens={"request": 3},
    )

    metadata = service.build_connector_meta(output)

    assert service.request_trackers["request"].request_token_len == expected_token_len
    assert len(metadata.store.requests) == expected_store_count
    if metadata.store.requests:
        assert metadata.store.requests[0].store_end_token == 8


@pytest.mark.parametrize(
    ("get_result", "invalid_block_ids"),
    [(None, {1, 2}), ([0, -1], {1}), ([0, 0], set())],
)
def test_classic_load_task_reports_failed_blocks(get_result, invalid_block_ids) -> None:
    class Database:
        def load_mask(self, block_hashes, token_len):
            assert token_len == 8
            return [[True, True]]

        def process_token_key_strings_with_block_ids(
            self, token_len, block_hashes, block_ids, mask_num, kv_cache_group_id, chunk_filter
        ):
            assert kv_cache_group_id == 0
            return [(0, 4, "first", 0, 1), (4, 8, "second", 0, 2)]

        def prepare_value(self, start, end, block_ids, kv_cache_group_id, block_id):
            assert kv_cache_group_id == 0
            return [block_id * 16], [16], block_id

    class Backend:
        def get(self, keys, addresses, sizes):
            assert keys == ["second", "first"]
            assert addresses == [[32], [16]]
            assert sizes == [[16], [16]]
            return get_result

    request = LoadRequest("request", 8, ((1, 2),), (b"a", b"b"), 0, 8)
    load_service = WorkerLoadService(
        make_unitary_coordinator(),
        ContiguousLoadTaskBuilder(Database(), {0: 4}, 1),
        SynchronousLoadExecutor(Backend()),
        4,
        uses_group_scoped_block_ids=False,
    )
    load_service.load(LoadRequestBatch((request,)))

    assert load_service.collect_result() == LoadResult(frozenset(), frozenset(), frozenset(invalid_block_ids))


@pytest.mark.parametrize("get_result", ([0], [0, 0, 0]))
def test_load_treats_misaligned_backend_results_as_failed(get_result) -> None:
    task = LoadTask(
        "request",
        (
            LoadChunk(0, "first", (100,), (16,), 1),
            LoadChunk(0, "second", (200,), (16,), 2),
        ),
    )
    load_service = WorkerLoadService(
        SimpleNamespace(select_load=lambda block_hashes, load_end_token: ()),
        SimpleNamespace(build=lambda request, load_end_token, selections: task),
        SynchronousLoadExecutor(SimpleNamespace(get=lambda keys, addresses, sizes: get_result)),
        4,
        uses_group_scoped_block_ids=False,
    )

    load_service.load(LoadRequestBatch((LoadRequest("request", 8, ((1, 2),), (b"a", b"b"), 0, 8),)))

    assert load_service.collect_result() == LoadResult(frozenset(), frozenset(), frozenset({1, 2}))


def test_grouped_load_builds_one_backend_call_from_each_group() -> None:
    database = ChunkedTokenDatabase(
        [KeyMetadata("model", 0, 0, 0, 0), KeyMetadata("model", 0, 0, 0, 1)],
        [4, 8],
        None,
        4,
    )
    database.set_group_buffers({0: [1000], 1: [2000]}, {0: [16], 1: [32]}, {0: [16], 1: [32]})

    class Reachability:
        group_ids = (0, 1)

        def select_load(self, block_hashes, token_len):
            assert token_len == 16
            return (
                ChunkSelection(0, (True, True, True, True)),
                ChunkSelection(1, (False, True)),
            )

    class Backend:
        def get(self, keys, addresses, sizes):
            assert ["@group:0@" in key for key in keys[:4]] == [True] * 4
            assert "@group:1@" in keys[4]
            assert addresses == [[1016], [1032], [1048], [1064], [2352]]
            assert sizes == [[16], [16], [16], [16], [32]]
            return [0] * 5

    request = LoadRequest("request", 16, ((1, 2, 3, 4), (10, 11)), (b"a", b"b", b"c", b"d"), 0, 16)
    task_builder = ContiguousLoadTaskBuilder(database, {0: 4, 1: 8}, 0)
    load_service = WorkerLoadService(
        Reachability(),
        task_builder,
        SynchronousLoadExecutor(Backend()),
        16,
        uses_group_scoped_block_ids=True,
    )

    load_service.load(LoadRequestBatch((request,)))

    assert load_service.collect_result() == LoadResult(frozenset(), frozenset(), frozenset())


def test_grouped_synchronous_load_fails_before_forward() -> None:
    task = LoadTask(
        "request",
        (
            LoadChunk(0, "group-0", (100,), (16,), 1),
            LoadChunk(1, "group-1", (200,), (32,), 1),
        ),
    )
    load_service = WorkerLoadService(
        SimpleNamespace(select_load=lambda block_hashes, load_end_token: ()),
        SimpleNamespace(build=lambda request, load_end_token, selections: task),
        SynchronousLoadExecutor(SimpleNamespace(get=lambda keys, addresses, sizes: [0, -1])),
        4,
        uses_group_scoped_block_ids=True,
    )

    with pytest.raises(RuntimeError, match="Hybrid KV Load failed for requests: \\['request'\\]"):
        load_service.load(LoadRequestBatch((LoadRequest("request", 4, ((1,), (1,)), (b"a",), 0, 4),)))

    assert load_service.collect_result() == LoadResult(frozenset(), frozenset({"request"}), frozenset())


def test_grouped_asynchronous_load_reports_request_failure() -> None:
    task = LoadTask(
        "request",
        (
            LoadChunk(0, "group-0", (100,), (16,), 1),
            LoadChunk(1, "group-1", (200,), (32,), 1),
        ),
    )
    backend = SimpleNamespace(set_device=lambda: None, get=lambda keys, addresses, sizes: [0, -1])
    load_service = WorkerLoadService(
        SimpleNamespace(select_load=lambda block_hashes, load_end_token: ()),
        SimpleNamespace(build=lambda request, load_end_token, selections: task),
        AsyncLoadExecutor(backend),
        4,
        uses_group_scoped_block_ids=True,
    )
    load_service.start()
    load_service.load(LoadRequestBatch((LoadRequest("request", 4, ((1,), (1,)), (b"a",), 0, 4),)))
    load_service._executor._task_queue.join()

    assert load_service.collect_result() == LoadResult(frozenset({"request"}), frozenset({"request"}), frozenset())
    load_service.close()


def test_async_load_aggregates_success_failure_and_empty_tasks() -> None:
    tasks = {
        "success": LoadTask("success", (LoadChunk(0, "success", (100,), (16,), 1),)),
        "failure": LoadTask("failure", (LoadChunk(1, "failure", (200,), (16,), 2),)),
        "empty": LoadTask("empty", ()),
    }

    class Backend:
        def set_device(self):
            return

        def get(self, keys, addresses, sizes):
            return [0] if keys == ["success"] else [-1]

    load_service = WorkerLoadService(
        SimpleNamespace(select_load=lambda block_hashes, load_end_token: ()),
        SimpleNamespace(build=lambda request, load_end_token, selections: tasks[request.request_id]),
        AsyncLoadExecutor(Backend()),
        4,
        uses_group_scoped_block_ids=True,
    )
    requests = tuple(LoadRequest(request_id, 4, ((1,), (2,)), (b"a",), 0, 4) for request_id in tasks)
    load_service.start()
    load_service.load(LoadRequestBatch(requests))
    load_service._executor._task_queue.join()

    assert load_service.collect_result() == LoadResult(
        frozenset({"success", "failure", "empty"}),
        frozenset({"failure"}),
        frozenset(),
    )
    load_service.close()


def test_async_load_reports_completion_and_failed_blocks() -> None:
    completed = Event()

    class Database:
        def load_mask(self, block_hashes, token_len):
            return [[True, True]]

        def process_token_key_strings_with_block_ids(
            self, token_len, block_hashes, block_ids, mask_num, kv_cache_group_id, chunk_filter
        ):
            assert kv_cache_group_id == 0
            return [(0, 4, "first", 0, 1), (4, 8, "second", 0, 2)]

        def prepare_value(self, start, end, block_ids, kv_cache_group_id, block_id):
            assert kv_cache_group_id == 0
            return [block_id * 16], [16], block_id

    class Backend:
        def set_device(self):
            return

        def get(self, keys, addresses, sizes):
            completed.set()
            return [0, -1]

    load_service = WorkerLoadService(
        make_unitary_coordinator(),
        ContiguousLoadTaskBuilder(Database(), {0: 4}, 0),
        AsyncLoadExecutor(Backend()),
        4,
        uses_group_scoped_block_ids=False,
    )
    load_service.start()
    load_service.load(LoadRequestBatch((LoadRequest("request", 8, ((1, 2),), (b"a", b"b"), 0, 8),)))
    assert completed.wait(2)
    load_service._executor._task_queue.join()

    assert load_service.collect_result() == LoadResult(frozenset({"request"}), frozenset(), frozenset({2}))
    assert load_service.collect_result() == LoadResult(frozenset(), frozenset(), frozenset())
    load_service.close()


def test_async_load_reports_terminal_request_after_late_completion(monkeypatch) -> None:
    load_started = Event()
    allow_load_to_finish = Event()

    class Database:
        def load_mask(self, block_hashes, token_len):
            return [[True]]

        def process_token_key_strings_with_block_ids(
            self, token_len, block_hashes, block_ids, mask_num, kv_cache_group_id, chunk_filter
        ):
            assert kv_cache_group_id == 0
            return [(0, 4, "key", 0, 1)]

        def prepare_value(self, start, end, block_ids, kv_cache_group_id, block_id):
            assert kv_cache_group_id == 0
            return [16], [16], block_id

    class Backend:
        def set_device(self):
            return

        def get(self, keys, addresses, sizes):
            load_started.set()
            assert allow_load_to_finish.wait(2)
            return [0]

    load_service = WorkerLoadService(
        make_unitary_coordinator(),
        ContiguousLoadTaskBuilder(Database(), {0: 4}, 0),
        AsyncLoadExecutor(Backend()),
        4,
        uses_group_scoped_block_ids=False,
    )
    load_service.start()
    load_service.load(LoadRequestBatch((LoadRequest("request", 4, ((1,),), (b"a",), 0, 4),)))
    assert load_started.wait(2)

    worker = worker_module.WorkerService.__new__(worker_module.WorkerService)
    worker._load_service = load_service
    worker._store_service = None
    instance = connector.AscendStoreV1Connector.__new__(connector.AscendStoreV1Connector)
    instance.worker = worker
    instance._pending_load_result = None
    monkeypatch.setattr(instance, "_get_connector_metadata", AscendStoreV1Metadata)

    assert instance.get_finished({"request"}) == (set(), set())
    assert instance.get_block_ids_with_load_errors() == set()
    allow_load_to_finish.set()
    load_service._executor._task_queue.join()
    assert instance.get_finished(set()) == (set(), {"request"})
    assert instance.get_block_ids_with_load_errors() == set()
    assert instance.get_finished(set()) == (set(), set())
    assert instance.get_block_ids_with_load_errors() == set()
    load_service.close()


@pytest.mark.parametrize(
    ("present", "max_model_len", "granularity", "expected_hit"),
    [
        ([1, 1, 1, 1, 0, 1], 12, 4, 4),
        ([1, 1, 1, 1, 1, 1], 8, 4, 8),
        ([1, 1, 1, 1, 1, 1], 12, 8, 8),
    ],
)
def test_classic_lookup_service_returns_continuous_rank_hit(present, max_model_len, granularity, expected_hit):
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)
    block_hashes = [b"a", b"b", b"c"]
    base_keys = [key for _, _, key, _ in database.process_token_key_strings(12, block_hashes)]

    class Backend:
        def exists(self, keys):
            assert keys[:3] == base_keys
            assert keys[3:] == [key.replace("@head_or_tp_rank:0@", "@head_or_tp_rank:1@") for key in base_keys]
            return present

    service = LookupService(
        make_unitary_coordinator(max_model_len=max_model_len, cache_transfer_granularity=granularity),
        LookupTaskBuilder(database, 2, 1, 1),
        LookupExecutor(Backend()),
    )
    assert service.lookup(LookupRequest(12, (0,), 0, tuple(block_hashes))) == LookupResult(expected_hit)


def test_classic_lookup_service_returns_zero_on_backend_error() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)

    class Backend:
        def exists(self, keys):
            raise RuntimeError("lookup unavailable")

    service = LookupService(
        make_unitary_coordinator(max_model_len=12),
        LookupTaskBuilder(database, 1, 1, 1),
        LookupExecutor(Backend()),
    )
    assert service.lookup(LookupRequest(4, (0,), 0, (b"a",))) == LookupResult(0)


@pytest.mark.parametrize("presence_codes", ((), (1, 1)))
def test_lookup_returns_zero_for_misaligned_backend_results(presence_codes) -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)
    service = LookupService(
        make_unitary_coordinator(max_model_len=4),
        LookupTaskBuilder(database, 1, 1, 1),
        LookupExecutor(SimpleNamespace(exists=lambda keys: presence_codes)),
    )

    assert service.lookup(LookupRequest(4, (0,), 0, (b"a",))) == LookupResult(0)


def test_lookup_empty_selection_skips_backend_and_preserves_observation() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)

    class Coordinator:
        group_ids = (0,)

        def select_lookup(self, lookup_end_token, local_cached_tokens):
            return (LookupChunkSelection(0, (False,), 0),)

        def resolve_lookup(self, block_hashes, lookup_end_token, local_cached_tokens, observations):
            assert observations == (LookupObservation(0, (), (), ()),)
            return 0

    service = LookupService(
        Coordinator(),
        LookupTaskBuilder(database, 1, 1, 1),
        LookupExecutor(SimpleNamespace(exists=lambda keys: pytest.fail("Backend must not receive an empty Lookup"))),
    )

    assert service.lookup(LookupRequest(4, (0,), 0, (b"a",))) == LookupResult(0)


def test_tp_mismatch_lookup_checks_every_effective_tp_rank() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)
    queried_keys = []

    class Backend:
        def exists(self, keys):
            queried_keys.extend(keys)
            return [1, 1, 1, 0]

    service = LookupService(
        make_unitary_coordinator(max_model_len=4),
        LookupTaskBuilder(database, 4, 1, 1),
        LookupExecutor(Backend()),
    )

    assert service.lookup(LookupRequest(4, (0,), 0, (b"a",))) == LookupResult(0)
    assert [f"@head_or_tp_rank:{rank}@" in key for rank, key in enumerate(queried_keys)] == [True] * 4


def test_grouped_lookup_returns_common_contiguous_hit() -> None:
    database = ChunkedTokenDatabase(
        [KeyMetadata("model", 0, 0, 0, 0), KeyMetadata("model", 0, 0, 0, 1)],
        [4, 8],
        None,
        4,
    )

    class Backend:
        def exists(self, keys):
            if "@group:0@" in keys[0]:
                return [1, 1, 1]
            return [1]

    class Coordinator:
        group_ids = (0, 1)

        def select_lookup(self, token_len, local_cached_tokens):
            assert token_len == 12
            assert local_cached_tokens == 0
            return LookupChunkSelection(0, None, 0), LookupChunkSelection(1, None, 0)

        def resolve_lookup(self, block_hashes, token_len, local_cached_tokens, observations):
            hit_ends = []
            for observation in observations:
                hit_end = 0
                for end, is_present in zip(observation.chunk_ends, observation.chunk_presence, strict=True):
                    if not is_present:
                        break
                    hit_end = end
                hit_ends.append(hit_end)
            return min(hit_ends)

    service = LookupService(
        Coordinator(),
        LookupTaskBuilder(database, 1, 1, 1),
        LookupExecutor(Backend()),
    )

    assert service.lookup(LookupRequest(12, (0, 1), 0, (b"a", b"b", b"c"))) == LookupResult(8)


def test_grouped_lookup_queries_only_reachable_chunks_after_hbm_hit() -> None:
    database = ChunkedTokenDatabase(
        [KeyMetadata("model", 0, 0, 0, 0), KeyMetadata("model", 0, 0, 0, 1)],
        [4, 8],
        None,
        4,
    )
    queries = {}

    class Backend:
        def exists(self, keys):
            group_id = 0 if "@group:0@" in keys[0] else 1
            queries[group_id] = keys
            return [1] * len(keys)

    class Reachability:
        group_ids = (0, 1)

        def select_lookup(self, token_len, local_cached_tokens):
            assert token_len == 16
            assert local_cached_tokens == 4
            return LookupChunkSelection(0, None, 4), LookupChunkSelection(1, (False, True), 0)

        def resolve_lookup(self, block_hashes, token_len, local_cached_tokens, observations):
            assert [observation.group_id for observation in observations] == [0, 1]
            assert [len(observation.chunk_hashes) for observation in observations] == [3, 1]
            assert all(all(observation.chunk_presence) for observation in observations)
            return 8

    service = LookupService(
        Reachability(),
        LookupTaskBuilder(database, 1, 1, 1),
        LookupExecutor(Backend()),
    )

    assert service.lookup(LookupRequest(16, (0, 1), 4, (b"a", b"b", b"c", b"d"))) == LookupResult(8)
    assert len(queries[0]) == 3
    assert len(queries[1]) == 1


def test_strided_load_task_maps_effective_rank_keys_to_head_slices() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 1, 0, 0)], [4], None, 4)
    database.set_group_buffers({0: [1000, 2000]}, {0: [64, 64]}, {0: [128, 128]})
    request = LoadRequest("request", 8, ((10, 11),), (b"a", b"b"), 0, 8)

    task = StridedLoadTaskBuilder(database, 4, 4, StridedKVPartitioner(database, 4, 1, 2)).build(
        request,
        8,
        (ChunkSelection(0, None),),
    )

    assert [chunk.block_id for chunk in task.chunks] == [10, 11, 11, 10]
    assert "@head_or_tp_rank:3@" in task.chunks[0].backend_key
    assert "@head_or_tp_rank:2@" in task.chunks[1].backend_key
    assert task.chunks[0].addresses == (2288, 2304, 2320, 2336, 3288, 3304, 3320, 3336)
    assert task.chunks[0].sizes == (8,) * 8


def test_strided_kv_partitioner_keeps_one_slice_when_local_tp_is_larger() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 3, 0, 0)], [4], None, 4)
    database.set_group_buffers({0: [1000]}, {0: [64]}, {0: [128]})
    base_key = next(database.process_token_key_strings(4, [b"a"]))[2]

    slices = list(StridedKVPartitioner(database, 4, 3, 1).partition(base_key, 2, 4))

    assert len(slices) == 1
    assert "@head_or_tp_rank:3@" in slices[0][0]
    assert slices[0][1] == (1256, 1272, 1288, 1304)
    assert slices[0][2] == (16,) * 4


def test_async_load_executes_strided_task() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)
    database.set_group_buffers({0: [1000]}, {0: [64]}, {0: [128]})
    loaded_keys = []

    class Backend:
        def set_device(self):
            return

        def get(self, keys, addresses, sizes):
            loaded_keys.extend(keys)
            return [0] * len(keys)

    task_builder = StridedLoadTaskBuilder(database, 4, 4, StridedKVPartitioner(database, 4, 0, 2))
    load_service = WorkerLoadService(
        make_unitary_coordinator(),
        task_builder,
        AsyncLoadExecutor(Backend()),
        4,
        uses_group_scoped_block_ids=False,
    )
    load_service.start()
    load_service.load(LoadRequestBatch((LoadRequest("request", 4, ((1,),), (b"a",), 0, 4),)))
    load_service._executor._task_queue.join()

    assert load_service.collect_result() == LoadResult(frozenset({"request"}), frozenset(), frozenset())
    assert len(loaded_keys) == 2
    load_service.close()


def test_async_strided_load_reports_failed_block() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)
    database.set_group_buffers({0: [1000]}, {0: [64]}, {0: [128]})

    class Backend:
        def set_device(self):
            return

        def get(self, keys, addresses, sizes):
            return [0, -1]

    task_builder = StridedLoadTaskBuilder(database, 4, 4, StridedKVPartitioner(database, 4, 0, 2))
    load_service = WorkerLoadService(
        make_unitary_coordinator(),
        task_builder,
        AsyncLoadExecutor(Backend()),
        4,
        uses_group_scoped_block_ids=False,
    )
    load_service.start()
    load_service.load(LoadRequestBatch((LoadRequest("request", 4, ((1,),), (b"a",), 0, 4),)))
    load_service._executor._task_queue.join()

    assert load_service.collect_result() == LoadResult(frozenset({"request"}), frozenset(), frozenset({1}))
    load_service.close()


def test_classic_worker_layout_keeps_rank_and_chunk_mapping(monkeypatch) -> None:
    monkeypatch.setattr(worker_layout, "get_tensor_model_parallel_rank", lambda: 3)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=7,
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(model="org/model/", use_mla=False, get_total_num_kv_heads=lambda: 2),
        cache_config=SimpleNamespace(block_size=4, prefix_match_unit=2),
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer", kv_connector_extra_config={}),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layers.0"], kv_cache_spec=SimpleNamespace(block_size=4))]
    )

    layout = worker_layout.resolve_worker_transfer_layout(vllm_config, kv_cache_config)

    assert (layout.tp_rank, layout.tp_size, layout.pp_size, layout.put_step) == (3, 4, 2, 2)
    assert (layout.cache_transfer_granularity, layout.hash_block_size) == (4, 4)
    assert layout.tp_partition == TPPartitionSpec(False, 2, 1)
    assert layout.kv_cache_groups == (KVCacheGroupLayout(0, 4, ("layers.0",), KeyMetadata("model", 1, 0, 1)),)
    assert layout.transfer_group_ids == (0,)


def test_worker_layout_preserves_per_group_geometry_and_key_namespace(monkeypatch) -> None:
    monkeypatch.setattr(worker_layout, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(worker_layout, "get_decode_context_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(worker_layout.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (16, 8))
    monkeypatch.setattr(
        worker_layout.kv_cache_utils,
        "resolve_dcp_kv_block_size",
        lambda spec, dcp_size: spec.block_size * dcp_size,
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=2,
        ),
        model_config=SimpleNamespace(model="model", use_mla=False, get_total_num_kv_heads=lambda: 1),
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer", kv_connector_extra_config={}),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layers.0"], kv_cache_spec=SimpleNamespace(block_size=4)),
            SimpleNamespace(layer_names=["layers.1"], kv_cache_spec=SimpleNamespace(block_size=8)),
        ]
    )

    layout = worker_layout.resolve_worker_transfer_layout(vllm_config, kv_cache_config)

    assert (layout.cache_transfer_granularity, layout.hash_block_size) == (16, 8)
    assert layout.kv_cache_groups == (
        KVCacheGroupLayout(0, 8, ("layers.0",), KeyMetadata("model", 0, 1, 0, 0)),
        KVCacheGroupLayout(1, 16, ("layers.1",), KeyMetadata("model", 0, 1, 0, 1)),
    )
    assert layout.transfer_group_ids == (0, 1)


@pytest.mark.parametrize(
    ("tp_size", "peer_tp_size", "tp_rank", "expected_partition"),
    [(2, 4, 1, TPPartitionSpec(True, 4, 2)), (4, 2, 3, TPPartitionSpec(True, 4, 1))],
)
def test_worker_layout_resolves_tp_mismatch_partition(
    monkeypatch, tp_size, peer_tp_size, tp_rank, expected_partition
) -> None:
    monkeypatch.setattr(worker_layout, "get_tensor_model_parallel_rank", lambda: tp_rank)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=tp_rank,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(model="model", use_mla=False, get_total_num_kv_heads=lambda: 8),
        cache_config=SimpleNamespace(block_size=4, prefix_match_unit=4),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_consumer", kv_connector_extra_config={"prefill_tp_size": peer_tp_size}
        ),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layers.0"], kv_cache_spec=SimpleNamespace(block_size=4))]
    )

    layout = worker_layout.resolve_worker_transfer_layout(vllm_config, kv_cache_config)

    assert layout.tp_partition == expected_partition


def test_tp_partition_uses_effective_tp_namespace_in_both_directions() -> None:
    def resolve(local_tp_size, peer_tp_size):
        return resolve_tp_partition(
            SimpleNamespace(
                parallel_config=SimpleNamespace(tensor_parallel_size=local_tp_size),
                model_config=SimpleNamespace(use_mla=False, get_total_num_kv_heads=lambda: 8),
                kv_transfer_config=SimpleNamespace(
                    kv_role="kv_consumer", kv_connector_extra_config={"prefill_tp_size": peer_tp_size}
                ),
            )
        )

    assert resolve(2, 4) == TPPartitionSpec(True, 4, 2)
    assert resolve(4, 2) == TPPartitionSpec(True, 4, 1)


@pytest.mark.parametrize(
    ("kv_role", "consumer_is_to_put", "starts_store"),
    [
        ("kv_producer", False, True),
        ("kv_both", False, True),
        ("kv_consumer", False, False),
        ("kv_consumer", True, True),
    ],
)
def test_classic_worker_keeps_registered_caches_and_starts_store_after_registration(
    monkeypatch, kv_role, consumer_is_to_put, starts_store
) -> None:
    calls = []

    class Database:
        def set_group_buffers(self, addresses, lengths, strides):
            calls.append(("database", addresses, lengths, strides))

    class Backend:
        requires_exists_before_put = True

        def __init__(self, parallel_config=None, extra_config=None) -> None:
            return

        def register_buffer(self, addresses, lengths):
            calls.append(("backend", addresses, lengths))

    backend_module = ModuleType("mooncake_backend")
    backend_module.MooncakeBackend = Backend
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
        backend_module,
    )
    monkeypatch.setattr(worker_layout, "get_tensor_model_parallel_rank", lambda: 0)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(model="model", max_model_len=64, use_mla=False, get_total_num_kv_heads=lambda: 1),
        cache_config=SimpleNamespace(block_size=4, prefix_match_unit=4),
        kv_transfer_config=SimpleNamespace(
            kv_role=kv_role, kv_connector_extra_config={"consumer_is_to_put": consumer_is_to_put}
        ),
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layers.2", "layers.10"],
                kv_cache_spec=SimpleNamespace(block_size=4),
            )
        ],
    )
    worker = service_factory.build_worker_service(vllm_config, kv_cache_config)
    assert (worker._store_service is not None) is starts_store
    monkeypatch.setattr(worker._cache_resources.token_database, "set_group_buffers", Database().set_group_buffers)

    def start_store(executor) -> None:
        calls.append(("start",))

    monkeypatch.setattr(StoreExecutor, "start_and_wait_ready", start_store)

    storage = torch.zeros(8, 4, 1)
    first, second = storage[:4], storage[4:]
    kv_caches = {"layers.10": second, "layers.2": first}
    worker.register_kv_caches(kv_caches)

    block_bytes = first[0].numel() * first.element_size()
    assert worker._cache_resources.kv_caches is kv_caches
    expected_calls = [
        ("database", {0: [first.data_ptr(), second.data_ptr()]}, {0: [block_bytes] * 2}, {0: [block_bytes] * 2}),
        ("backend", [storage.data_ptr()], [storage.numel() * storage.element_size()]),
    ]
    if starts_store:
        expected_calls.append(("start",))
    assert calls == expected_calls


def test_worker_cache_resources_registers_each_group_with_its_layers() -> None:
    registered = []
    backend = SimpleNamespace(register_buffer=lambda addresses, lengths: registered.append((addresses, lengths)))
    database = ChunkedTokenDatabase(
        [KeyMetadata("model", 0, 0, 0, 0), KeyMetadata("model", 0, 0, 0, 1)],
        [4, 8],
        None,
        4,
    )
    resources = worker_resources.WorkerCacheResources(
        backend,
        database,
        4,
        {0: ("layers.0",), 1: ("layers.1",)},
    )
    storage = torch.zeros(8, 4, 1)
    first, second = storage[:4], storage[4:]

    resources.register_kv_caches({"layers.0": first, "layers.1": second})

    block_bytes = first[0].numel() * first.element_size()
    assert database.group_kv_caches_base_addr == {0: [first.data_ptr()], 1: [second.data_ptr()]}
    assert database.group_block_len == {0: [block_bytes], 1: [block_bytes]}
    assert database.group_block_stride == {0: [block_bytes], 1: [block_bytes]}
    assert registered == [([storage.data_ptr()], [storage.numel() * storage.element_size()])]


@pytest.mark.parametrize("executor_type", [AsyncLoadExecutor, StoreExecutor])
def test_async_executor_start_and_close_are_idempotent(executor_type) -> None:
    device_selections = []
    backend = SimpleNamespace(set_device=lambda: device_selections.append(True))
    executor = executor_type(backend)

    executor.start_and_wait_ready()
    executor.start_and_wait_ready()
    executor.close()
    executor.close()

    assert device_selections == [True]
    assert not executor.is_alive()
    with pytest.raises(RuntimeError, match="is closed"):
        executor.start_and_wait_ready()


@pytest.mark.parametrize("executor_type", [AsyncLoadExecutor, StoreExecutor])
def test_async_executor_reports_startup_failure(executor_type) -> None:
    def fail_to_select_device() -> None:
        raise RuntimeError("device unavailable")

    executor = executor_type(SimpleNamespace(set_device=fail_to_select_device))

    with pytest.raises(RuntimeError, match="failed during") as error:
        executor.start_and_wait_ready()

    assert isinstance(error.value.__cause__, RuntimeError)
    assert not executor.is_alive()


def test_worker_close_stops_load_after_store_failure() -> None:
    closed_services = []

    def close_store() -> None:
        closed_services.append("store")
        raise RuntimeError("store failed")

    worker = worker_module.WorkerService.__new__(worker_module.WorkerService)
    worker._store_service = SimpleNamespace(close=close_store)
    worker._load_service = SimpleNamespace(close=lambda: closed_services.append("load"))
    worker._cache_resources = SimpleNamespace(close=lambda: closed_services.append("resources"))

    with pytest.raises(RuntimeError, match="store failed"):
        worker.close()

    assert closed_services == ["store", "load", "resources"]


def test_worker_stops_store_when_load_startup_fails() -> None:
    lifecycle = []
    worker = worker_module.WorkerService.__new__(worker_module.WorkerService)
    worker._cache_resources = SimpleNamespace(
        register_kv_caches=lambda caches: lifecycle.append("register"),
        close=lambda: lifecycle.append("close_resources"),
    )
    worker._store_service = SimpleNamespace(
        start=lambda: lifecycle.append("start_store"), close=lambda: lifecycle.append("close_store")
    )

    def start_load() -> None:
        lifecycle.append("start_load")
        raise RuntimeError("load failed")

    worker._load_service = SimpleNamespace(start=start_load, close=lambda: lifecycle.append("close_load"))

    with pytest.raises(RuntimeError, match="load failed"):
        worker.register_kv_caches({})

    assert lifecycle == ["register", "start_store", "start_load", "close_store", "close_load", "close_resources"]


@pytest.mark.parametrize(
    ("backend_name", "backend_type_name"),
    [
        ("mooncake", "MooncakeBackend"),
        ("memcache", "MemcacheBackend"),
        ("yuanrong", "YuanrongBackend"),
    ],
)
def test_worker_cache_resources_selects_configured_backend(monkeypatch, backend_name, backend_type_name) -> None:
    class SelectedBackend:
        def __init__(self, parallel_config, extra_config) -> None:
            self.parallel_config = parallel_config
            self.extra_config = extra_config

    imported_modules = []
    backend_module = ModuleType("selected_backend")
    setattr(backend_module, backend_type_name, SelectedBackend)
    monkeypatch.setattr(
        v1_backend.importlib,
        "import_module",
        lambda module_path: imported_modules.append(module_path) or backend_module,
    )
    parallel_config = object()
    extra_config = {"backend": backend_name}
    kv_group = KVCacheGroupLayout(0, 4, ("layers.0",), KeyMetadata("model", 0, 0, 0))

    resources = worker_resources.WorkerCacheResources.create(parallel_config, extra_config, (kv_group,), 4, 8)

    assert imported_modules == [v1_backend.BACKEND_IMPORTS[backend_name][0]]
    assert isinstance(resources.backend, SelectedBackend)
    assert resources.backend.parallel_config is parallel_config
    assert resources.backend.extra_config is extra_config


def test_worker_cache_resources_closes_backend_once_when_supported() -> None:
    close_calls = []
    backend = SimpleNamespace(close=lambda: close_calls.append("close"))
    resources = worker_resources.WorkerCacheResources(backend, SimpleNamespace(), 4, {0: ("layers.0",)})
    resources.kv_caches = {"layers.0": object()}

    resources.close()
    resources.close()

    assert close_calls == ["close"]
    assert resources.kv_caches is None


def test_worker_cache_resources_skips_backend_close_when_unsupported() -> None:
    resources = worker_resources.WorkerCacheResources(SimpleNamespace(), SimpleNamespace(), 4, {0: ("layers.0",)})

    resources.close()

    assert resources.kv_caches is None


def test_store_executor_reports_failure_while_waiting_for_previous_batch() -> None:
    executor = StoreExecutor(SimpleNamespace())
    executor.wait_for_previous_store()
    executor._previous_batch_barrier = SimpleNamespace(completed=Event())
    executor._fatal_error = RuntimeError("sender stopped")

    with pytest.raises(RuntimeError, match="failed during asynchronous transfer") as error:
        executor.wait_for_previous_store()
    assert isinstance(error.value.__cause__, RuntimeError)


def test_store_service_keeps_task_build_failures_inside_the_store_batch(monkeypatch) -> None:
    source_ready_event = SimpleNamespace(record=lambda: None)
    submitted_tasks = []

    def build_task(request, event):
        if request.request_id == "failed":
            raise RuntimeError("invalid Store task")
        return StoreTask(request.request_id, event, ())

    store_service = StoreService.__new__(StoreService)
    store_service._task_builder = SimpleNamespace(build=build_task)
    store_service._executor = SimpleNamespace(submit_batch=submitted_tasks.extend)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=lambda: source_ready_event), raising=False)

    store_service.submit(StoreRequestBatch((SimpleNamespace(request_id="failed"), SimpleNamespace(request_id="ready"))))

    assert [task.request_id for task in submitted_tasks] == ["failed", "ready"]
    assert all(task.source_ready_event is source_ready_event for task in submitted_tasks)


def test_connector_rejects_request_level_load_failure(monkeypatch) -> None:
    worker = worker_module.WorkerService.__new__(worker_module.WorkerService)
    worker._store_service = None
    worker._load_service = SimpleNamespace(
        collect_result=lambda: LoadResult(frozenset({"request"}), frozenset({"request"}), frozenset())
    )
    instance = connector.AscendStoreV1Connector.__new__(connector.AscendStoreV1Connector)
    instance.worker = worker
    instance._pending_load_result = None
    monkeypatch.setattr(instance, "_get_connector_metadata", AscendStoreV1Metadata)

    with pytest.raises(RuntimeError, match="cannot report request-level Load failures"):
        instance.get_finished(set())


def test_connector_leaves_finished_loads_to_vllm() -> None:
    instance = connector.AscendStoreV1Connector.__new__(connector.AscendStoreV1Connector)
    instance.scheduler = object()

    assert instance.update_connector_output(SimpleNamespace(finished_recving={"request"})) is None


def test_classic_connector_finished_hooks_do_not_delay_block_release() -> None:
    instance = connector.AscendStoreV1Connector.__new__(connector.AscendStoreV1Connector)
    instance.scheduler = object()
    request = SimpleNamespace(request_id="request")

    assert instance.request_finished(request, [1]) == (False, None)
    assert instance.request_finished_all_groups(request, ([1],)) == (False, None)


@pytest.mark.parametrize("backend_name", ["mooncake", "memcache", "yuanrong"])
def test_connector_accepts_classic_backends(monkeypatch, backend_name) -> None:
    worker = object()
    monkeypatch.setattr(connector, "build_worker_service", lambda vllm_config, kv_cache_config: worker)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=1),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"backend": backend_name}),
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])

    instance = connector.AscendStoreV1Connector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)

    assert instance.worker is worker


def test_connector_rejects_unknown_backend() -> None:
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=1),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"backend": "unknown"}),
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])

    with pytest.raises(ValueError, match="Unsupported AscendStore v1 backend: unknown"):
        connector.AscendStoreV1Connector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)


def test_connector_routes_store_batch_through_worker_service(monkeypatch) -> None:
    calls = []

    class Worker:
        def submit_store(self, metadata) -> None:
            calls.append(("store", metadata))

    monkeypatch.setattr(connector, "build_worker_service", lambda vllm_config, kv_cache_config: Worker())
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(rank=1),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    instance = connector.AscendStoreV1Connector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
    metadata = AscendStoreV1Metadata()

    def get_metadata():
        calls.append(("metadata",))
        return metadata

    monkeypatch.setattr(instance, "_get_connector_metadata", get_metadata)
    instance.wait_for_save()

    assert calls == [("metadata",), ("store", metadata.store)]


def test_strided_store_task_maps_effective_rank_keys_to_head_slices() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 1, 0, 0)], [4], None, 4)
    database.set_group_buffers({0: [1000, 2000]}, {0: [64, 64]}, {0: [128, 128]})
    source_ready_event = SimpleNamespace()
    request = StoreRequest("request", 8, ((10, 11),), (b"a", b"b"), 8)
    kv_partitioner = StridedKVPartitioner(database, 4, 1, 2)

    task = StridedStoreTaskBuilder(database, 4, 0, 1, kv_partitioner).build(
        request,
        source_ready_event,
        (ChunkSelection(0, None),),
    )

    assert task.source_ready_event is source_ready_event
    assert len(task.chunks) == 4
    assert "@head_or_tp_rank:2@" in task.chunks[0].backend_key
    assert "@head_or_tp_rank:3@" in task.chunks[1].backend_key
    assert task.chunks[0].addresses == (2280, 2296, 2312, 2328, 3280, 3296, 3312, 3328)
    assert task.chunks[1].addresses == (2288, 2304, 2320, 2336, 3288, 3304, 3320, 3336)
    assert task.chunks[0].sizes == task.chunks[1].sizes == (8,) * 8


def test_strided_store_shards_chunks_between_pcp_ranks_before_splitting_heads() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)
    database.set_group_buffers({0: [1000]}, {0: [64]}, {0: [128]})
    block_hashes = [b"a", b"b", b"c", b"d"]
    block_ids = [10, 11, 12, 13]
    token_chunks = database.process_token_key_strings_with_block_ids(16, block_hashes, block_ids)
    base_keys = [key for _, _, key, _, _ in token_chunks]
    request = StoreRequest("request", 16, (tuple(block_ids),), tuple(block_hashes), 16)

    task_builder = StridedStoreTaskBuilder(database, 4, 1, 2, StridedKVPartitioner(database, 4, 0, 2))
    task = task_builder.build(request, SimpleNamespace(), (ChunkSelection(0, None),))

    expected_keys = []
    for base_key in (base_keys[1], base_keys[3]):
        expected_keys.extend([base_key, base_key.replace("@head_or_tp_rank:0@", "@head_or_tp_rank:1@")])
    assert [chunk.backend_key for chunk in task.chunks] == expected_keys
    assert [chunk.addresses[0] for chunk in task.chunks] == [2408, 2416, 2664, 2672]


@pytest.mark.parametrize(
    ("exists_result", "requires_exists_before_put", "stored_indices"),
    [
        ([1, 0, 1], True, [1]),
        ([1, 1, 1], True, []),
        (RuntimeError("lookup unavailable"), True, [0, 1, 2]),
        (None, False, [0, 1, 2]),
    ],
)
def test_classic_store_task_filters_keys_before_reading_source(
    exists_result, requires_exists_before_put, stored_indices
) -> None:
    steps = []
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)
    database.set_group_buffers({0: [100]}, {0: [16]}, {0: [16]})
    block_hashes = [b"a", b"b", b"c"]
    keys = [key for _, _, key, _ in database.process_token_key_strings(12, block_hashes)]

    class Backend:
        def __init__(self):
            self.requires_exists_before_put = requires_exists_before_put

        def exists(self, queried_keys):
            steps.append(("exists", queried_keys))
            if isinstance(exists_result, Exception):
                raise exists_result
            return exists_result

        def put(self, selected_keys, addresses, sizes):
            steps.append(("put", selected_keys, addresses, sizes))

    source_ready_event = SimpleNamespace(synchronize=lambda: steps.append(("source_ready",)))
    request = StoreRequest("request", 12, ((1, 2, 3),), tuple(block_hashes), 12)
    task_builder = ContiguousStoreTaskBuilder(database, {0: 4}, 0, 0, 1, 1, 1, "kv_producer")
    task = task_builder.build(request, source_ready_event, (ChunkSelection(0, None),))
    executor = StoreExecutor(Backend())
    executor._execute_task(task)

    expected_steps = [("exists", keys)] if requires_exists_before_put else []
    if stored_indices:
        expected_steps.append(("source_ready",))
        expected_steps.append(
            (
                "put",
                [keys[index] for index in stored_indices],
                [[100 + (index + 1) * 16] for index in stored_indices],
                [[16] for _ in stored_indices],
            )
        )
    assert steps == expected_steps


def test_grouped_store_builds_one_backend_call_from_each_group() -> None:
    steps = []
    database = ChunkedTokenDatabase(
        [KeyMetadata("model", 0, 0, 0, 0), KeyMetadata("model", 0, 0, 0, 1)],
        [4, 8],
        None,
        4,
    )
    database.set_group_buffers({0: [1000], 1: [2000]}, {0: [16], 1: [32]}, {0: [16], 1: [32]})

    class Backend:
        requires_exists_before_put = False

        def put(self, keys, addresses, sizes):
            assert ["@group:0@" in key for key in keys[:4]] == [True] * 4
            assert "@group:1@" in keys[4]
            assert addresses == [[1016], [1032], [1048], [1064], [2352]]
            assert sizes == [[16], [16], [16], [16], [32]]
            steps.append("put")

    source_ready_event = SimpleNamespace(synchronize=lambda: steps.append("source_ready"))
    request = StoreRequest("request", 16, ((1, 2, 3, 4), (10, 11)), (b"a", b"b", b"c", b"d"), 16)
    task_builder = ContiguousStoreTaskBuilder(database, {0: 4, 1: 8}, 0, 0, 1, 1, 1, "kv_producer")
    task = task_builder.build(
        request,
        source_ready_event,
        (
            ChunkSelection(0, (True, True, True, True)),
            ChunkSelection(1, (False, True)),
        ),
    )

    assert [chunk.group_id for chunk in task.chunks] == [0, 0, 0, 0, 1]
    StoreExecutor(Backend())._execute_task(task)

    assert steps == ["source_ready", "put"]


def test_store_empty_selection_skips_source_wait_and_backend() -> None:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None, 4)
    source_ready_event = SimpleNamespace(synchronize=lambda: pytest.fail("Empty Store must not wait for its source"))
    request = StoreRequest("request", 4, ((1,),), (b"a",), 4)
    task = ContiguousStoreTaskBuilder(database, {0: 4}, 0, 0, 1, 1, 1, "kv_producer").build(
        request,
        source_ready_event,
        (ChunkSelection(0, (False,)),),
    )

    assert task.chunks == ()
    StoreExecutor(
        SimpleNamespace(
            requires_exists_before_put=False,
            put=lambda keys, addresses, sizes: pytest.fail("Backend must not receive an empty Store"),
        )
    )._execute_task(task)


def test_grouped_connector_preserves_original_group_ids_across_mainline(monkeypatch) -> None:
    events = []
    store_steps = []
    store_entered = Event()
    store_release = Event()

    class RecordedEvent:
        def __init__(self) -> None:
            events.append(self)
            self.recorded = False
            self.synchronized = False

        def record(self) -> None:
            self.recorded = True

        def synchronize(self) -> None:
            assert self.recorded
            self.synchronized = True
            store_steps.append("source_ready")

    lookup_callbacks = {}

    class InProcessLookupServer:
        def __init__(self, lookup, address) -> None:
            lookup_callbacks[address] = lookup
            self.address = address

        def close(self) -> None:
            lookup_callbacks.pop(self.address, None)

    class InProcessLookupClient:
        def __init__(self, address) -> None:
            self.address = address

        def lookup(self, request):
            return lookup_callbacks[self.address](request)

        def close(self) -> None:
            return

    class Backend:
        requires_exists_before_put = True
        instance = None

        def __init__(self, parallel_config, extra_config) -> None:
            self.store = SimpleNamespace(close=lambda: 0)
            self.existing_keys: set[str] = set()
            self.loaded_keys: list[str] = []
            self.stored_keys: list[str] = []
            self.loaded_addresses: list[list[int]] = []
            self.loaded_sizes: list[list[int]] = []
            self.stored_addresses: list[list[int]] = []
            self.stored_sizes: list[list[int]] = []
            Backend.instance = self

        def set_device(self) -> None:
            return

        def register_buffer(self, addresses, lengths) -> None:
            return

        def exists(self, keys):
            store_steps.append("exists")
            return [int(key in self.existing_keys) for key in keys]

        def get(self, keys, addresses, sizes):
            self.loaded_keys.extend(keys)
            self.loaded_addresses.extend(addresses)
            self.loaded_sizes.extend(sizes)
            return [0 if key in self.existing_keys else -1 for key in keys]

        def put(self, keys, addresses, sizes):
            assert events[0].synchronized
            store_steps.append("put")
            store_entered.set()
            assert store_release.wait(2)
            self.stored_keys.extend(keys)
            self.stored_addresses.extend(addresses)
            self.stored_sizes.extend(sizes)
            self.existing_keys.update(keys)

    backend_module = ModuleType("mooncake_backend")
    backend_module.MooncakeBackend = Backend
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend",
        backend_module,
    )
    monkeypatch.setattr(worker_layout, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(worker_layout.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda config, vllm: (8, 4))
    monkeypatch.setattr(connector, "LookupServer", InProcessLookupServer)
    monkeypatch.setattr(scheduler_lookup.service, "LookupClient", InProcessLookupClient)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=RecordedEvent), raising=False)

    with tempfile.TemporaryDirectory(prefix="v1-", dir="/tmp") as lookup_directory:
        lookup_path = f"ipc://{lookup_directory}/lookup"
        monkeypatch.setattr(
            connector.AscendStoreV1Connector, "_resolve_lookup_address", staticmethod(lambda config: lookup_path)
        )
        parallel_config = SimpleNamespace(
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_rank=0,
        )
        model_config = SimpleNamespace(
            model="model",
            max_model_len=64,
            use_mla=False,
            get_total_num_kv_heads=lambda: 1,
        )
        vllm_config = SimpleNamespace(
            parallel_config=parallel_config,
            model_config=model_config,
            cache_config=SimpleNamespace(block_size=4, prefix_match_unit=4),
            kv_transfer_config=SimpleNamespace(kv_role="kv_producer", kv_connector_extra_config={}),
        )
        kv_cache_groups = (
            KVCacheGroupSpec(
                ["layers.0"],
                FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
                enable_kv_transfer=False,
            ),
            KVCacheGroupSpec(
                ["layers.1"],
                FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
            ),
            KVCacheGroupSpec(
                ["layers.2"],
                FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32),
                enable_kv_transfer=False,
            ),
            KVCacheGroupSpec(
                ["layers.3"],
                SlidingWindowSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=8,
                ),
            ),
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=kv_cache_groups,
            transfer_group_ids=(1, 3),
            transfer_groups=(kv_cache_groups[1], kv_cache_groups[3]),
            prefix_cache_retention_interval=None,
            num_blocks=6,
        )
        worker = connector.AscendStoreV1Connector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
        group1_key_cache = torch.zeros(6, 4, 1)
        group1_value_cache = torch.zeros(6, 4, 1)
        group3_key_cache = torch.zeros(6, 8, 1)
        group3_value_cache = torch.zeros(6, 8, 1)
        worker.register_kv_caches(
            {
                "layers.0": (torch.zeros(6, 4, 1), torch.zeros(6, 4, 1)),
                "layers.1": (group1_key_cache, group1_value_cache),
                "layers.2": (torch.zeros(6, 8, 1), torch.zeros(6, 8, 1)),
                "layers.3": (group3_key_cache, group3_value_cache),
            }
        )
        scheduler_connector = connector.AscendStoreV1Connector(vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config)
        try:
            block_hashes = [b"a", b"b", b"c", b"d", b"e", b"f"]
            database = ChunkedTokenDatabase(
                [KeyMetadata("model", 0, 0, 0, group_id) for group_id in range(4)],
                [4, 4, 8, 8],
                None,
                4,
            )
            group1_keys = [
                key for _, _, key, _ in database.process_token_key_strings(24, block_hashes, kv_cache_group_id=1)
            ]
            group3_keys = [
                key for _, _, key, _ in database.process_token_key_strings(24, block_hashes, kv_cache_group_id=3)
            ]
            backend = Backend.instance
            assert backend is not None
            backend.existing_keys.update((*group1_keys[:4], *group3_keys[:2]))
            request = SimpleNamespace(
                request_id="request",
                num_prompt_tokens=24,
                num_tokens=24,
                num_computed_tokens=0,
                prompt_token_ids=[0] * 24,
                block_hashes=block_hashes,
            )

            assert scheduler_connector.get_num_new_matched_tokens(request, 0) == (16, False)
            scheduler_connector.update_state_after_alloc(
                request,
                SimpleNamespace(get_block_ids=lambda: ([20, 21, 22, 23], [0, 1, 2, 3], [30, 31], [0, 1])),
                16,
            )
            new_request = SimpleNamespace(
                req_id="request",
                block_ids=([20, 21, 22, 23], [0, 1, 2, 3], [30, 31], [0, 1]),
                num_computed_tokens=0,
            )
            empty_cached = SimpleNamespace(req_ids=[], new_block_ids=[])
            first_step = SimpleNamespace(
                scheduled_new_reqs=[new_request],
                scheduled_cached_reqs=empty_cached,
                num_scheduled_tokens={"request": 16},
                finished_req_ids=set(),
                preempted_req_ids=set(),
            )
            load_metadata = scheduler_connector.build_connector_meta(first_step)
            assert len(load_metadata.load.requests) == 1
            assert load_metadata.store.requests == ()
            assert load_metadata.load.requests[0].block_ids_by_group == (
                (20, 21, 22, 23),
                (0, 1, 2, 3),
                (30, 31),
                (0, 1),
            )
            worker.bind_connector_metadata(load_metadata)
            worker.start_load_kv(None)
            worker.wait_for_save()
            assert backend.loaded_keys == [*group1_keys[:4], group3_keys[1]]
            assert backend.loaded_addresses == [
                [group1_key_cache.data_ptr(), group1_value_cache.data_ptr()],
                [group1_key_cache.data_ptr() + 16, group1_value_cache.data_ptr() + 16],
                [group1_key_cache.data_ptr() + 32, group1_value_cache.data_ptr() + 32],
                [group1_key_cache.data_ptr() + 48, group1_value_cache.data_ptr() + 48],
                [group3_key_cache.data_ptr() + 32, group3_value_cache.data_ptr() + 32],
            ]
            assert backend.loaded_sizes == [[16, 16], [16, 16], [16, 16], [16, 16], [32, 32]]
            assert worker.get_finished(set()) == (set(), set())
            assert worker.get_block_ids_with_load_errors() == set()
            worker.clear_connector_metadata()

            request.num_computed_tokens = 16
            store_steps.clear()
            cached = SimpleNamespace(req_ids=["request"], new_block_ids=[([24, 25], [4, 5], [32], [2])])
            next_step = SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=cached,
                num_scheduled_tokens={"request": 8},
                finished_req_ids=set(),
                preempted_req_ids=set(),
            )
            store_metadata = scheduler_connector.build_connector_meta(next_step)
            assert store_metadata.load.requests == ()
            assert len(store_metadata.store.requests) == 1
            worker.bind_connector_metadata(store_metadata)
            worker.start_load_kv(None)
            worker.wait_for_save()
            assert not hasattr(store_metadata.store.requests[0], "current_event")
            worker.clear_connector_metadata()
            assert store_entered.wait(2)
            assert backend.stored_keys == []

            fence_done = Event()
            fence_errors = []

            def wait_for_next_step() -> None:
                try:
                    worker.handle_preemptions(store_metadata)
                except Exception as error:
                    fence_errors.append(error)
                finally:
                    fence_done.set()

            fence_thread = Thread(target=wait_for_next_step, daemon=True)
            fence_thread.start()
            assert not fence_done.wait(0.05)
            store_release.set()
            assert fence_done.wait(2)
            assert not fence_errors
            assert backend.stored_keys == [*group1_keys[4:], group3_keys[2]]
            assert backend.stored_addresses == [
                [group1_key_cache.data_ptr() + 64, group1_value_cache.data_ptr() + 64],
                [group1_key_cache.data_ptr() + 80, group1_value_cache.data_ptr() + 80],
                [group3_key_cache.data_ptr() + 64, group3_value_cache.data_ptr() + 64],
            ]
            assert backend.stored_sizes == [[16, 16], [16, 16], [32, 32]]
            assert store_steps == ["exists", "source_ready", "put"]
            assert len(events) == 1
            assert events[0].synchronized
        finally:
            store_release.set()
            scheduler_connector.shutdown()
            worker.shutdown()
