"""Build Workers and Scheduler inputs for isolated tests."""

from dataclasses import replace

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeResources,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import KVPoolConfig
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    LoadCommand,
    LookupRequest,
    LookupResult,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.lookup import resolve_load_candidate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import Worker


def make_worker(
    backend=None, *, topology=None, requires_exists=False, event_factory=FakeEvent, store=True, async_load=True
):
    backend = backend or FakeBackend()
    topology = topology or make_topology()
    topology = replace(topology, config=replace(topology.config, load_async=async_load))
    resources = FakeResources(backend, make_backend_spec(requires_exists_before_put=requires_exists), topology)
    worker = Worker(
        topology,
        resources,
        coordinator=make_coordinator(topology, 256),
        store_enabled=store,
        source_ready_event_factory=event_factory,
    )
    worker.bind_kv_caches({})
    return worker, resources, backend


def plan_load(
    config: KVPoolConfig,
    query: LookupRequest,
    result: LookupResult,
    *,
    request_id: str,
    block_ids_by_group: tuple[tuple[int, ...], ...],
    request_token_count: int,
) -> LoadCommand | None:
    """Build a Load command for test replay without changing the installed Scheduler."""
    candidate = resolve_load_candidate(
        config,
        result,
        query_end_token=query.end_token,
        local_cached_tokens=query.local_available_tokens,
        request_token_count=request_token_count,
    )
    if candidate is None or candidate.load_end_token <= candidate.load_start_token:
        return None
    return LoadCommand(
        request_id,
        candidate.load_start_token,
        candidate.load_end_token,
        block_ids_by_group,
        query.block_hashes,
        candidate.tail_key_boundaries,
    )
