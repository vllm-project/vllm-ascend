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
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlocks,
    FakeLookupClient,
    make_output,
    make_request,
    new_request_data,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import KVPoolConfig
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    LoadCommand,
    LookupRequest,
    LookupResult,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import SynchronousScheduler
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
    """Replay public Scheduler hooks to obtain a Load command for Worker tests."""
    lookup = FakeLookupClient(result.available_end_token, tail_key_boundaries=result.tail_key_boundaries)
    scheduler = SynchronousScheduler(replace(config, store_enabled=False), lookup)
    request = make_request(request_id, prompt_tokens=query.end_token, tokens=request_token_count)
    request.block_hashes = list(query.block_hashes)
    try:
        matched_tokens, _ = scheduler.get_num_new_matched_tokens(request, query.local_available_tokens)
        if matched_tokens == 0:
            return None
        blocks = tuple(list(group) for group in block_ids_by_group)
        scheduler.confirm_allocation(request, FakeBlocks(*blocks), matched_tokens)
        computed_tokens = query.local_available_tokens + matched_tokens
        step = scheduler.build_step(
            make_output(
                new=(new_request_data(request_id, blocks, computed_tokens),),
                scheduled_tokens={request_id: request_token_count - computed_tokens},
            )
        )
        return next(iter(step.load_commands), None)
    finally:
        scheduler.close()
