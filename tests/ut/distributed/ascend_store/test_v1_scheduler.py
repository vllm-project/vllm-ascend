"""Scheduler ownership and publication contracts for AscendStore v1."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import (
    AscendStoreV1Connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection.reachability import (
    HybridReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import LookupResult
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    StateCheckpointSource,
    StoreSourceReleaseMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import (
    AsynchronousBulkScheduler,
    LayerwiseScheduler,
    SchedulerConfig,
    SynchronousBulkScheduler,
)

from .v1.helpers import make_topology


class FakeRemoteLookup:
    def __init__(self, available_end_token: int = 0) -> None:
        self.available_end_token = available_end_token
        self.queries: list[tuple[TokenRange, tuple[int, ...], tuple[bytes, ...]]] = []
        self.closed = False

    def query(self, query_range, transfer_group_ids, block_hashes) -> LookupResult:
        self.queries.append((query_range, transfer_group_ids, block_hashes))
        return LookupResult(self.available_end_token)

    def close(self) -> None:
        self.closed = True


class FakeBlocks:
    def __init__(self, *groups: list[int]) -> None:
        self.groups = groups

    def get_block_ids(self):
        return self.groups


class FakeBlockPool:
    def __init__(self, block_count: int = 64) -> None:
        self.blocks = [SimpleNamespace(block_id=block_id) for block_id in range(block_count)]
        self.touched: list[tuple[int, ...]] = []
        self.freed: list[tuple[int, ...]] = []

    def touch(self, blocks) -> None:
        self.touched.append(tuple(block.block_id for block in blocks))

    def free_blocks(self, blocks) -> None:
        self.freed.append(tuple(block.block_id for block in blocks))


def make_config(
    *,
    granularity: int = 4,
    groups: tuple[int, ...] = (0,),
    align_groups: frozenset[int] = frozenset(),
    load: bool = True,
    store: bool = False,
    save_decode: bool = False,
    eagle: bool = False,
    private: bool = False,
    workers: int = 1,
) -> SchedulerConfig:
    return SchedulerConfig(
        cache_transfer_granularity=granularity,
        hash_block_size=4,
        transfer_group_ids=groups,
        align_state_group_ids=align_groups,
        load_enabled=load,
        store_enabled=store,
        save_decode_cache=save_decode,
        discard_partial_chunks=True,
        use_eagle_block_drop=eagle,
        has_private_state=private,
        expected_worker_count=workers,
    )


def make_request(
    request_id: str = "request",
    *,
    prompt_tokens: int = 8,
    tokens: int | None = None,
    hashes: int | None = None,
):
    token_count = prompt_tokens if tokens is None else tokens
    hash_count = (token_count + 3) // 4 if hashes is None else hashes
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=prompt_tokens,
        num_tokens=token_count,
        block_hashes=[bytes((index + 1,)) for index in range(hash_count)],
    )


def make_output(
    *,
    new=(),
    cached=(),
    resumed=(),
    scheduled_tokens=None,
    finished=(),
    preempted=(),
    checkpoints=None,
):
    cached = tuple(cached)
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[entry[0] for entry in cached],
            resumed_req_ids=set(resumed),
            new_block_ids=[entry[1] for entry in cached],
            num_computed_tokens=[entry[2] for entry in cached],
        ),
        num_scheduled_tokens={} if scheduled_tokens is None else scheduled_tokens,
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        kv_connector_block_state=(
            None if checkpoints is None else SimpleNamespace(boundary_state_offloads=checkpoints)
        ),
    )


def new_request_data(request_id: str, block_ids, num_computed_tokens: int):
    return SimpleNamespace(
        req_id=request_id,
        block_ids=block_ids,
        num_computed_tokens=num_computed_tokens,
    )


def test_load_publication_requires_allocation_and_is_exactly_once() -> None:
    for scheduler_type, is_deferred in (
        (SynchronousBulkScheduler, False),
        (AsynchronousBulkScheduler, True),
    ):
        lookup = FakeRemoteLookup(12)
        scheduler = scheduler_type(make_config(), lookup)
        request = make_request(prompt_tokens=12)

        assert scheduler.get_num_new_matched_tokens(request, 0) == (11, is_deferred)
        assert scheduler.build_step(make_output()).load.commands == ()
        scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), 11)
        # vLLM may confirm a second allocation for the same external prefix. It
        # may append blocks, but must not republish the Load.
        scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3, 4]), 0)

        scheduled = make_output(
            new=(new_request_data("request", ([1, 2, 3, 4],), 11),),
            scheduled_tokens={"request": 1},
        )
        if is_deferred:
            published = scheduler.build_step(make_output())
            assert published.load.commands[0].block_ids_by_group == ((1, 2, 3, 4),)
            assert scheduler.build_step(make_output()).load.commands == ()
        else:
            assert scheduler.build_step(make_output()).load.commands == ()
            published = scheduler.build_step(scheduled)
            assert published.load.commands[0].block_ids_by_group == ((1, 2, 3, 4),)
            assert scheduler.build_step(make_output()).load.commands == ()

        assert published.load.commands[0].load_range == TokenRange(0, 12)


def test_async_load_suppresses_store_if_request_is_already_scheduled() -> None:
    scheduler = AsynchronousBulkScheduler(
        make_config(store=True),
        FakeRemoteLookup(8),
    )
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(tokens=9)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (8, True)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), 8)

    step = scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1, 2, 3],), 8),),
            scheduled_tokens={"request": 1},
        )
    )

    assert len(step.load.commands) == 1
    assert step.store.commands == ()


def test_synchronous_load_stores_newly_computed_suffix_in_same_step() -> None:
    scheduler = SynchronousBulkScheduler(
        make_config(store=True),
        FakeRemoteLookup(4),
    )
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=8, tokens=9)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4, False)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), 4)

    step = scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1, 2, 3],), 4),),
            scheduled_tokens={"request": 4},
        )
    )

    assert [command.load_range for command in step.load.commands] == [TokenRange(0, 4)]
    assert [command.store_range for command in step.store.commands] == [TokenRange(4, 8)]


def test_layerwise_zero_new_allocation_still_publishes_load() -> None:
    lookup = FakeRemoteLookup(8)
    scheduler = LayerwiseScheduler(make_config(), lookup)
    request = make_request(prompt_tokens=8, tokens=9)

    assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
    assert lookup.queries[0][0] == TokenRange(0, 8)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)
    step = scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1, 2],), 8),),
            scheduled_tokens={"request": 1},
        )
    )

    assert [command.load_range for command in step.load.commands] == [TokenRange(0, 8)]


def test_load_only_resumed_request_publishes_scheduled_load() -> None:
    scheduler = SynchronousBulkScheduler(make_config(store=False), FakeRemoteLookup(8))
    request = make_request(prompt_tokens=8, tokens=10)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (8, False)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), 8)

    resumed = scheduler.build_step(
        make_output(
            cached=(("request", ([1, 2, 3],), 0),),
            resumed=("request",),
            scheduled_tokens={"request": 8},
        )
    )

    assert [command.load_range for command in resumed.load.commands] == [TokenRange(0, 8)]
    assert resumed.store.commands == ()


def test_lookup_discards_sub_transfer_chunk() -> None:
    lookup = FakeRemoteLookup(4)
    scheduler = SynchronousBulkScheduler(
        make_config(granularity=8),
        lookup,
    )
    request = make_request(prompt_tokens=4, tokens=5)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert lookup.queries == []


def test_new_and_running_publish_monotonic_store_without_new_blocks() -> None:
    scheduler = SynchronousBulkScheduler(
        make_config(
            granularity=8,
            load=False,
            store=True,
        ),
        FakeRemoteLookup(),
    )
    pool = FakeBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    request = make_request()
    scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)

    first = scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1, 2],), 0),),
            scheduled_tokens={"request": 6},
        )
    )
    second = scheduler.build_step(
        make_output(
            cached=(("request", None, 6),),
            scheduled_tokens={"request": 2},
        )
    )

    commands = first.store.commands + second.store.commands
    assert tuple(command.store_range for command in commands) == (TokenRange(0, 8),)
    assert all(command.store_range.end_token > command.store_range.start_token for command in commands)
    for command in commands:
        scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))


def test_resumed_prefill_frontier_survives_multiple_running_chunks() -> None:
    scheduler = SynchronousBulkScheduler(
        make_config(load=False, store=True),
        FakeRemoteLookup(),
    )
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=4, tokens=12, hashes=4)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3, 4]), 0)

    resumed = scheduler.build_step(
        make_output(
            cached=(("request", ([1, 2, 3, 4],), 4),),
            resumed=("request",),
            scheduled_tokens={"request": 4},
        )
    )
    first_reconstruction = scheduler.build_step(
        make_output(
            cached=(("request", None, 8),),
            scheduled_tokens={"request": 2},
        )
    )
    second_reconstruction = scheduler.build_step(
        make_output(
            cached=(("request", None, 10),),
            scheduled_tokens={"request": 2},
        )
    )
    request.num_tokens = 16
    decode = scheduler.build_step(
        make_output(
            cached=(("request", None, 12),),
            scheduled_tokens={"request": 4},
        )
    )

    reconstruction = resumed.store.commands + first_reconstruction.store.commands + second_reconstruction.store.commands
    assert tuple(command.store_range for command in reconstruction) == (
        TokenRange(0, 8),
        TokenRange(8, 12),
    )
    assert decode.store.commands == ()


def test_step_checkpoint_is_source_ready_and_preserves_exact_source() -> None:
    scheduler = SynchronousBulkScheduler(
        make_config(groups=(0, 1), align_groups=frozenset((1,)), load=False, store=True),
        FakeRemoteLookup(),
    )
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=12)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3], [10, 11]), 0)

    step = scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1, 2, 3], [10, 11]), 0),),
            scheduled_tokens={"request": 12},
            checkpoints={"request": [(1, 10, 8), (1, 11, 12)]},
        )
    )

    assert [command.store_range for command in step.store.source_pending_commands] == [TokenRange(0, 12)]
    checkpoints = step.store.source_ready_commands
    assert all(isinstance(command, CheckpointStoreCommand) for command in checkpoints)
    assert [command.sources for command in checkpoints] == [
        (StateCheckpointSource(1, 10, 8),),
        (StateCheckpointSource(1, 11, 12),),
    ]
    assert all(command.block_ids_by_group == ((1, 2, 3), (10, 11)) for command in checkpoints)


def test_finished_and_preempted_cancel_unpublished_load() -> None:
    for stage in ("lookup", "sync", "async"):
        for terminated_field in ("finished", "preempted"):
            scheduler_type = AsynchronousBulkScheduler if stage == "async" else SynchronousBulkScheduler
            scheduler = scheduler_type(make_config(), FakeRemoteLookup(8))
            request = make_request(tokens=9)
            scheduler.get_num_new_matched_tokens(request, 0)
            if stage != "lookup":
                scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 8)

            termination = {terminated_field: ("request",)}
            assert scheduler.build_step(make_output(**termination)).load.commands == (), (stage, terminated_field)
            assert scheduler.build_step(make_output()).load.commands == (), (stage, terminated_field)


def test_finished_tail_lease_survives_cleanup_until_every_worker_releases() -> None:
    scheduler = SynchronousBulkScheduler(
        make_config(
            groups=(0, 1),
            align_groups=frozenset((1,)),
            store=True,
            workers=2,
        ),
        FakeRemoteLookup(8),
    )
    pool = FakeBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    request = make_request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (7, False)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2], [10, 11]), 7)
    scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1, 2], [10, 11]), 7),),
            scheduled_tokens={"request": 1},
        )
    )

    assert not scheduler.register_finished_partial_tail(
        request,
        ([1, 2], [10, 11]),
        [(1, 11, 8)],
    )
    assert scheduler.has_pending_push_work()
    assert pool.touched == [(11, 1, 2)]

    finished = scheduler.build_step(make_output(finished=("request",)))
    assert len(finished.store.source_ready_commands) == 1
    command = finished.store.source_ready_commands[0]
    assert command.sources == (StateCheckpointSource(1, 11, 8),)
    assert scheduler.has_pending_push_work()

    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
    assert scheduler.has_pending_push_work()
    assert pool.freed == []
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
    assert not scheduler.has_pending_push_work()
    assert pool.freed == [(2, 1, 11)]


@pytest.mark.parametrize("eagle,expected_external_tokens", ((True, 0), (False, 3)))
def test_store_skip_survives_safe_frontier_trimming(eagle, expected_external_tokens) -> None:
    scheduler = SynchronousBulkScheduler(
        make_config(store=True, save_decode=True, eagle=eagle),
        FakeRemoteLookup(4),
    )
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=4)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (expected_external_tokens, False)
    scheduler.confirm_allocation(request, FakeBlocks([1]), expected_external_tokens)
    initial = scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1],), expected_external_tokens),),
            scheduled_tokens={"request": 4 - expected_external_tokens},
        )
    )
    assert initial.store.commands == ()

    request.num_tokens = 8
    request.block_hashes.append(b"b")
    decode = scheduler.build_step(
        make_output(
            cached=(("request", ([2],), 4),),
            scheduled_tokens={"request": 4},
        )
    )
    assert [command.store_range for command in decode.store.commands] == [TokenRange(4, 8)]


def test_zero_external_match_never_advertises_async_loading() -> None:
    scheduler = AsynchronousBulkScheduler(
        make_config(store=True, eagle=True),
        FakeRemoteLookup(4),
    )
    request = make_request(prompt_tokens=4)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)


@pytest.mark.parametrize(
    "readable_end,eagle,expected_safe_end",
    (
        (12, True, 8),
        (12, False, 8),
        (8, False, 8),
    ),
)
def test_private_state_recomputes_safe_frontier_without_erasing_store_skip(
    readable_end,
    eagle,
    expected_safe_end,
) -> None:
    scheduler = SynchronousBulkScheduler(
        make_config(store=True, private=True, eagle=eagle),
        FakeRemoteLookup(readable_end),
    )
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=12)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (expected_safe_end, False)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3, 4]), expected_safe_end)
    initial = scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1, 2, 3, 4],), expected_safe_end),),
            scheduled_tokens={"request": 0},
        )
    )
    assert initial.store.commands == ()

    request.num_tokens = 16
    request.block_hashes.append(b"next")
    running = scheduler.build_step(
        make_output(
            cached=(("request", None, expected_safe_end),),
            scheduled_tokens={"request": 16 - expected_safe_end},
        )
    )
    assert [command.store_range for command in running.store.commands] == [TokenRange(readable_end, 16)]


def test_worker_lookup_reports_raw_eagle_reachability() -> None:
    group = replace(make_topology().groups[0], is_eagle_group=True)
    reachability = HybridReachability((group,), 4, 4, 64, use_eagle=True)
    block_hashes = (b"a", b"b")
    observation = (
        (np.asarray((0, 4)), np.asarray((4, 4)), block_hashes),
        (True, True),
    )

    assert reachability.resolve_available_end(TokenRange(0, 8), block_hashes, (observation,)).end_token == 8

    scheduler = SynchronousBulkScheduler(make_config(eagle=True), FakeRemoteLookup(8))
    assert scheduler.get_num_new_matched_tokens(make_request(prompt_tokens=8), 0) == (4, False)


def test_factory_binds_role_capabilities(monkeypatch) -> None:
    cases: tuple[tuple[str, dict[str, bool], bool, bool], ...] = (
        ("kv_producer", {}, True, True),
        ("kv_both", {}, True, True),
        ("kv_consumer", {}, False, False),
        ("kv_consumer", {"consumer_is_to_load": True}, True, False),
        ("kv_consumer", {"consumer_is_to_put": True}, False, True),
    )
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )
    for role, extra, load_enabled, store_enabled in cases:
        lookup = FakeRemoteLookup(4)
        monkeypatch.setattr(vllm_adapter, "RemoteLookup", lambda _address, lookup=lookup: lookup)
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role=role, kv_connector_extra_config=extra),
            parallel_config=SimpleNamespace(world_size=1),
            speculative_config=None,
        )
        scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
        scheduler.bind_gpu_block_pool(FakeBlockPool())

        lookup_request = make_request("lookup", tokens=9)
        assert scheduler.get_num_new_matched_tokens(lookup_request, 0)[0] == (4 if load_enabled else 0), role
        assert bool(lookup.queries) is load_enabled, role

        store_request = make_request("store")
        scheduler.confirm_allocation(store_request, FakeBlocks([1, 2]), 0)
        step = scheduler.build_step(
            make_output(
                new=(new_request_data("store", ([1, 2],), 0),),
                scheduled_tokens={"store": 4},
            )
        )
        assert bool(step.store.commands) is store_enabled, role
        for command in step.store.commands:
            scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
        scheduler.close()


def test_factory_selects_one_concrete_publication_route(monkeypatch) -> None:
    cases = (
        (False, False, SynchronousBulkScheduler),
        (False, True, AsynchronousBulkScheduler),
        (True, False, LayerwiseScheduler),
        (True, True, LayerwiseScheduler),
    )
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )
    for use_layerwise, load_async, expected_type in cases:
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_both",
                kv_connector_extra_config={"use_layerwise": use_layerwise, "load_async": load_async},
            ),
            parallel_config=SimpleNamespace(world_size=1),
            speculative_config=None,
        )

        scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
        assert isinstance(scheduler, expected_type), (use_layerwise, load_async)
        scheduler.close()


@pytest.mark.parametrize("private_only", (False, True))
def test_factory_rejects_private_state_layerwise_and_private_only(monkeypatch, private_only) -> None:
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    cache_config = SimpleNamespace(
        transfer_group_ids=() if private_only else (0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
        if private_only
        else [SimpleNamespace(kv_cache_spec=object()), SimpleNamespace(kv_cache_spec=object())],
    )
    if private_only:

        def reject_private_only(_groups):
            raise AssertionError("no cacheable groups")

        monkeypatch.setattr(
            vllm_adapter,
            "infer_cacheable_group_ids",
            reject_private_only,
        )
        expected_message = "at least one prefix-cacheable"
    else:
        monkeypatch.setattr(vllm_adapter, "infer_cacheable_group_ids", lambda _groups: [0])
        expected_message = "private KV state requires non-layerwise"
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"use_layerwise": True},
        ),
        parallel_config=SimpleNamespace(world_size=1),
        speculative_config=None,
    )

    with pytest.raises(ValueError, match=expected_message):
        vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")


@pytest.mark.parametrize(
    "extra,force_reuse,expected_message",
    (
        (
            {"discard_partial_chunks": False},
            False,
            "does not support discard_partial_chunks=False",
        ),
        (
            {"use_layerwise": True, "discard_partial_chunks": False},
            False,
            "does not support discard_partial_chunks=False",
        ),
        (
            {"use_layerwise": True},
            True,
            "buffer reuse requires native partial-object support",
        ),
    ),
)
def test_factories_reject_unproven_partial_object_routes(
    monkeypatch,
    extra,
    force_reuse,
    expected_message,
) -> None:
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    monkeypatch.setattr(vllm_adapter, "_uses_layerwise_buffer_reuse", lambda *_: force_reuse)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config=extra,
        ),
        parallel_config=SimpleNamespace(world_size=1),
        speculative_config=None,
    )

    with pytest.raises(ValueError, match=expected_message):
        vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
    with pytest.raises(ValueError, match=expected_message):
        vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)


def test_scheduler_close_rejects_active_store_source_leases() -> None:
    lookup = FakeRemoteLookup()
    scheduler = SynchronousBulkScheduler(
        make_config(load=False, store=True),
        lookup,
    )
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=4)
    scheduler.confirm_allocation(request, FakeBlocks([1]), 0)
    step = scheduler.build_step(
        make_output(
            new=(new_request_data("request", ([1],), 0),),
            scheduled_tokens={"request": 4},
        )
    )
    command = step.store.commands[0]

    with pytest.raises(RuntimeError, match=r"source lease job ids=\(0,\)"):
        scheduler.close()
    assert lookup.closed

    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
    scheduler.close()


def test_connector_scheduler_hooks_are_thin_delegations() -> None:
    calls: list[tuple[Any, ...]] = []
    expected_step = KVTransferStep()

    def get_num_new_matched_tokens(*args):
        calls.append(("lookup", args))
        return 3, True

    def build_step(output):
        calls.append(("step", output))
        return expected_step

    def register_finished_partial_tail(*args):
        calls.append(("tail", args))
        return False

    scheduler = SimpleNamespace(
        get_num_new_matched_tokens=get_num_new_matched_tokens,
        confirm_allocation=lambda *args: calls.append(("allocation", args)),
        build_step=build_step,
        accept_worker_metadata=lambda metadata: calls.append(("worker", metadata)),
        register_finished_partial_tail=register_finished_partial_tail,
        bind_gpu_block_pool=lambda pool: calls.append(("pool", pool)),
        has_pending_push_work=lambda: True,
        close=lambda: calls.append(("close",)),
    )
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.scheduler = scheduler
    connector.worker = None
    connector.lookup_server = None
    request = make_request()
    blocks = FakeBlocks([1, 2])
    output = make_output()
    worker_metadata = object()

    assert connector.get_num_new_matched_tokens(request, 2) == (3, True)
    connector.update_state_after_alloc(request, blocks, 3)
    assert connector.build_connector_meta(output) is expected_step
    connector.update_connector_output(SimpleNamespace(kv_connector_worker_meta=worker_metadata))
    connector.register_finished_partial_tail(request, ([1, 2],), [(0, 2, 8)])
    connector.bind_gpu_block_pool("pool")
    assert connector.has_pending_push_work()
    connector.shutdown()

    assert calls == [
        ("lookup", (request, 2)),
        ("allocation", (request, blocks, 3)),
        ("step", output),
        ("worker", worker_metadata),
        ("tail", (request, ([1, 2],), [(0, 2, 8)])),
        ("pool", "pool"),
        ("close",),
    ]
