"""Lookup frontiers, allocation-gated publication, and source-lease release."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.v1.helpers import (
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlockPool,
    FakeBlocks,
    FakeRemoteLookup,
    make_config,
    make_output,
    make_request,
    new_request_data,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection.reachability import (
    HybridReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    StateCheckpointSource,
    StoreSourceReleaseMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import (
    AsynchronousBulkScheduler,
    LayerwiseScheduler,
    SynchronousBulkScheduler,
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


@pytest.mark.parametrize("scheduler_type", (SynchronousBulkScheduler, AsynchronousBulkScheduler, LayerwiseScheduler))
@pytest.mark.parametrize("save_decode", (False, True))
def test_store_uses_confirmed_hashes_and_current_progress_after_draft_rejection(scheduler_type, save_decode) -> None:
    scheduler = scheduler_type(make_config(load=False, store=True, save_decode=save_decode), FakeRemoteLookup())
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=8, hashes=2)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3, 4]), 0)
    commands = []
    try:
        initial = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2, 3, 4],), 0),), scheduled_tokens={"request": 12})
        )
        commands.extend(initial.store.commands)
        assert [command.store_range for command in initial.store.commands] == [TokenRange(0, 8)]

        request.num_tokens = 13
        # Scheduling reaches a third block, but its confirmed hash is not available yet.
        unhashed = scheduler.build_step(make_output(cached=(("request", None, 8),), scheduled_tokens={"request": 5}))
        assert unhashed.store.commands == ()

        request.block_hashes.append(b"accepted")
        # vLLM has rejected part of the previous draft and supplies the rolled-back position.
        rejected = scheduler.build_step(make_output(cached=(("request", None, 9),), scheduled_tokens={"request": 2}))
        assert rejected.store.commands == ()

        completed = scheduler.build_step(make_output(cached=(("request", None, 11),), scheduled_tokens={"request": 1}))
        commands.extend(completed.store.commands)
        assert [command.store_range for command in completed.store.commands] == (
            [TokenRange(8, 12)] if save_decode else []
        )
        if save_decode:
            assert completed.store.commands[0].block_hashes[-1] == b"accepted"
        repeated = scheduler.build_step(make_output(cached=(("request", None, 12),), scheduled_tokens={"request": 1}))
        assert repeated.store.commands == ()
    finally:
        for command in commands:
            scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
        scheduler.close()


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
    "prompt_tokens,readable_end,local_tokens,eagle,safe_end,load_end",
    (
        (8, 8, 0, True, 4, 4),
        (8, 8, 0, False, 7, 8),
        (9, 8, 0, True, 8, 8),
        (12, 8, 0, True, 8, 8),
        (8, 8, 6, True, 6, 0),
    ),
)
def test_speculative_lookup_recomputes_only_the_prompt_tail(
    prompt_tokens, readable_end, local_tokens, eagle, safe_end, load_end
) -> None:
    scheduler = SynchronousBulkScheduler(make_config(eagle=eagle), FakeRemoteLookup(readable_end))
    request = make_request(prompt_tokens=prompt_tokens)
    try:
        assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (safe_end - local_tokens, False)
        scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), safe_end - local_tokens)
        step = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2, 3],), safe_end),), scheduled_tokens={"request": 0})
        )
        assert [command.load_range for command in step.load.commands] == ([TokenRange(0, load_end)] if load_end else [])
    finally:
        scheduler.close()


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
