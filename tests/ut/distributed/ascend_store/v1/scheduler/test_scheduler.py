"""Lookup frontiers, allocation-gated publication, and source-lease release."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

from tests.ut.distributed.ascend_store.v1.align_state_fixtures import make_sparse_group_topology
from tests.ut.distributed.ascend_store.v1.helpers import (
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlockPool,
    FakeBlocks,
    FakeLookupClient,
    make_config,
    make_output,
    make_request,
    new_request_data,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.coordinator import (
    Coordinator,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    CheckpointStoreCommand,
    StateCheckpointSource,
    StoreSourceReleaseMetadata,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import (
    AsynchronousScheduler,
    SynchronousScheduler,
)


class _RefCountBlockPool(FakeBlockPool):
    """Keep the fixture's observations while mirroring upstream physical references."""

    def __init__(self) -> None:
        super().__init__()
        for block in self.blocks:
            block.ref_cnt = 0

    def touch(self, blocks) -> None:
        blocks = tuple(blocks)
        super().touch(blocks)
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks) -> None:
        blocks = tuple(blocks)
        super().free_blocks(blocks)
        for block in blocks:
            block.ref_cnt -= 1


@pytest.mark.parametrize("eagle,physical_end", ((False, 128), (True, 112)))
def test_sliding_load_uses_scheduler_physical_frontier_without_another_eagle_drop(eagle, physical_end) -> None:
    window = replace(make_sparse_group_topology(block_size=16, sliding_window=33).transfer_groups[1], group_id=0)
    config = replace(make_config(granularity=16, eagle=eagle), hash_block_size=16, group_specs=(window.kv_cache_spec,))
    coordinator = Coordinator((window,), 16, 16, 256, use_eagle=eagle)
    scheduler = SynchronousScheduler(config, FakeLookupClient(128))
    request = make_request(prompt_tokens=128, hashes=8)
    blocks = list(range(1, 9))
    try:
        matched_tokens, _ = scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.confirm_allocation(request, FakeBlocks(blocks), matched_tokens)
        (load,) = scheduler.build_step(
            make_output(
                new=(new_request_data("request", (blocks,), matched_tokens),),
                scheduled_tokens={"request": 128 - matched_tokens},
            )
        ).load_commands
        assert load.end_token == physical_end
        assert coordinator.select_for_load(load.block_hashes, load.end_token) == (
            (False,) * (physical_end // 16 - 2) + (True, True),
        )
    finally:
        scheduler.close()


def test_load_publication_requires_allocation_and_is_exactly_once() -> None:
    for scheduler_type, is_deferred in ((SynchronousScheduler, False), (AsynchronousScheduler, True)):
        lookup = FakeLookupClient(12)
        scheduler = scheduler_type(make_config(), lookup)
        request = make_request(prompt_tokens=12)

        assert scheduler.get_num_new_matched_tokens(request, 0) == (11, is_deferred)
        assert scheduler.build_step(make_output()).load_commands == ()
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
            assert published.load_commands[0].block_ids_by_group == ((1, 2, 3, 4),)
            assert scheduler.build_step(make_output()).load_commands == ()
        else:
            assert scheduler.build_step(make_output()).load_commands == ()
            published = scheduler.build_step(scheduled)
            assert published.load_commands[0].block_ids_by_group == ((1, 2, 3, 4),)
            assert scheduler.build_step(make_output()).load_commands == ()

        assert (published.load_commands[0].start_token, published.load_commands[0].end_token) == (0, 12)


@pytest.mark.parametrize("scheduler_type", (SynchronousScheduler, AsynchronousScheduler))
def test_emitted_commands_keep_their_snapshots_when_request_progress_advances(scheduler_type) -> None:
    scheduler = scheduler_type(make_config(store=True, save_decode=True), FakeLookupClient(4))
    request = make_request(prompt_tokens=8)
    block_ids = [1, 2]
    original_hashes = tuple(request.block_hashes)
    try:
        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.confirm_allocation(request, FakeBlocks(block_ids), 4)
        allocated = scheduler.build_step(make_output())
        initial = scheduler.build_step(
            make_output(new=(new_request_data("request", (block_ids,), 4),), scheduled_tokens={"request": 4})
        )
        (load,) = allocated.load_commands + initial.load_commands
        (store,) = initial.store.commands

        block_ids[0] = 99
        request.num_tokens = 12
        request.block_hashes.append(b"next")
        running = scheduler.build_step(make_output(cached=(("request", ([3],), 8),), scheduled_tokens={"request": 4}))
        (next_store,) = running.store.commands

        assert load.block_ids_by_group == store.block_ids_by_group == ((1, 2),)
        assert load.block_hashes == store.block_hashes == original_hashes
        assert (store.start_token, store.end_token) == (4, 8)
        assert (next_store.start_token, next_store.end_token) == (8, 12)
        assert next_store.block_ids_by_group == ((1, 2, 3),)
        assert next_store.block_hashes == (*original_hashes, b"next")
        assert not running.load_commands
    finally:
        scheduler.close()


@pytest.mark.parametrize("change", ("append", "replace"))
def test_hash_snapshots_refresh_without_changing_earlier_queries_or_commands(change) -> None:
    lookup = FakeLookupClient(4)
    scheduler = SynchronousScheduler(make_config(), lookup)
    request = make_request()
    original = tuple(request.block_hashes)
    try:
        scheduler.get_num_new_matched_tokens(request, 0)
        if change == "append":
            request.num_tokens += 4
            request.block_hashes.append(b"new")
        else:
            request.block_hashes = [b"replacement", request.block_hashes[1]]
        expected = tuple(request.block_hashes)
        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), 4)
        (load,) = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2, 3],), 4),), scheduled_tokens={"request": 4})
        ).load_commands
        request.num_tokens += 4
        request.block_hashes.append(b"later")

        assert lookup.queries[0][2] == original
        assert lookup.queries[1][2] == expected
        assert load.block_hashes == expected
    finally:
        scheduler.close()


@pytest.mark.parametrize("mode", ("bulk", "layerwise", "async"))
def test_unselected_load_only_route_keeps_empty_steps_and_request_release(mode) -> None:
    config = replace(make_config(), use_layerwise=mode == "layerwise", load_async=mode == "async")
    scheduler_type = AsynchronousScheduler if mode == "async" else SynchronousScheduler
    scheduler = scheduler_type(config, FakeLookupClient(4))
    request = make_request()
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (4, mode == "async")
        # Another connector won the match; real local blocks do not authorize a Load.
        scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)
        initial = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2],), 0),), scheduled_tokens={"request": 4})
        )
        request.num_tokens = 12
        request.block_hashes.append(b"next")
        running = scheduler.build_step(make_output(cached=(("request", ([3],), 4),), scheduled_tokens={"request": 4}))
        released = scheduler.build_step(make_output(preempted=("request",)))
        scheduler.confirm_allocation(request, FakeBlocks([4, 5, 6]), 0)
        resumed = scheduler.build_step(
            make_output(cached=(("request", ([4, 5, 6],), 0),), resumed=("request",), scheduled_tokens={"request": 4})
        )
        for step in (initial, running, resumed):
            assert step.load_commands == step.store.commands == ()
        assert released.released_request_ids == frozenset({"request"})
    finally:
        scheduler.close()


@pytest.mark.parametrize("entry", ("new", "resumed"))
def test_admission_rejects_unconfirmed_block_tables_without_consuming_load(entry) -> None:
    scheduler = SynchronousScheduler(make_config(), FakeLookupClient(4))
    request = make_request(prompt_tokens=8)
    scheduler.get_num_new_matched_tokens(request, 0)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 4)
    try:
        for block_ids, valid in (([9, 10], False), ([1, 2], True)):
            output = (
                make_output(new=(new_request_data("request", (block_ids,), 4),), scheduled_tokens={"request": 4})
                if entry == "new"
                else make_output(
                    cached=(("request", (block_ids,), 4),), resumed=("request",), scheduled_tokens={"request": 4}
                )
            )
            if not valid:
                with pytest.raises(ValueError, match="does not match its confirmed allocation"):
                    scheduler.build_step(output)
                continue
            (load,) = scheduler.build_step(output).load_commands
            assert load.block_ids_by_group == ((1, 2),)
            assert (load.start_token, load.end_token) == (0, 4)
        assert not scheduler.build_step(make_output()).load_commands
    finally:
        scheduler.close()


@pytest.mark.parametrize("scheduler_type", (SynchronousScheduler, AsynchronousScheduler))
def test_rejected_allocation_keeps_lookup_candidate_for_valid_confirmation(scheduler_type) -> None:
    scheduler = scheduler_type(make_config(), FakeLookupClient(4))
    request = make_request()
    try:
        scheduler.get_num_new_matched_tokens(request, 0)
        with pytest.raises(ValueError, match="expected either 0 or 4"):
            scheduler.confirm_allocation(request, FakeBlocks([9, 10]), 3)
        scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 4)
        step = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2],), 4),), scheduled_tokens={"request": 4})
        )
        (load,) = step.load_commands
        assert (load.start_token, load.end_token) == (0, 4)
        assert load.block_ids_by_group == ((1, 2),)
        assert not scheduler.build_step(make_output()).load_commands
    finally:
        scheduler.close()


@pytest.mark.parametrize("scheduler_type", (SynchronousScheduler, AsynchronousScheduler))
def test_failed_step_keeps_every_requests_pending_load_and_store_progress(scheduler_type) -> None:
    scheduler = scheduler_type(make_config(store=True, recycling=True), FakeLookupClient(4))
    pool = _RefCountBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    sources = [pool.blocks[index] for index in (1, 2, 3, 4)]
    pool.touch(sources)
    for request_id, blocks in (("first", [1, 2]), ("second", [3, 4])):
        request = make_request(request_id)
        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.confirm_allocation(request, FakeBlocks(blocks), 4)
    try:
        with pytest.raises(ValueError, match="does not match its confirmed allocation"):
            scheduler.build_step(
                make_output(
                    new=(new_request_data("first", ([1, 2],), 4), new_request_data("second", ([9, 10],), 4)),
                    scheduled_tokens={"first": 4, "second": 4},
                )
            )
        assert [block.ref_cnt for block in sources] == [1, 1, 1, 1]
        assert not scheduler.has_pending_push_work()

        step = scheduler.build_step(
            make_output(
                new=(new_request_data("first", ([1, 2],), 4), new_request_data("second", ([3, 4],), 4)),
                scheduled_tokens={"first": 4, "second": 4},
            )
        )
        assert [(command.request_id, command.start_token, command.end_token) for command in step.load_commands] == [
            ("first", 0, 4),
            ("second", 0, 4),
        ]
        assert [(command.start_token, command.end_token) for command in step.store.source_pending_commands] == (
            [(4, 8), (4, 8)] if scheduler_type is SynchronousScheduler else []
        )
        scheduler.accept_worker_metadata(
            StoreSourceReleaseMetadata(
                {command.store_job_id: 1 for command in step.store.commands if command.store_job_id is not None}
            )
        )
        assert [block.ref_cnt for block in sources] == [1, 1, 1, 1]
        assert not scheduler.build_step(make_output()).load_commands
    finally:
        scheduler.close()


def test_failed_publication_rolls_back_new_leases_and_keeps_finished_checkpoints() -> None:
    config = make_config(groups=(0, 1), align_groups=frozenset((1,)), load=False, store=True, workers=2, recycling=True)
    scheduler = SynchronousScheduler(config, FakeLookupClient())
    pool = _RefCountBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    old_sources = [pool.blocks[index] for index in (1, 2, 10, 11)]
    new_sources = [pool.blocks[index] for index in (4, 5, 12, 13)]
    pool.touch(old_sources + new_sources)
    finished_request = make_request("finished")
    scheduler.confirm_allocation(finished_request, FakeBlocks([1, 2], [10, 11]), 0)
    initial = scheduler.build_step(
        make_output(new=(new_request_data("finished", ([1, 2], [10, 11]), 0),), scheduled_tokens={"finished": 8})
    )
    initial_job = initial.store.commands[0].store_job_id
    assert initial_job is not None
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({initial_job: 2}))
    assert not scheduler.register_finished_partial_tail(finished_request, ([1, 2], [10, 11]), [(1, 10, 8)])
    pool.free_blocks(old_sources)
    assert pool.blocks[10].ref_cnt == 1

    request = make_request("new")
    scheduler.confirm_allocation(request, FakeBlocks([4, 5], [12, 13]), 0)
    with pytest.raises(IndexError):
        scheduler.build_step(
            make_output(
                finished=("finished",),
                new=(new_request_data("new", ([4, 5], [12, 13]), 0),),
                scheduled_tokens={"new": 8},
                checkpoints={"new": [(1, 1000, 8)]},
            )
        )
    assert [block.ref_cnt for block in new_sources] == [1, 1, 1, 1]
    assert pool.blocks[10].ref_cnt == 1
    assert scheduler.has_pending_push_work()

    step = scheduler.build_step(
        make_output(
            new=(new_request_data("new", ([4, 5], [12, 13]), 0),),
            scheduled_tokens={"new": 8},
            checkpoints={"new": [(1, 12, 8)]},
        )
    )
    (store,) = step.store.source_pending_commands
    assert (store.start_token, store.end_token) == (0, 8)
    assert [command.sources for command in step.store.source_ready_commands] == [
        (StateCheckpointSource(1, 12, 8),),
        (StateCheckpointSource(1, 10, 8),),
    ]
    scheduler.accept_worker_metadata(
        StoreSourceReleaseMetadata(
            {command.store_job_id: 2 for command in step.store.commands if command.store_job_id is not None}
        )
    )
    assert [block.ref_cnt for block in new_sources] == [1, 1, 1, 1]
    assert pool.blocks[10].ref_cnt == 0
    assert not scheduler.has_pending_push_work()
    assert not scheduler.build_step(make_output()).store.commands
    scheduler.close()


def test_publication_failure_keeps_async_load_without_a_scheduled_forward() -> None:
    config = make_config(groups=(0, 1), align_groups=frozenset((1,)), store=True, recycling=True)
    scheduler = AsynchronousScheduler(config, FakeLookupClient(4))
    pool = _RefCountBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    sources = [pool.blocks[index] for index in (3, 4, 12, 13)]
    pool.touch(sources)
    load_request = make_request("load")
    scheduler.get_num_new_matched_tokens(load_request, 0)
    scheduler.confirm_allocation(load_request, FakeBlocks([1, 2], [10, 11]), 4)
    cancelled = make_request("cancelled")
    scheduler.get_num_new_matched_tokens(cancelled, 0)
    scheduler.confirm_allocation(cancelled, FakeBlocks([5, 6], [14, 15]), 4)
    scheduler.confirm_allocation(make_request("store"), FakeBlocks([3, 4], [12, 13]), 0)

    with pytest.raises(IndexError):
        scheduler.build_step(
            make_output(
                preempted=("cancelled",),
                new=(new_request_data("store", ([3, 4], [12, 13]), 0),),
                scheduled_tokens={"store": 8},
                checkpoints={"store": [(1, 1000, 8)]},
            )
        )
    assert [block.ref_cnt for block in sources] == [1, 1, 1, 1]
    assert not scheduler.has_pending_push_work()

    # Cancellation stays effective; only the other allocation-ready Load survives the failed step.
    (load,) = scheduler.build_step(make_output()).load_commands
    assert load.request_id == "load"
    assert (load.start_token, load.end_token) == (0, 4)
    assert load.block_ids_by_group == ((1, 2), (10, 11))
    assert not scheduler.build_step(make_output()).load_commands
    scheduler.close()


@pytest.mark.parametrize("scheduler_type", (SynchronousScheduler, AsynchronousScheduler, SynchronousScheduler))
def test_store_publication_respects_load_timing(scheduler_type):
    scheduler = scheduler_type(make_config(store=True), FakeLookupClient(4))
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=8, tokens=9)
    deferred = scheduler_type is AsynchronousScheduler
    assert scheduler.get_num_new_matched_tokens(request, 0) == (4, deferred)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), 4)
    step = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1, 2, 3],), 4),), scheduled_tokens={"request": 4})
    )
    assert [(command.start_token, command.end_token) for command in step.load_commands] == [(0, 4)]
    assert [(command.start_token, command.end_token) for command in step.store.commands] == (
        [] if deferred else [(4, 8)]
    )
    scheduler.close()


def test_synchronous_local_prefix_does_not_publish_a_redundant_load() -> None:
    lookup = FakeLookupClient(8)
    scheduler = SynchronousScheduler(make_config(), lookup)
    request = make_request(prompt_tokens=8, tokens=9)

    assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
    assert lookup.queries == []
    scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)
    step = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1, 2],), 8),), scheduled_tokens={"request": 1})
    )

    assert not step.load_commands


def test_load_only_resumed_request_publishes_scheduled_load() -> None:
    scheduler = SynchronousScheduler(make_config(store=False), FakeLookupClient(8))
    request = make_request(prompt_tokens=8, tokens=10)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (8, False)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), 8)

    resumed = scheduler.build_step(
        make_output(cached=(("request", ([1, 2, 3],), 0),), resumed=("request",), scheduled_tokens={"request": 8})
    )

    assert [(command.start_token, command.end_token) for command in resumed.load_commands] == [(0, 8)]
    assert resumed.store.commands == ()


def test_lookup_discards_sub_transfer_chunk() -> None:
    lookup = FakeLookupClient(4)
    scheduler = SynchronousScheduler(make_config(granularity=8), lookup)
    request = make_request(prompt_tokens=4, tokens=5)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    assert lookup.queries == []


def test_new_and_running_publish_monotonic_store_without_new_blocks() -> None:
    scheduler = SynchronousScheduler(make_config(granularity=8, load=False, store=True), FakeLookupClient())
    pool = FakeBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    request = make_request()
    scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)

    first = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1, 2],), 0),), scheduled_tokens={"request": 6})
    )
    second = scheduler.build_step(make_output(cached=(("request", None, 6),), scheduled_tokens={"request": 2}))

    commands = first.store.commands + second.store.commands
    assert tuple((command.start_token, command.end_token) for command in commands) == ((0, 8),)
    assert all(command.end_token > command.start_token for command in commands)
    for command in commands:
        scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))


@pytest.mark.parametrize("scheduler_type", (SynchronousScheduler, AsynchronousScheduler, SynchronousScheduler))
def test_resumed_prefill_frontier_survives_multiple_running_chunks(scheduler_type) -> None:
    scheduler = scheduler_type(make_config(load=False, store=True), FakeLookupClient())
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=4, hashes=1)
    scheduler.confirm_allocation(request, FakeBlocks([1]), 0)
    initial = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1],), 0),), scheduled_tokens={"request": 4})
    )
    assert [(command.start_token, command.end_token) for command in initial.store.commands] == [(0, 4)]
    for computed_tokens, block_id in ((4, 2), (8, 3)):
        request.num_tokens = computed_tokens + 4
        request.block_hashes.append(bytes((block_id,)))
        step = scheduler.build_step(
            make_output(cached=(("request", ([block_id],), computed_tokens),), scheduled_tokens={"request": 4})
        )
        assert step.load_commands == step.store.commands == ()

    scheduler.build_step(make_output(preempted=("request",)))
    scheduler.confirm_allocation(request, FakeBlocks([4, 5, 6, 7]), 0)

    resumed = scheduler.build_step(
        make_output(cached=(("request", ([4, 5, 6, 7],), 4),), resumed=("request",), scheduled_tokens={"request": 4})
    )
    first_reconstruction = scheduler.build_step(
        make_output(cached=(("request", None, 8),), scheduled_tokens={"request": 2})
    )
    second_reconstruction = scheduler.build_step(
        make_output(cached=(("request", None, 10),), scheduled_tokens={"request": 2})
    )
    request.num_tokens = 16
    decode = scheduler.build_step(make_output(cached=(("request", None, 12),), scheduled_tokens={"request": 4}))

    reconstruction = resumed.store.commands + first_reconstruction.store.commands + second_reconstruction.store.commands
    assert tuple((command.start_token, command.end_token) for command in reconstruction) == ((0, 8), (8, 12))
    assert all(command.block_ids_by_group == ((4, 5, 6, 7),) for command in reconstruction)
    assert all(command.block_hashes == tuple(request.block_hashes) for command in reconstruction)
    assert decode.store.commands == ()


@pytest.mark.parametrize("scheduler_type", (SynchronousScheduler, AsynchronousScheduler, SynchronousScheduler))
@pytest.mark.parametrize("save_decode", (False, True))
def test_store_uses_confirmed_hashes_and_current_progress_after_draft_rejection(scheduler_type, save_decode) -> None:
    scheduler = scheduler_type(make_config(load=False, store=True, save_decode=save_decode), FakeLookupClient())
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=8, hashes=2)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3, 4]), 0)
    commands = []
    try:
        initial = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2, 3, 4],), 0),), scheduled_tokens={"request": 12})
        )
        commands.extend(initial.store.commands)
        assert [(command.start_token, command.end_token) for command in initial.store.commands] == [(0, 8)]

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
        assert [(command.start_token, command.end_token) for command in completed.store.commands] == (
            [(8, 12)] if save_decode else []
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
    scheduler = SynchronousScheduler(
        make_config(groups=(0, 1), align_groups=frozenset((1,)), load=False, store=True),
        FakeLookupClient(),
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

    assert [(command.start_token, command.end_token) for command in step.store.source_pending_commands] == [(0, 12)]
    checkpoints = step.store.source_ready_commands
    assert all(isinstance(command, CheckpointStoreCommand) for command in checkpoints)
    assert [command.sources for command in checkpoints] == [
        (StateCheckpointSource(1, 10, 8),),
        (StateCheckpointSource(1, 11, 12),),
    ]
    assert all(command.block_ids_by_group == ((1, 2, 3), (10, 11)) for command in checkpoints)

    # Even without decode-range Store, align-state offloads consume current
    # positions, confirmed hashes, and blocks appended during decode.
    request.num_tokens = 16
    request.block_hashes.append(b"decode")
    decode = scheduler.build_step(
        make_output(
            cached=(("request", ([4], [12]), 12),),
            scheduled_tokens={"request": 4},
            checkpoints={"request": [(1, 12, 16)]},
        )
    )
    assert decode.store.source_pending_commands == ()
    checkpoint = decode.store.source_ready_commands[0]
    assert checkpoint.sources == (StateCheckpointSource(1, 12, 16),)
    assert checkpoint.block_ids_by_group == ((1, 2, 3, 4), (10, 11, 12))
    assert checkpoint.block_hashes == tuple(request.block_hashes)


def test_decode_checkpoint_keeps_unchanged_groups_when_only_state_blocks_are_appended() -> None:
    scheduler = SynchronousScheduler(
        make_config(groups=(0, 1), align_groups=frozenset((1,)), load=False, store=True), FakeLookupClient()
    )
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=8, tokens=12)
    try:
        scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3], [10, 11]), 0)
        scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2, 3], [10, 11]), 0),), scheduled_tokens={"request": 8})
        )
        step = scheduler.build_step(
            make_output(
                cached=(("request", ([], [12]), 8),),
                scheduled_tokens={"request": 4},
                checkpoints={"request": [(1, 12, 12)]},
            )
        )
        assert not step.store.source_pending_commands
        (checkpoint,) = step.store.source_ready_commands
        assert checkpoint.block_ids_by_group == ((1, 2, 3), (10, 11, 12))
        assert checkpoint.block_hashes == tuple(request.block_hashes)
        assert checkpoint.sources == (StateCheckpointSource(1, 12, 12),)
        assert checkpoint.store_job_id is not None
        scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({checkpoint.store_job_id: 1}))
    finally:
        scheduler.close()


def test_finished_and_preempted_cancel_unpublished_load() -> None:
    for stage in ("lookup", "sync", "async"):
        for terminated_field in ("finished", "preempted"):
            scheduler_type = AsynchronousScheduler if stage == "async" else SynchronousScheduler
            scheduler = scheduler_type(make_config(), FakeLookupClient(8))
            request = make_request(tokens=9)
            scheduler.get_num_new_matched_tokens(request, 0)
            if stage != "lookup":
                scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 8)

            termination = {terminated_field: ("request",)}
            assert scheduler.build_step(make_output(**termination)).load_commands == (), (stage, terminated_field)
            assert scheduler.build_step(make_output()).load_commands == (), (stage, terminated_field)


@pytest.mark.parametrize("scheduler_type", (SynchronousScheduler, AsynchronousScheduler, SynchronousScheduler))
def test_step_reports_finished_and_preempted_request_union(scheduler_type) -> None:
    scheduler = scheduler_type(make_config(), FakeLookupClient())
    step = scheduler.build_step(make_output(finished=("finished", "both"), preempted=("preempted", "both")))

    assert step.released_request_ids == frozenset(("finished", "preempted", "both"))
    assert scheduler.build_step(make_output()).released_request_ids == frozenset()


@pytest.mark.parametrize("terminated_field", ("finished", "preempted"))
def test_released_request_blocks_remain_leased_until_all_workers_ack(terminated_field: str) -> None:
    scheduler = SynchronousScheduler(make_config(load=False, store=True, workers=2, recycling=True), FakeLookupClient())
    pool = _RefCountBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    blocks = [pool.blocks[1], pool.blocks[2]]
    pool.touch(blocks)
    request = make_request()
    scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)
    step = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1, 2],), 0),), scheduled_tokens={"request": 8})
    )
    command = step.store.commands[0]
    assert [block.ref_cnt for block in blocks] == [2, 2]
    assert scheduler.has_pending_push_work()
    assert not scheduler.has_pending_block_frees(), "Active request references cannot be reclaimed by Store"

    # The upstream Scheduler frees its own references before publishing the release metadata.
    pool.free_blocks(blocks)
    released = scheduler.build_step(make_output(**{terminated_field: ("request",)}))
    assert released.released_request_ids == frozenset(("request",))
    assert released.store.commands == ()
    assert scheduler.has_pending_push_work()
    assert scheduler.has_pending_block_frees()
    assert [block.ref_cnt for block in blocks] == [1, 1]

    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
    assert scheduler.has_pending_push_work()
    assert scheduler.has_pending_block_frees()
    assert [block.ref_cnt for block in blocks] == [1, 1]
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
    assert not scheduler.has_pending_push_work()
    assert not scheduler.has_pending_block_frees()
    assert [block.ref_cnt for block in blocks] == [0, 0]


@pytest.mark.parametrize("entry", ("new", "resumed"))
def test_old_store_release_cannot_release_reallocated_request_sources(entry) -> None:
    scheduler = SynchronousScheduler(make_config(load=False, store=True, workers=2, recycling=True), FakeLookupClient())
    pool = _RefCountBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    request = make_request()
    old_sources = [pool.blocks[index] for index in (1, 2)]
    new_sources = [pool.blocks[index] for index in (4, 5)]
    pool.touch(old_sources)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 0)
    old_step = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1, 2],), 0),), scheduled_tokens={"request": 8})
    )
    old_job = old_step.store.commands[0].store_job_id
    assert old_job is not None
    pool.free_blocks(old_sources)
    scheduler.build_step(make_output(preempted=("request",)))

    pool.touch(new_sources)
    scheduler.confirm_allocation(request, FakeBlocks([4, 5]), 0)
    output = (
        make_output(new=(new_request_data("request", ([4, 5],), 0),), scheduled_tokens={"request": 8})
        if entry == "new"
        else make_output(cached=(("request", ([4, 5],), 0),), resumed=("request",), scheduled_tokens={"request": 8})
    )
    (store,) = scheduler.build_step(output).store.commands
    assert (store.start_token, store.end_token) == (0, 8)
    assert store.block_ids_by_group == ((4, 5),)
    assert store.store_job_id is not None
    assert store.store_job_id != old_job

    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({old_job: 2}))
    assert [block.ref_cnt for block in old_sources] == [0, 0]
    assert [block.ref_cnt for block in new_sources] == [2, 2]
    assert scheduler.has_pending_push_work()
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({store.store_job_id: 2}))
    assert [block.ref_cnt for block in new_sources] == [1, 1]
    assert not scheduler.has_pending_push_work()
    scheduler.close()


def test_pending_block_frees_counts_duplicate_job_refs_not_worker_acknowledgements() -> None:
    scheduler = SynchronousScheduler(make_config(load=False, store=True, workers=2, recycling=True), FakeLookupClient())
    pool = _RefCountBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    block = pool.blocks[1]
    # Two requests may share a completed prefix; each Store job takes just one extra reference.
    pool.touch([block, block])
    for request_id in ("first", "second"):
        scheduler.confirm_allocation(make_request(request_id, prompt_tokens=4), FakeBlocks([1]), 0)
    step = scheduler.build_step(
        make_output(
            new=(new_request_data("first", ([1],), 0), new_request_data("second", ([1],), 0)),
            scheduled_tokens={"first": 4, "second": 4},
        )
    )
    first, second = step.store.commands
    assert block.ref_cnt == 4
    assert not scheduler.has_pending_block_frees()

    pool.free_blocks([block, block])
    scheduler.build_step(make_output(finished=("first", "second")))
    assert block.ref_cnt == 2
    assert scheduler.has_pending_block_frees(), "Both job references must be counted for the shared source"

    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({first.store_job_id: 1}))
    assert block.ref_cnt == 2
    assert scheduler.has_pending_block_frees()
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({first.store_job_id: 1}))
    assert block.ref_cnt == 1
    assert scheduler.has_pending_block_frees()
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({second.store_job_id: 1}))
    assert block.ref_cnt == 1
    assert scheduler.has_pending_block_frees()
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({second.store_job_id: 1}))
    assert block.ref_cnt == 0
    assert not scheduler.has_pending_block_frees()
    assert not scheduler.has_pending_push_work()


def test_pending_block_frees_detects_masked_block_without_freeing_active_blocks() -> None:
    scheduler = SynchronousScheduler(make_config(load=False, store=True, recycling=True), FakeLookupClient())
    pool = _RefCountBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    active, masked = pool.blocks[1], pool.blocks[2]
    pool.touch([active, masked])
    scheduler.confirm_allocation(make_request(), FakeBlocks([1, 2]), 0)
    step = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1, 2],), 0),), scheduled_tokens={"request": 8})
    )
    assert not scheduler.has_pending_block_frees()

    pool.free_blocks([masked])
    assert scheduler.has_pending_block_frees()
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({step.store.commands[0].store_job_id: 1}))
    assert active.ref_cnt == 1
    assert masked.ref_cnt == 0
    assert not scheduler.has_pending_block_frees()


@pytest.mark.skipif(not hasattr(BlockPool, "free_blocks"), reason="Allocator reuse requires the real vLLM BlockPool")
def test_running_window_sources_are_not_reused_until_every_worker_releases() -> None:
    pool = BlockPool(num_gpu_blocks=6, enable_caching=False, hash_block_size=4)
    sources = pool.get_new_blocks(2)
    source_ids = [block.block_id for block in sources]
    config = make_config(load=False, store=True, workers=2, recycling=True)
    scheduler = SynchronousScheduler(config, FakeLookupClient())
    scheduler.bind_gpu_block_pool(pool)
    scheduler.confirm_allocation(make_request(), FakeBlocks(source_ids), 0)
    step = scheduler.build_step(
        make_output(new=(new_request_data("request", (source_ids,), 0),), scheduled_tokens={"request": 8})
    )
    command = step.store.commands[0]

    manager = SlidingWindowManager(
        config.group_specs[0],
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=4,
    )
    manager.req_to_blocks["request"] = list(sources)
    manager.remove_skipped_blocks("request", 8)
    assert scheduler.build_step(make_output()).released_request_ids == frozenset()
    assert scheduler.has_pending_block_frees()
    available = pool.get_new_blocks(pool.get_num_free_blocks())
    assert set(source_ids).isdisjoint(block.block_id for block in available)
    assert pool.get_num_free_blocks() == 0

    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
    assert pool.get_num_free_blocks() == 0
    assert scheduler.has_pending_block_frees()
    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
    assert pool.get_num_free_blocks() == 1
    assert not scheduler.has_pending_block_frees()
    assert pool.get_new_blocks(1)[0].block_id == source_ids[0]
    assert sources[1].ref_cnt == 1, "The running request still owns its active window"


def test_finished_tail_lease_survives_cleanup_until_every_worker_releases() -> None:
    scheduler = SynchronousScheduler(
        make_config(groups=(0, 1), align_groups=frozenset((1,)), store=True, workers=2),
        FakeLookupClient(8),
    )
    pool = FakeBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    request = make_request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (7, False)
    scheduler.confirm_allocation(request, FakeBlocks([1, 2], [10, 11]), 7)
    scheduler.build_step(
        make_output(new=(new_request_data("request", ([1, 2], [10, 11]), 7),), scheduled_tokens={"request": 1})
    )

    assert not scheduler.register_finished_partial_tail(request, ([1, 2], [10, 11]), [(1, 11, 8)])
    assert scheduler.has_pending_push_work()
    assert pool.touched == [(11,)]

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
    assert pool.freed == [(11,)]


@pytest.mark.parametrize("eagle,expected_external_tokens", ((True, 0), (False, 3)))
def test_store_skip_survives_safe_frontier_trimming(eagle, expected_external_tokens) -> None:
    scheduler = SynchronousScheduler(make_config(store=True, save_decode=True, eagle=eagle), FakeLookupClient(4))
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
    decode = scheduler.build_step(make_output(cached=(("request", ([2],), 4),), scheduled_tokens={"request": 4}))
    assert [(command.start_token, command.end_token) for command in decode.store.commands] == [(4, 8)]


def test_zero_external_match_never_advertises_async_loading() -> None:
    scheduler = AsynchronousScheduler(make_config(store=True, eagle=True), FakeLookupClient(4))
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
    scheduler = SynchronousScheduler(make_config(eagle=eagle), FakeLookupClient(readable_end))
    request = make_request(prompt_tokens=prompt_tokens)
    try:
        assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (safe_end - local_tokens, False)
        scheduler.confirm_allocation(request, FakeBlocks([1, 2, 3]), safe_end - local_tokens)
        step = scheduler.build_step(
            make_output(new=(new_request_data("request", ([1, 2, 3],), safe_end),), scheduled_tokens={"request": 0})
        )
        assert [(command.start_token, command.end_token) for command in step.load_commands] == (
            [(0, load_end)] if load_end else []
        )
    finally:
        scheduler.close()


@pytest.mark.parametrize(
    "prompt_tokens,readable_end,attention_tail,eagle,private,safe_end,load_end",
    (
        (64, 32, 40, False, False, 32, 32),
        (128, 128, 128, False, False, 127, 128),
        (128, 128, 128, True, False, 64, 64),
        (128, 128, 128, False, True, 64, 64),
    ),
)
def test_load_keeps_lookup_tail_identity_only_when_physical_frontier_is_unchanged(
    prompt_tokens, readable_end, attention_tail, eagle, private, safe_end, load_end
) -> None:
    tail_keys = (TailKeyBoundary(0, attention_tail), TailKeyBoundary(1, readable_end))
    scheduler = SynchronousScheduler(
        replace(make_config(granularity=64, groups=(0, 1), eagle=eagle, private=private), hash_block_size=8),
        FakeLookupClient(readable_end, tail_key_boundaries=tail_keys),
    )
    request = make_request(prompt_tokens=prompt_tokens, hashes=prompt_tokens // 8)
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (safe_end, False)
        scheduler.confirm_allocation(request, FakeBlocks([5, 6], [7, 8, 9, 10]), safe_end)
        step = scheduler.build_step(
            make_output(
                new=(new_request_data("request", ([5, 6], [7, 8, 9, 10]), safe_end),), scheduled_tokens={"request": 0}
            )
        )
        (command,) = step.load_commands
        assert (command.start_token, command.end_token) == (0, load_end)
        assert command.tail_key_boundaries == (tail_keys if load_end == readable_end else ())
    finally:
        scheduler.close()


@pytest.mark.parametrize("readable_end,eagle,expected_safe_end", ((12, True, 8), (12, False, 8), (8, False, 8)))
def test_private_state_recomputes_safe_frontier_without_erasing_store_skip(
    readable_end,
    eagle,
    expected_safe_end,
) -> None:
    scheduler = SynchronousScheduler(
        make_config(store=True, private=True, eagle=eagle),
        FakeLookupClient(readable_end),
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
    assert [(command.start_token, command.end_token) for command in running.store.commands] == [(readable_end, 16)]


def test_eagle_hit_is_trimmed_once_by_scheduler() -> None:
    group = replace(make_topology().groups[0], is_eagle_group=True)
    coordinator = Coordinator((group,), 4, 4, 64, use_eagle=True)
    block_hashes = (b"a", b"b")
    observation = ((np.asarray((0, 4)), np.asarray((4, 4)), block_hashes), (True, True))

    assert coordinator.resolve_available_end(0, 8, block_hashes, (observation,)).end_token == 8

    scheduler = SynchronousScheduler(make_config(eagle=True), FakeLookupClient(8))
    assert scheduler.get_num_new_matched_tokens(make_request(prompt_tokens=8), 0) == (4, False)


def test_scheduler_close_rejects_active_store_source_leases() -> None:
    lookup = FakeLookupClient()
    scheduler = SynchronousScheduler(make_config(load=False, store=True, recycling=True), lookup)
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=4)
    scheduler.confirm_allocation(request, FakeBlocks([1]), 0)
    step = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1],), 0),), scheduled_tokens={"request": 4})
    )
    command = step.store.commands[0]

    with pytest.raises(RuntimeError, match=r"source lease job ids=\(0,\)"):
        scheduler.close()
    assert lookup.closed

    scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
    scheduler.close()


def test_private_align_checkpoints_do_not_publish_decode_stores():
    config = make_config(groups=(0,), align_groups=frozenset((1,)), private=True, load=False, store=True)
    scheduler = SynchronousScheduler(config, FakeLookupClient())
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    request = make_request(prompt_tokens=4)
    scheduler.confirm_allocation(request, FakeBlocks([1], [10]), 0)
    scheduler.build_step(
        make_output(new=(new_request_data("request", ([1], [10]), 0),), scheduled_tokens={"request": 4})
    )
    request.num_tokens = 8
    request.block_hashes.append(b"decode")
    step = scheduler.build_step(
        make_output(
            cached=(("request", ([2], [11]), 4),),
            scheduled_tokens={"request": 4},
            checkpoints={"request": [(1, 11, 8)]},
        )
    )
    assert step.store.commands == ()
    assert not step.load_commands
    assert not scheduler.has_pending_push_work()
    scheduler.close()
