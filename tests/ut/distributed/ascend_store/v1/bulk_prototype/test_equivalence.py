"""Compare native parameters and public results against the unchanged Bulk Worker."""

from __future__ import annotations

import pytest

from tests.ut.distributed.ascend_store.v1.bulk_prototype.fixtures import make_worker
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk.fixtures import make_align_state_topology
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype import LookupQuery
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import Resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommandBatch,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.results import LoadFailureLocation, LoadResult
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import BulkWorker as ReferenceWorker


def completion_values(completions, *, prototype=False):
    if prototype:
        return tuple(
            (
                result.request_id,
                result.store_job_id,
                result.succeeded,
                result.source_release_confirmed,
                tuple(zip(result.keys, result.result_codes, strict=True)),
            )
            for result in completions
        )
    return tuple(
        (
            completion.request_id,
            completion.store_job_id,
            completion.evidence.succeeded,
            completion.evidence.source_release_confirmed,
            tuple((item.source.key, item.result_code) for item in completion.evidence.transfer_evidence),
        )
        for completion in completions
    )


def observe(worker, backend, load, stores, query, *, prototype):
    if prototype:
        query = LookupQuery(query.end_token, query.transfer_group_ids, query.block_hashes, query.start_token)
    lookup = worker.lookup(query)
    worker.begin_step(KVTransferStep(load, StoreCommandBatch(stores)))
    worker.start_load()
    executor = worker.transfer.executor._write if prototype else worker._load_executor
    executor.submit(lambda: None).result(timeout=2)
    loaded = worker.collect_load_result()
    if not prototype:
        loaded = LoadResult(
            loaded.completed_request_ids,
            loaded.failed_request_ids,
            loaded.failed_block_ids,
            tuple(LoadFailureLocation(item.group_id, item.block_id) for item in loaded.failed_locations),
        )
    worker.finish_step()
    stored = worker.fence_previous_store()
    released = worker.take_released_store_job_ids()
    worker.end_step()
    # Independent queues may prepare Store before or after Load. Preserve every
    # parameter and the order within each operation, without imposing cross-queue order.
    calls = tuple(
        sorted((call for call in backend.calls if call[0] in ("exists", "get", "put")), key=lambda call: call[0])
    )
    return lookup, loaded, completion_values(stored, prototype=prototype), released, calls


@pytest.mark.parametrize(
    "topology",
    (
        make_topology(),
        make_topology(group_ids=(1, 3), physical_layers=(2, 5)),
        make_topology(
            physical_layers=(0,), tp_mismatch=True, tp_size=2, tp_rank=1, key_rank_count=4, key_slices_per_rank=2
        ),
        make_topology(
            physical_layers=(0,), tp_mismatch=True, tp_size=4, tp_rank=2, key_rank_count=4, key_slices_per_rank=1
        ),
        make_topology(target_layer_counts=(1, 1)),
        make_topology(tp_size=2, tp_rank=1, dcp_size=2, dcp_rank=1),
        make_topology(tp_size=2, tp_rank=1, put_step=2, pcp_size=2, pcp_rank=1),
    ),
    ids=("ordinary", "sparse_groups", "tp_split", "tp_single", "consumer_pp", "dcp", "pcp_writers"),
)
@pytest.mark.parametrize("observation", ("unconditional", "prefix", "partition"))
@pytest.mark.parametrize("native_failure", (False, True))
def test_native_arguments_and_failure_ownership_match_reference(topology, observation, native_failure):
    requires_exists = observation != "unconditional"
    prototype, proto_resources, proto_backend = make_worker(topology=topology, requires_exists=requires_exists)
    caches = proto_resources.kv_caches
    reference_backend = FakeBackend()
    reference_resources = Resources(
        reference_backend, make_backend_spec(requires_exists_before_put=requires_exists), topology.transfer_groups
    )
    reference = ReferenceWorker(
        topology,
        reference_resources,
        coordinator=make_coordinator(topology, 256),
        load_async=True,
        source_ready_event_factory=FakeEvent,
    )
    reference.bind_kv_caches(caches)
    for backend in (proto_backend, reference_backend):

        def exists(keys, owner=backend):
            owner.calls.append(("exists", tuple(keys)))
            return tuple(
                key.endswith("@61") and (observation != "partition" or "@head_or_tp_rank:2@" in key) for key in keys
            )

        def get(keys, addresses, sizes, owner=backend):
            owner.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
            return [-7 if native_failure and index == len(keys) - 1 else 0 for index in range(len(keys))]

        def put(keys, addresses, sizes, owner=backend):
            owner.calls.append(("put", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
            return [-8 if native_failure and index == len(keys) - 1 else 0 for index in range(len(keys))]

        backend.exists, backend.get, backend.put = exists, get, put
    blocks = tuple((1, 2) for _group in topology.groups)
    duplicate_blocks = tuple((3, 4) for _group in topology.groups)
    load = (LoadCommand("load", 0, 8, blocks, (b"a", b"b")),)
    stores = (
        RangeStoreCommand("first", 0, 8, blocks, (b"a", b"b"), 8, 17),
        RangeStoreCommand("duplicate", 0, 8, duplicate_blocks, (b"a", b"b"), 8, 18),
    )
    query = LookupRequest(0, 8, topology.config.transfer_group_ids, (b"a", b"b"))
    try:
        actual = observe(prototype, proto_backend, load, stores, query, prototype=True)
        expected = observe(reference, reference_backend, load, stores, query, prototype=False)
        assert actual == expected
    finally:
        prototype.close()
        reference.close()


@pytest.mark.parametrize("attention_block_size,boundary", ((16, 24), (64, 40)))
@pytest.mark.parametrize("tp_size", (1, 2))
def test_partial_state_checkpoint_companions_and_ready_submission_match_reference(
    attention_block_size, boundary, tp_size
):
    topology = make_align_state_topology(
        attention_block_size=attention_block_size, state_block_size=32, hash_block_size=8, tp_size=tp_size
    )
    prototype, proto_resources, proto_backend = make_worker(topology=topology)
    reference_backend = FakeBackend()
    reference_resources = Resources(reference_backend, make_backend_spec(), topology.transfer_groups)
    reference = ReferenceWorker(
        topology,
        reference_resources,
        coordinator=make_coordinator(topology),
        load_async=True,
        source_ready_event_factory=FakeEvent,
    )
    reference.bind_kv_caches(proto_resources.kv_caches)
    hashes = tuple(bytes([index]) for index in range(1, 9))
    command = CheckpointStoreCommand(
        "checkpoint", ((1, 2, 3, 4), ()), hashes, 0, (StateCheckpointSource(1, 7, boundary),), 17
    )
    observations = []
    try:
        for worker, backend in ((prototype, proto_backend), (reference, reference_backend)):
            worker.begin_step(KVTransferStep(store=StoreCommandBatch(source_ready_commands=(command,))))
            worker.finish_step()
            completions = worker.fence_previous_store()
            observations.append(
                (
                    completion_values(completions, prototype=worker is prototype),
                    worker.take_released_store_job_ids(),
                    tuple(call for call in backend.calls if call[0] == "put"),
                )
            )
            worker.end_step()
        assert observations[0] == observations[1]
    finally:
        prototype.close()
        reference.close()


@pytest.mark.parametrize("suffix_only", (False, True))
def test_multiple_load_requests_keep_group_block_and_return_code_ownership(suffix_only):
    topology = make_topology(group_ids=(1, 3), physical_layers=(0,))
    prototype, resources, proto_backend = make_worker(topology=topology)
    reference_backend = FakeBackend()
    reference = ReferenceWorker(
        topology,
        Resources(reference_backend, make_backend_spec(), topology.transfer_groups),
        coordinator=make_coordinator(topology, 256),
        load_async=True,
        source_ready_event_factory=FakeEvent,
    )
    reference.bind_kv_caches(resources.kv_caches)
    first_blocks = (3,) if suffix_only else (1, 2, 3)
    load = (
        LoadCommand("first", 4, 12, ((), first_blocks, (), first_blocks), (b"a", b"b", b"c")),
        LoadCommand("second", 0, 8, ((), (4, 5), (), (4, 5)), (b"d", b"e")),
        LoadCommand("empty", 0, 4, ((), (-1,), (), (-1,)), (b"f",)),
    )
    query = LookupRequest(0, 0, (1, 3), ())
    for backend in (proto_backend, reference_backend):

        def get(keys, addresses, sizes, owner=backend):
            owner.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
            return [-7 if key.endswith("@63") or key.endswith("@65") else 0 for key in keys]

        backend.get = get
    try:
        actual = observe(prototype, proto_backend, load, (), query, prototype=True)
        expected = observe(reference, reference_backend, load, (), query, prototype=False)
        assert actual == expected
        assert actual[1].completed_request_ids == {"first", "second", "empty"}
        assert actual[1].failed_request_ids == {"first", "second"}
        assert not actual[1].failed_block_ids
        assert {(location.group_id, location.block_id) for location in actual[1].failed_locations}.issuperset(
            {(1, 3), (3, 3), (1, 5), (3, 5)}
        )
    finally:
        prototype.close()
        reference.close()


@pytest.mark.parametrize("attention_block_size,boundary", ((16, 24), (64, 40)))
def test_partial_tail_load_uses_exact_checkpoint_hash_and_full_local_payload(attention_block_size, boundary):
    topology = make_align_state_topology(
        attention_block_size=attention_block_size, state_block_size=32, hash_block_size=8, tp_size=1
    )
    prototype, resources, proto_backend = make_worker(topology=topology)
    reference_backend = FakeBackend()
    reference = ReferenceWorker(
        topology,
        Resources(reference_backend, make_backend_spec(), topology.transfer_groups),
        coordinator=make_coordinator(topology, 256),
        load_async=True,
        source_ready_event_factory=FakeEvent,
    )
    reference.bind_kv_caches(resources.kv_caches)
    hashes = tuple(bytes([index]) for index in range(1, 9))
    tails = (TailKeyBoundary(0, boundary), TailKeyBoundary(1, boundary))
    command = LoadCommand("tail", 0, boundary, ((1, 2, 3, 4), (0, 5)), hashes, tails)
    query = LookupRequest(0, 0, (0, 1), ())
    try:
        actual = observe(prototype, proto_backend, (command,), (), query, prototype=True)
        expected = observe(reference, reference_backend, (command,), (), query, prototype=False)
        assert actual == expected
        assert actual[1].completed_request_ids == {"tail"}
        assert not actual[1].failed_locations
        copied = next(call for call in actual[-1] if call[0] == "get")
        assert any(key.endswith(f"@{hashes[boundary // 8 - 1].hex()}") for key in copied[1])
    finally:
        prototype.close()
        reference.close()


@pytest.mark.parametrize("tp_rank", (0, 1))
@pytest.mark.parametrize("missing_block", (False, True))
def test_shared_pairs_preserve_operation_masks_and_writer_selection_order(tp_rank, missing_block, monkeypatch):
    topology = make_topology(tp_size=2, tp_rank=tp_rank, put_step=2)
    prototype, resources, proto_backend = make_worker(topology=topology)
    coordinator = prototype.load.coordinator
    monkeypatch.setattr(coordinator, "select_for_load", lambda *_args: ((False, True, False, True),))
    monkeypatch.setattr(coordinator, "select_for_store", lambda *_args: ((False, True, True),))
    reference_backend = FakeBackend()
    reference = ReferenceWorker(
        topology,
        Resources(reference_backend, make_backend_spec(), topology.transfer_groups),
        coordinator=coordinator,
        load_async=True,
        source_ready_event_factory=FakeEvent,
    )
    reference.bind_kv_caches(resources.kv_caches)
    hashes = (b"a", b"b", b"c", b"d")
    block_ids = ((1, 2, -1 if missing_block else 3, 4),)
    load = (LoadCommand("load", 4, 16, block_ids, hashes),)
    stores = (RangeStoreCommand("store", 4, 16, block_ids, hashes, 16, 17),)
    query = LookupRequest(0, 0, (0,), ())
    try:
        actual = observe(prototype, proto_backend, load, stores, query, prototype=True)
        expected = observe(reference, reference_backend, load, stores, query, prototype=False)
        assert actual == expected
        copied = next(call for call in actual[-1] if call[0] == "get")
        assert [key.rsplit("@", 1)[-1] for key in copied[1]] == ["62", "64"]
        written = [key.rsplit("@", 1)[-1] for call in actual[-1] if call[0] == "put" for key in call[1]]
        assert written == ((["64"] if tp_rank == 0 else []) if missing_block else ["63" if tp_rank == 0 else "64"])
    finally:
        prototype.close()
        reference.close()
