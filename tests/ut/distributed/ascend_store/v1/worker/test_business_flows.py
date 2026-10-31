"""Operation-specific preparation preserves recovery, destination ownership and source safety."""

from itertools import product
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.gva_fixtures import FakeGVABackend
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    make_block_hash_mappings,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.fixtures import make_worker as make_bulk
from tests.ut.distributed.ascend_store.v1.worker.test_layerwise import layer_names
from tests.ut.distributed.ascend_store.v1.worker.test_layerwise import make_worker as make_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import GVARegion
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.coordinator import (
    ContiguousObservation,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.data.batch import LoadBatch
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.service.lookup import (
    LookupRequest,
    LookupService,
)


def test_store_reads_only_the_missing_middle_block():
    worker, resources, backend = make_bulk(requires_exists=True)
    commands = (RangeStoreCommand("r", 0, 12, ((1, 2, 3),), (b"a", b"b", b"c"), 12, 17),)
    backend.presence = [True, False, True]
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
        worker.finish_step()
        (result,) = worker.fence_previous_store()
        (copy,) = [call for call in backend.calls if call[0] == "put"]
        assert len(copy[1]) == 1 and copy[1][0].endswith("@62")
        assert copy[2] == (tuple(cache[2].data_ptr() for cache in resources.kv_caches.values()),)
        assert result.request_id == "r" and result.keys == copy[1] and result.succeeded
        assert result.source_release_confirmed and worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        worker.close()


@pytest.mark.parametrize("mode", ("bulk", "key_range", "gva"))
def test_compact_store_slots_keep_failure_owner_after_exists(mode, monkeypatch):
    worker, _, backend = (
        make_bulk(requires_exists=True)
        if mode == "bulk"
        else make_layerwise(backend=FakeGVABackend() if mode == "gva" else None, requires_exists=True)
    )
    monkeypatch.setattr(worker.transfer.backend, "exists", lambda keys: tuple(key.endswith("@61") for key in keys))
    if mode == "bulk":
        backend.put_result = [-7]
    elif mode == "key_range":
        backend.store_session_copy_result = [-7]
    else:
        backend.native_store.copy_result = -7
    commands = (
        RangeStoreCommand("present", 0, 4, ((1,),), (b"a",), 4, 17),
        RangeStoreCommand("missing", 0, 4, ((2,),), (b"b",), 4, 18),
        RangeStoreCommand("duplicate", 0, 4, ((3,),), (b"b",), 4, 19),
    )
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
        if mode != "bulk":
            for name in layer_names(worker):
                worker.save_layer(name)
        batch = worker.store._pending[-1]
        worker.finish_step()
        present, missing, duplicate = batch.completion.result(timeout=2)
        worker.fence_previous_store()
        assert present.succeeded and not present.keys
        assert not missing.succeeded and missing.result_codes == (-7,) and missing.keys[0].endswith("@62")
        assert duplicate.succeeded and not duplicate.keys
        assert all(result.source_release_confirmed for result in (present, missing, duplicate))
        assert worker.take_released_store_job_ids() == {17, 18, 19}
        worker.end_step()
    finally:
        if mode == "gva":
            backend.revoke_gva = lambda keys: (0,) * len(keys)
        worker.close()


@pytest.mark.parametrize("mode", ("key_range", "gva"))
def test_partial_open_scatter_preserves_shared_targets_and_only_projects_successes(mode, monkeypatch):
    worker, _, _ = make_layerwise(backend=FakeGVABackend() if mode == "gva" else None)
    commands = (
        LoadCommand("first", 0, 8, ((1, 2),), (b"a", b"b")),
        LoadCommand("second", 0, 4, ((3,),), (b"b",)),
    )
    selection = worker.load._prepare(commands)
    monkeypatch.setattr(worker.transfer.backend_sessions, "start_load", lambda *_args: (0, -7))
    if mode == "gva":
        monkeypatch.setattr(worker.transfer, "_load_regions", lambda keys, sizes: (GVARegion(1000, sizes[0]),))
    try:
        plan, codes, opened = worker.transfer.open(selection, to_remote=False)
        assert codes == [0, -7, -7] and opened == {0}
        assert all(ids.tolist() == [1] for ids in plan.block_ids.values())
        assert all(tuple(plan.result_indices(layer)) == (0,) for layer in plan.layers)
        batch = LoadBatch(("first", "second"), selection, codes=codes)
        worker.load._check_ready(batch)
        result = worker.load._complete(batch)
        assert result.failed_block_ids == {2, 3}
        assert result.completed_request_ids == {"first", "second"}
    finally:
        worker.close()


def test_load_rejects_alias_with_inconsistent_geometry_before_open(monkeypatch):
    worker, _, _ = make_layerwise(topology=make_topology(group_ids=(0, 1)))
    first, second = worker.context.partition_order
    worker.context.key_parts[1][second] = worker.context.key_parts[0][first]
    worker.context.object_sizes[second] = worker.context.object_sizes[first] + 1
    opening = Mock(side_effect=AssertionError("inconsistent identity must not reach Backend"))
    monkeypatch.setattr(worker.transfer.backend_sessions, "start_load", opening)
    try:
        with pytest.raises(ValueError, match="inconsistent object sizes"):
            worker.load._prepare((LoadCommand("r", 0, 4, ((1,), (2,)), (b"a",)),))
        opening.assert_not_called()
    finally:
        worker.close()


@pytest.mark.parametrize("groups", ((0,), (0, 1)))
def test_compact_lookup_matches_explicit_observations_for_local_prefixes_and_holes(groups, monkeypatch):
    topology = make_topology(group_ids=groups)
    backend, coordinator = FakeBackend(), make_coordinator(topology)
    lookup = LookupService(topology, backend, coordinator, block_hash_mappings=make_block_hash_mappings(topology))
    original = coordinator.resolve_available_end
    seen = []

    def compare(local, end, hashes, observations):
        compact = original(local, end, hashes, observations)
        explicit = []
        for observation in observations:
            if isinstance(observation, ContiguousObservation):
                seen.append(observation)
                start, size, hits = observation.start_token, observation.block_size, observation.available
                starts = tuple(range(start, start + len(hits) * size, size))
                chunks = hashes[start // size : start // size + len(hits)]
                explicit.append(((starts, (size,) * len(hits), chunks), hits))
            else:
                explicit.append(observation)
        assert compact == original(local, end, hashes, explicit)
        return compact

    coordinator.resolve_available_end = compare
    hashes = (b"a", b"b", b"c")
    for end in range(1, 14):
        for local in range(end + 1):
            for hits in product((False, True), repeat=3):
                monkeypatch.setattr(
                    backend,
                    "exists",
                    lambda keys, hits=hits: tuple(hits[int(key.rsplit("@", 1)[1], 16) - ord("a")] for key in keys),
                )
                recovered = lookup(
                    LookupRequest(
                        end_token=end, transfer_group_ids=groups, block_hashes=hashes, local_available_tokens=local
                    )
                ).available_end_token
                expected = min(local // 4, min(end, len(hashes) * 4) // 4)
                while expected < min(end, len(hashes) * 4) // 4 and hits[expected]:
                    expected += 1
                assert recovered == expected * 4
    assert seen


def test_lookup_deduplicates_cross_family_aliases_and_reduces_all_partitions(monkeypatch):
    topology = make_topology(pp_size=2)
    backend, coordinator = FakeBackend(), make_coordinator(topology)
    lookup = LookupService(topology, backend, coordinator, block_hash_mappings=make_block_hash_mappings(topology))
    # First candidate is shared across families; the second family's second candidate is absent.
    lookup._keys[0] = (("x", ""), ("", "x"))
    calls = []

    def exists(keys):
        calls.append(tuple(keys))
        return tuple(key != "abx" for key in keys)

    monkeypatch.setattr(backend, "exists", exists)
    result = lookup(
        LookupRequest(end_token=12, transfer_group_ids=(0,), block_hashes=("x", "ab", "x"), local_available_tokens=0)
    )
    assert calls == [("xx", "xab", "abx")]
    assert result.available_end_token == 4


@pytest.mark.parametrize("address", (1000, (1 << 63) + 3, (1 << 64) - 16))
@pytest.mark.parametrize("keep_regions", (False, True))
def test_gva_address_preparation_preserves_integer_precision_and_publication_ownership(
    address, keep_regions, monkeypatch
):
    worker, _, _ = make_layerwise(backend=FakeGVABackend())
    commands = (RangeStoreCommand("r", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),)
    selection = worker.store._candidates(commands)
    sizes = tuple(worker.context.object_sizes[partition] for partition, _, _ in selection.sources)
    regions = tuple(GVARegion(address + index, size) for index, size in enumerate(sizes))
    monkeypatch.setattr(worker.transfer.backend_sessions, "start_store", lambda *_args: (0, 0))
    resolve = Mock(return_value=regions)
    monkeypatch.setattr(worker.transfer, "_store_regions", resolve)
    calls = []

    def copy(remote, local, lengths):
        calls.append(remote)
        return 0

    monkeypatch.setattr(worker.transfer, "_store_gva", copy)
    try:
        plan, codes, opened = worker.transfer.open(selection, to_remote=True, keep_regions=keep_regions)
        resolve.assert_called_once_with(tuple(selection.keys), sizes)
        assert codes == [0, 0] and opened == {0, 1}
        assert plan.regions == (dict(zip(selection.keys, regions)) if keep_regions else {})
        for layer in plan.layers:
            worker.transfer._copy_gva(plan, layer, to_remote=True)
            expected: list[int] = []
            for coordinate in plan.layers[layer]:
                offsets = worker.transfer._geometry[layer][coordinate][2].ravel().tolist()
                expected.extend(
                    regions[index].address + offset for index in plan.result_slots[coordinate] for offset in offsets
                )
            assert calls[-1] == expected
            assert all(type(value) is int for value in calls[-1])
    finally:
        worker.close()
