"""Preparation follows object identity -> ownership -> admission -> execution views."""

from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.gva_fixtures import FakeGVABackend
from tests.ut.distributed.ascend_store.v1.helpers import FakeBackend, make_coordinator, make_topology
from tests.ut.distributed.ascend_store.v1.worker.fixtures import make_worker as make_bulk
from tests.ut.distributed.ascend_store.v1.worker.test_layerwise import make_worker as make_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import local_addresses
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import LoadCommand, RangeStoreCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.data.keys import KeyIndex
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.service.lookup import (
    LookupRequest,
    LookupService,
)


@pytest.mark.parametrize("hashes", ((b"\x0c", "0c", b"\x0c"), ("0c", b"\x0c", "0c")))
def test_equal_hash_representations_construct_one_wire_key(hashes):
    index = KeyIndex()
    assert [index.identify(("prefix@", "@suffix"), value) for value in hashes] == [0, 0, 0]
    assert index.keys == ["prefix@0c@suffix"]


@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize("boundary", (0, 2, 4))
def test_empty_ranges_have_no_selected_keys_or_blocks(operation, boundary):
    worker, _, _ = make_bulk()
    service = getattr(worker, operation)
    command = (
        LoadCommand("r", boundary, boundary, ((1,),), (b"a",))
        if operation == "load"
        else RangeStoreCommand("r", boundary, boundary, ((1,),), (b"a",), 4, 17)
    )
    try:
        selection = service._prepare((command,))
        assert not selection.keys
        assert not (selection.local if operation == "load" else selection.sources)
    finally:
        worker.close()


@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize("start,end", ((-1, 2), (3, 2)))
def test_invalid_ranges_are_rejected(operation, start, end):
    worker, _, _ = make_bulk()
    service = getattr(worker, operation)
    command = (
        LoadCommand("r", start, end, ((1,),), (b"a",))
        if operation == "load"
        else RangeStoreCommand("r", start, end, ((1,),), (b"a",), 4, 17)
    )
    try:
        with pytest.raises(ValueError, match="range requires"):
            service._prepare((command,))
    finally:
        worker.close()


def test_lookup_without_candidates_returns_a_miss_without_backend_io(monkeypatch):
    topology = make_topology()
    backend, coordinator = FakeBackend(), make_coordinator(topology)
    monkeypatch.setattr(coordinator, "select_for_lookup", lambda _end: ((),))
    result = LookupService(topology, backend, coordinator)(
        LookupRequest(end_token=4, transfer_group_ids=(0,), block_hashes=(b"a",), local_available_tokens=0)
    )
    assert result.available_end_token == 0 and not backend.calls


@pytest.mark.parametrize("operation", ("load", "store"))
def test_open_sizes_only_visit_unique_active_objects_and_keep_ownership(operation, monkeypatch):
    worker, _, backend = make_layerwise(requires_exists=True)
    if operation == "store":
        monkeypatch.setattr(backend, "exists", lambda _keys: (True, False))
        selection = worker.store._prepare((RangeStoreCommand("r", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),))
    else:
        selection = worker.load._prepare(
            (LoadCommand("a", 0, 4, ((1,),), (b"a",)), LoadCommand("b", 0, 4, ((2,),), (b"a",)))
        )
    opening = Mock(side_effect=lambda keys, _sizes: (0,) * len(keys))
    monkeypatch.setattr(worker.transfer.backend_sessions, f"start_{operation}", opening)
    try:
        plan, codes, opened = worker.transfer.open(selection, to_remote=operation == "store")
        assert len(opening.call_args.args[0]) == len(opening.call_args.args[1]) == 1
        assert opened == {0} and plan.layers and not any(codes)
        expected = [1, 2] if operation == "load" else [2]
        assert all(ids.tolist() == expected for ids in plan.block_ids.values())
        if operation == "load":
            (local,) = selection.local.values()
            assert local.block_ids == [1, 2] and local.request_indices == [0, 1]
        else:
            assert len(selection.sources) == 1 and selection.sources[0][1] == 2
    finally:
        worker.close()


@pytest.mark.parametrize("mode", ("key_range", "gva"))
@pytest.mark.parametrize("operation", ("load", "store"))
def test_failed_open_preserves_failure_codes_and_has_no_transfer_targets(operation, mode, monkeypatch):
    worker, _, _ = make_layerwise(backend=FakeGVABackend() if mode == "gva" else None)
    selection = (
        worker.load._prepare((LoadCommand("r", 0, 8, ((1, 2),), (b"a", b"b")),))
        if operation == "load"
        else worker.store._prepare((RangeStoreCommand("r", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),))
    )
    monkeypatch.setattr(worker.transfer.backend_sessions, f"start_{operation}", lambda keys, _sizes: (-7,) * len(keys))
    try:
        plan, codes, opened = worker.transfer.open(selection, to_remote=operation == "store")
        assert codes == [-7, -7] and not opened
        assert not plan.layers and not plan.block_ids and not plan.result_slots and not plan.range_indices
    finally:
        worker.close()


def test_layer_parameter_order_uses_bound_routes_even_for_cross_partition_aliases(monkeypatch):
    worker, _, _ = make_layerwise(topology=make_topology(group_ids=(0, 1)))
    first, second = worker.context.partition_order
    worker.context.key_parts[1][second] = worker.context.key_parts[0][first]
    # A's first partition does not participate in this layer. In the second partition,
    # B precedes A; the native Key order must still keep each Key's own addresses.
    commands = (
        LoadCommand("a-first", 0, 4, ((1,), ()), (b"a",)),
        LoadCommand("b", 0, 4, ((), (2,)), (b"b",)),
        LoadCommand("a-second", 0, 4, ((), (3,)), (b"a",)),
    )
    selection = worker.load._prepare(commands)
    layer = next(iter(worker.transfer._geometry))
    monkeypatch.setattr(worker.transfer, "_memberships", {(second,): [layer]})
    monkeypatch.setattr(worker.transfer.backend_sessions, "start_load", lambda keys, _sizes: (0,) * len(keys))
    copy = Mock(return_value=(0, 0))
    monkeypatch.setattr(worker.transfer, "_load_ranges", copy)
    try:
        plan, _, _ = worker.transfer.open(selection, to_remote=False)
        worker.transfer.copy(plan, layer, to_remote=False)
        keys, addresses, _, _ = copy.call_args.args
        span = worker.transfer._geometry[layer][second][0]
        expected = local_addresses(worker.context.local[second[0]][second[1]], [3, 2], span).tolist()
        assert keys == selection.keys and addresses == expected
    finally:
        worker.close()
