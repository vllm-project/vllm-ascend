"""Identity reuse, positional routing and terminal shortcuts preserve business outcomes."""

from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.bulk_prototype.fixtures import make_worker as make_bulk
from tests.ut.distributed.ascend_store.v1.bulk_prototype.test_layerwise import layer_names
from tests.ut.distributed.ascend_store.v1.bulk_prototype.test_layerwise import make_worker as make_layerwise
from tests.ut.distributed.ascend_store.v1.helpers import FakeBackend, make_coordinator, make_topology
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import FakeGVABackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype.data.keys import KeyIndex
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype.service.lookup import (
    LookupQuery,
    LookupService,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import BlockHashMapping
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
    StoreCommandBatch,
)


def test_identity_preserves_wire_aliases_across_hash_representations_and_naming_families():
    index = KeyIndex()
    assert index.identify(("ab", ""), b"\x0c") == index.identify(("ab", ""), "0c")
    assert index.identify(("ab", ""), "c") == index.identify(("a", ""), "bc")
    assert index.keys == ["ab0c", "abc"]


@pytest.mark.parametrize("operation", ("lookup", "load", "store"))
def test_shared_hash_reuses_each_object_and_keeps_its_first_source(operation):
    conversions = []

    class Hash(bytes):
        def hex(self):
            conversions.append(self)
            return super().hex()

    value = Hash(b"a")
    if operation == "lookup":
        topology = make_topology(pp_size=2)
        backend = FakeBackend()
        lookup = LookupService(topology, backend, make_coordinator(topology))
        result = lookup(LookupQuery(8, (0,), (value, value)))
        assert result.available_end_token == 8
        (query,) = backend.calls
        assert query[0] == "exists" and len(query[1]) == len(set(query[1])) == 2
    else:
        worker, _, _ = make_bulk(topology=make_topology(group_ids=(0, 1)))
        try:
            if operation == "load":
                commands = (
                    LoadCommand("a", 0, 4, ((1,), (2,)), (value,)),
                    LoadCommand("b", 0, 4, ((3,), (4,)), (value,)),
                )
                selection = worker.load._prepare(commands)
                assert selection.target_count == 4
                assert [local.block_ids for local in selection.local.values()] == [[1, 3], [2, 4]]
            else:
                commands = (
                    RangeStoreCommand("a", 0, 4, ((1,), (2,)), (value,), 4, 17),
                    RangeStoreCommand("b", 0, 4, ((3,), (4,)), (value,), 4, 18),
                )
                selection = worker.store._prepare(commands)
                assert len(selection.sources) == 2
                assert [block_id for _, block_id, _ in selection.sources] == [1, 2]
            assert len(selection.keys) == 2
        finally:
            worker.close()
    if operation != "store":
        assert conversions == [value]


def test_fully_local_lookup_skips_selection_but_keeps_coordinator_resolution(monkeypatch):
    topology = make_topology()
    backend, coordinator = FakeBackend(), make_coordinator(topology)
    monkeypatch.setattr(coordinator, "select_for_lookup", Mock(side_effect=AssertionError("no remote selection")))
    resolve = Mock(wraps=coordinator.resolve_available_end)
    monkeypatch.setattr(coordinator, "resolve_available_end", resolve)
    result = LookupService(topology, backend, coordinator)(LookupQuery(8, (0,), (b"a", b"b"), 8))
    assert result.available_end_token == 8 and not backend.calls
    resolve.assert_called_once()


def test_empty_load_skips_selection_and_empty_group_skips_pairing(monkeypatch):
    worker, _, _ = make_bulk()
    try:
        select = Mock(side_effect=AssertionError("empty command"))
        monkeypatch.setattr(worker.load.coordinator, "select_for_load", select)
        assert not worker.load._prepare((LoadCommand("empty", 0, 0, ((),), ()),)).keys
        select.assert_not_called()
        monkeypatch.setattr(worker.load.coordinator, "select_for_load", lambda *_args: ((),))
        monkeypatch.setattr(BlockHashMapping, "pair_block_ids", Mock(side_effect=AssertionError("empty group")))
        assert not worker.load._prepare((LoadCommand("empty", 0, 4, ((1,),), (b"a",)),)).keys
    finally:
        worker.close()


def test_bulk_shared_object_retains_independent_destination_failures():
    worker, _, backend = make_bulk()
    backend.get_result = [0, -7]
    commands = (LoadCommand("a", 0, 4, ((1,),), (b"a",)), LoadCommand("b", 0, 4, ((2,),), (b"a",)))
    try:
        worker.begin_step(KVTransferStep(commands))
        worker.start_load()
        batch = worker.load._pending[0]
        batch.completion.result(timeout=2)
        assert len(batch.selection.keys) == 1 and batch.selection.target_count == 2
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"a", "b"} and result.failed_block_ids == {2}
        worker.end_step()
    finally:
        worker.close()


def test_empty_store_skips_selection_and_all_present_skips_result_columns(monkeypatch):
    worker, _, backend = make_bulk(requires_exists=True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(worker.store.coordinator, "select_for_store", Mock(side_effect=AssertionError("empty range")))
            command = RangeStoreCommand("empty", 0, 4, ((),), (b"a",), 4, 17)
            assert not worker.store._prepare((command,)).keys
        backend.exists = lambda keys: (True,) * len(keys)
        command = RangeStoreCommand("present", 0, 4, ((1,),), (b"a",), 4, 18)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        batch = worker.store._pending[-1]
        batch.completion.result(timeout=2)
        assert not batch.codes
        (result,) = worker.fence_previous_store()
        assert result.succeeded and result.source_release_confirmed and not result.keys
        assert worker.take_released_store_job_ids() == {18}
        worker.finish_step()
        worker.end_step()
    finally:
        worker.close()


def test_key_range_failure_routes_to_all_and_only_its_bound_destinations():
    worker, _, backend = make_layerwise()
    backend.session_copy_result = [-7, 0]
    commands = (LoadCommand("a", 0, 8, ((1, 2),), (b"a", b"b")), LoadCommand("b", 0, 4, ((3,),), (b"a",)))
    try:
        worker.begin_step(KVTransferStep(commands))
        worker.start_load()
        plan = worker.transfer._write_batch.plan
        assert all(indices == (0, 1, 2) for indices in plan.range_indices.values())
        assert all(merges == (0, 1, 0) for merges in plan.key_merges.values())
        for name in layer_names(worker):
            worker.wait_for_layer_load(name)
        result = worker.collect_load_result()
        assert result.failed_block_ids == {1, 3} and result.completed_request_ids == {"a", "b"}
        worker.end_step()
    finally:
        worker.close()


@pytest.mark.parametrize("mode", ("key_range", "gva"))
def test_terminal_copy_codes_skip_later_io_but_finish_and_release_sources(mode, monkeypatch):
    worker, resources, backend = make_layerwise(backend=FakeGVABackend() if mode == "gva" else None)
    if mode == "gva":
        backend.native_store.copy_result = -7
    else:
        backend.store_session_copy_result = [-7]
    copy = Mock(wraps=worker.transfer.copy)
    monkeypatch.setattr(worker.transfer, "copy", copy)
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
        batch = worker.transfer._read_batch
        for name in layer_names(worker):
            worker.save_layer(name)
        worker.finish_step()
        (result,) = batch.completion.result(timeout=2)
        assert copy.call_count == 1
        assert result.result_codes == (-7,) and not result.succeeded
        assert result.source_release_confirmed and result.error is None
        assert worker.take_released_store_job_ids() == {17}
        if mode == "gva":
            assert not batch.plan.range_keys and not batch.plan.key_merges
        worker.end_step()
    finally:
        # The fake SDK copy is synchronous; confirm cleanup for this test's remote allocations.
        if mode == "gva":
            backend.revoke_gva = lambda keys: (0,) * len(keys)
        worker.close()
    assert resources.closed


def test_identity_switches_families_without_losing_old_or_new_wire_aliases():
    index = KeyIndex()
    for parts, value, expected in (
        (("a", ""), "b", 0),
        (("a", ""), "c", 1),
        (("", "b"), "a", 0),
        (("", "b"), "d", 2),
        (("a", ""), "db", 3),
        (("", "b"), "ad", 3),
        (("a", ""), "b", 0),
        (("", ""), "ac", 1),
    ):
        assert index.identify(parts, value) == expected
    assert index.keys == ["ab", "ac", "db", "adb"]
