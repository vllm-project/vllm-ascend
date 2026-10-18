"""Successful Loads notify completion without constructing per-object provenance."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeLoadStartGate, begin_step, make_topology, make_worker
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import make_gva_worker
from tests.ut.distributed.ascend_store.v1.worker.test_load_plan import make_group
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import LoadCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk.transfer import _bulk_load_completions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.transfer import (
    _layer_load_completions,
    make_layer_load_plan,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
    TransferSource,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.completion import (
    TransferEvidence,
    _failed_load_completions,
)


@pytest.mark.parametrize("route", ("bulk", "bulk_async", "key_range", "gva"))
def test_successful_load_never_constructs_object_evidence_and_preserves_notifications(route, monkeypatch):
    topology = make_topology(physical_layers=(2, 5, 9))
    if route == "gva":
        worker, resources, backend = make_gva_worker(topology=topology)
        worker._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    else:
        worker, resources, backend = make_worker(
            topology=topology,
            layerwise=route == "key_range",
            async_load=route == "bulk_async",
            store=False,
            start_gate_factory=lambda: FakeLoadStartGate(opened=True),
        )
    commands = (
        LoadCommand("first", TokenRange(0, 8), ((1, 2),), (b"a", b"b")),
        LoadCommand("second", TokenRange(0, 4), ((7,),), (b"c",)),
        LoadCommand("empty", TokenRange(0, 0), ((),), ()),
    )
    if route == "gva":
        batch = worker._build_load_batch(commands)
        backend.objects.update(
            {key: (10000 + index * 1000, 96, True) for index, key in enumerate(batch.selected_keys())}
        )
    source = Mock(side_effect=AssertionError("Successful Load must not construct TransferSource"))
    evidence = Mock(side_effect=AssertionError("Successful Load must not construct TransferEvidence"))
    monkeypatch.setattr(TransferSource, "__init__", source)
    monkeypatch.setattr(TransferEvidence, "__init__", evidence)
    try:
        begin_step(worker, load=commands)
        worker.start_load()
        if route == "bulk_async":
            worker._pending_loads[0].result(timeout=5)
        elif route in ("key_range", "gva"):
            for layer in (2, 2, 5, 5, 9):
                worker.wait_for_layer_load(f"layers.{layer}.group.0")
        result = worker.collect_load_result()
        assert result.completed_request_ids == ({"first", "second", "empty"} if route == "bulk_async" else set())
        assert not result.failed_locations and not result.failed_block_ids and not result.failed_request_ids
        assert not worker.collect_load_result().completed_request_ids
        if route in ("key_range", "gva"):
            copy_name = "copy" if route == "gva" else "batch_copy_get"
            assert sum(call[0] == copy_name for call in backend.calls) == 3
            assert not worker._backend_sessions.has_load_sessions
        source.assert_not_called()
        evidence.assert_not_called()
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("gva", (False, True), ids=("key_range", "gva"))
def test_all_successful_session_admission_reuses_batch_without_reselecting_objects(gva, monkeypatch):
    if gva:
        worker, _, backend = make_gva_worker()
    else:
        worker, _, backend = make_worker(layerwise=True)
    commands = (
        LoadCommand("first", TokenRange(0, 4), ((1,),), (b"same",)),
        LoadCommand("second", TokenRange(0, 4), ((7,),), (b"same",)),
    )
    batch = worker._build_load_batch(commands)
    if gva:
        backend.objects.update({key: (10000, batch.groups[0].object_size, True) for key in batch.selected_keys()})
    select = Mock(side_effect=AssertionError("Successful admission must reuse its input selection"))
    monkeypatch.setattr(KVTransferBatch, "select_keys", select)
    try:
        assert worker._submit_load(batch) == ()
        assert worker._load_plan.batch is batch
        assert worker._load_plan.groups[0].request_indices == (0, 1)
        assert worker._load_initial_completions == ()
        select.assert_not_called()
    finally:
        worker.close()


@pytest.mark.parametrize("failed_group", (0, 3))
def test_key_range_merged_failure_identifies_every_original_request_destination(failed_group, monkeypatch):
    topology = make_topology(group_ids=(0, 3), physical_layers=(2, 5))
    worker, resources, backend = make_worker(topology=topology, layerwise=True, store=False)
    commands = (
        LoadCommand("first", TokenRange(0, 4), ((1,), (), (), (2,)), (b"same",)),
        LoadCommand("second", TokenRange(0, 4), ((4,), (), (), (5,)), (b"same",)),
    )
    try:
        batch = worker._build_load_batch(commands)
        worker._prepare_load_layers(batch)
        backend.session_copy_result = (-7, 0) if failed_group == 0 else (0, -7)
        # Ownership must be resolved by the plan, not rediscovered for each failed layer.
        monkeypatch.setattr(
            KVGroupBatch, "request_index", Mock(side_effect=AssertionError("Repeated ownership lookup"))
        )
        completions = worker._copy_load_layer(2)
        assert [completion.request_id for completion in completions] == ["first", "second"]
        expected_blocks = (1, 4) if failed_group == 0 else (2, 5)
        for completion, block_id in zip(completions, expected_blocks, strict=True):
            (failure,) = completion.failures
            assert failure.result_code == -7
            assert (failure.source.group_id, failure.source.block_id) == (failed_group, block_id)
            assert failure.source.physical_layer_ids == (2,)
        copy = next(call for call in backend.calls if call[0] == "batch_copy_get")
        assert len(copy[1]) == 2  # Two Group keys, each merged across both requests.
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("code", (-9, None))
def test_layer_failure_only_expands_selected_rows_of_participating_groups(code):
    first = replace(make_group(2, (0, 1, 3), 1), physical_layer_ids=(2,))
    second = replace(make_group(7, (0, 2, 3), 1), physical_layer_ids=(5,))
    batch = KVTransferBatch(("first", "second"), (first, second)).select_keys(
        {first.key_axes[0][2], second.key_axes[0][0]}
    )
    plan = make_layer_load_plan(batch)
    completions = _layer_load_completions(plan, 2, code)
    assert [completion.request_id for completion in completions] == ["first", "second"]
    assert completions[0].failures == ()
    (failure,) = completions[1].failures
    assert (failure.result_code, failure.source.group_id, failure.source.block_id) == (code, 2, 12)
    assert failure.source.physical_layer_ids == (2,)
    # The outer exception path must apply the same physical-layer scope.
    fallback = _failed_load_completions(batch, 2)
    assert fallback[0].failures == ()
    assert [(item.source.group_id, item.source.block_id) for item in fallback[1].failures] == [(2, 12)]


def test_bulk_failure_uses_position_not_key_identity_and_keeps_empty_request_completion():
    first = replace(make_group(2, (0, 0, 1, 2), 1), key_axes=(("duplicate", "duplicate"),))
    second = make_group(7, (0, 0, 1, 2), 1)
    selection = np.array((False, True), dtype=np.bool_)
    selection.flags.writeable = False
    second = replace(second, selected_objects=selection)
    batch = KVTransferBatch(("empty", "first", "second"), (first, second))
    completions = _bulk_load_completions(batch, (-7, 0, -8))
    assert [completion.request_id for completion in completions] == ["empty", "first", "second"]
    assert completions[0].failures == ()
    assert [(item.result_code, item.source.group_id, item.source.block_id) for item in completions[1].failures] == [
        (-7, 2, 10)
    ]
    assert [(item.result_code, item.source.group_id, item.source.block_id) for item in completions[2].failures] == [
        (-8, 7, 11)
    ]


@pytest.mark.parametrize("route", ("key_range", "gva"))
@pytest.mark.parametrize("native_result", (-9, RuntimeError("copy failed")))
def test_layer_copy_failure_preserves_unknown_code_and_releases_load_sessions(route, native_result, monkeypatch):
    if route == "gva":
        worker, resources, backend = make_gva_worker()
        worker._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    else:
        worker, resources, backend = make_worker(
            layerwise=True, store=False, start_gate_factory=lambda: FakeLoadStartGate(opened=True)
        )
    command = LoadCommand("request", TokenRange(0, 4), ((3,),), (b"a",))
    if route == "gva":
        batch = worker._build_load_batch((command,))
        backend.objects[batch.selected_keys()[0]] = (10000, 64, True)
        backend.copy_result = native_result
    else:
        backend.session_copy_result = native_result if isinstance(native_result, Exception) else (native_result,)
    observed_codes = []
    record = worker._record_load_completion

    def record_failures(context, completion):
        observed_codes.extend(item.result_code for item in completion.failures)
        record(context, completion)

    monkeypatch.setattr(worker, "_record_load_completion", record_failures)
    try:
        begin_step(worker, load=(command,))
        worker.start_load()
        for layer in (0, 1):
            worker.wait_for_layer_load(f"layers.{layer}.group.0")
        expected_code = None if isinstance(native_result, Exception) else native_result
        assert observed_codes == [expected_code, expected_code]
        assert not worker._backend_sessions.has_load_sessions
        context = worker._active_step
        assert [(source.block_id, source.physical_layer_ids) for source in context.failed_sources] == [
            (3, (0,)),
            (3, (1,)),
        ]
        result = worker.collect_load_result()
        assert result.failed_block_ids == {3}
        assert [(item.group_id, item.block_id) for item in result.failed_locations] == [(0, 3)]
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed
