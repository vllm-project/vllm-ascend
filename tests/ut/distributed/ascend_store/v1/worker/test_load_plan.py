"""Load plans retain selected-object ownership without repeating it at each layer."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeLoadStartGate, begin_step, make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import LoadCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io.arguments import _layer_sources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
    make_layer_load_plan,
    make_layer_transfer_group,
    make_layer_transfer_plan,
)


def make_group(group_id, request_splits, axis_count=2):
    row_count = request_splits[-1]
    return KVGroupBatch(
        group_id,
        np.arange(row_count, dtype=np.uint64) + 10,
        np.full(row_count, 128, dtype=np.uint64),
        tuple(tuple(f"g{group_id}-axis{axis}-row{row}" for row in range(row_count)) for axis in range(axis_count)),
        np.array(request_splits, dtype=np.intp),
        (2, 5, 9),
        4096,
    )


@pytest.mark.parametrize("selected", (None, (0, 3, 5, 8, 9), (3, 4), ()))
def test_load_plan_resolves_selected_ownership_once_across_layers(selected, monkeypatch):
    groups = (make_group(2, (0, 0, 2, 2, 5)), make_group(7, (0, 1, 1, 4, 5)))
    batch = KVTransferBatch(("empty-or-first", "second", "empty-or-third", "last"), groups)
    if selected is not None:
        batch = batch.select_keys({group.key_axes[index // 5][index % 5] for group in groups for index in selected})
    expected = {
        (group.group_id, layer): tuple(
            group.source(int(index), layer) for index in make_layer_transfer_group(group).object_indices
        )
        for group in batch.groups
        for layer in group.physical_layer_ids
    }
    lookup = Mock(side_effect=KVGroupBatch.request_index)
    monkeypatch.setattr(KVGroupBatch, "request_index", lambda group, index: lookup(group, index))
    plan = make_layer_load_plan(batch)
    assert lookup.call_count == sum(len(group.keys) for group in plan.groups)
    lookup.side_effect = AssertionError("A layer must reuse the Load plan's request ownership")
    for group in plan.groups:
        assert isinstance(group.request_indices, tuple)
        assert group.request_indices == tuple(source.request_index for source in expected[group.group_id, 2])
        for layer in group.physical_layer_ids:
            assert _layer_sources(group, layer) == expected[group.group_id, layer]
            assert next(item for item in plan.groups_by_layer[layer] if item.group_id == group.group_id) is group


def test_new_load_plan_does_not_reuse_previous_batch_ownership():
    first = make_group(3, (0, 1, 3), axis_count=1)
    second = replace(first, request_splits=np.array((0, 2, 3), dtype=np.intp))
    first_plan = make_layer_load_plan(KVTransferBatch(("a", "b"), (first,)))
    second_plan = make_layer_load_plan(KVTransferBatch(("a", "b"), (second,)))
    assert first_plan.groups[0].request_indices == (0, 1, 1)
    assert second_plan.groups[0].request_indices == (0, 0, 1)
    assert tuple(source.request_index for source in _layer_sources(first_plan.groups[0], 2)) == (0, 1, 1)


def test_store_plan_does_not_prepare_load_ownership(monkeypatch):
    group = make_group(0, (0, 2, 5))
    monkeypatch.setattr(KVGroupBatch, "request_index", Mock(side_effect=AssertionError("Store must remain unchanged")))
    batch = KVTransferBatch(("a", "b"), (group,))
    plan = make_layer_transfer_plan(batch, (make_layer_transfer_group(group),))
    assert plan.groups[0].request_indices is None


def test_layer_sources_require_prepared_ownership():
    group = make_layer_transfer_group(make_group(0, (0, 2, 5)))
    with pytest.raises(RuntimeError, match="request ownership was not prepared"):
        _layer_sources(group, 2)


@pytest.mark.parametrize("copy_codes", ((0, 0), (0, -7)))
def test_partial_load_admission_preserves_original_failure_blocks(copy_codes, monkeypatch):
    worker, resources, backend = make_worker(
        layerwise=True,
        store=False,
        physical_layers=(0, 1, 2),
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    backend.session_start_result = [-1, 0, 0]
    backend.session_copy_result = copy_codes
    commands = (
        LoadCommand("first", TokenRange(0, 8), ((1, 2),), (b"a", b"b")),
        LoadCommand("second", TokenRange(0, 4), ((7,),), (b"c",)),
    )
    try:
        begin_step(worker, load=commands)
        worker.start_load()
        monkeypatch.setattr(
            KVGroupBatch, "request_index", Mock(side_effect=AssertionError("Ownership must be prepared before copies"))
        )
        for layer in range(3):
            worker.wait_for_layer_load(f"layers.{layer}.group.0")
        result = worker.collect_load_result()
        assert result.failed_block_ids == ({1} if copy_codes[1] == 0 else {1, 7})
        assert len([call for call in backend.calls if call[0] == "batch_copy_get"]) == 3
        assert len([call for call in backend.calls if call[0] == "batch_get_end"]) == 1
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed
