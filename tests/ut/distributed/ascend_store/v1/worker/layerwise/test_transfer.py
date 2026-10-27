"""Layer plans preserve selected-object ownership and per-layer failure provenance."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import make_topology, make_worker
from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import make_case
from tests.ut.distributed.ascend_store.v1.worker.layerwise.plan_fixtures import prepare_layer_group
from tests.ut.distributed.ascend_store.v1.worker.transfer.fixtures import make_group
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import LoadCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import transfer as layer_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.gva.worker import (
    materialize_layer_gva,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.key_range import (
    materialize_load_layer_ranges,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.transfer import (
    LayerTransferGroup,
    LayerTransferPlan,
    _layer_load_completions,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.completion import (
    _failed_load_completions,
)


def prepare_test_plan(batch, *, load=True):
    mappings = {}
    for group in batch.groups:
        case = make_case(layers=group.physical_layer_ids, head_slices=len(group.keys_by_partition))
        original_id = case.topology.groups[0].group_id
        case.topology.groups[0].group_id = group.group_id
        for entries in case.registration_arrays:
            entries[group.group_id] = entries.pop(original_id)
        mappings[group.group_id] = case.bind_layers()
    prepare = layer_transfer.prepare_layer_load if load else layer_transfer.prepare_layer_store
    return prepare(batch, mappings)


@pytest.mark.parametrize("code", (-9, None))
def test_layer_failure_only_expands_selected_rows_of_participating_groups(code):
    first = replace(make_group(2, (0, 1, 3), 1), physical_layer_ids=(2,))
    second = replace(make_group(7, (0, 2, 3), 1), physical_layer_ids=(5,))
    batch = KVTransferBatch(("first", "second"), (first, second)).select_keys(
        {first.keys_by_partition[0][2], second.keys_by_partition[0][0]}
    )
    plan = prepare_test_plan(batch)
    completions = _layer_load_completions(plan, 2, code)
    assert [completion.request_id for completion in completions] == ["first", "second"]
    assert completions[0].failures == ()
    (failure,) = completions[1].failures
    assert (failure.result_code, failure.source.group_id, failure.source.local_block_id) == (code, 2, 12)
    assert failure.source.physical_layer_ids == (2,)
    # The outer exception path must apply the same physical-layer scope.
    fallback = _failed_load_completions(batch, 2)
    assert fallback[0].failures == ()
    assert [(item.source.group_id, item.source.local_block_id) for item in fallback[1].failures] == [(2, 12)]


@pytest.mark.parametrize("selected", (None, (0, 3, 5, 8, 9), (3, 4), ()))
def test_load_plan_resolves_selected_ownership_once_across_layers(selected, monkeypatch):
    groups = (make_group(2, (0, 0, 2, 2, 5)), make_group(7, (0, 1, 1, 4, 5)))
    batch = KVTransferBatch(("empty-or-first", "second", "empty-or-third", "last"), groups)
    if selected is not None:
        batch = batch.select_keys(
            {group.keys_by_partition[index // 5][index % 5] for group in groups for index in selected}
        )
    expected = {
        (group.group_id, layer): tuple(
            group.source(index, layer)
            for index in range(group.transfer_candidate_count)
            if group.transfer_candidate_mask is None or group.transfer_candidate_mask[index]
        )
        for group in batch.groups
        for layer in group.physical_layer_ids
    }
    lookup = Mock(side_effect=KVGroupBatch.request_index)
    monkeypatch.setattr(KVGroupBatch, "request_index", lambda group, index: lookup(group, index))
    plan = prepare_test_plan(batch)
    assert lookup.call_count == sum(len(group.selected_keys) for group in plan.groups)
    lookup.side_effect = AssertionError("A layer must reuse the Load plan's request ownership")
    for group in plan.groups:
        assert isinstance(group.request_indices, tuple)
        assert group.request_indices == tuple(source.request_index for source in expected[group.group_id, 2])
        for layer in group.physical_layer_ids:
            assert next(item for item in plan.groups_by_layer[layer] if item.group_id == group.group_id) is group
    for layer in (2, 5, 9):
        completions = _layer_load_completions(plan, layer, None) if plan.groups else ()
        observed = [item.source for completion in completions for item in completion.failures]
        expected_sources = [source for group in plan.groups for source in expected[group.group_id, layer]]
        assert observed == sorted(expected_sources, key=lambda source: source.request_index)


def test_new_load_plan_does_not_reuse_previous_batch_ownership():
    first = make_group(3, (0, 1, 3), partition_count=1)
    second = replace(first, request_transfer_record_boundaries=np.array((0, 2, 3), dtype=np.intp))
    first_plan = prepare_test_plan(KVTransferBatch(("a", "b"), (first,)))
    second_plan = prepare_test_plan(KVTransferBatch(("a", "b"), (second,)))
    assert first_plan.groups[0].request_indices == (0, 1, 1)
    assert second_plan.groups[0].request_indices == (0, 0, 1)
    completions = _layer_load_completions(first_plan, 2, None)
    assert tuple(item.source.request_index for completion in completions for item in completion.failures) == (0, 1, 1)


def test_partition_slices_align_layer_keys_gva_bases_and_failure_ownership():
    case = make_case(layers=(2, 5, 9), pp_layer_counts=(4, 6), head_slices=2)
    mapping = case.bind_layers()
    keys = mapping.key_namer.keys(("repeat", "other", "repeat"))
    selected = (True, False, True, False, True, True, True, True, False, False, False, False)
    group = prepare_layer_group(mapping, keys, (2, 0, 1), selected_keys=selected).batch
    group = replace(group, request_transfer_record_boundaries=np.array((0, 1, 3), dtype=np.intp))
    batch = KVTransferBatch(("first", "second"), (group,))
    mappings = {group.group_id: mapping}
    bases = {key: 100000 + i * 1000 for i, key in enumerate(dict.fromkeys(key for part in keys for key in part))}
    plan = layer_transfer.prepare_layer_load(
        batch, mappings, lambda _group, ordered_keys: np.array([bases[key] for key in ordered_keys], dtype=np.uint64)
    )
    assert plan.selected_keys_by_layer[5] is plan.selected_keys_by_layer[9]
    assert plan.selected_keys_by_layer[2] != plan.selected_keys_by_layer[5]
    for layer_id in (2, 5, 9):
        participating = [
            partition_index * 3 + record_index
            for partition_index, layers in enumerate(group.physical_layer_ids_by_partition)
            if layer_id in layers
            for record_index in range(3)
            if selected[partition_index * 3 + record_index]
        ]
        expected_keys = [keys[index // 3][index % 3] for index in participating]
        assert plan.selected_keys_by_layer[layer_id] == expected_keys
        remote, local, lengths = materialize_layer_gva(mappings, plan, layer_id)
        expected_remote, expected_local, expected_lengths = [], [], []
        for layout in mapping.layer_layouts[layer_id]:
            segments = layout.local_segments
            strides = np.broadcast_to(segments.group_block_strides_bytes, segments.block_zero_segment_addresses.shape)
            sizes = np.broadcast_to(segments.segment_lengths_bytes, segments.block_zero_segment_addresses.shape)
            for index in participating:
                if index // 3 != layout.partition_index:
                    continue
                block_id = int(group.local_block_ids[index % 3])
                key = keys[index // 3][index % 3]
                expected_remote.extend(bases[key] + int(offset) for offset in layout.remote_segment_offsets_bytes)
                expected_local.extend(
                    int(base) + block_id * int(stride)
                    for base, stride in zip(segments.block_zero_segment_addresses, strides, strict=True)
                )
                expected_lengths.extend(sizes.tolist())
        assert (remote, local, lengths) == (expected_remote, expected_local, expected_lengths)
        arguments = materialize_load_layer_ranges(mappings, plan, layer_id)
        assert arguments.keys == list(dict.fromkeys(expected_keys))
        completions = _layer_load_completions(plan, layer_id, None)
        sources = [item.source for completion in completions for item in completion.failures]
        assert sorted(source.transfer_candidate_index for source in sources) == participating
        assert all(source.request_index == group.request_index(source.transfer_candidate_index) for source in sources)


def test_plan_omits_physical_layers_with_no_local_segments():
    case = make_case(pp_layer_counts=(2, 2))
    case.caches["layer1"] = (torch.empty(0),)
    for array in case.registration_arrays[:3]:
        del array[3][2:4]
    case.registration_arrays[3][3] = [0, 2, 2, 4, 6]
    mapping = case.bind_layers()
    group = prepare_layer_group(mapping, mapping.key_namer.keys(("a",)), (0,)).batch
    batch = KVTransferBatch(("request",), (group,))
    plan = layer_transfer.prepare_layer_load(batch, {group.group_id: mapping})
    assert tuple(plan.groups_by_layer) == (0, 2, 3)
    assert tuple(plan.selected_keys_by_layer) == (0, 2, 3)


def test_store_plan_does_not_prepare_load_ownership(monkeypatch):
    group = make_group(0, (0, 2, 5))
    monkeypatch.setattr(KVGroupBatch, "request_index", Mock(side_effect=AssertionError("Store must remain unchanged")))
    batch = KVTransferBatch(("a", "b"), (group,))
    plan = prepare_test_plan(batch, load=False)
    assert plan.groups[0].request_indices is None
    assert plan.groups[0].selected_transfer_candidate_indices is None


@pytest.mark.parametrize("load", (False, True), ids=("store", "load"))
@pytest.mark.parametrize("gva", (False, True), ids=("key_range", "gva"))
@pytest.mark.parametrize("selected", (False, True))
def test_preparation_constructs_each_group_and_final_plan_once(load, gva, selected, monkeypatch):
    topology = make_topology(group_ids=(0, 3), physical_layers=(2, 5, 9))
    worker, _, _ = make_worker(topology=topology, layerwise=True)
    try:
        commands = (
            LoadCommand("first", 0, 8, ((1, 2), (), (), (3, 4)), (b"a", b"b")),
            LoadCommand("second", 0, 4, ((7,), (), (), (6,)), (b"a",)),
        )
        batch = worker._build_load_batch(commands)
        if selected:
            # Preserve an empty Group alongside repeated keys belonging to distinct requests.
            batch = batch.select_keys({batch.groups[0].keys_by_partition[0][0]})
        group_init, plan_init = LayerTransferGroup.__init__, LayerTransferPlan.__init__
        constructed_groups, constructed_plans = [], []

        def create_group(self, *args, **kwargs):
            group_init(self, *args, **kwargs)
            constructed_groups.append(self)

        def create_plan(self, *args, **kwargs):
            plan_init(self, *args, **kwargs)
            constructed_plans.append(self)

        monkeypatch.setattr(LayerTransferGroup, "__init__", create_group)
        monkeypatch.setattr(LayerTransferPlan, "__init__", create_plan)
        regions = Mock(side_effect=lambda group, keys: np.arange(len(keys), dtype=np.uint64) + group.group_id * 1000)
        prepare = layer_transfer.prepare_layer_load if load else layer_transfer.prepare_layer_store
        plan = prepare(batch, worker._mappings, regions if gva else None)

        assert len(constructed_groups) == (1 if selected else len(batch.groups))
        assert constructed_plans == [plan]
        assert all(any(group is built for built in constructed_groups) for group in plan.groups)
        assert all(group.selected_transfer_slices_by_partition for group in constructed_groups)
        assert all((group.request_indices is not None) is load for group in constructed_groups)
        assert regions.call_count == (len(plan.groups) if gva else 0)
        assert tuple(plan.groups_by_layer) == (2, 5, 9)
        assert plan.selected_keys_by_layer[2] is plan.selected_keys_by_layer[5] is plan.selected_keys_by_layer[9]
        for group in plan.groups:
            assert all(any(item is group for item in plan.groups_by_layer[layer]) for layer in (2, 5, 9))
            if load:
                assert group.request_indices == tuple(
                    group.batch.request_index(int(i)) for i in group.selected_transfer_candidate_indices
                )
            if gva:
                np.testing.assert_array_equal(
                    plan.remote_object_bases_by_group[group.group_id],
                    np.arange(len(group.selected_keys)) + group.group_id * 1000,
                )
    finally:
        worker.close()


@pytest.mark.parametrize("load", (False, True))
@pytest.mark.parametrize("selection", ("partial", "full", "none", "zero_rows"))
def test_plan_reuses_mapping_rows_and_skips_empty_groups(load, selection, monkeypatch):
    topology = make_topology(physical_layers=(2, 5, 9))
    worker, _, _ = make_worker(topology=topology, layerwise=True)
    try:
        batch = worker._build_load_batch((LoadCommand("request", 0, 16, ((1, 2, 3, 4),), (b"a", b"b", b"c", b"d")),))
        if selection == "zero_rows":
            batch = worker._build_load_batch((LoadCommand("request", 0, 0, ((),), ()),))
        else:
            keys = batch.groups[0].keys_by_partition[0]
            accepted = keys[::2] if selection == "partial" else keys if selection == "full" else ()
            batch = batch.select_keys(set(accepted))
        prepare = Mock(side_effect=layer_transfer.make_layer_transfer_group)
        monkeypatch.setattr(layer_transfer, "make_layer_transfer_group", prepare)
        flatnonzero = Mock(side_effect=np.flatnonzero)
        monkeypatch.setattr(np, "flatnonzero", flatnonzero)
        regions = Mock(side_effect=lambda _group, keys: np.arange(len(keys), dtype=np.uint64))
        lower = layer_transfer.prepare_layer_load if load else layer_transfer.prepare_layer_store
        if selection in ("none", "zero_rows"):
            monkeypatch.setattr(LayerTransferGroup, "__init__", Mock(side_effect=AssertionError("empty Group wrapper")))
        plan = lower(batch, worker._mappings, regions)
        if selection in ("none", "zero_rows"):
            assert not plan.groups and not plan.groups_by_layer and not plan.selected_keys_by_layer
            prepare.assert_not_called()
            flatnonzero.assert_not_called()
            regions.assert_not_called()
            return
        prepare.assert_called_once()
        assert flatnonzero.call_count == (1 if selection == "partial" else 0)
        (group,) = plan.groups
        assert not hasattr(group, "local_block_ids") and not hasattr(group, "token_counts")
        if load and selection == "partial":
            assert group.selected_transfer_candidate_indices is group.selected_transfer_record_indices_by_partition[0]
            assert not group.selected_transfer_candidate_indices.flags.writeable
        elif load:
            assert isinstance(group.selected_transfer_candidate_indices, range)
        else:
            assert group.selected_transfer_candidate_indices is None and group.request_indices is None
        flatnonzero.side_effect = AssertionError("layer execution must not rescan selection")
        for layer in group.physical_layer_ids:
            materialize_load_layer_ranges(worker._mappings, plan, layer)
            if load:
                completion = _layer_load_completions(plan, layer, None)
                assert len(completion[0].failures) == len(group.selected_keys)
    finally:
        worker.close()
