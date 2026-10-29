"""Layer plans preserve selected-object ownership and per-layer failure provenance."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import make_topology, make_worker
from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.test_mapping import make_case
from tests.ut.distributed.ascend_store.v1.worker.transfer.fixtures import make_group
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import (
    build_key_range_remote,
    build_mappings,
    local_lengths,
    map_keys,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import LoadCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import transfer as layer_transfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.gva.worker import (
    materialize_layer_gva,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.key_range import (
    bind_key_range_layouts,
    materialize_load_layer_ranges,
    prepare_load_key_merges,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.transfer import (
    _layer_load_completions,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.completion import (
    _failed_load_completions,
)


def prepare_test_plan(batch):
    memberships = {
        group.group_id: {
            layer: (group.group_id, tuple(range(len(group.keys_by_partition)))) for layer in group.physical_layer_ids
        }
        for group in batch.groups
    }
    return layer_transfer.prepare_layer_transfer(batch, memberships)


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
def test_load_failures_follow_selected_destination_ownership(selected):
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
    plan = prepare_test_plan(batch)
    for layer in (2, 5, 9):
        if plan.groups:
            assert all(not c.failures for c in _layer_load_completions(plan, layer, {}))
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
    completions = _layer_load_completions(first_plan, 2, None)
    assert tuple(item.source.request_index for completion in completions for item in completion.failures) == (0, 1, 1)
    second_completions = _layer_load_completions(second_plan, 2, None)
    assert tuple(item.source.request_index for c in second_completions for item in c.failures) == (0, 0, 1)


def test_partition_slices_align_layer_keys_gva_bases_and_failure_ownership():
    topology, caches = make_case(layers=(2, 5, 9), pp_counts=(4, 6), head_partition_count=2)
    key_table, local, layers = build_mappings(topology, caches, group_id=3, layerwise=True)
    remote_table = build_key_range_remote(local)
    keys = tuple(map_keys(parts, ("repeat", "other", "repeat")) for parts in key_table.values())
    selected = (True, False, True, False, True, True, True, True, False, False, False, False)
    group = KVGroupBatch(
        3,
        np.asarray((2, 0, 1), dtype=np.uint64),
        keys,
        np.array((0, 1, 3), dtype=np.intp),
        (2, 5, 9),
        remote_table[0]["size_bytes"],
        np.asarray(selected, dtype=np.bool_),
        physical_layer_ids_by_partition=((2,), (2,), (5, 9), (5, 9)),
    )
    batch = KVTransferBatch(("first", "second"), (group,))
    locals_by_group, layers_by_group, remote_by_group = {3: local}, {3: layers}, {3: remote_table}
    bases = {key: 100000 + i * 1000 for i, key in enumerate(dict.fromkeys(key for part in keys for key in part))}
    plan = layer_transfer.prepare_layer_transfer(
        batch,
        layer_transfer.bind_layer_memberships(locals_by_group, layers_by_group),
        lambda _group, ordered_keys: np.array([bases[key] for key in ordered_keys], dtype=np.uint64),
    )
    assert plan.selected_keys_by_layer[2] != plan.selected_keys_by_layer[5]
    native_layouts = bind_key_range_layouts(locals_by_group, layers_by_group, remote_by_group)
    key_merges = prepare_load_key_merges(plan)
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
        remote, addresses, lengths = materialize_layer_gva(
            locals_by_group, layers_by_group, remote_by_group, plan, layer_id
        )
        expected_remote: list[int] = []
        expected_local: list[int] = []
        expected_lengths: list[int] = []
        for partition_index, ((pp, _, _), segments) in enumerate(local.items()):
            selected_slice = layers[pp].get(layer_id)
            if selected_slice is None:
                continue
            column_bases = segments["bases"][selected_slice]
            strides = np.broadcast_to(segments["block_strides"], segments["bases"].shape)[selected_slice]
            sizes = np.broadcast_to(local_lengths(segments, selected_slice), column_bases.shape)
            for index in participating:
                if index // 3 != partition_index:
                    continue
                block_id = int(group.local_block_ids[index % 3])
                key = keys[index // 3][index % 3]
                expected_remote.extend(
                    bases[key] + int(offset) for offset in remote_table[pp]["offsets"][selected_slice]
                )
                expected_local.extend(
                    int(base) + block_id * int(stride) for base, stride in zip(column_bases, strides, strict=True)
                )
                expected_lengths.extend(sizes.tolist())
        assert (remote, addresses, lengths) == (expected_remote, expected_local, expected_lengths)
        arguments = materialize_load_layer_ranges(native_layouts, plan, layer_id, key_merges[layer_id])
        assert arguments.keys == list(dict.fromkeys(expected_keys))
        completions = _layer_load_completions(plan, layer_id, None)
        sources = [item.source for completion in completions for item in completion.failures]
        assert sorted(source.transfer_candidate_index for source in sources) == participating
        assert all(source.request_index == group.request_index(source.transfer_candidate_index) for source in sources)


def test_plan_omits_physical_layers_with_no_local_segments():
    topology, caches = make_case(pp_counts=(2, 2))
    caches["layer1"] = (torch.empty(0),)
    key_table, local, layers = build_mappings(topology, caches, group_id=3, layerwise=True)
    keys = tuple(map_keys(parts, ("a",)) for parts in key_table.values())
    group = KVGroupBatch(3, np.array((0,), dtype=np.uint64), keys, np.array((0, 1), dtype=np.intp), (0, 1, 2, 3), 0)
    batch = KVTransferBatch(("request",), (group,))
    plan = layer_transfer.prepare_layer_transfer(batch, layer_transfer.bind_layer_memberships({3: local}, {3: layers}))
    assert tuple(plan.groups_by_layer) == (0, 2, 3)
    assert tuple(plan.selected_keys_by_layer) == (0, 2, 3)


@pytest.mark.parametrize("selection", ("partial", "full", "none", "zero_rows"))
def test_plan_filters_partial_full_and_empty_selections(selection):
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

        def regions(_group, keys):
            return np.arange(len(keys), dtype=np.uint64)

        lower = layer_transfer.prepare_layer_transfer
        plan = lower(batch, worker._layer_memberships, regions)
        if selection in ("none", "zero_rows"):
            assert not plan.groups and not plan.groups_by_layer and not plan.selected_keys_by_layer
            return
        (group,) = plan.groups
        assert group.selected_keys == tuple(accepted)
        expected_ids = [1, 3] if selection == "partial" else [1, 2, 3, 4]
        assert group.selected_local_block_ids_by_partition[0].tolist() == expected_ids
        key_merges = prepare_load_key_merges(plan)
        for layer in group.physical_layer_ids:
            materialize_load_layer_ranges(worker._key_range_layouts, plan, layer, key_merges[layer])
            completion = _layer_load_completions(plan, layer, None)
            assert len(completion[0].failures) == len(group.selected_keys)
    finally:
        worker.close()
