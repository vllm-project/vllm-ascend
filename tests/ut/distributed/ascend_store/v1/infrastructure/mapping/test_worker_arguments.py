"""Verify final Worker arguments against Tensor bytes across partitioned layouts."""

import ctypes
from itertools import product

import numpy as np
import pytest
import torch

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import (
    expected_group_bytes,
    expected_head_bytes,
    make_case,
    make_hybrid_case,
    read_bytes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import (
    build_key_range_remote,
    build_mappings,
    map_keys,
    select_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk.transfer import (
    bind_bulk_sizes,
    materialize_bulk_arguments,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.key_range import (
    bind_key_range_layouts,
    materialize_store_layer_ranges,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.transfer import (
    bind_layer_memberships,
    prepare_layer_transfer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
)


def transfer_arguments(case, group_id, hashes, ids, mask=None):
    keys, local, layers = build_mappings(
        case.topology, case.caches, group_id=group_id, kv_cache_layout=case.layout, layerwise=True
    )
    remote = build_key_range_remote(local)
    group = next(group for group in case.topology.transfer_groups if group.group_id == group_id)
    by_pp = select_layers(group, case.topology.config.pp_partition.target_layer_counts)
    batch = KVTransferBatch(
        ("request",),
        (
            KVGroupBatch(
                group_id,
                np.asarray(ids, dtype=np.uint64),
                tuple(map_keys(parts, hashes) for parts in keys.values()),
                np.array((0, len(ids)), dtype=np.intp),
                tuple(layer.physical_layer_id for layer in group.layers),
                next(iter(remote.values()))["size_bytes"],
                None if mask is None else np.asarray(mask, dtype=np.bool_),
                physical_layer_ids_by_partition=tuple(
                    tuple(layer.physical_layer_id for layer in by_pp[pp]) for pp, _, _ in keys
                ),
            ),
        ),
    )
    locals_by_group, layers_by_group, remote_by_group = {group_id: local}, {group_id: layers}, {group_id: remote}
    bulk = materialize_bulk_arguments(locals_by_group, batch, bind_bulk_sizes(locals_by_group))
    plan = prepare_layer_transfer(batch, bind_layer_memberships(locals_by_group, layers_by_group))
    layouts = bind_key_range_layouts(locals_by_group, layers_by_group, remote_by_group)
    per_layer = {
        layer: (plan.selected_keys_by_layer[layer], *materialize_store_layer_ranges(layouts, groups, layer))
        for layer, groups in plan.groups_by_layer.items()
    }
    return bulk, per_layer, keys, by_pp


@pytest.mark.parametrize(
    "layout,heads,kernels,padding,mask_kind",
    list(product(("NHD", "HND", "LBHNC"), (1, 2, 4), (1, 2), (0, 1), ("full", "mixed")))
    + [("NHD", heads, 1, 0, mask) for heads, mask in product((1, 2), ("none", "empty"))],
)
def test_bulk_and_layers_reassemble_tensor_bytes(layout, heads, kernels, padding, mask_kind):
    case = make_case(layout=layout, head_slices=heads, kernels=kernels, padding=padding, pp_layer_counts=(1, 0, 3))
    ids = [] if mask_kind == "empty" else [2, 0, 2]
    hashes = ["repeat", "other", "repeat"][: len(ids)]
    mask = (
        None
        if mask_kind in ("full", "empty")
        else [mask_kind == "mixed" and i % 3 != 1 for i in range(2 * heads * len(ids))]
    )
    bulk, per_layer, keys, by_pp = transfer_arguments(case, 3, hashes, ids, mask)
    expected = []
    for p, ((pp, _, head), parts) in enumerate(keys.items()):
        layers = tuple(layer.physical_layer_id for layer in by_pp[pp])
        for row, (key, block) in enumerate(zip(map_keys(parts, hashes), ids, strict=True)):
            if mask is None or mask[p * len(ids) + row]:
                expected.append((key, expected_head_bytes(case, layers, head % heads, block)))
    assert (
        list(zip(bulk.keys, (read_bytes(a, n) for a, n in zip(bulk.addresses, bulk.sizes, strict=True)), strict=True))
        == expected
    )
    objects = {key: bytearray(len(value)) for key, value in expected}
    for names, addresses, sizes, offsets in per_layer.values():
        for key, row_addresses, row_sizes, row_offsets in zip(names, addresses, sizes, offsets, strict=True):
            for address, size, offset in zip(row_addresses, row_sizes, row_offsets, strict=True):
                objects[key][offset : offset + size] = ctypes.string_at(address, size)
    assert objects == dict(expected)


@pytest.mark.parametrize("layout,replicas,source_layerwise", product(("NHD", "HND", "LBHNC"), (1, 2), (False, True)))
def test_round_trip_between_tp_pp_partitions_and_replicas(layout, replicas, source_layerwise):
    heads = 2 if replicas == 1 else 1
    producers = [
        make_case(
            layers=(pp * 2, pp * 2 + 1),
            head_slices=1,
            heads=heads,
            tp_rank=rank,
            replicas=replicas,
            pp_rank=pp,
            layout=layout,
        )
        for pp in range(2)
        for rank in range(4 * replicas)
    ]
    consumers = [
        make_case(head_slices=2, pp_layer_counts=(2, 2), heads=heads * 2, tp_rank=rank, pp_size=1, layout=layout)
        for rank in range(2)
    ]
    originals = {}
    for case in producers + consumers:
        group = case.topology.transfer_groups[0]
        for layer in group.layers:
            for component, tensor in enumerate(case.caches[layer.layer_names[0]]):
                if not tensor.numel():
                    continue
                axis = 1 if layout in ("HND", "LBHNC") else 2
                for head in range(tensor.shape[axis]):
                    tensor.select(axis, head).fill_(
                        layer.physical_layer_id * 1000
                        + component * 100
                        + group.key_metadata.head_or_tp_rank * tensor.shape[axis]
                        + head
                    )
                originals[tensor.data_ptr()] = tensor.clone()
    for sources, targets in ((producers, consumers), (consumers, producers)):
        objects: dict[str, bytearray] = {}
        for case in sources:
            bulk, layers, _, _ = transfer_arguments(case, 3, ("a", "b"), (1, 2))
            payloads = {key: bytearray(sum(sizes)) for key, sizes in zip(bulk.keys, bulk.sizes, strict=True)}
            calls = (
                layers.values()
                if source_layerwise
                else [
                    (
                        bulk.keys,
                        bulk.addresses,
                        bulk.sizes,
                        [np.cumsum([0] + sizes[:-1]).tolist() for sizes in bulk.sizes],
                    )
                ]
            )
            for names, addresses, lengths, offsets in calls:
                for key, row_addresses, sizes, starts in zip(names, addresses, lengths, offsets, strict=True):
                    for address, size, start in zip(row_addresses, sizes, starts, strict=True):
                        payloads[key][start : start + size] = ctypes.string_at(address, size)
            for key, value in payloads.items():
                assert objects.setdefault(key, value) == value
        for case in targets:
            for tensors in case.caches.values():
                for tensor in tensors:
                    tensor.zero_()
            bulk, layers, _, _ = transfer_arguments(case, 3, ("a", "b"), (1, 2))
            calls = (
                [
                    (
                        bulk.keys,
                        bulk.addresses,
                        bulk.sizes,
                        [np.cumsum([0] + sizes[:-1]).tolist() for sizes in bulk.sizes],
                    )
                ]
                if source_layerwise
                else layers.values()
            )
            for names, addresses, lengths, offsets in calls:
                for key, row_addresses, sizes, starts in zip(names, addresses, lengths, offsets, strict=True):
                    for address, size, start in zip(row_addresses, sizes, starts, strict=True):
                        ctypes.memmove(address, bytes(objects[key][start : start + size]), size)
            for tensors in case.caches.values():
                for tensor in tensors:
                    if tensor.numel():
                        assert torch.equal(tensor[1:], originals[tensor.data_ptr()][1:])


def test_hybrid_original_group_ids_and_mixed_dtype_pages():
    case = make_hybrid_case(pp_layer_counts=(2, 2), tp_rank=3)
    for group_id, ids in ((9, (2, 0, 2)), (3, (1, 0))):
        hashes = tuple(str(i) for i in range(len(ids)))
        bulk, layers, keys, by_pp = transfer_arguments(case, group_id, hashes, ids)
        expected = {
            key: expected_group_bytes(case, group_id, tuple(layer.physical_layer_id for layer in by_pp[pp]), block)
            for (pp, _, _), parts in keys.items()
            for key, block in zip(map_keys(parts, hashes), ids, strict=True)
        }
        assert (
            dict(
                zip(bulk.keys, (read_bytes(a, n) for a, n in zip(bulk.addresses, bulk.sizes, strict=True)), strict=True)
            )
            == expected
        )
        objects = {key: bytearray(len(value)) for key, value in expected.items()}
        for names, addresses, sizes, offsets in layers.values():
            for key, row_addresses, row_sizes, starts in zip(names, addresses, sizes, offsets, strict=True):
                for address, size, start in zip(row_addresses, row_sizes, starts, strict=True):
                    objects[key][start : start + size] = ctypes.string_at(address, size)
        assert objects == expected
