"""Object serialization and independent producer/consumer bindings over real CPU storage."""

import ctypes
from dataclasses import replace
from itertools import permutations, product

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import (
    build_key_range_remote,
    build_lookup_keys,
    build_mappings,
    local_addresses,
    local_lengths,
    map_keys,
)

from .fixtures import make_case
from .test_mapping import read_local


def layout_case(layout, rows, storage_kind, heads=4, partitions=1):
    kernel = 128
    case = make_case(
        layers=(0, 1),
        block_size=rows,
        kernels=rows // kernel,
        layout=layout,
        heads=heads,
        vector=3,
        head_slices=partitions,
        pp_size=1,
        key_rank_count=2,
    )
    if storage_kind != "separate":
        page_elements = kernel * heads * 3
        pitch = 2 * page_elements + (16 if storage_kind == "padded" else 0)
        for name, (key, _, value) in case.caches.items():
            storage = torch.full((5 + key.shape[0] * pitch,), -999, dtype=key.dtype)
            components = tuple(
                torch.as_strided(storage, key.shape, (pitch, *key.stride()[1:]), 5 + index * page_elements)
                for index in range(2)
            )
            for target, source in zip(components, (key, value), strict=True):
                target.copy_(source)
            case.caches[name] = components
    return case


def logical_values(case, block, head=0, partitions=1):
    """Independent Tensor indexing: values retain layer/component/token/head semantics."""
    values = []
    axis = 1 if case.layout == "HND" else 2
    kernels = case.topology.groups[0].block_size // 128
    for tensors in case.caches.values():
        for tensor in tensors:
            if tensor.numel():
                count = tensor.shape[axis] // partitions
                values.append(tensor[block * kernels : (block + 1) * kernels].narrow(axis, head * count, count).clone())
    return values


@pytest.mark.parametrize("layout", ("NHD", "HND"))
@pytest.mark.parametrize("rows", (128, 256, 640))
@pytest.mark.parametrize("storage_kind", ("separate", "packed", "padded"))
@pytest.mark.parametrize("partitions", (1, 2))
def test_independent_bindings_preserve_object_bytes_and_tp_destinations(layout, rows, storage_kind, partitions):
    producer = layout_case(layout, rows, storage_kind, partitions=partitions)
    consumer = layout_case(layout, rows, storage_kind, heads=4 // partitions)
    source_keys, source_local, source_layers = build_mappings(
        producer.topology, producer.caches, group_id=3, kv_cache_layout=layout, layerwise=True
    )
    bulk_keys, bulk_local, _ = build_mappings(producer.topology, producer.caches, group_id=3, kv_cache_layout=layout)
    for head in range(partitions):
        consumer.topology.groups[0].key_metadata.head_or_tp_rank = head
        target_keys, target_local, target_layers = build_mappings(
            consumer.topology, consumer.caches, group_id=3, kv_cache_layout=layout, layerwise=True
        )
        identity = (0, 0, head)
        parts = source_keys[identity]
        assert bulk_keys[identity] == target_keys[identity] == parts
        assert parts in build_lookup_keys(consumer.topology, 3)
        metadata = replace(producer.topology.groups[0].key_metadata, head_or_tp_rank=head)
        assert map_keys(parts, (b"hash",))[0] == PoolKey(metadata, "68617368").to_string()
        remote = build_key_range_remote(target_local)[0]
        source = source_local[identity]
        target = target_local[identity]
        assert len(target["bases"]) <= len(source["bases"])
        # Full local heads: two independent components, or one segment per padded page.
        expected_per_layer = 2 if storage_kind == "separate" else rows // 128 if storage_kind == "padded" else 1
        assert len(target["bases"]) == 2 * expected_per_layer
        if partitions == 1:
            original = KVPoolWorker.__new__(KVPoolWorker)
            original.num_blocks = 3
            original.use_layerwise = False
            original.tp_mismatch = False
            original.dcp_size = 1
            original.use_kvpp = False
            original._extra_config = {"discard_partial_chunks": True}
            original.grouped_block_size = [rows]
            original.cache_transfer_granularity = rows
            original.group_uses_align_state = [False]
            production = []
            for tensors in producer.caches.values():
                tensors = tuple(tensor for tensor in tensors if tensor.numel())
                segments = original._get_combined_cache_transfer_segments(tensors, 0)
                production.extend(
                    segments
                    if segments is not None
                    else [segment for tensor in tensors for segment in original._get_cache_transfer_segments(tensor)]
                )
            assert len(source["bases"]) <= len(production)
            assert read_local(source, 0) == b"".join(ctypes.string_at(base, size) for base, size, _ in production)
        for source_block, target_block in ((0, 2), (2, 0)):
            expected = logical_values(producer, source_block, head, partitions)
            payload = read_local(source, source_block)
            assert read_local(bulk_local[identity], source_block) == payload
            untouched = logical_values(consumer, 1)
            for layer_id, span in target_layers[0].items():
                source_span = source_layers[0][layer_id]
                assert (
                    read_local(source, source_block, source_span)
                    == payload[
                        int(remote["offsets"][span.start]) : int(remote["offsets"][span.start])
                        + sum(np.broadcast_to(local_lengths(target, span), (span.stop - span.start,)))
                    ]
                )
                for address, length, offset in zip(
                    local_addresses(target, (target_block,), span)[0],
                    np.broadcast_to(local_lengths(target, span), (span.stop - span.start,)),
                    remote["offsets"][span],
                    strict=True,
                ):
                    ctypes.memmove(int(address), payload[int(offset) : int(offset + length)], int(length))
            assert all(
                torch.equal(actual, value)
                for actual, value in zip(logical_values(consumer, target_block), expected, strict=True)
            )
            assert all(
                torch.equal(actual, value) for actual, value in zip(logical_values(consumer, 1), untouched, strict=True)
            )


@pytest.mark.parametrize("axis_order", tuple(permutations(range(3))))
@pytest.mark.parametrize("partitions", (1, 2))
def test_axis_order_and_padding_determine_minimum_selected_intervals(axis_order, partitions):
    case = make_case(layers=(0,), block_size=4, heads=4, vector=2, head_slices=partitions)
    shape = (4, 4, 2)
    strides = [0] * 3
    pitch = 1
    for axis in axis_order:
        strides[axis] = pitch
        pitch *= shape[axis]
    # A real gap between logical blocks, with a nonzero view offset.
    storage = torch.full((7 + 3 * (pitch + 5),), -999, dtype=torch.int32)
    tensor = torch.as_strided(storage, (3, *shape), (pitch + 5, *strides), 7)
    tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape))
    case.caches = {"layer0": tensor}
    _, local, _ = build_mappings(case.topology, case.caches, group_id=3)
    for head, segments in enumerate(local.values()):
        for block in (0, 2):
            selected = tensor[block, :, head * (4 // partitions) : (head + 1) * (4 // partitions)]
            addresses = sorted(
                selected[index].data_ptr() for index in product(*(range(size) for size in selected.shape))
            )
            minimum = 1 + sum(right != left + 4 for left, right in zip(addresses, addresses[1:]))
            assert len(segments["bases"]) == minimum
            assert read_local(segments, block) == b"".join(ctypes.string_at(address, 4) for address in addresses)


@pytest.mark.parametrize("axis_order", tuple(permutations(range(5))))
@pytest.mark.parametrize("partitions", (1, 2, 4))
@pytest.mark.parametrize("kernels", (1, 2))
@pytest.mark.parametrize("padding", (0, 3))
def test_component_axis_orders_preserve_bytes_and_tp_destinations(axis_order, partitions, kernels, padding):
    def make_views(heads, fill):
        shape = (3 * kernels, 2, 2, heads, 2)  # Kernel Block, component, Token, head, head_size.
        strides = [0] * len(shape)
        size = 1
        for position, axis in enumerate(axis_order):
            strides[axis] = size
            size *= shape[axis]
            if position == 1:
                size += padding
        storage = torch.full((size,), -999, dtype=torch.int32)
        tensor = torch.as_strided(storage, shape, strides)
        if fill:
            tensor.copy_(torch.arange(tensor.numel()).reshape(shape))
        return tensor[:, 0], tensor[:, 1]

    case = make_case(layers=(0,), block_size=2 * kernels, kernels=kernels, heads=4, vector=2, head_slices=partitions)
    source = make_views(4, True)
    _, local, _ = build_mappings(case.topology, {"layer0": source}, group_id=3)
    for head, segments in enumerate(local.values()):
        target_case = make_case(
            layers=(0,), block_size=2 * kernels, kernels=kernels, heads=4 // partitions, vector=2, head_slices=1
        )
        target = make_views(4 // partitions, False)
        _, target_local, _ = build_mappings(target_case.topology, {"layer0": target}, group_id=3)
        target_segments = next(iter(target_local.values()))
        for block in (0, 1, 2):
            selected = tuple(
                tensor[
                    block * kernels : (block + 1) * kernels,
                    :,
                    head * (4 // partitions) : (head + 1) * (4 // partitions),
                ]
                for tensor in source
            )
            addresses = sorted(
                tensor[index].data_ptr()
                for tensor in selected
                for index in product(*(range(size) for size in tensor.shape))
            )
            minimum = 1 + sum(right != left + 4 for left, right in zip(addresses, addresses[1:]))
            assert len(segments["bases"]) == minimum
            payload = read_local(segments, block)
            assert payload == b"".join(ctypes.string_at(address, 4) for address in addresses)
            target_block = 2 - block
            offset = 0
            for address, length in zip(
                local_addresses(target_segments, (target_block,))[0],
                np.broadcast_to(local_lengths(target_segments), target_segments["bases"].shape),
                strict=True,
            ):
                length = int(length)
                ctypes.memmove(int(address), payload[offset : offset + length], length)
                offset += length
            assert offset == len(payload)
            for tensor, expected in zip(target, selected, strict=True):
                assert torch.equal(tensor[target_block * kernels : (target_block + 1) * kernels], expected)


def test_adjacent_components_with_different_block_strides_do_not_merge():
    case = make_case(layers=(0,), block_size=2, heads=1, vector=1, head_slices=1)
    storage = torch.arange(64, dtype=torch.int32)
    left = torch.as_strided(storage, (3, 2, 1, 1), (12, 1, 1, 1), 0)
    right = torch.as_strided(storage, (3, 2, 1, 1), (16, 1, 1, 1), 2)
    _, local, _ = build_mappings(case.topology, {"layer0": (left, right)}, group_id=3)
    segments = next(iter(local.values()))
    assert len(segments["bases"]) == 2
    for block in (0, 2):
        assert read_local(segments, block) == left[block].numpy().tobytes() + right[block].numpy().tobytes()


def test_adjacent_first_head_is_not_enough_to_merge_partition_geometry():
    case = make_case(layers=(0,), block_size=2, heads=2, vector=1, head_slices=2)
    storage = torch.arange(96, dtype=torch.int32)
    left = torch.as_strided(storage, (3, 2, 2, 1), (32, 1, 4, 1), 0)
    right = torch.as_strided(storage, (3, 2, 2, 1), (32, 1, 6, 1), 2)
    _, local, _ = build_mappings(case.topology, {"layer0": (left, right)}, group_id=3)
    for head, segments in enumerate(local.values()):
        assert len(segments["bases"]) == 2
        for block in (0, 2):
            assert (
                read_local(segments, block)
                == left[block, :, head].numpy().tobytes() + right[block, :, head].numpy().tobytes()
            )


@pytest.mark.parametrize(
    "components,expected_segments",
    (
        pytest.param(
            ((0, (2, 16), (64, 1)), (16, (2, 2, 8), (64, 16, 1)), (24, (2, 2, 8), (64, 16, 1))),
            2,
            id="nested-component-repetition",
        ),
        pytest.param(((0, (2, 4), (16, 1)), (4, (12,), (1,))), 1, id="different-pattern-seams"),
        pytest.param(((0, (2, 4), (8, 1)), (12, (2, 4), (8, 1))), 3, id="single-layout-seams"),
        pytest.param(((0, (3, 4), (8, 1)), (12, (3, 4), (8, 1))), 3, id="interleaved-streams"),
        pytest.param(((0, (4,), (1,)), (8, (4,), (1,)), (16, (4,), (1,))), 3, id="extend-component-axis"),
        pytest.param(((0, (4,), (1,)), (4, (4,), (1,)), (8, (4,), (1,))), 1, id="three-contiguous-components"),
        pytest.param(((0, (4,), (1,)), (8, (4,), (1,)), (20, (4,), (1,))), 3, id="uneven-component-offsets"),
        pytest.param(((0, (2, 16), (24, 1)), (16, (2, 8), (24, 1))), 1, id="unequal-component-widths"),
        pytest.param(((0, (2, 4), (8, 1)), (16, (3, 4), (8, 1))), 5, id="extend-strided-content-axis"),
    ),
)
def test_shared_storage_components_preserve_bytes_across_repetition_patterns(components, expected_segments):
    case = make_case(layers=(0,), block_size=1, heads=1, vector=1, head_slices=1)
    block_stride = 160
    storage = torch.arange(3 * block_stride).remainder(251).to(torch.uint8)
    tensors = tuple(
        torch.as_strided(storage, (3, *shape), (block_stride, *strides), offset)
        for offset, shape, strides in components
    )
    case.topology.groups[0].kv_cache_spec = MambaSpec(
        block_size=1,
        shapes=tuple(shape for _, shape, _ in components),
        dtypes=(torch.uint8,) * len(components),
        page_size_padded=block_stride,
    )
    _, local, _ = build_mappings(case.topology, {"layer0": tensors}, group_id=3)
    segments = next(iter(local.values()))
    assert len(segments["bases"]) == expected_segments
    for block in (0, 2):
        addresses = sorted(
            tensor[block][index].data_ptr()
            for tensor in tensors
            for index in product(*(range(size) for size in tensor.shape[1:]))
        )
        assert len(set(addresses)) == len(addresses)
        assert read_local(segments, block) == b"".join(ctypes.string_at(address, 1) for address in addresses)
