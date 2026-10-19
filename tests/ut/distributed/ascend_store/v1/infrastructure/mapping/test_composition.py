"""Each access mode binds only its own capabilities, once, without adapter views."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import make_case
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import BoundLayerwiseKeys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import (
    BulkMapping,
    LayerwiseMapping,
    bulk,
    layerwise,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.object_layout import (
    ObjectLayout,
)


@pytest.mark.parametrize("module,mapping_type", ((bulk, BulkMapping), (layerwise, LayerwiseMapping)))
def test_custom_names_bind_once_to_coordinates_without_encoding_default_prefixes(module, mapping_type):
    case = make_case(head_slices=2, pp_layer_counts=(1, 3), padding=1)
    coordinates: list[tuple[int, int]] = []

    def custom_names(group_id, positions, *, lookup_head_count):
        assert group_id == 3 and lookup_head_count == 4
        coordinates.extend((position.pp_rank, position.head_rank) for position in positions)
        parts = tuple((f"stage{pp}@head{head}@", "@custom") for pp, head in coordinates)
        return BoundLayerwiseKeys(parts, parts)

    builder = SimpleNamespace(bind=Mock(side_effect=custom_names))
    other_type = LayerwiseMapping if mapping_type is BulkMapping else BulkMapping
    with (
        patch.object(module, "bind_local_layout", wraps=module.bind_local_layout) as bind_local,
        patch.object(PoolKey, "to_string", side_effect=AssertionError("unused canonical codec")),
        patch.object(module, "bind_canonical_keys", side_effect=AssertionError("unused default naming")),
        patch.object(other_type, "__init__", side_effect=AssertionError("unused access mode")),
    ):
        mapping = mapping_type(case.topology, *case.registration_arrays, kv_caches=case.caches, key_builder=builder)
        keys = mapping.keys((b"a", "b"))
        for _ in range(2):
            assert mapping.lookup_keys((b"a", "b")) == keys
            if mapping_type is LayerwiseMapping:
                selection = mapping.prepare(keys, (0, 2))
                for layer_id in mapping.local.physical_layer_ids:
                    mapping.arguments(selection, layer_id=layer_id)
            else:
                mapping.arguments(keys, (0, 2))
    assert coordinates == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert keys[0] == ("stage0@head0@61@custom", "stage0@head0@b@custom")
    bind_local.assert_called_once()
    builder.bind.assert_called_once()
    assert not hasattr(mapping, "binding")
    if mapping_type is LayerwiseMapping:
        for layouts in mapping.layer_layouts.values():
            for layout in layouts:
                original = mapping.local.objects[layout.object_index].memory_segments
                assert np.shares_memory(
                    layout.local_memory_segments.block_zero_addresses, original.block_zero_addresses
                )


def test_custom_object_layout_replaces_default_and_is_not_rebound_during_traffic():
    case = make_case(layers=(0, 1), heads=1)

    def padded_layout(local):
        lengths = local.objects[0].memory_segments.segment_lengths_bytes
        offsets = np.asarray([64, *[64 + int(value) for value in np.cumsum(lengths + 8)]], dtype=np.uint64)
        offsets.flags.writeable = False
        return (ObjectLayout(int(offsets[-1]), offsets),)

    build_objects = Mock(side_effect=padded_layout)
    with patch.object(
        layerwise, "build_default_object_layouts", side_effect=AssertionError("unused default placement")
    ):
        layers = case.bind_layers(object_layout_factory=build_objects)
        local = layers.local.objects[0]
        offsets = layers.object_layouts[0].segment_offsets_bytes
        lengths = local.memory_segments.segment_lengths_bytes
        selection = layers.prepare(layers.keys(("a",)), (2,))
        for _ in range(2):
            for index, layer_id in enumerate(local.physical_layer_ids):
                _, addresses, sizes, positions = layers.arguments(selection, layer_id=layer_id)
                start, end = local.layer_segment_boundaries[index : index + 2]
                assert positions == [offsets[start:end].tolist()]
                assert sizes == [lengths[start:end].tolist()]
                assert addresses[0] == [
                    int(base) + 2 * int(stride)
                    for base, stride in zip(
                        local.memory_segments.block_zero_addresses[start:end],
                        np.broadcast_to(local.memory_segments.group_block_strides_bytes, lengths.shape)[start:end],
                        strict=True,
                    )
                ]
    build_objects.assert_called_once_with(layers.local)


@pytest.mark.parametrize("phase", ("naming", "object_layout"))
def test_custom_binding_failure_propagates_without_falling_back(phase):
    case = make_case()
    failure = Mock(side_effect=RuntimeError("wire binding failed"))
    names = BoundLayerwiseKeys((("custom@", ""),), (("custom@", ""),))
    builder = SimpleNamespace(bind=failure if phase == "naming" else Mock(return_value=names))
    with (
        patch.object(layerwise, "bind_canonical_keys", side_effect=AssertionError("unexpected default naming")),
        patch.object(
            layerwise, "build_default_object_layouts", side_effect=AssertionError("unexpected default placement")
        ),
        pytest.raises(RuntimeError, match="wire binding failed"),
    ):
        case.bind_layers(key_builder=builder, object_layout_factory=failure if phase == "object_layout" else None)
    failure.assert_called_once()


def test_bulk_does_not_initialize_layerwise_object_offsets():
    case = make_case()
    with patch.object(np, "cumsum", side_effect=AssertionError("unused layer offsets")):
        mapping = case.bind()
        mapping.arguments(mapping.keys(("a",)), (1,))
