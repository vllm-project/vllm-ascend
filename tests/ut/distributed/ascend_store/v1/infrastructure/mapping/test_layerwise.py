"""Complete-object layer layer_layouts, CPU byte equivalence and static-work reuse."""

import ctypes
import itertools
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import (
    expected_group_bytes,
    expected_head_bytes,
    make_case,
    make_hybrid_case,
    read_bytes,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk.argument_fixtures import bulk_arguments
from tests.ut.distributed.ascend_store.v1.worker.layerwise.plan_fixtures import layer_arguments, prepare_layer_group
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import (
    layerwise,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.local_layout import head_slicing


def assemble_layers(mapping, selection, object_sizes):
    """Emulate object-offset writes, detecting overlap and incomplete coverage."""
    payloads = {key: bytearray(size) for key, size in object_sizes.items()}
    coverage = {key: bytearray(size) for key, size in object_sizes.items()}
    for layer_id in reversed(mapping.layer_layouts):  # Offsets, not call order, locate the bytes.
        keys, addresses, lengths, offsets = layer_arguments(mapping, selection, layer_id=layer_id)
        for key, sources, sizes, starts in zip(keys, addresses, lengths, offsets, strict=True):
            for source, size, start in zip(sources, sizes, starts, strict=True):
                end = start + size
                assert 0 <= start < end <= object_sizes[key]
                assert not any(coverage[key][start:end])
                payloads[key][start:end] = ctypes.string_at(source, size)
                coverage[key][start:end] = b"\1" * size
    assert all(all(mask) for mask in coverage.values())
    return {key: bytes(payload) for key, payload in payloads.items()}


class LayerwiseMappingTests(unittest.TestCase):
    def test_layer_layout_distinguishes_local_segments_from_object_byte_offsets(self):
        case = make_case(layers=(0,), block_size=2, heads=4, vector=1, head_slices=2)
        bulk = case.bind()
        mapping = case.bind_layers(group_id=bulk.local_layout.group_id)
        self.assertFalse(hasattr(layerwise, "LayerwiseView"))
        self.assertFalse(hasattr(layerwise, "LayerRange"))
        keys = bulk.key_namer.keys(["a", "a"])
        selection = prepare_layer_group(mapping, keys, [1, 1])
        self.assertEqual(len(selection.selected_keys), 4)  # Duplicate keys are still distinct input rows.
        for prefix_index, layout in enumerate(mapping.layer_layouts[0]):
            self.assertEqual(layout.partition_index, prefix_index)
            self.assertEqual(
                tuple(obj.layer_segment_index_boundaries for obj in bulk.local_layout.partitions)[prefix_index], (0, 4)
            )
            self.assertEqual(layout.remote_object_size_bytes, 32)
            self.assertEqual(layout.remote_segment_offsets_bytes.tolist(), [0, 8, 16, 24])
            segments = layout.local_segments
            self.assertEqual(segments.segment_lengths_bytes.tolist(), [8])
            self.assertEqual(segments.group_block_strides_bytes.tolist(), [32])
            k_base, v_base = case.registration_arrays[0][3]
            self.assertEqual(
                segments.block_zero_segment_addresses.tolist(),
                [base + prefix_index * 8 + token * 16 for base in (k_base, v_base) for token in range(2)],
            )
        peer = case.bind_layers(group_id=bulk.local_layout.group_id)
        self.assertEqual(layer_arguments(peer, selection, layer_id=0), layer_arguments(mapping, selection, layer_id=0))

    def test_layers_reassemble_exact_bulk_and_tensor_bytes(self):
        comparisons = 0
        for layout, head_slices, padding, kernels, pp in itertools.product(
            ("NHD", "HND", "LBHNC"), (1, 2, 4), (0, 1), (1, 2), (False, True)
        ):
            case = make_case(
                layout=layout,
                head_slices=head_slices,
                padding=padding,
                kernels=kernels,
                pp_layer_counts=(1, 3) if pp else None,
            )
            bulk = case.bind()
            layers = case.bind_layers(group_id=bulk.local_layout.group_id)
            self.assertIsNot(layers.local_layout, bulk.local_layout)
            ids = [2, 0, 2, 1]
            keys = bulk.key_namer.keys(["a", "b", "c", "d"])
            for mask_kind in ("none", "all", "empty", "mixed"):
                with self.subTest(
                    layout=layout, head_slices=head_slices, padding=padding, kernels=kernels, pp=pp, mask=mask_kind
                ):
                    mask = (
                        None
                        if mask_kind == "none"
                        else [
                            mask_kind == "all" or (mask_kind == "mixed" and index % 3 == 1)
                            for index in range(len(keys) * len(ids))
                        ]
                    )
                    actual_keys, addresses, lengths = bulk_arguments(bulk, keys, ids, selected_keys=mask)
                    expected = {
                        keys[prefix_index][row]: expected_head_bytes(
                            case, physical_layers, prefix_index % head_slices, block_id
                        )
                        for prefix_index, physical_layers in enumerate(
                            tuple(obj.physical_layer_ids for obj in bulk.local_layout.partitions)
                        )
                        for row, block_id in enumerate(ids)
                        if mask is None or mask[prefix_index * len(ids) + row]
                    }
                    self.assertEqual(
                        dict(zip(actual_keys, (read_bytes(a, n) for a, n in zip(addresses, lengths, strict=True)))),
                        expected,
                    )
                    selection = prepare_layer_group(layers, keys, ids, selected_keys=mask)
                    self.assertEqual(len(selection.selected_keys), len(actual_keys))
                    self.assertEqual(
                        assemble_layers(layers, selection, {k: len(v) for k, v in expected.items()}), expected
                    )
                    for layer_id in layers.layer_layouts:
                        layer_keys, sources, sizes, _ = layer_arguments(layers, selection, layer_id=layer_id)
                        selected = [
                            (prefix_index, row)
                            for prefix_index, physical_layers in enumerate(
                                tuple(obj.physical_layer_ids for obj in bulk.local_layout.partitions)
                            )
                            if layer_id in physical_layers
                            for row in range(len(ids))
                            if mask is None or mask[prefix_index * len(ids) + row]
                        ]
                        self.assertEqual(layer_keys, [keys[p][r] for p, r in selected])
                        self.assertEqual(
                            [read_bytes(a, n) for a, n in zip(sources, sizes, strict=True)],
                            [expected_head_bytes(case, (layer_id,), p % head_slices, ids[r]) for p, r in selected],
                        )
                    comparisons += 1
        self.assertEqual(comparisons, 288)

    def test_target_pp_objects_start_at_zero_but_layer_selection_preserves_offsets(self):
        for head_slices in (None, 1, 2):
            case = make_case(layers=(4, 5, 6, 7), head_slices=head_slices, block_size=96, stored_tokens=(3, 4, 3, 4))
            whole = case.bind()
            whole_layers = case.bind_layers()
            case.topology.consumer_pipeline_partitions = (5, 0, 3)
            split = case.bind()
            split_layers = case.bind_layers()
            self.assertEqual(
                tuple(obj.physical_layer_ids for obj in split.local_layout.partitions),
                ((4,),) * case.head_slices + ((5, 6, 7),) * case.head_slices,
            )
            for bound, layer_map in ((whole, whole_layers), (split, split_layers)):
                for prefix_index, (ids, boundaries, segments) in enumerate(
                    zip(
                        tuple(obj.physical_layer_ids for obj in bound.local_layout.partitions),
                        tuple(obj.layer_segment_index_boundaries for obj in bound.local_layout.partitions),
                        tuple(obj.local_segments for obj in bound.local_layout.partitions),
                        strict=True,
                    )
                ):
                    self.assertEqual(len(boundaries), len(ids) + 1)
                    self.assertEqual(boundaries[0], 0)
                    self.assertEqual(boundaries[-1], len(segments.block_zero_segment_addresses))
                    for index, layer_id in enumerate(ids):
                        layer_layout = next(
                            v for v in layer_map.layer_layouts[layer_id] if v.partition_index == prefix_index
                        )
                        preceding = expected_head_bytes(case, ids[:index], prefix_index % case.head_slices, 0)
                        self.assertEqual(int(layer_layout.remote_segment_offsets_bytes[0]), len(preceding))
                        self.assertEqual(
                            layer_layout.remote_object_size_bytes,
                            tuple(obj.payload_bytes for obj in bound.local_layout.partitions)[prefix_index],
                        )
            self.assertGreater(int(whole_layers.layer_layouts[5][0].remote_segment_offsets_bytes[0]), 0)
            self.assertEqual(int(split_layers.layer_layouts[5][0].remote_segment_offsets_bytes[0]), 0)
            self.assertEqual(
                int(split_layers.layer_layouts[6][0].remote_segment_offsets_bytes[0]),
                len(expected_head_bytes(case, (5,), 0, 0)),
            )

    def test_offsets_and_layer_layouts_share_static_arrays_without_rebinding(self):
        for padding in (0, 1):
            case = make_case(head_slices=4, padding=padding, pp_layer_counts=(1, 3))
            bulk = case.bind()
            layouts = {
                (id(r.segment_lengths_bytes), len(r.block_zero_segment_addresses))
                for r in tuple(obj.local_segments for obj in bulk.local_layout.partitions)
            }
            with (
                patch.object(head_slicing, "inspect_entries", wraps=head_slicing.inspect_entries) as inspect,
                patch.object(head_slicing, "compile_geometries", wraps=head_slicing.compile_geometries) as compile,
                patch.object(np, "cumsum", wraps=np.cumsum) as scans,
            ):
                mapping = case.bind_layers(group_id=bulk.local_layout.group_id)
            self.assertEqual(scans.call_count, len(layouts))
            inspect.assert_called_once()
            compile.assert_called_once()
            for layer_layouts in mapping.layer_layouts.values():
                for layer_layout in layer_layouts:
                    source = tuple(obj.local_segments for obj in mapping.local_layout.partitions)[
                        layer_layout.partition_index
                    ]
                    for name in ("block_zero_segment_addresses", "group_block_strides_bytes", "segment_lengths_bytes"):
                        array = getattr(layer_layout.local_segments, name)
                        self.assertTrue(np.shares_memory(array, getattr(source, name)))
                        self.assertFalse(array.flags.writeable)
                    self.assertFalse(layer_layout.remote_segment_offsets_bytes.flags.writeable)
                    self.assertTrue(
                        np.shares_memory(
                            layer_layout.remote_segment_offsets_bytes, layer_layouts[0].remote_segment_offsets_bytes
                        )
                    )
            with self.assertRaises(TypeError):
                mapping.layer_layouts[100] = ()
            selection = prepare_layer_group(mapping, bulk.key_namer.keys(["a", "b"]), [0, 2])
            self.assertIs(
                selection.selected_local_block_ids_by_partition[0], selection.selected_local_block_ids_by_partition[1]
            )
            with (
                patch.object(np, "cumsum", side_effect=AssertionError("runtime offsets")),
                patch.object(np, "flatnonzero", side_effect=AssertionError("runtime selection")),
                patch.object(np, "count_nonzero", side_effect=AssertionError("runtime mask scan")),
            ):
                for layer_id in mapping.layer_layouts:
                    layer_arguments(mapping, selection, layer_id=layer_id)

    def test_bulk_does_not_build_object_offsets(self):
        for head_slices, pp in itertools.product((None, 1, 2), (None, (1, 3))):
            with patch.object(np, "cumsum", side_effect=AssertionError("Bulk has no use for byte offsets")):
                make_case(head_slices=head_slices, pp_layer_counts=pp).bind()

    def test_layer_evaluation_visits_only_its_key_prefixes(self):
        class CountedPrefixSelections(tuple):
            visits: list[int]

            def __new__(cls, values):
                result = super().__new__(cls, values)
                result.visits = []
                return result

            def __getitem__(self, index):
                self.visits.append(index)
                return super().__getitem__(index)

        case = make_case(head_slices=2, pp_layer_counts=(1, 3))
        bulk = case.bind()
        mapping = case.bind_layers(group_id=bulk.local_layout.group_id)
        selection = prepare_layer_group(mapping, bulk.key_namer.keys(["a", "b"]), [0, 1])
        prefix_selections = CountedPrefixSelections(selection.selected_local_block_ids_by_partition)
        selection = replace(selection, selected_local_block_ids_by_partition=prefix_selections)
        with patch.object(type(bulk), "__init__", side_effect=AssertionError("no all-layer binding")):
            layer_arguments(mapping, selection, layer_id=0)
            self.assertEqual(prefix_selections.visits, [0, 1])
            prefix_selections.visits.clear()
            layer_arguments(mapping, selection, layer_id=3)
            self.assertEqual(prefix_selections.visits, [2, 3])

    def test_ordinary_segments_preserve_non_token_divisible_lengths(self):
        for pp_layer_counts in (None, (1, 3)):
            case = make_case(pp_layer_counts=pp_layer_counts)
            case.registration_arrays[1][3][:] = [length - 1 for length in case.registration_arrays[1][3]]
            bulk = case.bind()
            mapping = case.bind_layers(group_id=bulk.local_layout.group_id)
            keys = bulk.key_namer.keys(["a", "b"])
            names, addresses, lengths = bulk_arguments(bulk, keys, [0, 2])
            expected = {key: read_bytes(a, n) for key, a, n in zip(names, addresses, lengths, strict=True)}
            selection = prepare_layer_group(mapping, keys, [0, 2])
            self.assertEqual(assemble_layers(mapping, selection, {k: len(v) for k, v in expected.items()}), expected)

    def test_empty_layer_bounds_and_absent_or_unselected_work_exit_early(self):
        for head_slices in (None, 1, 2):
            case = make_case(head_slices=head_slices, pp_layer_counts=(2, 2))
            case.caches["layer1"] = (torch.empty(0),)
            for array in case.registration_arrays[:3]:
                del array[3][2:4]
            case.registration_arrays[3][3] = [0, 2, 2, 4, 6]
            bulk = case.bind()
            mapping = case.bind_layers(group_id=bulk.local_layout.group_id)
            self.assertNotIn(1, mapping.layer_layouts)
            self.assertEqual(
                tuple(obj.layer_segment_index_boundaries for obj in bulk.local_layout.partitions)[0][-1],
                tuple(obj.layer_segment_index_boundaries for obj in bulk.local_layout.partitions)[0][-2],
            )
            keys = bulk.key_namer.keys(["a"])
            all_rows = prepare_layer_group(mapping, keys, [0])
            no_rows = prepare_layer_group(mapping, keys, [0], selected_keys=[False] * len(keys))
            empty_rows = prepare_layer_group(mapping, bulk.key_namer.keys([]), [])
            only_pp0 = prepare_layer_group(
                mapping, keys, [0], selected_keys=[p < case.head_slices for p in range(len(keys))]
            )
            with (
                patch.object(np, "broadcast_to", side_effect=AssertionError("unused segments")),
                patch.object(np, "empty", side_effect=AssertionError("unused allocation")),
            ):
                for selection, layer_id in (
                    (all_rows, 1),
                    (all_rows, 999),
                    (no_rows, 0),
                    (empty_rows, 0),
                    (only_pp0, 2),
                ):
                    self.assertEqual(layer_arguments(mapping, selection, layer_id=layer_id), ([], [], [], []))

    def test_preparation_snapshots_inputs_and_outputs_do_not_alias(self):
        case = make_case(head_slices=2)
        bulk = case.bind()
        mapping = case.bind_layers(group_id=bulk.local_layout.group_id)
        keys = [list(prefix_keys) for prefix_keys in bulk.key_namer.keys(["same", "other", "same"])]
        ids = np.array([2, 1, 2], dtype=np.uint64)
        mask = [True, False, True, True, True, True]
        selection = prepare_layer_group(mapping, keys, ids, selected_keys=mask)
        expected = layer_arguments(mapping, selection, layer_id=0)
        ids[:] = 0
        keys[0][0] = "mutated"
        mask[:] = [False] * len(mask)
        self.assertEqual(layer_arguments(mapping, selection, layer_id=0), expected)
        self.assertEqual(expected[0][0], expected[0][1])  # Duplicate input selection are not silently deduplicated.
        self.assertEqual(expected[1][0], expected[1][1])
        for ids in selection.selected_local_block_ids_by_partition:
            self.assertFalse(ids.flags.writeable)
        result = layer_arguments(mapping, selection, layer_id=0)
        for field in result[1:]:
            first = field[0][0]
            field[0][0] = first + 1
            self.assertEqual(field[1][0], first)
        self.assertEqual(layer_arguments(mapping, selection, layer_id=0), expected)

    def test_hybrid_groups_have_independent_rows_and_complete_mixed_dtype_payloads(self):
        case = make_hybrid_case(pp_layer_counts=(2, 2), tp_rank=3)
        for group_id, ids in ((3, [2, 0]), (9, [1, 2, 0])):
            bulk = case.bind(group_id=group_id)
            mapping = case.bind_layers(group_id=bulk.local_layout.group_id)
            keys = bulk.key_namer.keys([str(i) for i in range(len(ids))])
            selection = prepare_layer_group(mapping, keys, ids)
            expected = {
                keys[p][i]: expected_group_bytes(case, group_id, layer_ids, block_id)
                for p, layer_ids in enumerate(tuple(obj.physical_layer_ids for obj in bulk.local_layout.partitions))
                for i, block_id in enumerate(ids)
            }
            self.assertEqual(assemble_layers(mapping, selection, {k: len(v) for k, v in expected.items()}), expected)

    def test_bulk_layerwise_round_trip_across_tp_pp_and_replicas(self):
        for layout, replicas in itertools.product(("NHD", "HND", "LBHNC"), (1, 2)):
            producer_heads = 2 if replicas == 1 else 1
            producers = [
                make_case(
                    layers=(pp * 2, pp * 2 + 1),
                    head_slices=1,
                    heads=producer_heads,
                    tp_rank=rank,
                    replicas=replicas,
                    pp_rank=pp,
                    layout=layout,
                )
                for pp in range(2)
                for rank in range(4 * replicas)
            ]
            consumers = [
                make_case(
                    head_slices=2,
                    pp_layer_counts=(2, 2),
                    heads=producer_heads * 2,
                    tp_rank=rank,
                    pp_size=1,
                    layout=layout,
                )
                for rank in range(2)
            ]
            bindings = {}
            originals = {}
            for case in producers + consumers:
                bulk = case.bind()
                bindings[id(case)] = bulk, case.bind_layers(group_id=bulk.local_layout.group_id)
                group = case.topology.transfer_groups[0]
                for layer in group.layers:
                    for kind, tensor in enumerate(case.caches[layer.layer_names[0]]):
                        if not tensor.numel():
                            continue
                        head_axis = 1 if layout in ("HND", "LBHNC") else 2
                        heads = tensor.shape[head_axis]
                        for head in range(heads):
                            value = (
                                layer.physical_layer_id * 1000
                                + kind * 100
                                + group.key_metadata.head_or_tp_rank * heads
                                + head
                            )
                            (tensor[:, head] if head_axis == 1 else tensor[:, :, head]).fill_(value)
                        originals[tensor.data_ptr()] = tensor.clone()
            for sources, targets in ((producers, consumers), (consumers, producers)):
                for source_layerwise in (False, True):
                    objects: dict[str, bytes] = {}
                    for case in sources:
                        bulk, mapping = bindings[id(case)]
                        keys = bulk.key_namer.keys(["a", "b"])
                        if source_layerwise:
                            sizes = {
                                key: size
                                for prefix_keys, size in zip(
                                    keys, tuple(obj.payload_bytes for obj in bulk.local_layout.partitions), strict=True
                                )
                                for key in prefix_keys
                            }
                            payloads = assemble_layers(mapping, prepare_layer_group(mapping, keys, [1, 2]), sizes)
                        else:
                            names, addresses, lengths = bulk_arguments(bulk, keys, [1, 2])
                            payloads = {
                                key: read_bytes(a, n) for key, a, n in zip(names, addresses, lengths, strict=True)
                            }
                        for key, value in payloads.items():
                            self.assertEqual(objects.setdefault(key, value), value)
                    for case in targets:
                        for tensors in case.caches.values():
                            for tensor in tensors:
                                tensor.zero_()
                        bulk, mapping = bindings[id(case)]
                        keys = bulk.key_namer.keys(["a", "b"])
                        if source_layerwise:
                            names, addresses, lengths = bulk_arguments(bulk, keys, [1, 2])
                            for key, destinations, sizes in zip(names, addresses, lengths, strict=True):
                                cursor = 0
                                for address, size in zip(destinations, sizes, strict=True):
                                    ctypes.memmove(address, objects[key][cursor : cursor + size], size)
                                    cursor += size
                        else:
                            selection = prepare_layer_group(mapping, keys, [1, 2])
                            for layer_id in mapping.layer_layouts:
                                names, addresses, lengths, offsets = layer_arguments(
                                    mapping, selection, layer_id=layer_id
                                )
                                for key, destinations, sizes, starts in zip(
                                    names, addresses, lengths, offsets, strict=True
                                ):
                                    for address, size, start in zip(destinations, sizes, starts, strict=True):
                                        ctypes.memmove(address, objects[key][start : start + size], size)
                        for tensors in case.caches.values():
                            for tensor in tensors:
                                if tensor.numel():
                                    self.assertTrue(torch.equal(tensor[1:], originals[tensor.data_ptr()][1:]))


if __name__ == "__main__":
    unittest.main()
