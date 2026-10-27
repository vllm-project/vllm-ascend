"""Mapping contracts and CPU byte oracles; no device transfer or Worker integration."""

import ctypes
import itertools
import unittest
from dataclasses import fields
from unittest.mock import patch

import numpy as np
import torch
from numpy.typing import NDArray

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import (
    expected_group_bytes,
    expected_head_bytes,
    make_case,
    make_hybrid_case,
    read_bytes,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk.argument_fixtures import bulk_arguments
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import mapping
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.local_layout import (
    head_slicing,
    layer_partitioning,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk.transfer import bind_bulk_sizes


class MappingTests(unittest.TestCase):
    def test_four_routes_share_one_key_to_byte_contract(self):
        for head_slices, pp in itertools.product((None, 2), (False, True)):
            with self.subTest(head_slices=head_slices, pp=pp):
                case = make_case(head_slices=head_slices, pp_layer_counts=(2, 2) if pp else None, tp_rank=1)
                bound = case.bind()
                heads = head_slices or 1
                self.assertEqual(len(bound.key_namer.key_prefixes), heads * (2 if pp else 1))

                if pp:
                    self.assertEqual(
                        tuple(obj.physical_layer_ids for obj in bound.local_layout.partitions),
                        ((0, 1),) * heads + ((2, 3),) * heads,
                    )
                else:
                    self.assertEqual(
                        tuple(obj.physical_layer_ids for obj in bound.local_layout.partitions), ((0, 1, 2, 3),) * heads
                    )
                self.assertEqual(len(bound.key_namer.lookup_keys(["hash"])), case.topology.pp_size * 4)
                for pp_rank in range(2 if pp else 1):
                    for head in range(heads):
                        prefix = bound.key_namer.key_prefixes[pp_rank * heads + head]
                        self.assertIn(f"@pp_rank:{pp_rank}@", prefix)
                        self.assertIn(f"@head_or_tp_rank:{heads + head}@", prefix)

    def test_local_transfer_and_lookup_use_target_pp_layers_not_execution_pp(self):
        case = make_case(head_slices=2, pp_size=1)
        original = case.bind()
        case.topology.consumer_pipeline_partitions = (2, 2)
        combined = case.bind()
        self.assertEqual(len(original.key_namer.lookup_keys(["a"])), 4)
        self.assertEqual(len(combined.key_namer.lookup_keys(["a"])), 8)
        self.assertEqual(
            tuple(obj.physical_layer_ids for obj in combined.local_layout.partitions), ((0, 1),) * 2 + ((2, 3),) * 2
        )
        for key, layers, size in zip(
            combined.key_namer.key_prefixes,
            tuple(obj.physical_layer_ids for obj in combined.local_layout.partitions),
            tuple(obj.payload_bytes for obj in combined.local_layout.partitions),
            strict=True,
        ):
            pp_rank = 0 if layers == (0, 1) else 1
            self.assertIn(f"@pp_rank:{pp_rank}@", key)
            self.assertEqual(size, tuple(obj.payload_bytes for obj in original.local_layout.partitions)[0] // 2)
        required = [prefix_keys[0] for prefix_keys in combined.key_namer.lookup_keys(["a"])]
        present = set(prefix_keys[0] for prefix_keys in original.key_namer.lookup_keys(["a"]))
        self.assertFalse(all(key in present for key in required))  # Missing PP1 cannot be a complete hit.

    def test_replicas_share_keys_but_bind_their_own_memory(self):
        for head_slices, pp in itertools.product((None, 1), (False, True)):
            # Normalized topology facts: two replicas for each unique local Head shard.
            # Mapping does not infer peer TP or change the production admission gates.
            cases = [
                make_case(
                    head_slices=head_slices, pp_layer_counts=(2, 2) if pp else None, tp_rank=rank, replicas=2, heads=1
                )
                for rank in (2, 3)
            ]
            left, right = (case.bind() for case in cases)
            self.assertEqual(left.key_namer.keys(["hash"]), right.key_namer.keys(["hash"]))
            expected_first_head = head_slices or 1
            self.assertIn(f"@head_or_tp_rank:{expected_first_head}@", left.key_namer.key_prefixes[0])
            self.assertNotEqual(
                bulk_arguments(left, left.key_namer.keys(["h"]), [0])[1],
                bulk_arguments(right, right.key_namer.keys(["h"]), [0])[1],
            )
            lookup = left.key_namer.lookup_keys(["h"])
            self.assertEqual(len(lookup), 8)
            self.assertEqual(len(set(prefix_keys[0] for prefix_keys in lookup)), 8)

    def test_local_prefixes_reuse_canonical_lookup_encoding(self):
        for head_slices, pp_layer_counts in itertools.product((None, 1, 2), (None, (2, 2))):
            case = make_case(head_slices=head_slices, pp_layer_counts=pp_layer_counts, tp_rank=1, pp_rank=1)
            case.topology.dcp_size = 2
            case.topology.transfer_groups[0].key_metadata.dcp_rank = 1
            with patch.object(PoolKey, "to_string", autospec=True, side_effect=PoolKey.to_string) as encode:
                bound = case.bind()
            self.assertEqual(encode.call_count, len(bound.key_namer.lookup_key_prefixes))
            for index, prefix in enumerate(bound.key_namer.key_prefixes):
                pp_rank = index // (head_slices or 1) if pp_layer_counts else 1
                self.assertIn(f"@pp_rank:{pp_rank}@", prefix)
                self.assertIn("@dcp:1@", prefix)
                self.assertIn(f"@head_or_tp_rank:{(head_slices or 1) + index % (head_slices or 1)}@", prefix)
                canonical = next(value for value in bound.key_namer.lookup_key_prefixes if value == prefix)
                self.assertIs(prefix, canonical)

    def test_producer_and_consumer_store_the_same_object_bytes_with_replicas(self):
        for layout, replicas in itertools.product(("NHD", "HND"), (1, 2)):
            # H=8: P(PP2,TP4) -> D(PP1,TP2). H=4: P(PP2,TP8) -> D(PP1,TP2).
            producer_heads = 2 if replicas == 1 else 1
            producer_cases = [
                make_case(
                    layers=tuple(range(pp * 4, pp * 4 + 4)),
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
            consumer_cases = [
                make_case(
                    layers=tuple(range(8)),
                    head_slices=2,
                    pp_layer_counts=(4, 4),
                    heads=producer_heads * 2,
                    tp_rank=rank,
                    pp_size=1,
                    layout=layout,
                )
                for rank in range(2)
            ]
            payloads = []
            for cases in (producer_cases, consumer_cases):
                objects: dict[str, bytes] = {}
                for case in cases:
                    group = case.topology.transfer_groups[0]
                    for layer in group.layers:
                        for kind, tensor in enumerate(case.caches[layer.layer_names[0]]):
                            if not tensor.numel():
                                continue
                            head_axis = 1 if layout == "HND" else 2
                            local_heads = tensor.shape[head_axis]
                            first_head = group.key_metadata.head_or_tp_rank * local_heads
                            for head in range(local_heads):
                                values = tensor[:, head] if layout == "HND" else tensor[:, :, head]
                                values.fill_(layer.physical_layer_id * 1000 + kind * 100 + first_head + head)
                    transfer = case.bind()
                    keys, addresses, lengths = bulk_arguments(
                        transfer, transfer.key_namer.keys(["zero", "one", "two", "repeat"]), [0, 1, 2, 0]
                    )
                    for key, address, length in zip(keys, addresses, lengths, strict=True):
                        value = read_bytes(address, length)
                        if key in objects:
                            self.assertEqual(objects[key], value)  # Replica identity, not extra objects.
                        objects[key] = value
                payloads.append(objects)
            self.assertEqual(len(payloads[0]), 8 * 4)
            self.assertEqual(payloads[0], payloads[1])

    def test_combined_full_object_byte_oracle_masks_and_layouts(self):
        comparisons = 0
        for layout, head_slices, padding, kernels in itertools.product(
            ("NHD", "HND", "LBHNC"), (1, 2, 4), (0, 1), (1, 2)
        ):
            case = make_case(
                layout=layout, head_slices=head_slices, padding=padding, kernels=kernels, pp_layer_counts=(1, 3)
            )
            bound = case.bind()
            ids: NDArray[np.uint64] = np.arange(9, dtype=np.uint64) % 3
            hashes = [f"h{index}" for index in range(9)]
            for transfer in (bound, bound):
                keys = transfer.key_namer.keys(hashes)
                for mask_kind in ("none", "all", "empty", "mixed"):
                    mask = (
                        None
                        if mask_kind == "none"
                        else [
                            mask_kind == "all" or (mask_kind == "mixed" and index % 3 == 1)
                            for index in range(len(keys) * 9)
                        ]
                    )
                    actual_keys, addresses, lengths = bulk_arguments(transfer, keys, ids, selected_keys=mask)
                    expected_keys, expected_bytes = [], []
                    for prefix_index, layers in enumerate(
                        tuple(obj.physical_layer_ids for obj in transfer.local_layout.partitions)
                    ):
                        for row, block_id in enumerate(ids):
                            if mask is not None and not mask[prefix_index * 9 + row]:
                                continue
                            expected_keys.append(keys[prefix_index][row])
                            expected_bytes.append(
                                expected_head_bytes(case, layers, prefix_index % head_slices, int(block_id))
                            )
                    self.assertEqual(actual_keys, expected_keys)
                    self.assertEqual(
                        [read_bytes(a, n) for a, n in zip(addresses, lengths, strict=True)], expected_bytes
                    )
                    comparisons += 1
                full = bulk_arguments(transfer, transfer.key_namer.keys(["full"]), [0])
                self.assertEqual(
                    tuple(map(sum, full[2])), tuple(obj.payload_bytes for obj in transfer.local_layout.partitions)
                )
        self.assertEqual(comparisons, 288)

    def test_pp_store_load_round_trip_both_directions(self):
        for layout, mismatch, replicas in itertools.product(("NHD", "HND", "LBHNC"), (False, True), (1, 2)):
            with self.subTest(layout=layout, mismatch=mismatch, replicas=replicas):
                # P owns four canonical head keys; D either owns one or two keys per rank.
                # Replicas bind the same keys at distinct addresses and all must Load.
                producer_heads = 2 if replicas == 1 else 1
                consumer_slices = 2 if mismatch else 1
                producer = [
                    make_case(
                        layers=tuple(range(pp * 2, pp * 2 + 2)),
                        head_slices=1 if mismatch else None,
                        heads=producer_heads,
                        tp_rank=rank,
                        replicas=replicas,
                        pp_rank=pp,
                        layout=layout,
                    )
                    for pp in range(2)
                    for rank in range(4 * replicas)
                ]
                consumer = [
                    make_case(
                        head_slices=consumer_slices if mismatch else None,
                        pp_layer_counts=(2, 2),
                        heads=producer_heads * consumer_slices,
                        tp_rank=rank,
                        pp_size=1,
                        layout=layout,
                    )
                    for rank in range(4 // consumer_slices)
                ]
                expected = {}
                for case in producer + consumer:
                    group = case.topology.transfer_groups[0]
                    for layer in group.layers:
                        for kind, tensor in enumerate(case.caches[layer.layer_names[0]]):
                            if not tensor.numel():
                                continue
                            head_axis = 1 if layout in ("HND", "LBHNC") else 2
                            local_heads = tensor.shape[head_axis]
                            first_head = group.key_metadata.head_or_tp_rank * local_heads
                            for head in range(local_heads):
                                values = tensor[:, head] if head_axis == 1 else tensor[:, :, head]
                                values.copy_(
                                    torch.arange(values.numel()).reshape(values.shape)
                                    + layer.physical_layer_id * 100000
                                    + kind * 10000
                                    + (first_head + head) * 1000
                                )
                            expected[tensor.data_ptr()] = tensor.clone()
                bound = {id(case): case.bind() for case in producer + consumer}
                for sources, targets in ((producer, consumer), (consumer, producer)):
                    objects: dict[str, bytes] = {}
                    for case in sources:
                        transfer = bound[id(case)]
                        keys, addresses, lengths = bulk_arguments(
                            transfer, transfer.key_namer.keys(["a", "b", "c"]), [0, 1, 2]
                        )
                        for key, address, length in zip(keys, addresses, lengths, strict=True):
                            payload = read_bytes(address, length)
                            if key in objects:
                                self.assertEqual(objects[key], payload)
                            objects[key] = payload
                    for case in targets:
                        for tensors in case.caches.values():
                            for tensor in tensors:
                                tensor.zero_()
                        mapping_for_case = bound[id(case)]
                        self.assertEqual(
                            {
                                key
                                for prefix_keys in mapping_for_case.key_namer.lookup_keys(["a", "b", "c"])
                                for key in prefix_keys
                            },
                            objects.keys(),
                        )
                        transfer = mapping_for_case
                        keys, addresses, lengths = bulk_arguments(
                            transfer, transfer.key_namer.keys(["a", "b", "c"]), [0, 1, 2]
                        )
                        for key, address, length in zip(keys, addresses, lengths, strict=True):
                            payload = objects[key]
                            self.assertEqual(sum(length), len(payload))
                            cursor = 0
                            for destination, size in zip(address, length, strict=True):
                                ctypes.memmove(destination, payload[cursor : cursor + size], size)
                                cursor += size
                        for tensors in case.caches.values():
                            for tensor in tensors:
                                if tensor.numel():
                                    self.assertTrue(torch.equal(tensor, expected[tensor.data_ptr()]))

    def test_mixed_geometry_and_unequal_pp_sizes_preserve_complete_objects(self):
        # Logical block size is equal, but entries have different physical extents.
        for layout in ("NHD", "HND"):
            case = make_case(
                layout=layout, head_slices=2, pp_layer_counts=(1, 3), block_size=96, stored_tokens=(3, 4, 3, 4)
            )
            bound = case.bind()
            ids = [2, 0, 1, 2, 1]
            for transfer in (bound, bound):
                keys = transfer.key_namer.keys([str(index) for index in range(len(ids))])
                for masked in (False, True):
                    mask = [index % 3 != 2 for index in range(len(keys) * len(ids))] if masked else None
                    _, addresses, lengths = bulk_arguments(transfer, keys, ids, selected_keys=mask)
                    expected = [
                        expected_head_bytes(case, layers, prefix_index % 2, block_id)
                        for prefix_index, layers in enumerate(
                            tuple(obj.physical_layer_ids for obj in transfer.local_layout.partitions)
                        )
                        for row, block_id in enumerate(ids)
                        if mask is None or mask[prefix_index * len(ids) + row]
                    ]
                    self.assertEqual([read_bytes(a, n) for a, n in zip(addresses, lengths, strict=True)], expected)

    def test_plain_and_pp_preserve_registered_lengths_without_token_division(self):
        case = make_case(pp_layer_counts=(1, 3), layout="HND")
        # Registered payload lengths need not divide the logical token block size.
        case.registration_arrays[1][3][:] = [length - 1 for length in case.registration_arrays[1][3]]
        bound = case.bind()
        for transfer in (bound, bound):
            keys = transfer.key_namer.keys(["a", "b", "c"])
            _, addresses, lengths = bulk_arguments(transfer, keys, [0, 2, 1])
            for prefix_index, layers in enumerate(
                tuple(obj.physical_layer_ids for obj in transfer.local_layout.partitions)
            ):
                entries = [2 * layer + kind for layer in layers for kind in range(2)]
                for row, block_id in enumerate((0, 2, 1)):
                    index = prefix_index * 3 + row
                    self.assertEqual(
                        addresses[index],
                        [
                            case.registration_arrays[0][3][entry] + block_id * case.registration_arrays[2][3][entry]
                            for entry in entries
                        ],
                    )
                    self.assertEqual(lengths[index], [case.registration_arrays[1][3][entry] for entry in entries])

    def test_absolute_layer_ids_empty_pp_spans_and_single_local_pp_span(self):
        case = make_case(layers=(4, 5, 6, 7), pp_layer_counts=(4, 0, 4), pp_rank=2, head_slices=2)
        with patch.object(head_slicing, "compile_geometries", wraps=head_slicing.compile_geometries) as geometry:
            bound = case.bind()
        self.assertEqual(len(geometry.call_args.args[-1]), 9)  # Layer boundaries also need Entry -> segment boundaries.

        self.assertEqual(tuple(obj.physical_layer_ids for obj in bound.local_layout.partitions), ((4, 5, 6, 7),) * 2)
        case.topology.groups[0].key_metadata.pp_rank = 0
        bound = case.bind()

        self.assertTrue(all("@pp_rank:2@" in prefix for prefix in bound.key_namer.key_prefixes))
        case = make_case(layers=(1, 3, 5, 7), pp_layer_counts=(4, 4))
        self.assertEqual(tuple(obj.physical_layer_ids for obj in case.bind().local_layout.partitions), ((1, 3), (5, 7)))

    def test_pp_partitioning_does_not_rescan_layers_per_rank(self):
        class CountedLayers(tuple):
            reads = 0

            def __getitem__(self, index):
                if isinstance(index, int):
                    self.reads += 1
                return super().__getitem__(index)

        layers = CountedLayers(range(64))
        pp_spans = layer_partitioning.partition_layers(layers, list(range(0, 129, 2)), (8,) * 8)
        self.assertEqual(len(pp_spans), 8)
        self.assertEqual(layers.reads, 64 + 7)  # Each layer, plus one boundary probe per non-final PP span.
        self.assertEqual(
            [(item.entry_start_index, item.entry_end_index) for item in pp_spans],
            [(i * 16, (i + 1) * 16) for i in range(8)],
        )

    def test_disabled_routes_skip_all_variant_work(self):
        for pp_layer_counts in (None, (4,), (2, 2)):
            case = make_case(pp_layer_counts=pp_layer_counts)
            with patch.object(head_slicing, "inspect_entries", side_effect=AssertionError("unused TP")):
                if pp_layer_counts in (None, (4,)):
                    with patch.object(layer_partitioning, "partition_layers", side_effect=AssertionError("unused PP")):
                        case.bind()

                else:
                    case.bind()
        case = make_case(head_slices=2)
        with patch.object(layer_partitioning, "partition_layers", side_effect=AssertionError("unused PP")):
            case.bind()

    def test_packed_tp_and_combined_exit_before_scatter(self):
        for pp_layer_counts in (None, (2, 2)):
            case = make_case(head_slices=1, pp_layer_counts=pp_layer_counts)
            with (
                patch.object(head_slicing, "compile_geometries", side_effect=AssertionError("packed geometry")),
                patch.object(head_slicing, "materialize_segments", side_effect=AssertionError("packed allocation")),
                patch.object(head_slicing.np, "unique", side_effect=AssertionError("packed dedup")),
                patch.object(head_slicing.np, "arange", side_effect=AssertionError("packed enumeration")),
            ):
                bound = case.bind()
            self.assertEqual(
                sum(
                    len(segments.block_zero_segment_addresses)
                    for segments in tuple(obj.local_segments for obj in bound.local_layout.partitions)
                ),
                8,
            )
            self.assertEqual(
                bulk_arguments(bound, bound.key_namer.keys(["h"]), [0])[2],
                [
                    [case.registration_arrays[1][3][2 * layer + kind] for layer in layers for kind in range(2)]
                    for layers in tuple(obj.physical_layer_ids for obj in bound.local_layout.partitions)
                ],
            )

    def test_combination_reuses_geometry_addresses_and_metadata(self):
        case = make_case(layers=tuple(range(32)), head_slices=4, pp_layer_counts=(8, 8, 8, 8))
        with (
            patch.object(head_slicing, "_read_head_layout", wraps=head_slicing._read_head_layout) as reads,
            patch.object(
                head_slicing, "_compile_head_slice_geometry", wraps=head_slicing._compile_head_slice_geometry
            ) as compile_geometry,
            patch.object(head_slicing.np, "add", wraps=np.add) as adds,
            patch.object(head_slicing.np, "unique", wraps=np.unique) as uniques,
            patch.object(
                layer_partitioning, "slice_layer_segments", wraps=layer_partitioning.slice_layer_segments
            ) as slice_layers,
        ):
            bound = case.bind()
        self.assertEqual(reads.call_count, 64)
        self.assertEqual(compile_geometry.call_count, 1)
        self.assertEqual(adds.call_count, 64)
        self.assertEqual(uniques.call_count, 0)  # No logical-coverage metadata is built.
        self.assertTrue(all("out" in call.kwargs for call in adds.call_args_list))
        for index, segments in enumerate(tuple(obj.local_segments for obj in bound.local_layout.partitions)):
            self.assertTrue(
                np.shares_memory(
                    segments.block_zero_segment_addresses,
                    slice_layers.call_args_list[0].args[0][index % 4].block_zero_segment_addresses,
                )
            )
            self.assertIs(
                segments.group_block_strides_bytes,
                tuple(obj.local_segments for obj in bound.local_layout.partitions)[
                    index // 4 * 4
                ].group_block_strides_bytes,
            )
            self.assertIs(
                segments.segment_lengths_bytes,
                tuple(obj.local_segments for obj in bound.local_layout.partitions)[
                    index // 4 * 4
                ].segment_lengths_bytes,
            )
            for field in fields(segments):
                value = getattr(segments, field.name)
                if isinstance(value, np.ndarray):
                    self.assertFalse(value.flags.writeable)
        with (
            patch.object(head_slicing, "_read_head_layout", side_effect=AssertionError("runtime layout")),
            patch.object(head_slicing, "_compile_head_slice_geometry", side_effect=AssertionError("runtime geometry")),
        ):
            bulk_arguments(bound, bound.key_namer.keys(["a", "b"]), [0, 1])

    def test_returned_rows_do_not_alias_each_other_or_bound_segments(self):
        for head_slices, pp_layer_counts in itertools.product((None, 1, 2), (None, (1, 3))):
            case = make_case(head_slices=head_slices, pp_layer_counts=pp_layer_counts)
            transfer = case.bind()
            keys = transfer.key_namer.keys(["a", "b"])
            _, addresses, lengths = bulk_arguments(transfer, keys, [1, 1])
            self.assertEqual(addresses[0], addresses[1])
            self.assertEqual(lengths[0], lengths[1])
            original_address, original_length = addresses[1][0], lengths[1][0]
            addresses[0][0] = lengths[0][0] = 0
            self.assertEqual(addresses[1][0], original_address)
            self.assertEqual(lengths[1][0], original_length)
            _, addresses, lengths = bulk_arguments(transfer, keys, [1, 1])
            self.assertEqual(addresses[0][0], original_address)
            self.assertEqual(lengths[0][0], original_length)

    def test_invalid_inputs_and_empty_runtime_rows(self):
        for invalid_pp_layer_counts in ((-1, 5), (1, 1)):
            with self.assertRaises(ValueError):
                make_case(pp_layer_counts=invalid_pp_layer_counts).bind()
        case = make_case(head_slices=3)
        with self.assertRaisesRegex(ValueError, "divide"):
            case.bind()
        case = make_case(head_slices=2)
        with self.assertRaisesRegex(ValueError, "registered KV tensors"):
            mapping.BulkMapping(case.topology, *case.registration_arrays)
        for head_slices, pp_layer_counts in itertools.product((None, 2), (None, (2, 2))):
            transfer = make_case(head_slices=head_slices, pp_layer_counts=pp_layer_counts).bind()
            self.assertEqual(bulk_arguments(transfer, transfer.key_namer.keys([]), []), ([], [], []))
            with self.assertRaises(ValueError):
                bulk_arguments(transfer, transfer.key_namer.keys(["a"]), [[0]])
            with self.assertRaises(ValueError):
                bulk_arguments(
                    transfer,
                    transfer.key_namer.keys(["a"]),
                    [0],
                    selected_keys=[True] * (len(transfer.key_namer.key_prefixes) + 1),
                )

    def test_bulk_empty_or_unselected_rows_skip_segment_materialization(self):
        for head_slices, pp_layer_counts in itertools.product((None, 2), (None, (1, 3))):
            bound = make_case(head_slices=head_slices, pp_layer_counts=pp_layer_counts).bind()
            keys = bound.key_namer.keys(["a", "b"])
            sizes = bind_bulk_sizes({bound.local_layout.group_id: bound})
            with (
                patch.object(np, "broadcast_to", side_effect=AssertionError("unused segment lengths")),
                patch.object(np, "flatnonzero", side_effect=AssertionError("unused selected indices")),
                patch.object(np, "empty", side_effect=AssertionError("unused address allocation")),
            ):
                self.assertEqual(
                    bulk_arguments(bound, bound.key_namer.keys([]), [], sizes_by_group=sizes), ([], [], [])
                )
                self.assertEqual(
                    bulk_arguments(bound, keys, [0, 1], selected_keys=[False] * (len(keys) * 2), sizes_by_group=sizes),
                    ([], [], []),
                )

    def test_hybrid_groups_bind_original_ids_and_independent_rows(self):
        case = make_hybrid_case(pp_layer_counts=(2, 2), tp_rank=3)
        with self.assertRaisesRegex(ValueError, "original group_id"):
            case.bind()
        for group_id in (0, 1):
            with self.assertRaisesRegex(KeyError, "Unknown transferable"):
                case.bind(group_id=group_id)
        inputs = {
            3: (["tail40", b"\x12"], [2, 0], [True, False, False, True]),
            9: (["state32", "state64", "state32"], [1, 2, 1], [False, True, True, True, False, True]),
        }
        for group_id, (hashes, ids, mask) in inputs.items():
            with patch.object(head_slicing, "inspect_entries", side_effect=AssertionError("unused TP binding")):
                bound = case.bind(group_id=group_id)
            self.assertEqual(bound.local_layout.group_id, group_id)
            self.assertEqual(bound.local_layout.hash_block_size, 8)

            transfer = bound
            keys = transfer.key_namer.keys(hashes)
            actual_keys, addresses, lengths = bulk_arguments(transfer, keys, ids, selected_keys=mask)
            selected = [
                (prefix_index, row)
                for prefix_index in range(len(keys))
                for row in range(len(ids))
                if mask[prefix_index * len(ids) + row]
            ]
            self.assertEqual(actual_keys, [keys[prefix_index][row] for prefix_index, row in selected])
            self.assertEqual(
                [read_bytes(address, length) for address, length in zip(addresses, lengths, strict=True)],
                [
                    expected_group_bytes(
                        case,
                        group_id,
                        tuple(obj.physical_layer_ids for obj in transfer.local_layout.partitions)[prefix_index],
                        ids[row],
                    )
                    for prefix_index, row in selected
                ],
            )
            self.assertEqual(
                list(map(sum, lengths)),
                [tuple(obj.payload_bytes for obj in transfer.local_layout.partitions)[p] for p, _ in selected],
            )
            self.assertEqual(bulk_arguments(transfer, transfer.key_namer.keys([]), []), ([], [], []))
            self.assertEqual(bulk_arguments(transfer, keys, ids, selected_keys=[False] * len(mask)), ([], [], []))

    def test_hybrid_full_payload_excludes_padding_and_preserves_repeated_rows(self):
        for pp_layer_counts in (None, (2, 2)):
            case = make_hybrid_case(pp_layer_counts=pp_layer_counts)
            for group_id in (3, 9):
                bound = case.bind(group_id=group_id)
                transfer = bound
                ids = [2, 0, 1, 2]
                keys = transfer.key_namer.keys(["a", "b", "c", "a"])
                _, addresses, lengths = bulk_arguments(transfer, keys, ids)
                self.assertEqual(
                    [read_bytes(address, length) for address, length in zip(addresses, lengths, strict=True)],
                    [
                        expected_group_bytes(case, group_id, layers, block_id)
                        for layers in tuple(obj.physical_layer_ids for obj in transfer.local_layout.partitions)
                        for block_id in ids
                    ],
                )
                for segments in tuple(obj.local_segments for obj in transfer.local_layout.partitions):
                    for field in fields(segments):
                        value = getattr(segments, field.name)
                        if isinstance(value, np.ndarray):
                            self.assertFalse(value.flags.writeable)
                # State payloads exclude the inter-page padding in registered strides.
                if group_id == 9:
                    self.assertTrue(
                        all(
                            a < b
                            for a, b in zip(case.registration_arrays[1][9], case.registration_arrays[2][9], strict=True)
                        )
                    )

    def test_hybrid_attention_replicas_and_state_ranks_have_different_key_axes(self):
        cases = [make_hybrid_case(tp_rank=rank, pp_layer_counts=(2, 2)) for rank in (2, 3)]
        for group_id, key_rank_count, family in ((3, 2, "c16"), (9, 4, "mamba")):
            left, right = (case.bind(group_id=group_id) for case in cases)
            self.assertEqual(len(left.key_namer.lookup_keys(["h"])), 2 * key_rank_count)
            self.assertEqual(left.key_namer.lookup_keys(["h"]), right.key_namer.lookup_keys(["h"]))
            self.assertEqual(left.key_namer.keys(["h"]) == right.key_namer.keys(["h"]), group_id == 3)
            for bound, rank in ((left, 2), (right, 3)):
                key_rank = rank // 2 if group_id == 3 else rank
                for prefix in bound.key_namer.key_prefixes:
                    self.assertIn(f"@group:{group_id}@", prefix)
                    self.assertIn(f"@cache_family:{family}@", prefix)
                    self.assertIn(f"@head_or_tp_rank:{key_rank}@", prefix)
                    self.assertIn(prefix, bound.key_namer.lookup_key_prefixes)

    def test_single_align_group_uses_whole_objects_and_raw_tp_ranks(self):
        case = make_hybrid_case(tp_rank=3)
        case.topology.transfer_groups = (case.topology.transfer_groups[0],)
        bound = case.bind()
        self.assertEqual(bound.local_layout.group_id, 9)
        self.assertEqual(len(bound.key_namer.lookup_keys(["h"])), 8)
        self.assertIn("@head_or_tp_rank:3@", bound.key_namer.key_prefixes[0])
        _, addresses, lengths = bulk_arguments(bound, bound.key_namer.keys(["h"]), [1])
        self.assertEqual(read_bytes(addresses[0], lengths[0]), expected_group_bytes(case, 9, (1, 3), 1))

    def test_multi_group_whole_objects_do_not_require_recurrent_state(self):
        case = make_hybrid_case(tp_rank=3)
        other_group = case.topology.transfer_groups[0]
        other_group.uses_align_state = False
        other_group.key_metadata.head_or_tp_rank = 1
        other_group.key_metadata.cache_family = "c4"
        for group_id in (3, 9):
            bound = case.bind(group_id=group_id)
            self.assertEqual(len(bound.key_namer.lookup_keys(["h"])), 4)
            self.assertIn("@head_or_tp_rank:1@", bound.key_namer.key_prefixes[0])
            _, addresses, lengths = bulk_arguments(bound, bound.key_namer.keys(["h"]), [2])
            self.assertEqual(
                read_bytes(addresses[0], lengths[0]),
                expected_group_bytes(case, group_id, bound.local_layout.physical_layer_ids, 2),
            )

    def test_hybrid_pp_round_trip_preserves_group_objects_in_both_directions(self):
        # Both groups appear on both PP stages; empty-group ownership is Scheduler work.
        producers = [
            make_hybrid_case(attention_layers=(pp * 2,), state_layers=(pp * 2 + 1,), pp_rank=pp, tp_rank=rank)
            for pp in range(2)
            for rank in range(4)
        ]
        consumers = [make_hybrid_case(pp_layer_counts=(2, 2), pp_size=1, tp_rank=rank) for rank in range(4)]
        bound = {(id(case), group): case.bind(group_id=group) for case in producers + consumers for group in (3, 9)}
        originals = {
            tensor.data_ptr(): tensor.clone()
            for case in producers + consumers
            for tensors in case.caches.values()
            for tensor in tensors
            if tensor.numel()
        }
        for sources, targets in ((producers, consumers), (consumers, producers)):
            objects: dict[str, bytes] = {}
            for case in sources:
                for group_id in (3, 9):
                    transfer = bound[id(case), group_id]
                    keys, addresses, lengths = bulk_arguments(transfer, transfer.key_namer.keys(["a", "b"]), [2, 1])
                    for key, address, length in zip(keys, addresses, lengths, strict=True):
                        payload = read_bytes(address, length)
                        if key in objects:
                            self.assertEqual(objects[key], payload)
                        objects[key] = payload
            self.assertEqual(len(objects), 2 * (2 + 4) * 2)  # PP x distinct group ranks x rows.
            for case in targets:
                for tensors in case.caches.values():
                    for tensor in tensors:
                        tensor.zero_()
                for group_id in (3, 9):
                    group = bound[id(case), group_id]
                    self.assertTrue(
                        all(
                            key in objects
                            for prefix_keys in group.key_namer.lookup_keys(["a", "b"])
                            for key in prefix_keys
                        )
                    )
                    transfer = group
                    keys, addresses, lengths = bulk_arguments(transfer, transfer.key_namer.keys(["a", "b"]), [2, 1])
                    for key, address, length in zip(keys, addresses, lengths, strict=True):
                        payload = objects[key]
                        self.assertEqual(sum(length), len(payload))
                        cursor = 0
                        for destination, size in zip(address, length, strict=True):
                            ctypes.memmove(destination, payload[cursor : cursor + size], size)
                            cursor += size
                for tensors in case.caches.values():
                    for tensor in tensors:
                        if tensor.numel():
                            self.assertTrue(torch.equal(tensor[1:], originals[tensor.data_ptr()][1:]))


if __name__ == "__main__":
    unittest.main()
