"""Paired before/after CPU benchmarks for Bulk and Layerwise mapping.

Run in the UT environment with --baseline /path/to/snapshot/__init__.py.
The snapshot must expose BulkMapping and, with --layerwise, LayerwiseMapping.
Historical interfaces are resolved outside timing; current calls use Worker
argument adapters. Both sides use the same tensors, keys, block IDs, masks and layer order.
Exact argument-list equality is checked before every measured workload.

Binding, keys + arguments, arguments alone, keys + selection preparation, prepared
layer sweeps and complete layer sweeps are measured separately. Scenarios cover
64-row full/mixed/all-false selection, zero rows and one row. Layer sweeps always
visit all 32 layers, including layers with no selected keys.

Current adapters include construction of their test batches; historical Mapping
interfaces do not. Use worker.layerwise.load_plan_benchmark for matched production
Worker costs with batch construction outside timing. These adapter comparisons
verify geometry and expose construction costs, not isolated serving regressions.

Samples alternate before/after order, with GC disabled. Short operations are
batched to reduce timer noise; timing includes output allocation and disposal.
Report medians, p10/p90, paired percentage changes and a bootstrap interval for
the paired median. No imports, tensor allocation, byte comparisons, Backend/NPU
execution or device synchronization is timed. These are CPU parameter-building
measurements, not end-to-end performance evidence."""

import argparse
import gc
import importlib.util
import json
import platform
import statistics
import sys
import time

import numpy as np

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.fixtures import make_case
from tests.ut.distributed.ascend_store.v1.worker.bulk.argument_fixtures import bulk_arguments
from tests.ut.distributed.ascend_store.v1.worker.layerwise.plan_fixtures import layer_arguments, prepare_layer_group
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import (
    build_key_range_remote,
    build_lookup_keys,
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
    materialize_group_layer_ranges,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.transfer import (
    make_layer_transfer_group,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
)

_WARMUP_CALLS = 5


_MAX_REPETITIONS = 1000


_BOOTSTRAP_RESAMPLES = 1000


_BOOTSTRAP_SEED = 0


def measure(before, after, samples, target_sample_ms):
    functions = (before, after)
    estimates = []
    for function in functions:
        start = time.perf_counter_ns()
        for _ in range(_WARMUP_CALLS):
            result = function()
            del result
        estimates.append((time.perf_counter_ns() - start) / _WARMUP_CALLS)
    repetitions = max(1, min(_MAX_REPETITIONS, int(target_sample_ms * 1_000_000 / max(estimates))))
    timings: tuple[list[float], list[float]] = ([], [])
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for sample in range(samples):
            for side in (0, 1) if sample % 2 else (1, 0):
                function = functions[side]
                start = time.perf_counter_ns()
                for _ in range(repetitions):
                    result = function()
                    del result
                timings[side].append((time.perf_counter_ns() - start) / (1000 * repetitions))
    finally:
        if was_enabled:
            gc.enable()
    medians = [statistics.median(values) for values in timings]
    paired_changes = (np.asarray(timings[1]) / timings[0] - 1) * 100
    bootstrap = np.random.default_rng(_BOOTSTRAP_SEED).choice(
        paired_changes, size=(_BOOTSTRAP_RESAMPLES, samples), replace=True
    )
    return {
        "before_us": medians[0],
        "after_us": medians[1],
        "after_before": medians[1] / medians[0],
        "paired_change_pct": float(np.median(paired_changes)),
        "paired_change_ci95_pct": np.quantile(np.median(bootstrap, axis=1), (0.025, 0.975)).tolist(),
        "p10_p90_us": [np.quantile(values, (0.1, 0.9)).tolist() for values in timings],
        "repetitions_per_sample": repetitions,
    }


def bulk_workload(bound, transfer_record_count, mask_kind):
    # Compatibility belongs only to the benchmark, never to production mapping.
    transfer = getattr(bound, "store", bound)
    naming = getattr(transfer, "key_namer", getattr(transfer, "naming", transfer))
    hashes = tuple(index.to_bytes(4, "little") for index in range(transfer_record_count))
    ids = np.zeros(transfer_record_count, dtype=np.uint64)
    keys = naming.keys(hashes)
    mask = None
    if mask_kind != "full":
        mask = [mask_kind == "mixed" and index % 3 != 1 for index in range(len(keys) * transfer_record_count)]

    if hasattr(transfer, "arguments"):
        materialize = transfer.arguments
    else:
        materialize = lambda keys, ids, **options: bulk_arguments(transfer, keys, ids, **options)

    def keys_and_arguments():
        return materialize(naming.keys(hashes), ids, selected_keys=mask)

    def arguments_only():
        return materialize(keys, ids, selected_keys=mask)

    return transfer, hashes, ids, mask, keys_and_arguments, arguments_only


def selection_signature(selection):
    if hasattr(selection, "selected_keys"):
        return tuple(
            None if not len(ids) else (selection.selected_keys[part], ids.tolist())
            for part, ids in zip(
                selection.selected_transfer_slices_by_partition,
                selection.selected_local_block_ids_by_partition,
                strict=True,
            )
        )
    selections = getattr(
        selection, "by_partition", getattr(selection, "by_object", getattr(selection, "by_key_prefix", None))
    )
    if selections is None:
        selections = selection.partitions
    return tuple(
        None
        if item is None
        else (item.keys, (item.local_block_ids if hasattr(item, "local_block_ids") else item.block_ids).tolist())
        for item in selections
    )


def layer_workload(layer_mapping, bulk_work):
    transfer, hashes, ids, mask, _, _ = bulk_work
    naming = getattr(transfer, "key_namer", getattr(transfer, "naming", transfer))
    if hasattr(layer_mapping, "local_layout"):
        layer_ids = layer_mapping.local_layout.physical_layer_ids
    elif hasattr(layer_mapping, "local"):
        layer_ids = layer_mapping.local.physical_layer_ids
    elif hasattr(layer_mapping, "binding"):
        layer_ids = layer_mapping.binding.local_layout.physical_layer_ids
    else:
        layer_ids = layer_mapping.bulk.physical_layer_ids

    if hasattr(layer_mapping, "prepare"):
        prepare = layer_mapping.prepare
        arguments = layer_mapping.arguments
    else:
        prepare = lambda keys, ids, **options: prepare_layer_group(layer_mapping, keys, ids, **options)
        arguments = lambda group, **options: layer_arguments(layer_mapping, group, **options)

    def keys_and_prepare():
        return prepare(naming.keys(hashes), ids, selected_keys=mask)

    prepared = keys_and_prepare()

    def prepared_layers():
        return [arguments(prepared, layer_id=layer_id) for layer_id in layer_ids]

    def keys_prepare_all_layers():
        selection = keys_and_prepare()
        return [arguments(selection, layer_id=layer_id) for layer_id in layer_ids]

    return keys_and_prepare, prepared_layers, keys_prepare_all_layers


def current_workloads(case, bound, row_count, mask_kind):
    key_table, local, layers = bound
    hashes = tuple(index.to_bytes(4, "little") for index in range(row_count))
    ids = np.zeros(row_count, dtype=np.uint64)
    mask = (
        None
        if mask_kind == "full"
        else np.asarray(
            [mask_kind == "mixed" and i % 3 != 1 for i in range(len(key_table) * row_count)], dtype=np.bool_
        )
    )
    group = case.topology.transfer_groups[0]
    by_pp = select_layers(group, case.topology.consumer_pipeline_partitions)
    memberships = tuple(tuple(layer.physical_layer_id for layer in by_pp[pp]) for pp, _, _ in local)
    sizes = bind_bulk_sizes({3: local})
    remote = build_key_range_remote(local) if layers is not None else None
    layouts = bind_key_range_layouts({3: local}, {3: layers}, {3: remote})[3] if layers is not None else None

    def keys():
        return tuple(map_keys(parts, hashes) for parts in key_table.values())

    def batch(names):
        return KVGroupBatch(
            3,
            ids,
            names,
            np.array((0, row_count), dtype=np.intp),
            tuple(layer.physical_layer_id for layer in group.layers),
            sum(sizes[3][0]),
            mask,
            physical_layer_ids_by_partition=memberships,
        )

    def bulk(names):
        result = materialize_bulk_arguments({3: local}, KVTransferBatch(("request",), (batch(names),)), sizes)
        return result.keys, result.addresses, result.sizes

    names = keys()
    bulk_runs = (lambda: bulk(keys()), lambda: bulk(names))
    if layouts is None:
        return bulk_runs, None

    def prepare():
        value = batch(keys())
        return make_layer_transfer_group(value, selected_keys=value.selected_keys())

    def sweep(selection):
        return [
            (
                [
                    key
                    for native in layouts[layer]
                    for key in selection.selected_keys[
                        selection.selected_transfer_slices_by_partition[native.partition_index]
                    ]
                ],
                *materialize_group_layer_ranges(layouts[layer], selection),
            )
            for layer in (item.physical_layer_id for item in group.layers)
        ]

    prepared = prepare()
    return bulk_runs, (prepare, lambda: sweep(prepared), lambda: sweep(prepare()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--target-sample-ms", type=float, default=2)
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--routes", nargs="+", help="Optional exact profile names for focused reruns")
    args = parser.parse_args()
    if args.samples < 5 or args.target_sample_ms <= 0:
        parser.error("Use at least five paired samples and a positive sample duration")
    baseline_name = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure._mapping_baseline"
    spec = importlib.util.spec_from_file_location(baseline_name, args.baseline)
    if spec is None or spec.loader is None:
        parser.error(f"Not an importable mapping snapshot: {args.baseline}")
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    profiles: tuple[tuple[str, dict[str, object]], ...] = (
        ("default", {}),
        ("Consumer PP", {"pp_layer_counts": (8, 8, 8, 8)}),
        ("TP packed", {"head_slices": 1}),
        ("TP NHD", {"head_slices": 2}),
        ("TP HND", {"head_slices": 2, "layout": "HND"}),
        ("PP + TP NHD", {"head_slices": 2, "pp_layer_counts": (8, 8, 8, 8)}),
        ("PP + TP HND", {"head_slices": 2, "pp_layer_counts": (8, 8, 8, 8), "layout": "HND"}),
        ("replicas", {"replicas": 2, "heads": 1}),
        ("PP replicas", {"replicas": 2, "heads": 1, "pp_layer_counts": (8, 8, 8, 8)}),
    )
    if args.routes and set(args.routes) - {name for name, _ in profiles}:
        parser.error("Unknown profile name")
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "samples": args.samples,
                "entries": 64,
                "block_size_tokens": 128,
                "baseline": args.baseline,
                "target_sample_ms": args.target_sample_ms,
            }
        ),
        flush=True,
    )
    for name, options in profiles:
        if args.routes and name not in args.routes:
            continue
        case = make_case(layers=tuple(range(32)), block_size=128, vector=16, blocks=1, fill=False, **options)
        before = case.bind(baseline)
        bind_current = lambda case=case: build_mappings(
            case.topology, case.caches, group_id=3, kv_cache_layout=case.layout
        )
        after = bind_current()
        assert getattr(before, "key_namer", getattr(before, "naming", before)).lookup_keys(["a", "b"]) == tuple(
            map_keys(parts, ("a", "b")) for parts in build_lookup_keys(case.topology, 3)
        )
        result = {
            "route": name,
            "bulk_binding": measure(
                lambda case=case: case.bind(baseline), bind_current, args.samples, args.target_sample_ms
            ),
        }
        if args.layerwise:
            before_layer = case.bind_layers(baseline)

            def bind_current_layer(case=case):
                tables = build_mappings(
                    case.topology, case.caches, group_id=3, kv_cache_layout=case.layout, layerwise=True
                )
                return tables, build_key_range_remote(tables[1])

            after_layer, _ = bind_current_layer()
            result["layer_binding"] = measure(
                lambda case=case: case.bind_layers(baseline),
                bind_current_layer,
                args.samples,
                args.target_sample_ms,
            )
        workloads = {}
        for scenario, transfer_record_count, mask_kind in (
            ("full", 64, "full"),
            ("mixed", 64, "mixed"),
            ("all_false", 64, "none"),
            ("empty", 0, "full"),
            ("single_row", 1, "full"),
        ):
            old_bulk = bulk_workload(before, transfer_record_count, mask_kind)
            new_bulk, new_layer = current_workloads(
                case, after_layer if args.layerwise else after, transfer_record_count, mask_kind
            )
            assert old_bulk[4]() == new_bulk[0]()
            measurements = {
                "bulk_keys_arguments": measure(old_bulk[4], new_bulk[0], args.samples, args.target_sample_ms),
                "bulk_arguments_only": measure(old_bulk[5], new_bulk[1], args.samples, args.target_sample_ms),
            }
            if args.layerwise:
                old_layer = layer_workload(before_layer, old_bulk)
                assert selection_signature(old_layer[0]()) == selection_signature(new_layer[0]())
                assert old_layer[1]() == new_layer[1]()
                assert old_layer[2]() == new_layer[2]()
                for phase, old_run, new_run in zip(
                    ("layer_keys_prepare", "layer_prepared_all_layers", "layer_keys_prepare_all_layers"),
                    old_layer,
                    new_layer,
                    strict=True,
                ):
                    measurements[phase] = measure(old_run, new_run, args.samples, args.target_sample_ms)
            workloads[scenario] = measurements
        result["workloads"] = workloads
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
