"""Paired before/after CPU benchmarks for Bulk and Layerwise mapping.

Run in the UT environment with --baseline /path/to/snapshot/__init__.py.
The snapshot must expose BulkMapping and, with --layerwise, LayerwiseMapping.
Old BulkMapping.store is resolved OUTSIDE timing; current callers use BulkMapping
directly. Both sides use the same tensors, keys, block IDs, masks and layer order.
Exact argument-list equality is checked before every measured workload.

Binding, keys + arguments, arguments alone, keys + Layerwise.prepare, prepared
layer sweeps and complete layer sweeps are measured separately. Scenarios cover
64-row full/mixed/all-false selection, zero rows and one row. Layer sweeps always
visit all 32 layers, including layers with no selected keys.

Samples alternate before/after order, with GC disabled. Short operations are
batched to reduce timer noise; timing includes output allocation and disposal.
Report medians, p10/p90, paired percentage changes and a bootstrap interval for
the paired median. No imports, tensor allocation, byte comparisons, Backend/NPU
execution or device synchronization is timed. These are CPU parameter-building
measurements, not end-to-end performance evidence.
"""

import argparse
import gc
import importlib.util
import json
import platform
import statistics
import sys
import time

import numpy as np

from tests.ut.distributed.ascend_store.v1.infrastructure.mapping_fixtures import make_case
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import mapping

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
    timings = ([], [])
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


def bulk_workload(bound, row_count, mask_kind):
    # Compatibility belongs only to the benchmark, never to production mapping.
    transfer = getattr(bound, "store", bound)
    hashes = tuple(index.to_bytes(4, "little") for index in range(row_count))
    ids = np.zeros(row_count, dtype=np.uint64)
    mask = None
    if mask_kind != "full":
        mask = [mask_kind == "mixed" and index % 3 != 1 for index in range(len(transfer.key_prefixes) * row_count)]
    keys = transfer.keys(hashes)

    def keys_and_arguments():
        return transfer.arguments(transfer.keys(hashes), ids, selected_keys=mask)

    def arguments_only():
        return transfer.arguments(keys, ids, selected_keys=mask)

    return transfer, hashes, ids, mask, keys_and_arguments, arguments_only


def selection_signature(selection):
    selections = getattr(selection, "by_key_prefix", None)
    if selections is None:
        selections = selection.partitions
    return tuple(None if item is None else (item.keys, item.block_ids.tolist()) for item in selections)


def layer_workload(layer_mapping, bulk_work):
    transfer, hashes, ids, mask, _, _ = bulk_work
    layer_ids = layer_mapping.bulk.physical_layer_ids

    def keys_and_prepare():
        return layer_mapping.prepare(transfer.keys(hashes), ids, selected_keys=mask)

    prepared = keys_and_prepare()

    def prepared_layers():
        return [layer_mapping.arguments(prepared, layer_id=layer_id) for layer_id in layer_ids]

    def keys_prepare_all_layers():
        selection = keys_and_prepare()
        return [layer_mapping.arguments(selection, layer_id=layer_id) for layer_id in layer_ids]

    return keys_and_prepare, prepared_layers, keys_prepare_all_layers


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
    spec = importlib.util.spec_from_file_location("mapping_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    profiles = (
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
        before, after = case.bind(baseline), case.bind()
        assert before.lookup_keys(["a", "b"]) == after.lookup_keys(["a", "b"])
        result = {
            "route": name,
            "bulk_binding": measure(
                lambda case=case: case.bind(baseline), case.bind, args.samples, args.target_sample_ms
            ),
        }
        if args.layerwise:
            before_layer = baseline.LayerwiseMapping(before)
            after_layer = mapping.LayerwiseMapping(after)
            result["layer_binding"] = measure(
                lambda before=before: baseline.LayerwiseMapping(before),
                lambda after=after: mapping.LayerwiseMapping(after),
                args.samples,
                args.target_sample_ms,
            )
        workloads = {}
        for scenario, row_count, mask_kind in (
            ("full", 64, "full"),
            ("mixed", 64, "mixed"),
            ("all_false", 64, "none"),
            ("empty", 0, "full"),
            ("single_row", 1, "full"),
        ):
            old_bulk, new_bulk = bulk_workload(before, row_count, mask_kind), bulk_workload(after, row_count, mask_kind)
            assert old_bulk[4]() == new_bulk[4]()
            measurements = {
                "bulk_keys_arguments": measure(old_bulk[4], new_bulk[4], args.samples, args.target_sample_ms),
                "bulk_arguments_only": measure(old_bulk[5], new_bulk[5], args.samples, args.target_sample_ms),
            }
            if args.layerwise:
                old_layer, new_layer = layer_workload(before_layer, old_bulk), layer_workload(after_layer, new_bulk)
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
