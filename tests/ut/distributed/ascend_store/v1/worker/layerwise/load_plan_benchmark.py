"""Compare Load result costs against a saved pre-change v1 package, with no native copies.

Run in the CPU UT environment with --baseline-root /path/to/saved/v1. Each variant
uses its own Worker, Mapping and Transfer modules. Timed work includes plan/argument
preparation, copies returning fixed success, and Worker failure consumption. Results
are retained until after timing, matching Session Futures; checks and release are untimed."""

import argparse
import gc
import importlib
import importlib.util
import inspect
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeResources,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.infrastructure.mapping.benchmark import measure
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import FakeGVABackend, make_gva_spec

_PACKAGE = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1"


_LAYERS = 36


_BLOCK_SIZE = 128


_BLOCKS_PER_REQUEST = 64


_SEGMENT_BYTES = 262144


class ConstantCopyBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.copy_calls = 0

    def load(self, keys, addresses, sizes):
        self.copy_calls += 1
        return (0,) * len(keys)

    def copy_key_range_load(self, keys, addresses, sizes, offsets):
        self.copy_calls += 1
        return (0,) * len(keys)


class ConstantGVABackend(FakeGVABackend):
    def __init__(self):
        super().__init__()
        self.copy_calls = 0

    def copy_from_gva(self, remote_addresses, local_addresses, sizes):
        self.copy_calls += 1
        return 0


def load_baseline(path):
    path = Path(path).resolve()
    name = _PACKAGE.rsplit(".", 1)[0] + "._load_result_baseline"
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    if spec is None or spec.loader is None:
        raise ValueError(f"Not an importable v1 snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def make_case(request_count, route="key_range", package=_PACKAGE):
    mapping = importlib.import_module(package + ".infrastructure.mapping")
    batches = importlib.import_module(package + ".worker.transfer.batch")
    topology = make_topology(physical_layers=tuple(range(_LAYERS)))
    topology = replace(
        topology,
        hash_block_size=_BLOCK_SIZE,
        cache_transfer_granularity=_BLOCK_SIZE,
        groups=tuple(
            replace(group, kv_cache_spec=replace(group.kv_cache_spec, block_size=_BLOCK_SIZE))
            for group in topology.groups
        ),
    )
    registration_arrays = (
        {0: [0x100000000 * (index + 1) for index in range(2 * _LAYERS)]},
        {0: [_SEGMENT_BYTES] * (2 * _LAYERS)},
        {0: [_SEGMENT_BYTES] * (2 * _LAYERS)},
        {0: list(range(0, 2 * _LAYERS + 1, 2))},
    )
    count = request_count * _BLOCKS_PER_REQUEST
    keys = tuple(f"key-{index:064x}" for index in range(count))
    remote_object_size_bytes = 2 * _LAYERS * _SEGMENT_BYTES
    legacy_counts = (
        (np.full(count, _BLOCK_SIZE, dtype=np.uint64),)
        if "token_counts" in batches.KVGroupBatch.__dataclass_fields__
        else ()
    )
    group = batches.KVGroupBatch(
        0,
        np.arange(count, dtype=np.uint64),
        *legacy_counts,
        (keys,),
        np.arange(0, count + 1, _BLOCKS_PER_REQUEST, dtype=np.intp),
        tuple(range(_LAYERS)),
        remote_object_size_bytes,
    )
    batch = batches.KVTransferBatch(tuple(f"request-{index}" for index in range(request_count)), (group,)).select_keys(
        set(keys)
    )
    backend = ConstantGVABackend() if route == "gva" else ConstantCopyBackend()
    backend_spec = make_gva_spec() if route == "gva" else make_backend_spec()
    resources = FakeResources(backend, backend_spec, topology)
    if route == "bulk":
        worker_type = importlib.import_module(package + ".worker.bulk").BulkWorker
    else:
        layerwise = importlib.import_module(package + ".worker.layerwise")
        worker_type = layerwise.GVALayerwiseWorker if route == "gva" else layerwise.KeyRangeLayerwiseWorker
    worker = worker_type(topology, resources, coordinator=make_coordinator(topology), store_enabled=False)
    if hasattr(mapping, "build_mappings"):
        # Bound numeric fixtures match historical synthetic addresses; no KV bytes are read.
        segments = {
            "bases": np.asarray(registration_arrays[0][0], dtype=np.uint64),
            "lengths": np.asarray((_SEGMENT_BYTES,), dtype=np.uint64),
            "block_strides": np.asarray((_SEGMENT_BYTES,), dtype=np.uint64),
        }
        worker._local = {0: {(0, 0, 0): segments}}
        if route == "bulk":
            transfer = importlib.import_module(package + ".worker.bulk.transfer")
            worker._bulk_sizes = transfer.bind_bulk_sizes(worker._local)
        else:
            transfer = importlib.import_module(package + ".worker.layerwise.transfer")
            worker._layer_slices = {0: {0: {layer: slice(2 * layer, 2 * layer + 2) for layer in range(_LAYERS)}}}
            worker._layer_memberships = transfer.bind_layer_memberships(worker._local, worker._layer_slices)
            worker._remote = {0: mapping.build_key_range_remote(worker._local[0])}
            if route == "key_range":
                key_range = importlib.import_module(package + ".worker.layerwise.key_range")
                worker._key_range_layouts = key_range.bind_key_range_layouts(
                    worker._local, worker._layer_slices, worker._remote
                )
    elif "topology" in inspect.signature(mapping.LayerwiseMapping).parameters:
        worker._bind_mappings(
            dict(
                zip(
                    ("base_addresses", "block_lengths", "block_strides", "layer_entry_offsets"),
                    registration_arrays,
                    strict=True,
                )
            )
        )
    else:
        # Historical snapshots bind their Bulk view first. This compatibility
        # belongs to the paired benchmark, never to the production Mapping API.
        bind = getattr(mapping, "bind_mapping", mapping.BulkMapping)
        binding = bind(topology, *registration_arrays)
        worker._mappings = {0: binding}
        if route != "bulk":
            worker._layer_mappings = {0: mapping.LayerwiseMapping(binding)}
        elif hasattr(mapping, "bind_mapping"):
            worker._bulk_mappings = {0: mapping.BulkMapping(binding)}
    if route == "gva":
        # Lease synthetic addresses once; no tensor or native memory is read.
        assert isinstance(backend, ConstantGVABackend) and backend.native_store is not None
        backend.native_store.objects.update(
            {
                key: (0x10000000000 + index * remote_object_size_bytes, remote_object_size_bytes, True)
                for index, key in enumerate(keys)
            }
        )
        assert worker._backend_sessions.start_load(list(keys), [remote_object_size_bytes] * count) == (0,) * count
    return worker, batch


def run_layers(worker, batch):
    worker._prepare_load_layers(batch)
    return tuple(worker._copy_load_layer(layer) for layer in range(_LAYERS))


def run_load(worker, batch, context_type, step, route):
    context = context_type(step)
    layers = (worker._execute_bulk_load(batch),) if route == "bulk" else run_layers(worker, batch)
    for completions in layers:
        for completion in completions:
            worker._record_load_completion(context, completion)
    return layers, context


def check_success(result, request_ids, route):
    layers, context = result
    assert len(layers) == (1 if route == "bulk" else _LAYERS)
    assert not context.failed_request_ids and not context.failed_block_ids and not context.failed_sources
    for completions in layers:
        assert tuple(completion.request_id for completion in completions) == request_ids
        for completion in completions:
            evidence = completion.failures if hasattr(completion, "failures") else completion.transfer_evidence
            assert all(item.result_code == 0 for item in evidence)


def constructor_counts(operation):
    counts: Counter[str] = Counter()

    def record(frame, event, _arg):
        if event == "call" and frame.f_code.co_name == "__init__":
            cls = type(frame.f_locals.get("self"))
            # Count Worker values regardless of their subdirectory, excluding test fixtures.
            if cls.__module__.startswith("vllm_ascend.") and ".worker." in cls.__module__:
                counts[cls.__name__] += 1

    previous = sys.getprofile()
    sys.setprofile(record)
    try:
        result = operation()
    finally:
        sys.setprofile(previous)
    del result
    return dict(counts)


def capture_arguments(worker, batch, route):
    """Capture native call arguments before timing, without copying any KV bytes."""
    method = {"bulk": "load", "key_range": "copy_key_range_load", "gva": "copy_from_gva"}[route]
    original = getattr(worker._backend, method)
    calls = []

    def capture(*args):
        calls.append(args)
        return original(*args)

    setattr(worker._backend, method, capture)
    try:
        if route == "bulk":
            worker._execute_bulk_load(batch)
        else:
            run_layers(worker, batch)
    finally:
        setattr(worker._backend, method, original)
    return calls


def phase_operations(worker, batch, route, package):
    if route == "bulk":
        return {"native_arguments": partial(worker._materialize_bulk_arguments, batch)}
    worker._prepare_load_layers(batch)
    module_name = ".worker.layerwise.gva.worker" if route == "gva" else ".worker.layerwise.key_range"
    module = importlib.import_module(package + module_name)
    materialize = module.materialize_layer_gva if route == "gva" else module.materialize_load_layer_ranges

    if route == "key_range" and hasattr(worker, "_load_key_merges"):

        def all_layer_arguments():
            return tuple(
                materialize(worker._key_range_layouts, worker._load_plan, layer, worker._load_key_merges[layer])
                for layer in range(_LAYERS)
            )
    elif route == "gva" and hasattr(worker, "_local"):

        def all_layer_arguments():
            return tuple(
                materialize(worker._local, worker._layer_slices, worker._remote, worker._load_plan, layer)
                for layer in range(_LAYERS)
            )
    else:

        def all_layer_arguments():
            return tuple(materialize(worker._mappings, worker._load_plan, layer) for layer in range(_LAYERS))

    return {
        "plan_preparation": partial(worker._prepare_load_layers, batch),
        "all_layer_native_arguments": all_layer_arguments,
    }


def compare(request_count, route, repeats, baseline):
    cases = {}
    timings: dict[str, list[float]] = {"before": [], "after": []}
    rng = random.Random(71)
    gc_was_enabled = gc.isenabled()
    try:
        for name, package in (("before", baseline), ("after", _PACKAGE)):
            worker, batch = make_case(request_count, route, package)
            worker_base = importlib.import_module(package + ".worker.base")
            cases[name] = worker, batch, worker_base.KVPoolStepContext, worker_base.KVTransferStep()
        assert capture_arguments(*cases["before"][:2], route) == capture_arguments(*cases["after"][:2], route)
        phase_runs = {
            name: phase_operations(*cases[name][:2], route, package)
            for name, package in (("before", baseline), ("after", _PACKAGE))
        }
        phases = {
            phase: measure(phase_runs["before"][phase], phase_runs["after"][phase], repeats, 2)
            for phase in phase_runs["before"]
        }
        gc.collect()
        gc.disable()
        for iteration in range(repeats + 3):
            order = list(timings)
            rng.shuffle(order)
            for name in order:
                worker, batch, context_type, step = cases[name]
                previous_calls = worker._backend.copy_calls
                start = time.perf_counter_ns()
                result = run_load(worker, batch, context_type, step, route)
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                check_success(result, batch.request_ids, route)
                assert worker._backend.copy_calls - previous_calls == (1 if route == "bulk" else _LAYERS)
                if iteration >= 3:
                    timings[name].append(elapsed_ms)
                del result
        allocations = {}
        for name, (worker, batch, context_type, step) in cases.items():
            allocations[name] = constructor_counts(partial(run_load, worker, batch, context_type, step, route))
    finally:
        if gc_was_enabled:
            gc.enable()
        for worker, *_ in cases.values():
            worker._load_plan = None
            worker.close()
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "route": route,
        "requests": request_count,
        "objects": request_count * _BLOCKS_PER_REQUEST,
        "layers": _LAYERS,
        "repeats": repeats,
        "median_ms": medians,
        "min_ms": {name: min(values) for name, values in timings.items()},
        "max_ms": {name: max(values) for name, values in timings.items()},
        "reduction_percent": 100 * (1 - medians["after"] / medians["before"]),
        "constructors_per_invocation": allocations,
        "phases": phases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, help="Saved pre-change v1 package directory")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 7, 8])
    parser.add_argument(
        "--routes", choices=("bulk", "key_range", "gva"), nargs="+", default=["bulk", "key_range", "gva"]
    )
    args = parser.parse_args()
    if args.repeats <= 0 or any(count <= 0 for count in args.requests):
        parser.error("--repeats and --requests must be positive")
    baseline = load_baseline(args.baseline_root)
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "baseline_root": str(Path(args.baseline_root).resolve()),
        "scope": "CPU plan/arguments, fixed-success copy, result consumption; no native copy or execution overlap",
        "gc": "disabled equally; assertions and completion release outside timing; interleaved shuffled variants",
        "results": [compare(count, route, args.repeats, baseline) for route in args.routes for count in args.requests],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
