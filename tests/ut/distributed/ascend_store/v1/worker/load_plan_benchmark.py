"""Compare Load result costs against a saved pre-change v1 package, with no native copies.

Run in the CPU UT environment with --baseline-root /path/to/saved/v1. Each variant
uses its own Worker, Mapping and Transfer modules. Timed work includes plan/argument
preparation, copies returning fixed success, and Worker failure consumption. Results
are retained until after timing, matching Session Futures; checks and release are untimed.
"""

import argparse
import gc
import importlib
import importlib.util
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
    make_reachability,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import FakeGVABackend, make_gva_spec

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
    bulk = mapping.BulkMapping(
        topology,
        {0: [0x100000000 * (index + 1) for index in range(2 * _LAYERS)]},
        {0: [_SEGMENT_BYTES] * (2 * _LAYERS)},
        {0: [_SEGMENT_BYTES] * (2 * _LAYERS)},
        {0: list(range(0, 2 * _LAYERS + 1, 2))},
    )
    count = request_count * _BLOCKS_PER_REQUEST
    keys = tuple(f"key-{index:064x}" for index in range(count))
    object_size = 2 * _LAYERS * _SEGMENT_BYTES
    group = batches.KVGroupBatch(
        0,
        np.arange(count, dtype=np.uint64),
        np.full(count, _BLOCK_SIZE, dtype=np.uint64),
        (keys,),
        np.arange(0, count + 1, _BLOCKS_PER_REQUEST, dtype=np.intp),
        tuple(range(_LAYERS)),
        object_size,
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
    worker = worker_type(topology, resources, reachability=make_reachability(topology), store_enabled=False)
    worker._mappings = {0: bulk}
    if route != "bulk":
        worker._layer_mappings = {0: mapping.LayerwiseMapping(bulk)}
    if route == "gva":
        # Lease synthetic addresses once; no tensor or native memory is read.
        backend.native_store.objects.update(
            {key: (0x10000000000 + index * object_size, object_size, True) for index, key in enumerate(keys)}
        )
        assert worker._backend_sessions.start_load(list(keys), [object_size] * count) == (0,) * count
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
    counts = Counter()

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


def compare(request_count, route, repeats, baseline):
    cases = {}
    timings = {"before": [], "after": []}
    rng = random.Random(71)
    gc_was_enabled = gc.isenabled()
    try:
        for name, package in (("before", baseline), ("after", _PACKAGE)):
            worker, batch = make_case(request_count, route, package)
            worker_base = importlib.import_module(package + ".worker.base")
            protocol = importlib.import_module(package + ".protocol.transfer")
            cases[name] = worker, batch, worker_base.KVPoolStepContext, protocol.KVTransferStep()
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
