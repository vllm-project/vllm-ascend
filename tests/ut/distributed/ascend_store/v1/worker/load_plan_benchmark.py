"""CPU-only Load comparison including plan preparation; run as a module in the UT environment.

The baseline rebuilds ownership per layer. Both paths retain all layer completions,
use the same real addressing/evidence code and replace device copies with fixed success.
No timing thresholds are imposed on CI; test_load_plan checks the work-count bound.
"""

import argparse
import gc
import json
import platform
import random
import statistics
import time
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    make_backend_spec,
    make_reachability,
    make_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import KeyRangeLayerwiseProjectionBinder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.io import arguments, key_range
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVGroupBatch,
    KVTransferBatch,
    make_layer_transfer_group,
    make_layer_transfer_plan,
)

_LAYERS = 36
_BLOCK_SIZE = 128
_BLOCKS_PER_REQUEST = 64
_SEGMENT_BYTES = 262144


class ConstantCopyBackend(FakeBackend):
    def copy_key_range_load(self, keys, addresses, sizes, offsets):
        return (0,) * len(keys)


def baseline_plan(batch):
    return make_layer_transfer_plan(batch, tuple(make_layer_transfer_group(group) for group in batch.groups))


def baseline_sources(group, layer_id):
    return tuple(group.batch.source(int(index), layer_id) for index in group.object_indices)


def make_case(request_count):
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
    binder = KeyRangeLayerwiseProjectionBinder(
        topology, lambda group, value, head, stage: str(value), make_reachability(topology, 8192)
    )
    projection = binder.bind(
        {0: [0x100000000 * (index + 1) for index in range(2 * _LAYERS)]},
        {0: [_SEGMENT_BYTES] * (2 * _LAYERS)},
        {0: [_SEGMENT_BYTES] * (2 * _LAYERS)},
        {0: list(range(0, 2 * _LAYERS + 1, 2))},
    )
    count = request_count * _BLOCKS_PER_REQUEST
    keys = tuple(f"key-{index:064x}" for index in range(count))
    group = KVGroupBatch(
        0,
        np.arange(count, dtype=np.uint64),
        np.full(count, _BLOCK_SIZE, dtype=np.uint64),
        (keys,),
        np.arange(0, count + 1, _BLOCKS_PER_REQUEST, dtype=np.intp),
        tuple(range(_LAYERS)),
        2 * _LAYERS * _SEGMENT_BYTES,
    )
    batch = KVTransferBatch(tuple(f"request-{index}" for index in range(request_count)), (group,)).select_keys(
        set(keys)
    )
    io = key_range.KeyRangeBackendIO(ConstantCopyBackend(), make_backend_spec())
    io.bind_projection(projection)
    return io, batch


def run_layers(io, batch):
    io.prepare_load_layers(batch)
    return tuple(io.load_layer(layer) for layer in range(_LAYERS))


def compare(request_count, repeats):
    io, batch = make_case(request_count)
    expected = run_layers(io, batch)
    timings: dict[str, list[float]] = {"baseline": [], "prepared_ownership": []}
    rng = random.Random(71)
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for iteration in range(repeats + 3):
            order = list(timings)
            rng.shuffle(order)
            for name in order:
                with ExitStack() as stack:
                    if name == "baseline":
                        stack.enter_context(patch.object(key_range, "prepare_layer_load", baseline_plan))
                        stack.enter_context(patch.object(arguments, "_layer_sources", baseline_sources))
                    start = time.perf_counter_ns()
                    result = run_layers(io, batch)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                assert result == expected
                if iteration >= 3:
                    timings[name].append(elapsed_ms)
                del result
    finally:
        if gc_was_enabled:
            gc.enable()
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "requests": request_count,
        "objects_per_layer": request_count * _BLOCKS_PER_REQUEST,
        "layers": _LAYERS,
        "repeats": repeats,
        "median_ms": medians,
        "min_ms": {name: min(values) for name, values in timings.items()},
        "max_ms": {name: max(values) for name, values in timings.items()},
        "reduction_percent": 100 * (1 - medians["prepared_ownership"] / medians["baseline"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "scope": "CPU BackendIO including Load plan preparation; no native copy, device or Timeline overlap",
        "gc": "disabled for both variants; completion release and equality checks are outside timing",
        "results": [compare(request_count, args.repeats) for request_count in (1, 7, 8)],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
