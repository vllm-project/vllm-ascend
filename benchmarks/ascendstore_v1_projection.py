# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired CPU benchmark for legacy and bound-rule layerwise materialization.

No Backend, NPU, event, queue or worker thread is involved.  Static cache
registration is outside the timer for both implementations.

Run: NUMBA_DISABLE_JIT=1 PYTHONPATH=.:../vllm python benchmarks/ascendstore_v1_projection.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import LayerBatchBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerTransferTask,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules.memory import (
    KVMemoryRule,
    gva_arguments,
    key_range_arguments,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.batch import (
    KVGroupBatch,
)

AccessKind = Literal["gva", "key_range"]
Scope = Literal["materialize", "request_bind_and_materialize"]
Operation = Callable[[], int]


@dataclass(frozen=True, slots=True)
class Scenario:
    requests: int
    chunks_per_request: int
    layers: int

    @property
    def name(self) -> str:
        return f"r{self.requests}-c{self.chunks_per_request}-l{self.layers}"


@dataclass(slots=True)
class BenchmarkCase:
    scenario: Scenario
    legacy_builder: LayerBatchBuilder
    legacy_task: LayerTransferTask
    legacy_requests: tuple[ReqMeta, ...]
    block_ids: tuple[int, ...]
    keys: tuple[str, ...]
    request_splits: tuple[int, ...]
    object_size: int
    memory: KVMemoryRule
    sessions: dict[str, tuple[int, int]]

    def legacy_request(self) -> tuple[object, ...]:
        tasks = tuple(
            LayerTransferTask(
                layer,
                [LayerBlockRange(request, 0, self.scenario.chunks_per_request) for request in self.legacy_requests],
                group_id=0,
                layer_idx_in_group=layer,
                use_key_major_ranges=self.legacy_task.use_key_major_ranges,
            )
            for layer in range(self.scenario.layers)
        )
        shared = self.legacy_builder.build_shared(tasks[0], is_save=True)
        assert shared is not None
        return tuple(self.legacy_builder.build_addrs(shared, layer) for layer in range(self.scenario.layers))

    def bind_group(self) -> KVGroupBatch:
        block_ids = _readonly(self.block_ids)
        token_counts = _readonly([1] * len(block_ids))
        request_splits = np.asarray(self.request_splits, dtype=np.intp)
        request_splits.flags.writeable = False
        return KVGroupBatch(
            0,
            block_ids,
            token_counts,
            (self.keys,),
            request_splits,
            tuple(range(self.scenario.layers)),
            self.object_size,
        )

    def project(self, group: KVGroupBatch, access: AccessKind, layer: int, object_bases: np.ndarray | None):
        ranges = self.memory.partial(
            group.group_id,
            group.block_ids,
            group.token_counts,
            layer_id=layer,
        )
        if access == "gva":
            assert object_bases is not None
            return gva_arguments(ranges, object_bases, group.selection)
        return key_range_arguments(group.key_axes, ranges, group.selection)

    def gva_bases(self, group: KVGroupBatch) -> np.ndarray:
        result = np.asarray([self.sessions[key][0] for key in group.selected_keys()], dtype=np.uint64)
        result.flags.writeable = False
        return result

    def new_request(self, access: AccessKind) -> tuple[object, ...]:
        group = self.bind_group()
        object_bases = self.gva_bases(group) if access == "gva" else None
        return tuple(self.project(group, access, layer, object_bases) for layer in range(self.scenario.layers))

    def prepared(self, access: AccessKind) -> tuple[Operation, Operation]:
        shared = self.legacy_builder.build_shared(self.legacy_task, is_save=True)
        assert shared is not None
        group = self.bind_group()
        object_bases = self.gva_bases(group) if access == "gva" else None

        def legacy() -> int:
            result = tuple(self.legacy_builder.build_addrs(shared, layer) for layer in range(self.scenario.layers))
            return _consume(result)

        def new() -> int:
            result = tuple(self.project(group, access, layer, object_bases) for layer in range(self.scenario.layers))
            return _consume(result)

        return legacy, new


def make_case(scenario: Scenario, access: AccessKind, segment_count: int = 2) -> BenchmarkCase:
    block_size = 1
    segment_size = 4096
    object_size = scenario.layers * segment_count * segment_size
    bases = [
        1_000_000 + (layer * segment_count + segment) * 100_000_000
        for layer in range(scenario.layers)
        for segment in range(segment_count)
    ]
    sizes = [segment_size] * len(bases)
    strides = [segment_size] * len(bases)
    layer_offsets = [layer * segment_count for layer in range(scenario.layers + 1)]

    database = ChunkedTokenDatabase([KeyMetadata("benchmark", 0, 0, 0)], [block_size], None, block_size)
    database.set_group_buffers(
        {0: bases},
        {0: sizes},
        {0: strides},
        group_layer_cache_entry_offsets={0: layer_offsets},
    )
    builder = LayerBatchBuilder(database, object_size, scenario.layers)

    ranges = []
    requests = []
    block_ids: list[int] = []
    keys: list[str] = []
    request_splits = [0]
    sessions: dict[str, tuple[int, int]] = {}
    global_row = 0
    for request_index in range(scenario.requests):
        request_block_ids = np.arange(
            global_row + 1,
            global_row + scenario.chunks_per_request + 1,
            dtype=np.int64,
        )
        gvas = np.arange(
            10_000_000 + global_row * object_size,
            10_000_000 + (global_row + scenario.chunks_per_request) * object_size,
            object_size,
            dtype=np.int64,
        )
        request_keys = [f"key-{global_row + row}" for row in range(scenario.chunks_per_request)]
        request = ReqMeta(
            f"request-{request_index}",
            block_ids_by_group=[request_block_ids.tolist()],
            block_ids_by_group_np=[request_block_ids],
            block_gvas_by_group_np=[gvas],
            save_block_keys=request_keys,
        )
        requests.append(request)
        ranges.append(LayerBlockRange(request, 0, scenario.chunks_per_request))
        block_ids.extend(map(int, request.block_ids_by_group[0]))
        keys.extend(request_keys)
        request_splits.append(len(block_ids))
        sessions.update((key, (int(gva), object_size)) for key, gva in zip(request_keys, gvas, strict=True))
        global_row += scenario.chunks_per_request

    task = LayerTransferTask(
        0,
        ranges,
        group_id=0,
        layer_idx_in_group=0,
        use_key_major_ranges=access == "key_range",
    )
    memory = KVMemoryRule(
        group_ids=(0,),
        block_sizes={0: block_size},
        align_state_group_ids=frozenset(),
        physical_layers={0: tuple(range(scenario.layers))},
        base_addresses={0: bases},
        block_lengths={0: sizes},
        block_strides={0: strides},
        layer_entry_offsets={0: layer_offsets},
        strided_slice_count=1,
        consumer_pipeline_partitions=None,
        store_pipeline_ranks=None,
        data_plane=access,
        requires_global_offsets=False,
        object_sizes={0: object_size} if access == "gva" else None,
        object_offsets=None,
    )
    case = BenchmarkCase(
        scenario,
        builder,
        task,
        tuple(requests),
        tuple(block_ids),
        tuple(keys),
        tuple(request_splits),
        object_size,
        memory,
        sessions,
    )
    _validate_oracle(case, access)
    return case


def _validate_oracle(case: BenchmarkCase, access: AccessKind) -> None:
    legacy = case.legacy_request()
    current = case.new_request(access)
    for legacy_layer, current_layer in zip(legacy, current, strict=True):
        if access == "gva":
            remote_addresses, local_addresses, sizes = current_layer
            assert legacy_layer.addr_array.tolist() == local_addresses.tolist()
            assert legacy_layer.size_array.tolist() == sizes.tolist()
            assert legacy_layer.gvas_array.tolist() == remote_addresses.tolist()
        else:
            keys, addresses, sizes, offsets = current_layer
            assert legacy_layer.keys == keys
            assert legacy_layer.all_buffers == addresses
            assert legacy_layer.all_sizes == sizes
            assert legacy_layer.all_offsets == offsets


def paired_measure(legacy: Operation, new: Operation, samples: int, minimum_sample_ms: float) -> dict[str, float]:
    iterations = 1
    while _elapsed_ms(legacy, iterations) < minimum_sample_ms or _elapsed_ms(new, iterations) < minimum_sample_ms:
        iterations *= 2
    legacy_samples = []
    new_samples = []
    gc.collect()
    collect_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for sample in range(samples):
            order = ((legacy, legacy_samples), (new, new_samples))
            if sample % 2:
                order = tuple(reversed(order))
            for operation, destination in order:
                start = time.perf_counter_ns()
                checksum = 0
                for _ in range(iterations):
                    checksum ^= operation()
                elapsed = time.perf_counter_ns() - start
                if checksum == -1:
                    raise AssertionError("unreachable checksum")
                destination.append(elapsed / iterations / 1_000_000)
    finally:
        if collect_was_enabled:
            gc.enable()
    legacy_median = statistics.median(legacy_samples)
    new_median = statistics.median(new_samples)
    return {
        "iterations": iterations,
        "legacy_ms": legacy_median,
        "new_ms": new_median,
        "ratio": new_median / legacy_median,
        "delta_ms": new_median - legacy_median,
        "p25_ratio": _quantile(
            [new_value / legacy_value for legacy_value, new_value in zip(legacy_samples, new_samples, strict=True)],
            0.25,
        ),
        "p75_ratio": _quantile(
            [new_value / legacy_value for legacy_value, new_value in zip(legacy_samples, new_samples, strict=True)],
            0.75,
        ),
    }


def classify(result: dict[str, float]) -> str:
    legacy_ms = result["legacy_ms"]
    delta_ms = result["delta_ms"]
    if delta_ms > max(legacy_ms * 0.10, 0.05):
        return "hard_regression"
    if abs(delta_ms) <= max(legacy_ms * 0.05, 0.02):
        return "equivalent"
    return "improvement" if delta_ms < 0 else "review"


def benchmark_case(
    scenario: Scenario,
    access: AccessKind,
    scope: Scope,
    samples: int,
    minimum_sample_ms: float,
) -> dict[str, object]:
    case = make_case(scenario, access)
    if scope == "materialize":
        legacy, new = case.prepared(access)
    else:
        legacy = lambda: _consume(case.legacy_request())
        new = lambda: _consume(case.new_request(access))
    result = paired_measure(legacy, new, samples, minimum_sample_ms)
    return {
        "scenario": scenario.name,
        "access": access,
        "scope": scope,
        "payloads": scenario.layers,
        **result,
        "admission": classify(result),
    }


def _consume(values: tuple[object, ...]) -> int:
    if not values:
        return 0
    value = values[-1]
    if isinstance(value, tuple):
        for selected in reversed(value):
            if len(selected) == 0:
                continue
            tail = selected[-1]
            return int(tail[-1] if isinstance(tail, list) else tail)
    for attribute in ("gvas_array", "remote_addresses", "all_offsets", "offsets"):
        selected = getattr(value, attribute, None)
        if selected is None or len(selected) == 0:
            continue
        tail = selected[-1]
        return int(tail[-1] if isinstance(tail, list) else tail)
    return len(values)


def _elapsed_ms(operation: Operation, iterations: int) -> float:
    start = time.perf_counter_ns()
    checksum = 0
    for _ in range(iterations):
        checksum ^= operation()
    if checksum == -1:
        raise AssertionError("unreachable checksum")
    return (time.perf_counter_ns() - start) / 1_000_000


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _readonly(values: Sequence[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.uint64)
    result.flags.writeable = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--minimum-sample-ms", type=float, default=100.0)
    args = parser.parse_args()
    scenarios = (
        Scenario(1, 4, 24),
        Scenario(1, 32, 32),
        Scenario(1, 32, 80),
        Scenario(1, 128, 32),
        Scenario(8, 32, 32),
    )
    results = []
    for scenario in scenarios:
        for access in ("gva", "key_range"):
            for scope in ("materialize", "request_bind_and_materialize"):
                result = benchmark_case(scenario, access, scope, args.samples, args.minimum_sample_ms)
                results.append(result)
                print(json.dumps(result), flush=True)
    ratios = [float(result["ratio"]) for result in results if result["scope"] == "request_bind_and_materialize"]
    summary = {
        "summary": True,
        "geometric_mean_ratio": math.prod(ratios) ** (1 / len(ratios)),
        "hard_regressions": sum(result["admission"] == "hard_regression" for result in results),
        "bounded_policy": "at most two targeted optimization rounds before architecture review",
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
