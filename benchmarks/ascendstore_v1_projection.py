# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired CPU benchmark for legacy and bound-plan layerwise materialization.

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
from collections.abc import Callable
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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.lowering import (
    bind_transfer_rows,
    enumerate_transfer_work,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.representation import (
    KVBlockAssignment,
    KVBlockAssignmentBatch,
    KVChunk,
    PhysicalCoordinate,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.selection import merge_transfer_work
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.transfer import (
    BoundGroupPlan,
    ContiguousLayoutPlan,
    SubmissionPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend.arguments import (
    materialize_gva,
    materialize_key_ranges,
    resolve_gva_sessions,
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
    assignments: tuple[KVBlockAssignmentBatch, ...]
    plan: BoundGroupPlan
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

    def new_request(self, access: AccessKind) -> tuple[object, ...]:
        rows = tuple(bind_transfer_rows(self.plan, batch) for batch in self.assignments)
        work_by_request = tuple(enumerate_transfer_work((request_rows,)) for request_rows in rows)
        merged = tuple(
            merge_transfer_work(tuple(request_work[layer] for request_work in work_by_request))
            for layer in range(self.scenario.layers)
        )
        if access == "gva":
            resolved_sessions = {}
            resolved_batches = {}
            for work in merged:
                resolve_gva_sessions(work, self.sessions, resolved_sessions)
            return tuple(materialize_gva(work, self.sessions, resolved_sessions, resolved_batches) for work in merged)
        return tuple(materialize_key_ranges(work) for work in merged)

    def prepared(self, access: AccessKind) -> tuple[Operation, Operation]:
        shared = self.legacy_builder.build_shared(self.legacy_task, is_save=True)
        assert shared is not None
        rows = tuple(bind_transfer_rows(self.plan, batch) for batch in self.assignments)
        work_by_request = tuple(enumerate_transfer_work((request_rows,)) for request_rows in rows)
        merged = tuple(
            merge_transfer_work(tuple(request_work[layer] for request_work in work_by_request))
            for layer in range(self.scenario.layers)
        )
        resolved_sessions = {}
        resolved_batches = {}
        if access == "gva":
            for work in merged:
                resolve_gva_sessions(work, self.sessions, resolved_sessions)

        def legacy() -> int:
            result = tuple(self.legacy_builder.build_addrs(shared, layer) for layer in range(self.scenario.layers))
            return _consume(result)

        def new() -> int:
            if access == "gva":
                result = tuple(
                    materialize_gva(work, self.sessions, resolved_sessions, resolved_batches) for work in merged
                )
            else:
                result = tuple(materialize_key_ranges(work) for work in merged)
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
    assignments = []
    sessions: dict[str, tuple[int, int]] = {}
    global_row = 0
    for request_index in range(scenario.requests):
        block_ids = np.arange(global_row + 1, global_row + scenario.chunks_per_request + 1, dtype=np.int64)
        gvas = np.arange(
            10_000_000 + global_row * object_size,
            10_000_000 + (global_row + scenario.chunks_per_request) * object_size,
            object_size,
            dtype=np.int64,
        )
        keys = [f"key-{global_row + row}" for row in range(scenario.chunks_per_request)]
        request = ReqMeta(
            f"request-{request_index}",
            block_ids_by_group=[block_ids.tolist()],
            block_ids_by_group_np=[block_ids],
            block_gvas_by_group_np=[gvas],
            save_block_keys=keys,
        )
        requests.append(request)
        ranges.append(LayerBlockRange(request, 0, scenario.chunks_per_request))
        chunks = tuple(KVChunk(0, row, TokenRange(row, row + 1), key) for row, key in enumerate(keys))
        assignments.append(
            KVBlockAssignmentBatch(
                0,
                tuple(
                    KVBlockAssignment(chunk, int(block_id), 1)
                    for chunk, block_id in zip(chunks, block_ids, strict=True)
                ),
            )
        )
        sessions.update((key, (int(gva), object_size)) for key, gva in zip(keys, gvas, strict=True))
        global_row += scenario.chunks_per_request

    task = LayerTransferTask(
        0,
        ranges,
        group_id=0,
        layer_idx_in_group=0,
        use_key_major_ranges=access == "key_range",
    )
    layouts = []
    submissions = []
    for layer in range(scenario.layers):
        start = layer * segment_count
        end = start + segment_count
        offsets = np.asarray(
            [start * segment_size + segment * segment_size for segment in range(segment_count)],
            dtype=np.uint64,
        )
        layouts.append(
            ContiguousLayoutPlan(
                (layer,),
                0,
                object_size,
                layer,
                _readonly(bases[start:end]),
                _readonly(strides[start:end]),
                _readonly(sizes[start:end]),
                offsets,
                1,
            )
        )
        submissions.append(SubmissionPlan(layer, (layer,)))
    plan = BoundGroupPlan(
        0,
        (PhysicalCoordinate(),),
        ("",),
        tuple(layouts),
        tuple(submissions),
        scenario.requests * scenario.chunks_per_request + 1,
    )
    case = BenchmarkCase(scenario, builder, task, tuple(requests), tuple(assignments), plan, sessions)
    _validate_oracle(case, access)
    return case


def _validate_oracle(case: BenchmarkCase, access: AccessKind) -> None:
    legacy = case.legacy_request()
    current = case.new_request(access)
    for legacy_layer, current_layer in zip(legacy, current, strict=True):
        if access == "gva":
            assert legacy_layer.addr_array.tolist() == current_layer.local_addresses.tolist()
            assert legacy_layer.size_array.tolist() == current_layer.sizes.tolist()
            assert legacy_layer.gvas_array.tolist() == current_layer.remote_addresses.tolist()
        else:
            assert legacy_layer.keys == current_layer.keys
            assert legacy_layer.all_buffers == current_layer.addresses
            assert legacy_layer.all_sizes == current_layer.sizes
            assert legacy_layer.all_offsets == current_layer.offsets


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


def _readonly(values: list[int]) -> np.ndarray:
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
