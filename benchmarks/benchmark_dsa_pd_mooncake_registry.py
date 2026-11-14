# SPDX-License-Identifier: Apache-2.0
"""CPU-only before/after timings for the two Mooncake registry lookups.

Run from the checkout root with TORCH_DEVICE_BACKEND_AUTOLOAD=0::

    python benchmarks/benchmark_dsa_pd_mooncake_registry.py \
        --baseline-ref 1e72fc4 --sizes 1 156 1024 --output registry-benchmark.json

The synthetic tensors expose only address/length metadata. No KV payload,
transfer engine, profiler, or accelerator is used. Timings are reported rather
than asserted, so CPU scheduling noise does not make correctness UTs flaky.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import types
from functools import partial
from pathlib import Path

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.transfer_targets import MooncakeKVRegistry

SOURCE_PATH = "vllm_ascend/kv_offload/dsa_pd_mooncake.py"
BASE_ADDRESS = 1 << 40
RANGE_BYTES = 4 * 1024 * 1024
ADDRESS_STRIDE = 2 * RANGE_BYTES
DIRECT_QUERIES = 21


class _TensorStub:
    def __init__(self, ptr: int) -> None:
        self.ptr = ptr

    def data_ptr(self) -> int:
        return self.ptr

    def numel(self) -> int:
        return RANGE_BYTES

    def element_size(self) -> int:
        return 1


def _prepare_registry(registry_type, tensors):
    registry = registry_type()
    for tensor in tensors:
        registry.register_cpu_buffer(tensor.data_ptr(), tensor.numel())
        registry.register_offloaded_tensor(tensor)
    return registry


def _elapsed_ns(operation, iterations):
    start = time.perf_counter_ns()
    for _ in range(iterations):
        operation()
    return time.perf_counter_ns() - start


def _measure_pair(operations, samples, target_ns):
    iterations = {}
    timings = {name: [] for name in operations}
    for name, operation in operations.items():
        operation()  # Warm lazy indexes before calibration and measured samples.
        count = 1
        while _elapsed_ns(operation, count) < target_ns:
            count *= 2
        iterations[name] = count
    for sample in range(samples):
        names = list(operations)
        if sample % 2:
            names.reverse()
        for name in names:
            count = iterations[name]
            timings[name].append(_elapsed_ns(operations[name], count) / count / 1000)
    result = {
        name: {
            "iterations_per_sample": iterations[name],
            "samples_us": values,
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
        }
        for name, values in timings.items()
    }
    result["speedup"] = result["baseline"]["median_us"] / result["optimized"]["median_us"]
    return result


def _receive_lookup_pattern(registry, queries):
    # Mirrors the observed preflight + receive-loop lookup counts, without
    # adding byte-view preparation, transfer planning, or device work.
    any(registry.resolve_offloaded_tensor(ptr, size) is not None for ptr, size in queries)
    for ptr, size in queries:
        if registry.resolve_offloaded_tensor(ptr, size) is None:
            registry.get_register_location(ptr, size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="1e72fc4")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 156, 1024])
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--min-sample-ms", type=float, default=10)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 3 or args.min_sample_ms <= 0 or any(size < 1 for size in args.sizes):
        parser.error("sizes must be positive, samples >= 3, and min-sample-ms > 0")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})

    root = Path(__file__).resolve().parents[1]
    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:{SOURCE_PATH}"], cwd=root, text=True
    )
    baseline_module = types.ModuleType("_dsa_registry_benchmark_baseline")
    exec(compile(baseline_source, f"{args.baseline_ref}:{SOURCE_PATH}", "exec"), baseline_module.__dict__)
    classes = {"baseline": baseline_module._DSAPDMooncakeKVRegistry, "optimized": MooncakeKVRegistry}
    result = {
        "environment": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "python": sys.version,
            "baseline_ref": args.baseline_ref,
            "baseline_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
            "optimized_sha256": hashlib.sha256(
                (root / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_sfa/transfer_targets.py").read_bytes()
            ).hexdigest(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "gc_enabled": gc.isenabled(),
            "profiler_enabled": False,
            "samples": args.samples,
            "min_sample_ms": args.min_sample_ms,
        },
        "cases": [],
        "cold_first_lookup": [],
        "limitations": [
            "Synthetic address metadata; no model, KV payload, network, or accelerator is exercised.",
            "Warm timings exclude registration/index construction; cold first-lookups are reported separately.",
            "Receive lookup pattern measures only these two functions and their caller loop.",
            "Times include the same Python harness overhead for both implementations; "
            "no CI latency threshold is asserted.",
        ],
    }
    print("ranges  case                                      baseline_us optimized_us speedup")
    for size in args.sizes:
        tensors = [_TensorStub(BASE_ADDRESS + i * ADDRESS_STRIDE) for i in range(size)]
        registries = {name: _prepare_registry(cls, tensors) for name, cls in classes.items()}
        first = (tensors[0].ptr + 128, 1024)
        last = (tensors[-1].ptr + 128, 1024)
        gap = (tensors[0].ptr + RANGE_BYTES + 128, 1024)
        miss = (BASE_ADDRESS + (size + 1) * ADDRESS_STRIDE, 1024)
        queries = [(miss[0] + i * ADDRESS_STRIDE, 1024) for i in range(DIRECT_QUERIES)]
        queries += [(tensor.ptr + 128, 1024) for tensor in tensors]

        # Compare observable return values before entering timed sections.
        for ptr, length in [first, last, gap, miss, *queries]:
            for method in ["get_register_location", "resolve_offloaded_tensor"]:
                old = getattr(registries["baseline"], method)(ptr, length)
                new = getattr(registries["optimized"], method)(ptr, length)
                if method == "resolve_offloaded_tensor" and old is not None:
                    assert new is not None and old[0] is new[0] and old[1] == new[1]
                else:
                    assert old == new

        for method in ["get_register_location", "resolve_offloaded_tensor"]:
            for scenario, query in [("first_hit", first), ("last_hit", last), ("gap_miss", gap), ("after_miss", miss)]:
                operations = {name: partial(getattr(registry, method), *query) for name, registry in registries.items()}
                row = {"ranges": size, "case": f"{method}/{scenario}", "unit": "us_per_lookup"}
                row.update(_measure_pair(operations, args.samples, args.min_sample_ms * 1_000_000))
                result["cases"].append(row)
                print(
                    f"{size:6}  {row['case']:40} "
                    f"{row['baseline']['median_us']:11.3f} {row['optimized']['median_us']:12.3f} {row['speedup']:6.2f}x"
                )

        operations = {
            name: partial(_receive_lookup_pattern, registry, queries) for name, registry in registries.items()
        }
        row = {
            "ranges": size,
            "case": "receive_lookup_pattern",
            "unit": "us_per_request_pattern",
            "resolver_calls": DIRECT_QUERIES + 1 + len(queries),
            "cpu_location_calls": DIRECT_QUERIES,
        }
        row.update(_measure_pair(operations, args.samples, args.min_sample_ms * 1_000_000))
        result["cases"].append(row)
        print(
            f"{size:6}  {row['case']:40} "
            f"{row['baseline']['median_us']:11.3f} {row['optimized']['median_us']:12.3f} {row['speedup']:6.2f}x"
        )

        for name, cls in classes.items():
            cold = {method: [] for method in ["get_register_location", "resolve_offloaded_tensor"]}
            for _ in range(3):
                registry = _prepare_registry(cls, tensors)
                for method, values in cold.items():
                    fn = getattr(registry, method)
                    start = time.perf_counter_ns()
                    fn(*last)
                    values.append((time.perf_counter_ns() - start) / 1000)
            result["cold_first_lookup"].append(
                {
                    "ranges": size,
                    "implementation": name,
                    "median_us": {k: statistics.median(v) for k, v in cold.items()},
                }
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
