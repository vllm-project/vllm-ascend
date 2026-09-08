# SPDX-License-Identifier: Apache-2.0

"""Compare uniform pages with Jenga-style exact-LCM/bounded superpages.

Example:
    python -m benchmarks.kv_cache.run_jenga_experiment \
        benchmarks/kv_cache/scenarios/hybrid_demo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.kv_cache.jenga_allocator import (
    HeterogeneousPageAllocator,
    PageType,
    RequestDemand,
    UniformPageAllocator,
    exact_lcm_page_size,
    replay_workload,
)


def _mib(value: int) -> str:
    return f"{value / 1024**2:.2f}"


def _load_scenario(
    path: Path,
) -> tuple[int, list[PageType], list[RequestDemand], list[int], int | None, dict[str, int]]:
    with path.open(encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)
    memory_bytes = int(data["memory_bytes"])
    page_types = [PageType(item["name"], int(item["page_size_bytes"])) for item in data["page_types"]]
    requests = [
        RequestDemand(
            request_id=item["request_id"],
            start_step=int(item["start_step"]),
            duration_steps=int(item["duration_steps"]),
            page_counts={name: int(count) for name, count in item["page_counts"].items()},
        )
        for item in data["requests"]
    ]
    superpage_sizes = [int(size) for size in data.get("bounded_superpage_sizes_bytes", [])]
    uniform_page_size = data.get("uniform_page_size_bytes")
    uniform_slots = {name: int(count) for name, count in data.get("uniform_slots_per_page", {}).items()}
    return memory_bytes, page_types, requests, superpage_sizes, uniform_page_size, uniform_slots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    memory_bytes, page_types, requests, bounded_sizes, uniform_page_size, uniform_slots = _load_scenario(args.scenario)
    exact_size = exact_lcm_page_size(page_types)
    experiments = [
        (
            "uniform-shared-page",
            UniformPageAllocator(
                memory_bytes,
                page_types,
                physical_page_size_bytes=uniform_page_size,
                objects_per_page=uniform_slots,
            ),
        )
    ]
    skipped: dict[str, str] = {}
    if exact_size <= memory_bytes:
        experiments.append(("jenga-exact-lcm", HeterogeneousPageAllocator(memory_bytes, page_types, exact_size)))
    else:
        skipped["jenga-exact-lcm"] = (
            f"LCM page {_mib(exact_size)} MiB exceeds the {_mib(memory_bytes)} MiB memory budget"
        )
    for size in bounded_sizes:
        name = f"typed-superpage-{_mib(size)}MiB"
        try:
            allocator = HeterogeneousPageAllocator(memory_bytes, page_types, size)
        except ValueError as error:
            skipped[name] = str(error)
        else:
            experiments.append((name, allocator))

    results: dict[str, dict[str, object]] = {}
    print(
        f"{'allocator':<30} {'accepted':>8} {'rejected':>8} "
        f"{'peak useful MiB':>16} {'peak reserved MiB':>18} {'peak waste MiB':>15}"
    )
    for name, allocator in experiments:
        result = replay_workload(allocator, requests)
        results[name] = result.to_dict()
        print(
            f"{name:<30} {result.accepted_requests:>8} {result.rejected_requests:>8} "
            f"{_mib(result.peak_useful_bytes):>16} {_mib(result.peak_reserved_bytes):>18} "
            f"{_mib(result.peak_fragmentation_bytes):>15}"
        )

    for name, reason in skipped.items():
        results[name] = {"skipped": reason}
        print(f"{name:<30} {'SKIPPED':>8}  {reason}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
