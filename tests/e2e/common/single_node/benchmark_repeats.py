"""Expand benchmark repetitions while retaining independent result names."""

from copy import deepcopy
from typing import Any


def expand_benchmark_repeats(benchmarks: dict[str, Any] | None) -> dict[str, Any] | None:
    """Run repeated benchmarks on the same service and preserve each result."""
    if not benchmarks:
        return benchmarks
    expanded = {}
    for name, benchmark in benchmarks.items():
        if not benchmark:
            expanded[name] = benchmark
            continue
        count = benchmark.get("repeat_count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name}: repeat_count must be a positive integer")
        for number in range(1, count + 1):
            key = name if count == 1 else f"{name}_round{number:02d}"
            if key in expanded or (count > 1 and key in benchmarks):
                raise ValueError(f"Repeated benchmark name collides with configured case: {key}")
            case = deepcopy(benchmark)
            case.pop("repeat_count", None)
            if count > 1:
                case["case_name"] = f"{benchmark.get('case_name', name)}_round{number:02d}"
            expanded[key] = case
    return expanded
