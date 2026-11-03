# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small CPU-side precision/trace reporters; never infer a hardware pass."""

import json
import math
from collections import defaultdict


def precision_result(actual, reference, dtype):
    if len(actual) != len(reference):
        raise ValueError("Different output lengths")
    tolerances = {"float16": (2**-9, 2**-9, 0.1), "bfloat16": (2**-6, 2**-6, 1.0), "float32": (2**-10, 2**-16, 0.01)}
    if dtype in ("int32", "int64"):
        exact = all(isinstance(a, int) and isinstance(b, int) and a == b for a, b in zip(actual, reference))
        return dict(passed=bool(actual) and exact, elements=len(actual), exact=exact, dtype=dtype)
    if dtype not in tolerances:
        raise ValueError("Unsupported output dtype")
    rtol, atol, maximum = tolerances[dtype]
    count, max_error, finite = 0, 0.0, True
    for a, b in zip(actual, reference):
        if not math.isfinite(a) or not math.isfinite(b):
            finite = False
            max_error = math.inf
            continue
        error = abs(a - b)
        count += error <= atol + rtol * abs(b)
        max_error = max(max_error, error)
    ratio = count / len(actual) if actual else 0.0
    return dict(
        passed=bool(actual) and finite and ratio >= 0.99 and max_error <= maximum,
        elements=len(actual),
        dtype=dtype,
        rtol=rtol,
        atol=atol,
        mixed_tolerance_fraction=ratio,
        maximum_absolute_error=max_error,
        maximum_absolute_error_limit=maximum,
        all_finite=finite,
    )


def summarize_traces(paths):
    # Keep rank/process/cache scopes separate: the same key in another worker
    # is not a redundant compile inside the original worker.
    output = []
    for path in paths:
        stats = defaultdict(
            lambda: dict(shapes=set(), misses=0, completed=0, keys=set(), warmup_keys=set(), inference_keys=set())
        )
        installed = completed = False
        relevant_calls = 0
        conflicts = []
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                installed |= row["event"] == "installed"
                if row["event"] == "summary":
                    completed = True
                    conflicts.extend(row["restoration_conflicts"])
                    relevant_calls += sum(call["count"] for call in row.get("calls", []))
                name = row.get("kernel")
                if not name:
                    continue
                entry = stats[name]
                if row["event"] == "shape":
                    entry["shapes"].add(row["signature_key"])
                elif row["event"] == "jit_cache_miss":
                    entry["misses"] += 1
                    entry["keys"].add(row["key"])
                    if row["phase"] == "inference":
                        entry["inference_keys"].add(row["key"])
                    else:
                        entry["warmup_keys"].add(row["key"])
                elif row["event"] == "compile_or_load_completed":
                    entry["completed"] += 1
        for entry in stats.values():
            entry["new_inference_keys"] = entry["inference_keys"] - entry["warmup_keys"]
        output.append(
            dict(
                path=str(path),
                trace_complete=installed and completed and not conflicts,
                relevant_calls_observed=bool(stats) and relevant_calls > 0,
                relevant_call_count=relevant_calls,
                restoration_conflicts=conflicts,
                kernels={
                    name: {key: len(value) if isinstance(value, set) else value for key, value in values.items()}
                    for name, values in stats.items()
                },
            )
        )
    return output
