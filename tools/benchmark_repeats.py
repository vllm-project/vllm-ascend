# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serial benchmark repetitions on one running service, with independent case names."""

import re
from copy import deepcopy
from typing import Any


def expand_benchmark_repeats(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand explicitly repeated cases; leave legacy cases unchanged."""
    expanded = []
    names = set()
    for case in cases:
        if "repeat_count" not in case:
            expanded.append(case)
            continue
        count = case["repeat_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("repeat_count must be a positive integer")
        name = str(case.get("benchmark_name", case.get("case_name", "unknown")))
        artifact = str(case.get("benchmark_artifact_name", name))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", artifact) or artifact in {".", ".."}:
            raise ValueError("Repeated benchmark artifact name must be a safe directory name")
        if artifact in names:
            raise ValueError(f"Duplicate repeated benchmark artifact name: {artifact}")
        names.add(artifact)
        for index in range(1, count + 1):
            item = deepcopy(case)
            item["source_benchmark_name"] = name
            item["repeat_index"] = index
            item["repeat_artifact_group"] = artifact
            suffix = f"-round-{index:02d}-of-{count:02d}"
            item["benchmark_name"] = name + suffix
            item["case_name"] = name + suffix
            item["benchmark_artifact_name"] = artifact + suffix
            expanded.append(item)
    return expanded
