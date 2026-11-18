#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Load packaged pattern.json once and compile alert thresholds for the native detector."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_CONFIG_PATH = Path(__file__).resolve().parent / "pattern.json"
_OP_GE = 0
_OP_LE = 1
_MAX_ATOMS = 4
_NUM_FEATURE_RULES = 11
_NUM_RULES = 14  # 2 exact + run_max + 11 feature


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"pattern.json {field} must be an int")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"pattern.json {field} must be an int")
    return int(value)


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"pattern.json {field} must be a finite float")
    return float(value)


@lru_cache(maxsize=1)
def load_pattern_rules(path: str | None = None) -> dict[str, Any]:
    """Parse and validate packaged thresholds. Cached for process lifetime."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError("pattern.json must be an object")

    exact = data.get("exact")
    if not isinstance(exact, list) or len(exact) != 2:
        raise ValueError("pattern.json exact must be a list of length 2")
    exact_params = np.zeros((2, 3), dtype=np.int32)
    exact_persist = np.zeros(2, dtype=np.int32)
    reasons: list[str] = []
    persist = np.zeros(_NUM_RULES, dtype=np.int32)
    for i, row in enumerate(exact):
        if not isinstance(row, dict):
            raise ValueError(f"pattern.json exact[{i}] must be an object")
        exact_params[i, 0] = _as_int(row.get("pmin"), f"exact[{i}].pmin")
        exact_params[i, 1] = _as_int(row.get("pmax"), f"exact[{i}].pmax")
        exact_params[i, 2] = _as_int(row.get("copies"), f"exact[{i}].copies")
        exact_persist[i] = _as_int(row.get("persist"), f"exact[{i}].persist")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"pattern.json exact[{i}].reason must be a non-empty string")
        reasons.append(reason)
        persist[i] = exact_persist[i]

    run_max = data.get("run_max")
    if not isinstance(run_max, dict):
        raise ValueError("pattern.json run_max must be an object")
    run_thr = _as_int(run_max.get("threshold"), "run_max.threshold")
    run_persist = _as_int(run_max.get("persist"), "run_max.persist")
    run_reason = run_max.get("reason")
    if not isinstance(run_reason, str) or not run_reason:
        raise ValueError("pattern.json run_max.reason must be a non-empty string")
    reasons.append(run_reason)
    persist[2] = run_persist

    feature_rules = data.get("feature_rules")
    if not isinstance(feature_rules, list) or len(feature_rules) != _NUM_FEATURE_RULES:
        raise ValueError(f"pattern.json feature_rules must be a list of length {_NUM_FEATURE_RULES}")
    # Flat layout: [n_atoms, (f, op, pad)*MAX] per rule; values in parallel float array.
    atom_meta = np.zeros((_NUM_FEATURE_RULES, 1 + _MAX_ATOMS * 2), dtype=np.int32)
    atom_thr = np.zeros((_NUM_FEATURE_RULES, _MAX_ATOMS), dtype=np.float64)
    for i, rule in enumerate(feature_rules):
        if not isinstance(rule, dict):
            raise ValueError(f"pattern.json feature_rules[{i}] must be an object")
        clauses = rule.get("all")
        if not isinstance(clauses, list) or not (1 <= len(clauses) <= _MAX_ATOMS):
            raise ValueError(f"pattern.json feature_rules[{i}].all must have 1..{_MAX_ATOMS} atoms")
        atom_meta[i, 0] = len(clauses)
        for j, atom in enumerate(clauses):
            if not isinstance(atom, dict) or "f" not in atom:
                raise ValueError(f"pattern.json feature_rules[{i}].all[{j}] invalid")
            feat = _as_int(atom["f"], f"feature_rules[{i}].all[{j}].f")
            if not 0 <= feat <= 8:
                raise ValueError(f"pattern.json feature index out of range: {feat}")
            if "ge" in atom and "le" not in atom:
                op, thr = _OP_GE, _as_float(atom["ge"], f"feature_rules[{i}].all[{j}].ge")
            elif "le" in atom and "ge" not in atom:
                op, thr = _OP_LE, _as_float(atom["le"], f"feature_rules[{i}].all[{j}].le")
            else:
                raise ValueError(f"pattern.json feature_rules[{i}].all[{j}] needs exactly one of ge/le")
            atom_meta[i, 1 + j * 2] = feat
            atom_meta[i, 1 + j * 2 + 1] = op
            atom_thr[i, j] = thr
        rule_persist = _as_int(rule.get("persist"), f"feature_rules[{i}].persist")
        reason = rule.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"pattern.json feature_rules[{i}].reason must be a non-empty string")
        reasons.append(reason)
        persist[3 + i] = rule_persist

    if len(reasons) != _NUM_RULES:
        raise ValueError("pattern.json reason count mismatch")
    if np.any(persist < 1):
        raise ValueError("pattern.json persist values must be >= 1")

    return {
        "reasons": tuple(reasons),
        "persist": np.ascontiguousarray(persist),
        "exact_params": np.ascontiguousarray(exact_params),
        "run_max_threshold": run_thr,
        "atom_meta": np.ascontiguousarray(atom_meta),
        "atom_thr": np.ascontiguousarray(atom_thr),
        "path": str(cfg_path),
    }
