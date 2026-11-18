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
_MAX_FEATURE_RULES = 64
_NUM_FEATURES = 9
_FEATURE_WINDOW_KEYS = (
    "a3_n4",
    "a5_n4",
    "novel_n4",
    "p_n8",
    "wrep_long",
    "wrep_short",
    "marker",
    "sdup",
    "sep",
)


def _as_int(value: Any, field: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"pattern.json {field} must be an int")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"pattern.json {field} must be an int")
    iv = int(value)
    if min_value is not None and iv < min_value:
        raise ValueError(f"pattern.json {field} must be >= {min_value}")
    return iv


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"pattern.json {field} must be a finite float")
    return float(value)


def _as_title_reason(row: dict[str, Any], field: str) -> tuple[str, str]:
    title = row.get("title")
    reason = row.get("reason")
    if not isinstance(title, str) or not title:
        raise ValueError(f"pattern.json {field}.title must be a non-empty string")
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"pattern.json {field}.reason must be a non-empty string")
    return title, reason


@lru_cache(maxsize=1)
def load_pattern_rules(path: str | None = None) -> dict[str, Any]:
    """Parse and validate packaged thresholds. Cached for process lifetime."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError("pattern.json must be an object")

    windows = data.get("feature_windows")
    if not isinstance(windows, dict):
        raise ValueError("pattern.json feature_windows must be an object")
    feature_windows = np.zeros(_NUM_FEATURES, dtype=np.int32)
    for i, key in enumerate(_FEATURE_WINDOW_KEYS):
        if key not in windows:
            raise ValueError(f"pattern.json feature_windows missing {key}")
        feature_windows[i] = _as_int(windows[key], f"feature_windows.{key}", min_value=1)

    exact = data.get("exact")
    if not isinstance(exact, list) or len(exact) != 2:
        raise ValueError("pattern.json exact must be a list of length 2")
    exact_params = np.zeros((2, 3), dtype=np.int32)
    titles: list[str] = []
    reasons: list[str] = []
    persist_vals: list[int] = []
    for i, row in enumerate(exact):
        if not isinstance(row, dict):
            raise ValueError(f"pattern.json exact[{i}] must be an object")
        exact_params[i, 0] = _as_int(row.get("pmin"), f"exact[{i}].pmin", min_value=1)
        exact_params[i, 1] = _as_int(row.get("pmax"), f"exact[{i}].pmax", min_value=1)
        exact_params[i, 2] = _as_int(row.get("copies"), f"exact[{i}].copies", min_value=2)
        persist_vals.append(_as_int(row.get("persist"), f"exact[{i}].persist", min_value=1))
        title, reason = _as_title_reason(row, f"exact[{i}]")
        titles.append(title)
        reasons.append(reason)

    run_max = data.get("run_max")
    if not isinstance(run_max, dict):
        raise ValueError("pattern.json run_max must be an object")
    run_thr = _as_int(run_max.get("threshold"), "run_max.threshold", min_value=1)
    persist_vals.append(_as_int(run_max.get("persist"), "run_max.persist", min_value=1))
    title, reason = _as_title_reason(run_max, "run_max")
    titles.append(title)
    reasons.append(reason)

    feature_rules = data.get("feature_rules")
    if not isinstance(feature_rules, list) or not (1 <= len(feature_rules) <= _MAX_FEATURE_RULES):
        raise ValueError(f"pattern.json feature_rules must have 1..{_MAX_FEATURE_RULES} entries")
    n_feat = len(feature_rules)
    atom_meta = np.zeros((n_feat, 1 + _MAX_ATOMS * 2), dtype=np.int32)
    atom_thr = np.zeros((n_feat, _MAX_ATOMS), dtype=np.float64)
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
            if not 0 <= feat < _NUM_FEATURES:
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
        persist_vals.append(_as_int(rule.get("persist"), f"feature_rules[{i}].persist", min_value=1))
        title, reason = _as_title_reason(rule, f"feature_rules[{i}]")
        titles.append(title)
        reasons.append(reason)

    num_rules = 2 + 1 + n_feat
    if len(titles) != num_rules or len(reasons) != num_rules or len(persist_vals) != num_rules:
        raise ValueError("pattern.json rule metadata length mismatch")
    if len(set(titles)) != len(titles):
        raise ValueError("pattern.json titles must be unique")

    return {
        "titles": tuple(titles),
        "reasons": tuple(reasons),
        "persist": np.ascontiguousarray(persist_vals, dtype=np.int32),
        "exact_params": np.ascontiguousarray(exact_params),
        "run_max_threshold": run_thr,
        "feature_windows": np.ascontiguousarray(feature_windows),
        "atom_meta": np.ascontiguousarray(atom_meta),
        "atom_thr": np.ascontiguousarray(atom_thr),
        "num_feature_rules": n_feat,
        "num_rules": num_rules,
        "path": str(cfg_path),
    }
