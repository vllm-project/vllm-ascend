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

"""Deep-merge / normalize helpers for runtime_config payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _leaf_changes(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Return ``path: old -> new`` strings for leaf values that differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        keys = set(old) | set(new)
        out: list[str] = []
        for key in sorted(keys):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old:
                out.append(f"{path}: <missing> -> {new[key]!r}")
            elif key not in new:
                out.append(f"{path}: {old[key]!r} -> <missing>")
            else:
                out.extend(_leaf_changes(old[key], new[key], path))
        return out
    if old != new:
        path = prefix or "<root>"
        return [f"{path}: {old!r} -> {new!r}"]
    return []


def _normalize_ascend_log_section_into(ascend: dict[str, Any]) -> None:
    """Normalize ``ascend_log`` in place (level, debug list, modules dict)."""
    if "level" not in ascend:
        ascend["level"] = "INFO"
    debug = ascend.get("debug", [])
    if debug is None:
        debug = []
    if isinstance(debug, str):
        debug = [debug]
    if not isinstance(debug, list):
        raise ValueError("ascend_log.debug must be a list of module name strings")
    ascend["debug"] = [str(item).strip() for item in debug if str(item).strip()]
    modules = ascend.get("modules", {})
    if modules is None:
        modules = {}
    if not isinstance(modules, dict):
        raise ValueError("ascend_log.modules must be an object")
    normalized_modules: dict[str, str] = {}
    for key, val in modules.items():
        name = str(key).strip()
        if not name:
            continue
        normalized_modules[name] = str(val).strip().upper()
    ascend["modules"] = normalized_modules


def _normalize_config_sections_into(data: dict[str, Any]) -> None:
    """Normalize ``ascend_log`` shape in place (coerce debug/modules).

    Unknown keys are rejected by :func:`validate_runtime_config`, not stripped here.
    """
    if not isinstance(data, dict):
        return
    ascend = data.get("ascend_log")
    if not isinstance(ascend, dict):
        ascend = {}
        data["ascend_log"] = ascend
    _normalize_ascend_log_section_into(ascend)


def _normalize_config_sections(data: dict[str, Any]) -> dict[str, Any]:
    """Copy-then-normalize wrapper around :func:`_normalize_config_sections_into`."""
    out = dict(data)
    ascend = out.get("ascend_log")
    out["ascend_log"] = dict(ascend) if isinstance(ascend, dict) else {}
    _normalize_config_sections_into(out)
    return out
