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

"""Declarative runtime_config field schema (detector sections)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FieldKind = Literal["bool", "int", "float", "list_int"]


@dataclass(frozen=True, slots=True)
class ConfigField:
    """One JSON key under a detector (or shared) section."""

    name: str
    default: Any
    kind: FieldKind = "bool"
    min_value: float | int | None = None
    max_value: float | int | None = None
    help: str = ""


@dataclass(frozen=True, slots=True)
class DetectorSchema:
    """Self-description for one ``detector.<section_key>`` JSON object."""

    section_key: str
    fields: tuple[ConfigField, ...]
    retired_keys: frozenset[str] = frozenset()
    help: str = ""
    stage: str = ""  # e.g. before_sample / after_sample / after_spec (panel hint)

    def defaults_dict(self) -> dict[str, Any]:
        return {f.name: f.default for f in self.fields}

    def param_keys(self) -> frozenset[str]:
        return frozenset(f.name for f in self.fields)


def coerce_bool_inplace(container: dict[str, Any], key: str, field: str) -> None:
    val = container.get(key)
    if val is None or isinstance(val, bool):
        return
    if val in (0, 1):
        container[key] = bool(val)
        return
    raise ValueError(f"{field} must be bool")


def coerce_int(value: Any, field: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"{field} must be a number, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    iv = int(value)
    if min_value is not None and iv < min_value:
        raise ValueError(f"{field} must be >= {min_value}, got {iv}")
    return iv


def coerce_float(
    value: Any,
    field: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"{field} must be a number, got {value!r}")
    fv = float(value)
    if min_value is not None and fv < min_value:
        raise ValueError(f"{field} must be >= {min_value}, got {fv}")
    if max_value is not None and fv > max_value:
        raise ValueError(f"{field} must be <= {max_value}, got {fv}")
    return fv


def coerce_list_int(raw: Any, field: str) -> list[int]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field} must be a list of ints, got {type(raw).__name__}")
    out: list[int] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field}[{i}] must be int, got {item!r}")
        out.append(int(item))
    return out


def validate_detector_section(schema: DetectorSchema, section: dict[str, Any]) -> None:
    """Coerce known fields on ``section`` in place (action override keys untouched)."""
    prefix = f"detector.{schema.section_key}"
    for f in schema.fields:
        path = f"{prefix}.{f.name}"
        if f.name not in section or section[f.name] is None:
            section[f.name] = f.default
            continue
        if f.kind == "bool":
            coerce_bool_inplace(section, f.name, path)
            section[f.name] = bool(section[f.name])
        elif f.kind == "int":
            section[f.name] = coerce_int(
                section[f.name],
                path,
                min_value=int(f.min_value) if f.min_value is not None else None,
            )
        elif f.kind == "float":
            section[f.name] = coerce_float(
                section[f.name],
                path,
                min_value=float(f.min_value) if f.min_value is not None else None,
                max_value=float(f.max_value) if f.max_value is not None else None,
            )
        elif f.kind == "list_int":
            section[f.name] = coerce_list_int(section[f.name], path)
        else:
            raise ValueError(f"unknown ConfigField.kind {f.kind!r} for {path}")
