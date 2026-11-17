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

"""Registered detector schemas → defaults / keys / validate.

Add or remove a detector: declare ``schema`` on the class, then edit
:data:`REGISTERED_DETECTOR_TYPES` below. Defaults and validate consume this
catalog only.
"""

from __future__ import annotations

from typing import Any

from vllm_ascend.observability.runtime_config.schema import (
    DetectorSchema,
    validate_detector_section,
)
from vllm_ascend.observability.runtime_guard.detector.logits_finite import LogitsFiniteDetector
from vllm_ascend.observability.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector
from vllm_ascend.observability.runtime_guard.detector.token_repeat import TokenRepeatDetector

# Registration order = DETECTOR_SECTIONS order.
REGISTERED_DETECTOR_TYPES: tuple[type, ...] = (
    SpecAcceptanceDetector,
    TokenRepeatDetector,
    LogitsFiniteDetector,
)

# Entire detector sections retired from product JSON (soft-pop on validate).
RETIRED_DETECTOR_SECTIONS: frozenset[str] = frozenset({"output_substring"})


def _schema_of(cls: type) -> DetectorSchema:
    schema = getattr(cls, "schema", None)
    if not isinstance(schema, DetectorSchema):
        raise TypeError(f"{cls.__name__} must declare a DetectorSchema as class attribute 'schema'")
    if schema.section_key != getattr(cls, "section_key", schema.section_key):
        raise ValueError(f"{cls.__name__}: schema.section_key != section_key")
    return schema


DETECTOR_SCHEMAS: tuple[DetectorSchema, ...] = tuple(_schema_of(cls) for cls in REGISTERED_DETECTOR_TYPES)
DETECTOR_SECTIONS: tuple[str, ...] = tuple(s.section_key for s in DETECTOR_SCHEMAS)


def build_detector_defaults() -> dict[str, Any]:
    """``detector`` object defaults for ``runtime_config.json``."""
    return {s.section_key: s.defaults_dict() for s in DETECTOR_SCHEMAS}


def retired_detector_keys() -> dict[str, frozenset[str]]:
    return {s.section_key: s.retired_keys for s in DETECTOR_SCHEMAS if s.retired_keys}


def detector_param_keys() -> dict[str, frozenset[str]]:
    return {s.section_key: s.param_keys() for s in DETECTOR_SCHEMAS}


def validate_registered_detectors(detector: dict[str, Any]) -> None:
    """Coerce fields for every registered section (sections must already exist)."""
    by_key = {s.section_key: s for s in DETECTOR_SCHEMAS}
    for name, schema in by_key.items():
        sec = detector.get(name)
        if not isinstance(sec, dict):
            raise ValueError(f"detector.{name} must be an object")
        validate_detector_section(schema, sec)
