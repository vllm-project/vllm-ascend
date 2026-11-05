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

"""Detector schema catalog drives defaults / validate."""

from __future__ import annotations

from copy import deepcopy

from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
from vllm_ascend.observability.runtime_config.config import validate_runtime_config
from vllm_ascend.observability.runtime_config.detector_catalog import (
    DETECTOR_SCHEMAS,
    DETECTOR_SECTIONS,
    REGISTERED_DETECTOR_TYPES,
)


def test_catalog_sections_match_defaults_and_classes():
    assert tuple(s.section_key for s in DETECTOR_SCHEMAS) == DETECTOR_SECTIONS
    assert set(DETECTOR_SECTIONS) == set(_DEFAULTS["detector"])
    for cls, schema in zip(REGISTERED_DETECTOR_TYPES, DETECTOR_SCHEMAS, strict=True):
        assert cls.section_key == schema.section_key
        assert cls.schema is schema
        assert _DEFAULTS["detector"][schema.section_key] == schema.defaults_dict()


def test_validate_uses_schema_defaults_for_missing_fields():
    data = deepcopy(_DEFAULTS)
    data["detector"]["token_repeat"] = {"enabled": True}
    validate_runtime_config(data)
    tr = data["detector"]["token_repeat"]
    assert tr["enabled"] is True
    assert tr["window"] == 32
    assert tr["repeat_sum_threshold"] == 64
    assert tr["ignore_token_ids"] == []
