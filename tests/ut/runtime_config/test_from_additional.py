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

"""UT: build_runtime_config_from_additional bootstrap entry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_ascend.observability.runtime_config import (
    ADDITIONAL_CONFIG_STRIP_KEYS,
    build_runtime_config_from_additional,
)
from vllm_ascend.observability.runtime_config._defaults import HOT_RELOAD_INTERVAL_SECONDS


def test_strip_keys_cover_runtime_keys():
    assert "runtime_config" in ADDITIONAL_CONFIG_STRIP_KEYS
    assert "runtime_config_path" in ADDITIONAL_CONFIG_STRIP_KEYS
    assert "runtime_config_hot_reload" in ADDITIONAL_CONFIG_STRIP_KEYS
    assert "runtime_report_dir" in ADDITIONAL_CONFIG_STRIP_KEYS
    assert "runtime_dump_dir" in ADDITIONAL_CONFIG_STRIP_KEYS


def test_from_additional_builds_overlay(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    boot = build_runtime_config_from_additional(
        {
            "runtime_config_path": str(cfg_path),
            "runtime_config_hot_reload": False,
            "runtime_report_dir": str(tmp_path / "reports"),
            "runtime_config": {
                "detector": {"token_repeat": {"enabled": True, "window": 12}},
            },
        }
    )
    assert boot.path == str(cfg_path)
    assert boot.hot_reload is False
    assert boot.runtime_config.hot_reload_enabled is False
    assert boot.runtime_config.detector_get("token_repeat", "enabled") is True
    assert boot.runtime_config.detector_get("token_repeat", "window") == 12


def test_from_additional_hot_reload_uses_fixed_interval(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    boot = build_runtime_config_from_additional(
        {
            "runtime_config_path": str(cfg_path),
            "runtime_config_hot_reload": True,
        }
    )
    assert boot.hot_reload is True
    assert boot.runtime_config.reload_interval_seconds == HOT_RELOAD_INTERVAL_SECONDS


def test_from_additional_rejects_bad_path_type():
    with pytest.raises(ValueError, match="runtime_config_path must be a string"):
        build_runtime_config_from_additional({"runtime_config_path": 123})


def test_from_additional_rejects_non_bool_hot_reload():
    with pytest.raises(ValueError, match="runtime_config_hot_reload must be a bool"):
        build_runtime_config_from_additional({"runtime_config_hot_reload": 5})


def test_from_additional_rejects_non_dict_overlay(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_config must be a dict"):
        build_runtime_config_from_additional(
            {
                "runtime_config_path": str(cfg_path),
                "runtime_config": json.dumps({"detector": {}}),
            }
        )
