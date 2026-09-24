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

"""P0 UT: RuntimeConfig load / soft-fail / hot-reload / dump flags."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vllm_ascend.observability.runtime_config.config import RuntimeConfig


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_defaults_dump_and_detectors_off(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.hot_reload_enabled is False
    assert cfg.dump_enabled() is False
    assert cfg.dump_max_times() == 0
    assert cfg.detectors_enabled_in(cfg._data) is False


def test_startup_overlay_enables_detector(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
        startup_overlay={
            "detector": {
                "token_repeat": {"enabled": True, "window": 8, "repeat_sum_threshold": 4},
            }
        },
    )
    assert cfg.detector_get("token_repeat", "enabled") is True
    assert cfg.detector_get("token_repeat", "window") == 8


def test_malformed_json_keeps_previous(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0.01,
        sync_mode="file",
        startup_overlay={
            "detector": {"token_repeat": {"enabled": True, "window": 16}},
        },
    )
    assert cfg.detector_get("token_repeat", "enabled") is True
    # Corrupt file; reload must soft-fail and keep in-memory config.
    cfg_path.write_text("{not-json", encoding="utf-8")
    time.sleep(0.02)
    changed = cfg.reload(force=True)
    assert changed is False or cfg.detector_get("token_repeat", "enabled") is True
    assert cfg.detector_get("token_repeat", "window") == 16


def test_hot_reload_picks_up_detector_enable(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0.01,
        sync_mode="file",
    )
    assert cfg.detector_get("token_repeat", "enabled") is False
    _write(
        cfg_path,
        {
            "detector": {
                "token_repeat": {
                    "enabled": True,
                    "window": 8,
                    "repeat_sum_threshold": 4,
                    "min_tokens": 4,
                }
            }
        },
    )
    time.sleep(0.02)
    assert cfg.reload(force=True) is True
    assert cfg.detector_get("token_repeat", "enabled") is True


def test_dump_auto_and_manual_exclusive_or_derived(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=1,
    )
    assert cfg.dump_enabled() is False
    _write(cfg_path, {"dump": {"auto_max_times": 3, "manual_dump": False}})
    time.sleep(0.01)
    cfg.reload(force=True)
    assert cfg.dump_max_times() == 3
    assert cfg.dump_enabled() is True


def test_actions_default_on_trigger_includes_report(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
    )
    actions = cfg.actions_default_on_trigger()
    assert "report" in actions


def test_detector_on_trigger_override_accepted(tmp_path: Path):
    """Per-detector on_trigger / dump_kv must validate (ActionExecutor reads them)."""
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=1,
        sync_mode="file",
        startup_overlay={
            "detector": {
                "token_repeat": {
                    "enabled": True,
                    "on_trigger": ["report", "dump_kv"],
                    "dump_kv": {"scope": "request"},
                },
                "manual_trigger": {"on_trigger": ["report"]},
            }
        },
    )
    sec = cfg.detector_section("token_repeat")
    assert sec.get("on_trigger") == ["report", "dump_kv"]
    assert cfg.detector_section("manual_trigger").get("on_trigger") == ["report"]


def test_dump_manual_trigger_unknown_key_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config._validate import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["dump"]["manual_trigger"] = 2
    with pytest.raises(ValueError, match="unknown key"):
        validate_runtime_config(data)


def test_token_repeat_window_wrong_type_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config._validate import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["detector"]["token_repeat"]["window"] = "x"
    with pytest.raises(ValueError, match="detector.token_repeat.window"):
        validate_runtime_config(data)


def test_token_repeat_window_non_integer_float_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config._validate import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["detector"]["token_repeat"]["window"] = 2.7
    with pytest.raises(ValueError, match="must be an integer"):
        validate_runtime_config(data)


def test_report_save_sensitive_wrong_type_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config._validate import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["report"]["save_sensitive_info"] = "yes"
    with pytest.raises(ValueError, match="save_sensitive_info"):
        validate_runtime_config(data)


def test_ascend_log_enabled_unknown_key_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config._validate import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["ascend_log"]["enabled"] = True
    with pytest.raises(ValueError, match="unknown key"):
        validate_runtime_config(data)


def test_startup_overlay_reload_interval_overridden_by_ctor(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=5,
        startup_overlay={
            "reload_interval_seconds": 999,
            "detector": {"token_repeat": {"enabled": True}},
        },
    )
    # Ctor interval is authoritative; overlay's copy is ignored.
    assert cfg.reload_interval_seconds == 5.0
    assert cfg.hot_reload_enabled is True
    assert cfg._data["reload_interval_seconds"] == 5.0
    # Overlay still applies other (non-frozen) keys.
    assert cfg.detector_get("token_repeat", "enabled") is True


def test_startup_overlay_sync_mode_overridden_by_ctor(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="broadcast",
        startup_overlay={"sync_mode": "file"},
    )
    assert cfg.sync_mode == "broadcast"
    assert cfg._data["sync_mode"] == "broadcast"


def test_pp_gt1_forces_sync_mode_file(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {"sync_mode": "broadcast"})
    monkeypatch.setattr(
        "vllm_ascend.observability.runtime_config.config._pp_forces_file_sync",
        lambda: True,
    )
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="broadcast",
    )
    assert cfg.sync_mode == "file"
    assert cfg._sync_mode_frozen == "file"
    assert cfg._data["sync_mode"] == "file"


def test_startup_overlay_dump_dir_overridden_by_ctor(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    overlay_dir = tmp_path / "from_overlay"
    ctor_dir = tmp_path / "from_ctor"
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        dump_dir=str(ctor_dir),
        startup_overlay={"dump": {"dump_dir": str(overlay_dir)}},
    )
    assert cfg.dump_root() == ctor_dir.resolve()


def test_startup_overlay_non_dict_raises(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    with pytest.raises(ValueError, match="must be a dict"):
        RuntimeConfig(
            config_path=cfg_path,
            report_dir=tmp_path / "report",
            startup_overlay=["not", "a", "dict"],
        )


def test_startup_overlay_unknown_key_soft_fails_to_defaults(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        startup_overlay={"detector": {"fatal_error": {"enabled": True}}},
    )
    # Bad overlay must not kill the service; fall back to defaults.
    assert cfg.detectors_enabled_in(cfg._data) is False
    assert cfg.detector_get("token_repeat", "enabled") is False


def test_report_dir_explicit(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    reports = tmp_path / "reports"
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=reports,
        ensure_file=True,
    )
    assert cfg.report_dir == reports.resolve()
    assert cfg.dump_root() == reports.resolve() / "kv_cache"


def test_manual_dump_persist_only_when_count_reaches_zero(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=1,
        sync_mode="file",
    )
    _write(cfg_path, {"dump": {"manual_dump": 3, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.manual_trigger_count() == 3

    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_trigger_count() == 2
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["manual_dump"] == 3

    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_trigger_count() == 1
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["manual_dump"] == 3

    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_trigger_count() == 0
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["manual_dump"] is False


def test_manual_dump_hand_edit_reloads_while_count_nonzero(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=1,
        sync_mode="file",
    )
    _write(cfg_path, {"dump": {"manual_dump": 3, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_trigger_count() == 2

    # Hand-edit while memory still >0: reload must pick up the new value.
    _write(cfg_path, {"dump": {"manual_dump": 5, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.manual_trigger_count() == 5
