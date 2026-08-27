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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.dfx.runtime_config import (
    DfxRuntimeConfig,
    _dfx_config_sync_group_or_none,
    _is_json_writer,
    _leaf_changes,
)


def _msprobe_path(tmp_path: Path, *, dump_enable: bool | None = None) -> str:
    p = tmp_path / "msprobe.json"
    data: dict = {"dump_path": str(tmp_path / "msprobe_out")}
    if dump_enable is not None:
        data["dump_enable"] = dump_enable
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_leaf_changes_reports_only_diffs():
    old = {"dump": {"auto_max_times": 0}, "ascend_log": {"level": "INFO"}}
    new = {"dump": {"auto_max_times": 3}, "ascend_log": {"level": "INFO"}}
    assert _leaf_changes(old, new) == ["dump.auto_max_times: 0 -> 3"]


def test_startup_overlay_dfx_config_enables_detectors(tmp_path: Path):
    """additional_config.dfx_config uses the same schema as the JSON file."""
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        startup_overlay={
            "detector": {
                "block_kv": {"enabled": True},
                "logits_finite": {"enabled": True},
            },
            "dump": {"auto_max_times": 0, "manual_dump": False},
        },
    )
    assert cfg.detector_get("block_kv", "enabled", False) is True
    assert cfg.detector_get("logits_finite", "enabled", False) is True
    assert cfg.detector_get("position_alignment", "enabled", False) is False
    assert cfg.dump_enabled() is False
    assert cfg.any_detector_enabled() is True


def test_startup_overlay_overrides_explicit_json(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"detector": {"block_kv": {"enabled": False}, "logits_finite": {"enabled": True}}}),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        startup_overlay={"detector": {"block_kv": {"enabled": True}}},
    )
    assert cfg.detector_get("block_kv", "enabled", False) is True
    # Unmentioned sections keep file values.
    assert cfg.detector_get("logits_finite", "enabled", False) is True


def test_startup_overlay_must_be_dict(tmp_path: Path):
    with pytest.raises(ValueError, match="dfx_config must be a dict"):
        DfxRuntimeConfig(
            tmp_path / "dfx_config.json",
            report_dir=tmp_path / "report",
            ensure_file=True,
            startup_overlay=["not", "a", "dict"],  # type: ignore[arg-type]
        )


def test_dfx_config_hot_reload_and_defaults(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    report_dir = tmp_path / "report"
    msprobe = _msprobe_path(tmp_path)
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=report_dir,
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
        msprobe_config_path=msprobe,
    )

    assert cfg_path.exists()
    assert cfg.hot_reload_enabled is True
    assert cfg.dump_max_times() == 0
    assert cfg.ascend_log_level() == "INFO"
    assert cfg.detector_get("spec_acceptance", "enabled") is False

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["auto_max_times"] = 3
    payload["dump"]["manual_dump"] = False
    payload["ascend_log"]["level"] = "DEBUG"
    payload["detector"]["token_logprob"]["enabled"] = True
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    # Bypass interval gate.
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is True
    assert cfg.dump_max_times() == 3
    assert cfg.ascend_log_level() == "DEBUG"
    assert cfg.detector_get("token_logprob", "enabled") is True
    # Nested threshold reload (covers former hot_reload_updates_from_json).
    payload["detector"]["spec_acceptance"]["window"] = 33
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is True
    assert cfg.detector_get("spec_acceptance", "window") == 33


def test_reload_detects_same_mtime_same_size_content_change(tmp_path: Path):
    """Bug #11: same-second, same-size edits must not be skipped (S8 size proxy)."""
    import os

    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["detector"]["spec_acceptance"]["window"] = 10
    payload["detector"]["spec_acceptance"]["enabled"] = True
    payload["ascend_log"]["level"] = "INFO"
    text1 = json.dumps(payload, separators=(",", ":"))
    cfg_path.write_text(text1, encoding="utf-8")
    assert cfg.reload(force=True) is True
    assert cfg.detector_get("spec_acceptance", "window") == 10

    # Same-length edits: 10→33, true→fals (pad), INFO→WARN — keep st_size.
    payload["detector"]["spec_acceptance"]["window"] = 33
    payload["ascend_log"]["level"] = "WARN"
    text2 = json.dumps(payload, separators=(",", ":"))
    assert len(text1.encode("utf-8")) == len(text2.encode("utf-8")), (
        f"test setup requires equal size: {len(text1)} vs {len(text2)}"
    )
    mtime = cfg_path.stat().st_mtime
    cfg_path.write_text(text2, encoding="utf-8")
    os.utime(cfg_path, (mtime, mtime))
    assert cfg_path.stat().st_mtime == mtime
    assert cfg_path.stat().st_size == len(text1.encode("utf-8"))

    assert cfg.reload(force=False) is True
    assert cfg.detector_get("spec_acceptance", "window") == 33
    assert cfg.ascend_log_level() == "WARN"

    # Identical rewrite at same mtime+size must be a no-op.
    cfg_path.write_text(text2, encoding="utf-8")
    os.utime(cfg_path, (mtime, mtime))
    assert cfg.reload(force=False) is False


def test_msprobe_config_path_seeded_and_reload_flag(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = str(tmp_path / "msprobe.json")
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
        msprobe_config_path=msprobe,
    )
    assert cfg.dump_msprobe_config_path() == msprobe
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["dump"]["msprobe_config_path"] == msprobe
    assert payload["dump"]["reload_msprobe"] is False

    payload["dump"]["reload_msprobe"] = True
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is True
    assert cfg.dump_reload_msprobe() is True
    assert cfg.consume_reload_msprobe() is True
    assert cfg.dump_reload_msprobe() is False


def test_msprobe_dump_enable_omitted_seeds_manual_dump(tmp_path: Path):
    msprobe = tmp_path / "msprobe.json"
    msprobe.write_text(json.dumps({"task": "statistics", "dump_path": str(tmp_path / "out")}), encoding="utf-8")
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=str(msprobe),
    )
    assert cfg.dump_enabled() is True
    assert cfg.manual_trigger() is True


def test_msprobe_dump_enable_false_does_not_seed(tmp_path: Path):
    msprobe = tmp_path / "msprobe.json"
    msprobe.write_text(json.dumps({"dump_enable": False, "dump_path": str(tmp_path / "out")}), encoding="utf-8")
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=str(msprobe),
    )
    assert cfg.dump_enabled() is False
    assert cfg.manual_trigger() is False


def test_msprobe_dump_enable_respects_user_dump_off_explicit(tmp_path: Path):
    """DFX explicit dump off + msprobe explicit true must abort startup."""
    msprobe = tmp_path / "msprobe.json"
    msprobe.write_text(json.dumps({"dump_enable": True, "dump_path": str(tmp_path / "out")}), encoding="utf-8")
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 0, "manual_dump": False}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DFX dump off but msprobe dump_enable=true explicitly"):
        DfxRuntimeConfig(
            cfg_path,
            report_dir=tmp_path / "report",
            ensure_file=True,
            msprobe_config_path=str(msprobe),
        )


def test_msprobe_dump_enable_dfx_on_msprobe_false_is_idle_ok(tmp_path: Path):
    """dump_enable=false is the live idle gate; DFX dump capability may stay on."""
    msprobe = tmp_path / "msprobe.json"
    msprobe.write_text(json.dumps({"dump_enable": False, "dump_path": str(tmp_path / "out")}), encoding="utf-8")
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=str(msprobe),
        startup_overlay={"dump": {"manual_dump": True}},
    )
    assert cfg.dump_enabled() is True
    assert cfg.manual_trigger() is True
    # Policy must not flip the live gate back on.
    assert json.loads(msprobe.read_text(encoding="utf-8"))["dump_enable"] is False


def test_msprobe_dump_enable_dfx_on_missing_file_seeds_idle_false(tmp_path: Path):
    """DFX dump on + missing msprobe file → create stub with dump_enable=false."""
    msprobe = tmp_path / "missing_msprobe.json"
    assert not msprobe.exists()
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=str(msprobe),
        startup_overlay={"dump": {"auto_max_times": 2, "manual_dump": False}},
    )
    assert cfg.dump_enabled() is True
    assert msprobe.exists()
    assert json.loads(msprobe.read_text(encoding="utf-8"))["dump_enable"] is False


def test_reload_succeeds_when_idle_gate_closed_and_dump_on(tmp_path: Path):
    """After idle close (dump_enable=false), hot-reload must still apply detector edits."""
    msprobe = tmp_path / "msprobe.json"
    msprobe.write_text(json.dumps({"dump_enable": True, "dump_path": str(tmp_path / "out")}), encoding="utf-8")
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=str(msprobe),
        startup_overlay={"dump": {"auto_max_times": 5, "manual_dump": False}},
        reload_interval_seconds=5,
    )
    assert cfg.dump_enabled() is True
    # Simulate Dumper idle close.
    DfxRuntimeConfig._write_msprobe_dump_enable_file(str(msprobe), False)
    assert json.loads(msprobe.read_text(encoding="utf-8"))["dump_enable"] is False

    # Edit detector via disk (as ops would).
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data.setdefault("detector", {}).setdefault("token_repeat", {})["enabled"] = True
    data["detector"]["token_repeat"]["window"] = 42
    cfg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    assert cfg.reload(force=True) is True
    assert bool(cfg.detector_get("token_repeat", "enabled", False)) is True
    assert int(cfg.detector_get("token_repeat", "window", 0)) == 42
    assert cfg.dump_enabled() is True
    # Still idle-closed; dumper owns reopening.
    assert json.loads(msprobe.read_text(encoding="utf-8"))["dump_enable"] is False


def test_msprobe_dump_enable_aligned_explicit_ok(tmp_path: Path):
    msprobe = tmp_path / "msprobe.json"
    msprobe.write_text(json.dumps({"dump_enable": True, "dump_path": str(tmp_path / "out")}), encoding="utf-8")
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=str(msprobe),
        startup_overlay={"dump": {"auto_max_times": 3, "manual_dump": False}},
    )
    assert cfg.dump_enabled() is True
    assert cfg.manual_trigger() is False


def test_overlay_dump_off_explicit_not_overwritten_by_msprobe_seed(tmp_path: Path):
    """Overlay explicit dump off vs msprobe explicit on → conflict error."""
    msprobe = tmp_path / "msprobe.json"
    msprobe.write_text(json.dumps({"dump_enable": True, "dump_path": str(tmp_path / "out")}), encoding="utf-8")
    with pytest.raises(ValueError, match="DFX dump off but msprobe dump_enable=true explicitly"):
        DfxRuntimeConfig(
            tmp_path / "dfx_config.json",
            report_dir=tmp_path / "report",
            ensure_file=True,
            msprobe_config_path=str(msprobe),
            startup_overlay={"dump": {"auto_max_times": 0, "manual_dump": False}},
        )


def test_msprobe_dump_enable_unreadable_path_does_not_seed(tmp_path: Path):
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=str(tmp_path / "missing_msprobe.json"),
    )
    assert cfg.dump_enabled() is False
    assert cfg.manual_trigger() is False


# ---- dump schema / derived flags -------------------------------------------------


@pytest.mark.parametrize(
    ("auto_max_times", "manual_dump", "expect_active", "expect_auto", "expect_manual"),
    [
        (0, False, False, False, False),
        (3, False, True, True, False),
        (0, True, True, False, True),
        (0, 2, True, False, True),
    ],
)
def test_dump_enabled_derived_from_auto_or_manual(
    tmp_path: Path,
    auto_max_times: int,
    manual_dump: bool | int,
    expect_active: bool,
    expect_auto: bool,
    expect_manual: bool,
):
    # DFX dump "on" needs a msprobe path; dump_enable may be idle false.
    if expect_active:
        msprobe = _msprobe_path(tmp_path)  # omitted dump_enable → default on
    else:
        msprobe = _msprobe_path(tmp_path, dump_enable=False)
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": auto_max_times,
                    "manual_dump": manual_dump,
                    "msprobe_config_path": msprobe,
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.dump_enabled() is expect_active
    assert cfg.auto_dump_on() is expect_auto
    assert cfg.manual_dump_on() is expect_manual
    assert cfg.dump_max_times() == auto_max_times


def test_read_dfx_dump_state_classifies_absent_on_off_explicit():
    assert DfxRuntimeConfig._read_dfx_dump_state(None, None) == "absent"
    assert (
        DfxRuntimeConfig._read_dfx_dump_state({"dump": {"auto_max_times": 3, "manual_dump": False}}, None)
        == "on"
    )
    assert (
        DfxRuntimeConfig._read_dfx_dump_state({"dump": {"auto_max_times": 0, "manual_dump": True}}, None)
        == "on"
    )
    assert (
        DfxRuntimeConfig._read_dfx_dump_state({"dump": {"auto_max_times": 0, "manual_dump": False}}, None)
        == "off_explicit"
    )
    # Overlay wins for "on".
    assert (
        DfxRuntimeConfig._read_dfx_dump_state(
            {"dump": {"auto_max_times": 0, "manual_dump": False}},
            {"dump": {"manual_dump": True}},
        )
        == "on"
    )


def test_read_msprobe_dump_enable_tuple():
    assert DfxRuntimeConfig._read_msprobe_dump_enable(None) == (False, False)
    assert DfxRuntimeConfig._read_msprobe_dump_enable("") == (False, False)


def test_dump_schema_rejects_legacy_cooldown_seconds():
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['cooldown_seconds'\]"):
        DfxRuntimeConfig._validate(_valid_dfx_data(cooldown_seconds=60))


def test_auto_cooldown_seconds_must_be_non_negative():
    with pytest.raises(ValueError, match="auto_cooldown_seconds must be >= 0"):
        DfxRuntimeConfig._validate(_valid_dfx_data(auto_cooldown_seconds=-1))


def test_interaction_mode_summary_detect_plus_auto_and_manual_only(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path)  # default on — required when DFX dump is on
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 2,
                    "manual_dump": False,
                    "msprobe_config_path": msprobe,
                },
                "detector": {"spec_acceptance": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)
    summary = cfg.interaction_mode_summary()
    assert "detect+auto_dump" in summary
    assert "dump.active=True" in summary

    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 0,
                    "manual_dump": True,
                    "msprobe_config_path": msprobe,
                },
            }
        ),
        encoding="utf-8",
    )
    cfg2 = DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)
    assert "manual_dump_only" in cfg2.interaction_mode_summary()


# ---- msprobe bootstrap / reload policy (decision matrix) -------------------------


def test_bootstrap_absent_msprobe_explicit_true_seeds_manual_dump(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path, dump_enable=True)
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=msprobe,
    )
    assert cfg.dump_enabled() is True
    assert cfg.manual_trigger() is True
    assert cfg.auto_dump_on() is False
    assert cfg.dump_max_times() == 0


def test_bootstrap_dump_section_only_msprobe_path_is_off_explicit_not_seed(tmp_path: Path):
    """Explicit dump section (even only path) must not inherit msprobe manual seed."""
    msprobe = _msprobe_path(tmp_path)  # default on
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(json.dumps({"dump": {"msprobe_config_path": msprobe}}), encoding="utf-8")
    cfg = DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)
    assert cfg.dump_enabled() is False
    assert cfg.manual_trigger() is False
    assert json.loads(Path(msprobe).read_text(encoding="utf-8"))["dump_enable"] is False


def test_bootstrap_on_msprobe_default_on_aligned_without_write(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path)  # omitted dump_enable → default on
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 2,
                    "manual_dump": False,
                    "msprobe_config_path": msprobe,
                }
            }
        ),
        encoding="utf-8",
    )
    before = json.loads(Path(msprobe).read_text(encoding="utf-8"))
    assert "dump_enable" not in before
    cfg = DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)
    assert cfg.dump_enabled() is True
    after = json.loads(Path(msprobe).read_text(encoding="utf-8"))
    assert "dump_enable" not in after


def test_bootstrap_on_no_msprobe_file_seeds_msprobe_dump_enable_true(tmp_path: Path):
    msprobe = str(tmp_path / "new_msprobe.json")
    assert not Path(msprobe).exists()
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 1,
                    "manual_dump": False,
                    "msprobe_config_path": msprobe,
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)
    assert cfg.dump_enabled() is True
    assert Path(msprobe).exists()
    assert json.loads(Path(msprobe).read_text(encoding="utf-8"))["dump_enable"] is True


def test_bootstrap_off_explicit_msprobe_default_on_warns_and_writes_false(tmp_path: Path, caplog):
    import logging

    msprobe = _msprobe_path(tmp_path)  # default on (no dump_enable key)
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 0, "manual_dump": False, "msprobe_config_path": msprobe}}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="vllm_ascend.dfx.runtime_config"):
        cfg = DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)
    assert cfg.dump_enabled() is False
    assert any("overriding msprobe default dump_enable=true to false" in r.message for r in caplog.records)
    assert json.loads(Path(msprobe).read_text(encoding="utf-8"))["dump_enable"] is False


def test_bootstrap_off_explicit_msprobe_explicit_false_stays_off(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path, dump_enable=False)
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 0, "manual_dump": False, "msprobe_config_path": msprobe}}),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)
    assert cfg.dump_enabled() is False
    assert json.loads(Path(msprobe).read_text(encoding="utf-8"))["dump_enable"] is False


def test_bootstrap_off_explicit_no_msprobe_file_all_off(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 0, "manual_dump": False}}),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)
    assert cfg.dump_enabled() is False


def test_bootstrap_on_without_msprobe_path_raises(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 3, "manual_dump": False}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dump.msprobe_config_path is not set"):
        DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)


def test_bootstrap_absent_no_msprobe_file_all_off(tmp_path: Path):
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=str(tmp_path / "missing.json"),
    )
    assert cfg.dump_enabled() is False
    assert cfg.manual_trigger() is False


def test_reload_rejects_mutually_exclusive_dump(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path, dump_enable=False)
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
        msprobe_config_path=msprobe,
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["auto_max_times"] = 3
    payload["dump"]["manual_dump"] = True
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is False


def test_manual_dump_count_decrements_on_consume(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    msprobe = _msprobe_path(tmp_path, dump_enable=False)
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=msprobe,
    )
    assert cfg.save({"dump": {"manual_dump": 3, "auto_max_times": 0}})
    assert cfg.manual_trigger_count() == 3
    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_trigger_count() == 2
    reloaded = json.loads(cfg.config_path.read_text(encoding="utf-8"))
    assert reloaded["dump"]["manual_dump"] == 2


def test_manual_dump_true_stays_continuous_on_consume(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path, dump_enable=False)
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=msprobe,
    )
    assert cfg.save({"dump": {"manual_dump": True, "auto_max_times": 0}})
    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_trigger_continuous() is True
    assert cfg.manual_trigger() is True


def test_manual_dump_stops_after_reload_to_false(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path)  # default on — manual_dump on requires alignment
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
        msprobe_config_path=msprobe,
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["manual_dump"] = True
    payload["dump"]["auto_max_times"] = 0
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is True
    assert cfg.manual_trigger() is True

    payload["dump"]["manual_dump"] = False
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is True
    assert cfg.manual_trigger() is False
    assert cfg.dump_enabled() is False


def test_disable_dump_unavailable_clears_auto_and_manual(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    msprobe = _msprobe_path(tmp_path, dump_enable=False)
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        msprobe_config_path=msprobe,
    )
    assert cfg.save({"dump": {"manual_dump": True, "auto_max_times": 0}})
    assert cfg.dump_enabled() is True
    assert cfg.disable_dump_unavailable(reason="test") is True
    assert cfg.dump_enabled() is False
    assert cfg.manual_trigger() is False
    assert cfg.dump_max_times() == 0
    saved = json.loads(cfg.config_path.read_text(encoding="utf-8"))
    assert saved["dump"]["manual_dump"] is False
    assert saved["dump"]["auto_max_times"] == 0


def test_reload_off_explicit_msprobe_explicit_true_rejected(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path, dump_enable=False)
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 0, "manual_dump": False, "msprobe_config_path": msprobe}}),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    # External msprobe edit: explicit true while DFX dump stays off → reload conflict.
    Path(msprobe).write_text(
        json.dumps({"dump_path": str(tmp_path / "out"), "dump_enable": True}),
        encoding="utf-8",
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["ascend_log"]["level"] = "DEBUG"  # touch DFX JSON to trigger reload
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is False


def test_reload_on_missing_msprobe_file_seeds_dump_enable_true(tmp_path: Path):
    msprobe = str(tmp_path / "reload_msprobe.json")
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 2, "manual_dump": False, "msprobe_config_path": msprobe}}),
        encoding="utf-8",
    )
    assert not Path(msprobe).exists()
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert Path(msprobe).exists()
    assert json.loads(Path(msprobe).read_text(encoding="utf-8"))["dump_enable"] is True
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["auto_max_times"] = 4
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is True
    assert cfg.dump_max_times() == 4


def test_manual_dump_active_static_helper():
    assert DfxRuntimeConfig._manual_dump_active(False) is False
    assert DfxRuntimeConfig._manual_dump_active(True) is True
    assert DfxRuntimeConfig._manual_dump_active(3) is True
    assert DfxRuntimeConfig._manual_dump_active(0) is False


def test_auto_on_from_dump_static_helper():
    assert DfxRuntimeConfig._auto_on_from_dump({"auto_max_times": 0}) is False
    assert DfxRuntimeConfig._auto_on_from_dump({"auto_max_times": 1}) is True


def test_bootstrap_mutual_exclusive_auto_and_manual_raises(tmp_path: Path):
    msprobe = _msprobe_path(tmp_path)
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 2,
                    "manual_dump": True,
                    "msprobe_config_path": msprobe,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        DfxRuntimeConfig(cfg_path, report_dir=tmp_path / "report", ensure_file=True)


def test_reload_omitted_msprobe_config_path_keeps_seed(tmp_path: Path):
    """JSON omitting dump.msprobe_config_path must not wipe a bootstrap seed."""
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = str(tmp_path / "msprobe.json")
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
        msprobe_config_path=msprobe,
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"].pop("msprobe_config_path", None)
    payload["ascend_log"]["level"] = "DEBUG"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is True
    assert cfg.dump_msprobe_config_path() == msprobe
    assert cfg.ascend_log_level() == "DEBUG"


def test_reload_explicit_null_msprobe_config_path_clears(tmp_path: Path):
    """Explicit JSON null clears dump.msprobe_config_path (omit does not)."""
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = str(tmp_path / "msprobe.json")
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
        msprobe_config_path=msprobe,
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["msprobe_config_path"] = None
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is True
    assert cfg.dump_msprobe_config_path() is None


@pytest.mark.parametrize(
    "payload, getter, default, expected, err_match",
    [
        (
            {"detector": {"stop_after_alert": 0}},
            lambda c: c.stop_after_alert(),
            True,
            False,
            "stop_after_alert",
        ),
        (
            {"detector": {"output_substring": {"match_prefix": 1}}},
            lambda c: c.detector_get("output_substring", "match_prefix"),
            False,
            True,
            "match_prefix",
        ),
    ],
)
def test_detector_bool_knob_defaults_and_validation(tmp_path: Path, payload, getter, default, expected, err_match):
    """Shared bool knobs: default, 0/1 normalize, reject non-bool."""
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert getter(cfg) is default

    good_path = tmp_path / "explicit.json"
    good_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg2 = DfxRuntimeConfig(
        good_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert getter(cfg2) is expected

    bad = json.loads(json.dumps(payload))
    if "stop_after_alert" in payload.get("detector", {}):
        bad["detector"]["stop_after_alert"] = "yes"
    else:
        bad["detector"]["output_substring"]["match_prefix"] = "yes"
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=err_match):
        DfxRuntimeConfig(
            bad_path,
            report_dir=tmp_path / "report",
            ensure_file=True,
            sync_mode="file",
            reload_interval_seconds=0,
        )


def test_dfx_hot_reload_disabled_when_interval_zero(tmp_path: Path):
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.hot_reload_enabled is False
    assert cfg.sync_dfx_config() is False
    # File change must not apply while hot-reload is off.
    payload = json.loads(cfg.config_path.read_text(encoding="utf-8"))
    payload["dump"]["auto_max_times"] = 9
    cfg.config_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is False
    assert cfg.dump_max_times() == 0


def test_dump_auto_without_detector_allowed(tmp_path: Path):
    """Auto dump and detectors are orthogonal; auto-only dump is valid."""
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = _msprobe_path(tmp_path)
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 3,
                    "manual_dump": False,
                    "msprobe_config_path": msprobe,
                },
                "detector": {
                    "spec_acceptance": {"enabled": False},
                    "token_logprob": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_enabled() is True
    assert cfg.any_detector_enabled() is False
    assert "auto_dump_only" in cfg.interaction_mode_summary()
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["auto_max_times"] == 3


def test_manual_dump_not_consumed_when_dump_inactive(tmp_path: Path):
    from vllm_ascend.dfx.manual_trigger import ManualTriggerManager

    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 0, "manual_dump": False}}),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    runner = SimpleNamespace(tp_rank=0, input_batch=SimpleNamespace(req_ids=["r1"]))
    mgr = ManualTriggerManager(dfx_config=cfg, runner=runner)
    assert mgr.consume_once(allow_arm=True) is None
    assert cfg.manual_trigger() is False


def _valid_dfx_data(**dump_overrides):
    dump = {
        "auto_max_times": 0,
        "auto_cooldown_seconds": 300,
        "manual_dump": False,
        "msprobe_config_path": None,
        "reload_msprobe": False,
    }
    dump.update(dump_overrides)
    return {
        "sync_mode": "file",
        "reload_interval_seconds": 0,
        "dump": dump,
        "ascend_log": {"level": "INFO", "debug": []},
        "detector": {
            "stop_after_alert": True,
            "spec_acceptance": {},
            "token_logprob": {},
            "output_substring": {},
            "token_repeat": {},
        },
        "input_filter": {"filters": [], "print_input_token_ids_once": False},
        "log": {
            "print_sampling_meta": False,
            "print_output_on_finish": False,
        },
        "report": {
            "save_sensitive_info": False,
            "decode_token_ids": True,
            "max_prompt_token_ids": 1000,
            "max_output_token_ids": 1000,
            "include_block_ids": True,
            "include_slot_mapping": False,
            "block_last_write_wave": False,
            "block_last_writer": False,
        },
    }


def _merge_deep(base: dict, override: dict) -> None:
    """Recursively merge ``override`` into ``base`` in place (test helper)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_deep(base[key], value)
        else:
            base[key] = value


def test_dump_rejects_unknown_keys_including_dump_once():
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['dump_once'\]"):
        DfxRuntimeConfig._validate(_valid_dfx_data(dump_once=True))

    with pytest.raises(ValueError, match=r"unknown key\(s\) \['foo'\]"):
        DfxRuntimeConfig._validate(_valid_dfx_data(foo=1))

    DfxRuntimeConfig._validate(_valid_dfx_data(auto_max_times=3))
    DfxRuntimeConfig._validate(_valid_dfx_data(manual_dump=True))
    DfxRuntimeConfig._validate(_valid_dfx_data(manual_dump=3))
    with pytest.raises(ValueError, match="mutually exclusive"):
        DfxRuntimeConfig._validate(_valid_dfx_data(auto_max_times=3, manual_dump=True))
    with pytest.raises(ValueError, match="manual_dump"):
        DfxRuntimeConfig._validate(_valid_dfx_data(manual_dump=-1))
    with pytest.raises(ValueError, match="manual_dump"):
        DfxRuntimeConfig._validate(_valid_dfx_data(manual_dump="yes"))
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['enabled'\]"):
        DfxRuntimeConfig._validate(_valid_dfx_data(enabled=True))


def test_dump_schema_rejects_legacy_keys():
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['max_times'\]"):
        DfxRuntimeConfig._validate(_valid_dfx_data(max_times=3))
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['manual_trigger'\]"):
        DfxRuntimeConfig._validate(_valid_dfx_data(manual_trigger=True))


def test_dump_once_in_json_fails_bootstrap(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 0, "dump_once": True}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['dump_once'\]"):
        DfxRuntimeConfig(
            cfg_path,
            report_dir=tmp_path / "report",
            ensure_file=False,
            sync_mode="file",
            reload_interval_seconds=0,
        )


@pytest.mark.parametrize(
    "override, err_match",
    [
        (
            {"detector": {"token_repeat": {"window": 0}}},
            "detector.token_repeat.window must be >= 1",
        ),
        (
            {"detector": {"token_repeat": {"repeat_sum_threshold": -1}}},
            "detector.token_repeat.repeat_sum_threshold must be >= 0",
        ),
        (
            {"detector": {"token_repeat": {"consecutive_hits": 0}}},
            "detector.token_repeat.consecutive_hits must be >= 1",
        ),
    ],
)
def test_token_repeat_validation_rejects_invalid_values(override, err_match):
    """Negative validation: token_repeat knobs must satisfy their ranges."""
    data = _valid_dfx_data()
    _merge_deep(data, override)
    with pytest.raises(ValueError, match=err_match):
        DfxRuntimeConfig._validate(data)


def test_token_repeat_validation_rejects_non_int_ignore_token_ids():
    data = _valid_dfx_data()
    data["detector"]["token_repeat"]["ignore_token_ids"] = ["x", 1]
    with pytest.raises(ValueError, match=r"ignore_token_ids\[0\] must be int"):
        DfxRuntimeConfig._validate(data)


def test_log_print_output_on_finish_rejects_non_bool():
    data = _valid_dfx_data()
    data["log"]["print_output_on_finish"] = "yes"
    with pytest.raises(ValueError, match="log.print_output_on_finish must be bool"):
        DfxRuntimeConfig._validate(data)


def test_report_rejects_legacy_print_keys():
    data = _valid_dfx_data()
    data["report"]["print_sampling_meta"] = True
    with pytest.raises(ValueError, match=r"report has unknown key\(s\) \['print_sampling_meta'\]"):
        DfxRuntimeConfig._validate(data)


def test_input_filters_roundtrip(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.save(
        {
            "input_filter": {
                "filters": [
                    {
                        "type": "input_token_id_prefix",
                        "mode": "include",
                        "prefixes": [[1, 2], [9]],
                    },
                    {
                        "type": "prompt_length",
                        "mode": "exclude",
                        "op": "lt",
                        "value": 8,
                    },
                ]
            }
        }
    )
    configs = cfg.input_filter_configs()
    assert len(configs) == 2
    assert configs[0]["type"] == "input_token_id_prefix"
    assert configs[0]["prefixes"] == [[1, 2], [9]]
    assert configs[1]["type"] == "prompt_length"
    reloaded = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert reloaded["input_filter"]["filters"][0]["prefixes"] == [[1, 2], [9]]


def test_input_filters_rejects_bad_type(tmp_path: Path):
    import pytest

    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig as Cfg

    cfg_path = tmp_path / "bad_filters.json"
    cfg_path.write_text(
        json.dumps({"input_filter": {"filters": [{"type": "nope"}]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported"):
        Cfg(
            cfg_path,
            report_dir=tmp_path / "report",
            ensure_file=False,
            sync_mode="file",
            reload_interval_seconds=0,
        )


def test_explicit_path_reads_json(tmp_path: Path):
    """Explicit dfx_config_path reads JSON and merges with defaults."""
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    msprobe = _msprobe_path(tmp_path)
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 9,
                    "auto_cooldown_seconds": 10,
                    "manual_dump": False,
                    "msprobe_config_path": msprobe,
                },
                "detector": {
                    "token_logprob": {"enabled": True},
                    "spec_acceptance": {"window": 33},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 9
    assert cfg.detector_get("token_logprob", "enabled") is True
    assert cfg.detector_get("spec_acceptance", "window") == 33
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["auto_max_times"] == 9
    assert saved["detector"]["spec_acceptance"]["window"] == 33


def test_no_explicit_path_resets_to_defaults(tmp_path: Path, monkeypatch):
    """Without dfx_config_path, default path overwrites any prior content (leader)."""
    monkeypatch.setenv("RANK", "0")
    root = tmp_path / "cwd"
    root.mkdir()
    monkeypatch.chdir(root)
    cfg_path = root / "dfx" / "config" / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {"auto_max_times": 9},
                "ascend_log": {"level": "DEBUG"},
                "detector": {"spec_acceptance": {"window": 33}},
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        None,  # no explicit path → default under cwd
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.config_path == cfg_path.resolve()
    assert cfg.dump_max_times() == 0
    assert cfg.ascend_log_level() == "INFO"
    assert cfg.detector_get("spec_acceptance", "window") == 10
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["auto_max_times"] == 0
    assert saved["ascend_log"]["level"] == "INFO"
    assert saved["detector"]["spec_acceptance"]["window"] == 10


def test_bootstrap_and_save_skip_persist_on_non_leader(tmp_path: Path, monkeypatch):
    """Non-leader ranks keep in-memory merge but must not write JSON."""
    monkeypatch.setenv("RANK", "1")
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = _msprobe_path(tmp_path)
    prior = {
        "dump": {
            "auto_max_times": 2,
            "auto_cooldown_seconds": 10,
            "manual_dump": False,
            "msprobe_config_path": msprobe,
        },
        "ascend_log": {"level": "INFO"},
        "detector": {"spec_acceptance": {"window": 33}},
    }
    cfg_path.write_text(json.dumps(prior), encoding="utf-8")
    before = cfg_path.read_text(encoding="utf-8")

    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 2
    assert cfg_path.read_text(encoding="utf-8") == before  # disk unchanged
    assert cfg.save({"dump": {"auto_max_times": 1}}) is False
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["auto_max_times"] == 2


def test_ensure_persisted_deferred_to_worker_leader(tmp_path: Path, monkeypatch):
    """AscendConfig-style: no write at ctor; leader ensure_persisted writes once."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 0  # default
    assert not cfg_path.exists()
    assert cfg.ensure_persisted() is True
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["auto_max_times"] == 0
    # Idempotent.
    mtime = cfg_path.stat().st_mtime
    assert cfg.ensure_persisted() is True
    assert cfg_path.stat().st_mtime == mtime


def test_ensure_persisted_skips_rewrite_when_file_exists(tmp_path: Path, monkeypatch):
    """Existing JSON must not be rewritten on restart (mtime churn / clobber)."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = _msprobe_path(tmp_path)
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {"auto_max_times": 7, "msprobe_config_path": msprobe},
                "ascend_log": {"level": "WARNING"},
            }
        ),
        encoding="utf-8",
    )
    mtime_before = cfg_path.stat().st_mtime
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 7
    assert cfg.ensure_persisted() is True
    assert cfg_path.stat().st_mtime == mtime_before
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["auto_max_times"] == 7


def test_ensure_persisted_backfills_omitted_msprobe_config_path(tmp_path: Path, monkeypatch):
    """Explicit existing JSON that omits the key gets that key only, not a full rewrite."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"auto_max_times": 7}, "ascend_log": {"level": "WARNING"}}),
        encoding="utf-8",
    )
    msprobe = str(tmp_path / "msprobe.json")
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
        msprobe_config_path=msprobe,
    )
    assert cfg.dump_msprobe_config_path() == msprobe
    assert cfg.ensure_persisted() is True
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["msprobe_config_path"] == msprobe
    assert saved["dump"]["auto_max_times"] == 7
    assert saved["ascend_log"]["level"] == "WARNING"
    assert "debug" not in saved["ascend_log"]


def test_save_prefers_disk_over_stale_memory(tmp_path: Path, monkeypatch):
    """save() must not wipe hand-edits that landed on disk after bootstrap."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = _msprobe_path(tmp_path)
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 0,
                    "manual_dump": True,
                    "msprobe_config_path": msprobe,
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.dump_max_times() == 0
    # Concurrent hand-edit on disk (stale memory still has max_times=0).
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {
                    "auto_max_times": 5,
                    "manual_dump": False,
                    "msprobe_config_path": msprobe,
                }
            }
        ),
        encoding="utf-8",
    )
    assert cfg.save({"dump": {"manual_dump": False}}) is True
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["auto_max_times"] == 5
    assert saved["dump"]["manual_dump"] is False
    assert cfg.dump_max_times() == 5


def test_overwrite_deferred_keeps_file_until_leader_persists(tmp_path: Path, monkeypatch):
    """API/non-persist bootstrap must not delete default JSON (race with leader)."""
    monkeypatch.delenv("RANK", raising=False)
    root = tmp_path / "cwd"
    root.mkdir()
    monkeypatch.chdir(root)
    cfg_path = root / "dfx" / "config" / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps({"dump": {"auto_max_times": 9}, "ascend_log": {"level": "DEBUG"}}), encoding="utf-8")

    cfg = DfxRuntimeConfig(
        None,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    # In-memory defaults; disk kept until leader ensure_persisted.
    assert cfg.dump_max_times() == 0
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["auto_max_times"] == 9
    monkeypatch.setenv("RANK", "0")
    assert cfg.ensure_persisted() is True
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["auto_max_times"] == 0


def test_non_leader_bootstrap_does_not_delete_leader_json(tmp_path: Path, monkeypatch):
    """RANK!=0 must not unlink the default-path file the leader just wrote."""
    root = tmp_path / "cwd"
    root.mkdir()
    monkeypatch.chdir(root)
    cfg_path = root / "dfx" / "config" / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True)
    # Leader already materialized defaults.
    cfg_path.write_text(json.dumps({"dump": {"auto_max_times": 0}, "ascend_log": {"level": "INFO"}}), encoding="utf-8")

    monkeypatch.setenv("RANK", "1")
    cfg = DfxRuntimeConfig(
        None,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert cfg.ensure_persisted() is False
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["auto_max_times"] == 0


def test_ensure_persisted_skip_non_leader(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.ensure_persisted() is False
    assert not cfg_path.exists()


@pytest.mark.parametrize("env_key", ["RANK", "LOCAL_RANK"])
def test_non_worker_background_reload_skips_when_worker_env_set(tmp_path: Path, monkeypatch, env_key):
    for key in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(env_key, "0")
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert cfg.start_non_worker_background_reload() is False


def test_non_worker_background_reload_starts_without_rank(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("VLLM_DP_RANK", raising=False)
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert cfg.start_non_worker_background_reload() is True
    assert cfg._bg_thread is not None and cfg._bg_thread.is_alive()
    # Idempotent.
    assert cfg.start_non_worker_background_reload() is False


def test_reload_noop_when_file_missing(tmp_path: Path):
    cfg_path = tmp_path / "missing.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=5,
    )
    assert not cfg_path.exists()
    cfg._last_reload_ts = 0.0
    assert cfg.reload(force=False) is False
    assert cfg.dump_max_times() == 0


def test_apply_ascend_log_level_sets_vllm_ascend_loggers(tmp_path: Path):
    import logging

    from vllm_ascend.logger import init_logger_ascend

    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.save({"ascend_log": {"level": "WARNING", "debug": []}})
    cfg.apply_ascend_log_level()
    assert logging.root.level == logging.WARNING
    assert logging.getLogger("vllm_ascend").level == logging.WARNING
    assert logging.getLogger("vllm").level == logging.WARNING

    # Root INFO + debug whitelist: only listed modules become DEBUG.
    other = init_logger_ascend("vllm_ascend.worker.model_runner_v1")
    dfx_logger = init_logger_ascend("vllm_ascend.dfx.runtime_config")
    assert cfg.save({"ascend_log": {"level": "INFO", "debug": ["dfx"]}})
    cfg.apply_ascend_log_level()
    assert logging.getLogger("vllm_ascend").level == logging.INFO
    assert logging.getLogger("vllm_ascend.dfx").level == logging.DEBUG
    assert dfx_logger.level == logging.DEBUG
    assert dfx_logger.isEnabledFor(logging.DEBUG)
    assert other.level == logging.INFO
    assert not other.isEnabledFor(logging.DEBUG)

    # Full-package DEBUG.
    assert cfg.save({"ascend_log": {"level": "DEBUG", "debug": []}})
    cfg.apply_ascend_log_level()
    assert logging.getLogger("vllm_ascend").level == logging.DEBUG
    assert dfx_logger.isEnabledFor(logging.DEBUG)


def test_ascend_log_debug_string_and_enabled_stripped(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = _msprobe_path(tmp_path)
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {"auto_max_times": 1, "manual_dump": False, "msprobe_config_path": msprobe},
                "ascend_log": {"enabled": True, "level": "WARNING", "debug": "dfx"},
            }
        ),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.ascend_log_level() == "WARNING"
    assert cfg.ascend_log_debug_modules() == ["dfx"]
    assert "enabled" not in cfg.ascend_log


def test_ascend_log_debug_full_logger_name(tmp_path: Path):
    import logging

    from vllm_ascend.logger import init_logger_ascend

    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    child = init_logger_ascend("vllm_ascend.dfx.dumper")
    assert cfg.save({"ascend_log": {"level": "INFO", "debug": ["vllm_ascend.dfx.dumper"]}})
    cfg.apply_ascend_log_level()
    assert logging.getLogger("vllm_ascend").level == logging.INFO
    assert child.level == logging.DEBUG
    assert child.isEnabledFor(logging.DEBUG)


def test_ascend_log_defaults_include_empty_debug(tmp_path: Path):
    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    assert cfg.ascend_log_debug_modules() == []
    assert cfg.ascend_log_modules() == {}
    saved = json.loads(cfg.config_path.read_text(encoding="utf-8"))
    assert saved["ascend_log"]["debug"] == []
    assert saved["ascend_log"]["modules"] == {}
    assert "enabled" not in saved["ascend_log"]


def test_ascend_log_modules_dict_per_logger(tmp_path: Path):
    import logging

    from vllm_ascend.logger import init_logger_ascend

    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    vllm_child = logging.getLogger("vllm.worker.test_mod")
    dfx_logger = init_logger_ascend("vllm_ascend.dfx.runtime_config")
    assert cfg.save(
        {
            "ascend_log": {
                "level": "WARNING",
                "debug": [],
                "modules": {
                    "vllm.worker": "ERROR",
                    "dfx": "DEBUG",
                },
            }
        }
    )
    cfg.apply_ascend_log_level()
    assert logging.root.level == logging.WARNING
    assert logging.getLogger("vllm").level == logging.WARNING
    assert vllm_child.level == logging.ERROR
    assert logging.getLogger("vllm_ascend").level == logging.WARNING
    assert dfx_logger.level == logging.DEBUG
    assert dfx_logger.isEnabledFor(logging.DEBUG)


def test_ascend_log_modules_override_debug_list(tmp_path: Path):
    import logging

    from vllm_ascend.logger import init_logger_ascend

    cfg = DfxRuntimeConfig(
        tmp_path / "dfx_config.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    dfx_logger = init_logger_ascend("vllm_ascend.dfx.runtime_config")
    assert cfg.save(
        {
            "ascend_log": {
                "level": "INFO",
                "debug": ["dfx"],
                "modules": {"dfx": "ERROR"},
            }
        }
    )
    cfg.apply_ascend_log_level()
    assert dfx_logger.level == logging.ERROR
    assert not dfx_logger.isEnabledFor(logging.DEBUG)


def test_multi_dp_without_inner_world_falls_back_to_file_poll(tmp_path: Path):
    """Full-world broadcast under multi-DP deadlocks on one-sided dummy; use file."""
    cfg_path = tmp_path / "dfx_config.json"
    msprobe = _msprobe_path(tmp_path)
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="broadcast",
        reload_interval_seconds=5,
        msprobe_config_path=msprobe,
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["auto_max_times"] = 4
    payload["dump"]["manual_dump"] = False
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    world = MagicMock(world_size=32)
    with (
        patch("vllm_ascend.dfx.runtime_config._dfx_config_sync_group_or_none", return_value=None),
        patch("vllm_ascend.dfx.runtime_config._world_group_or_none", return_value=world),
        patch("vllm_ascend.dfx.runtime_config._dp_world_size_or_one", return_value=2),
    ):
        cfg._last_reload_ts = 0.0
        assert cfg.sync_dfx_config() is True
    assert cfg.dump_max_times() == 4


def test_sync_group_prefers_inner_dp_never_full_multi_dp_world():
    world = MagicMock(world_size=32)
    inner = MagicMock(world_size=16)
    with (
        patch("vllm_ascend.dfx.runtime_config._world_group_or_none", return_value=world),
        patch("vllm_ascend.dfx.runtime_config._dp_world_size_or_one", return_value=1),
    ):
        assert _dfx_config_sync_group_or_none() is world
    with (
        patch("vllm_ascend.dfx.runtime_config._world_group_or_none", return_value=world),
        patch("vllm_ascend.dfx.runtime_config._dp_world_size_or_one", return_value=2),
        patch("vllm_ascend.dfx.runtime_config._inner_dp_world_or_none", return_value=inner),
    ):
        assert _dfx_config_sync_group_or_none() is inner
    with (
        patch("vllm_ascend.dfx.runtime_config._world_group_or_none", return_value=world),
        patch("vllm_ascend.dfx.runtime_config._dp_world_size_or_one", return_value=2),
        patch("vllm_ascend.dfx.runtime_config._inner_dp_world_or_none", return_value=None),
    ):
        assert _dfx_config_sync_group_or_none() is None


def test_json_writer_is_per_dp_leader_not_only_global_rank0():
    """DP=2 must allow each replica's local leader to persist its JSON."""
    inner_leader = MagicMock(world_size=16, is_first_rank=True)
    inner_follower = MagicMock(world_size=16, is_first_rank=False)
    with patch("vllm_ascend.dfx.runtime_config._inner_dp_world_or_none", return_value=inner_leader):
        assert _is_json_writer() is True
    with patch("vllm_ascend.dfx.runtime_config._inner_dp_world_or_none", return_value=inner_follower):
        assert _is_json_writer() is False

    tp0 = MagicMock(is_first_rank=True)
    pp0 = MagicMock(is_first_rank=True)
    tp1 = MagicMock(is_first_rank=False)
    with (
        patch("vllm_ascend.dfx.runtime_config._inner_dp_world_or_none", return_value=None),
        patch("vllm_ascend.dfx.runtime_config._dp_world_size_or_one", return_value=2),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp0),
        patch("vllm.distributed.parallel_state.get_pp_group", return_value=pp0),
    ):
        assert _is_json_writer() is True
    with (
        patch("vllm_ascend.dfx.runtime_config._inner_dp_world_or_none", return_value=None),
        patch("vllm_ascend.dfx.runtime_config._dp_world_size_or_one", return_value=2),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=tp1),
        patch("vllm.distributed.parallel_state.get_pp_group", return_value=pp0),
    ):
        assert _is_json_writer() is False


def test_broadcast_sync_applies_leader_payload_to_follower(tmp_path: Path):
    leader = DfxRuntimeConfig(
        tmp_path / "leader.json",
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="broadcast",
        reload_interval_seconds=5,
    )
    follower = DfxRuntimeConfig(
        tmp_path / "follower.json",
        report_dir=tmp_path / "report2",
        ensure_file=True,
        sync_mode="broadcast",
        reload_interval_seconds=5,
    )
    # Simulate leader edited dump.max_times.
    assert leader.save({"dump": {"auto_max_times": 7}})

    world = MagicMock()
    world.world_size = 2
    world.cpu_group = object()
    world.is_first_rank = True

    # Leader path: build payload via real reload, then hand object to follower.
    with (
        patch("vllm_ascend.dfx.runtime_config._dfx_config_sync_group_or_none", return_value=world),
        patch("vllm_ascend.dfx.runtime_config._world_group_or_none", return_value=world),
        patch("torch.distributed.all_reduce") as ar,
        patch.object(world, "broadcast_object", side_effect=lambda obj, src=0: obj),
    ):
        ar.side_effect = lambda t, op=None, group=None: t.fill_(1.0)
        leader._last_reload_ts = 0.0
        leader._initial_broadcast_done = False
        assert leader.sync_dfx_config() is True

    world.is_first_rank = False
    payload = {"version": float(leader._version), "data": leader._data}
    with (
        patch("vllm_ascend.dfx.runtime_config._dfx_config_sync_group_or_none", return_value=world),
        patch("vllm_ascend.dfx.runtime_config._world_group_or_none", return_value=world),
        patch("torch.distributed.all_reduce") as ar,
        patch.object(world, "broadcast_object", side_effect=lambda obj, src=0: payload),
    ):
        ar.side_effect = lambda t, op=None, group=None: t.fill_(1.0)
        follower._last_reload_ts = 0.0
        follower._initial_broadcast_done = False
        # Local follower.json was created in the same fs timestamp tick as
        # leader.json's save — without this the version-tie makes the payload
        # look already-applied (flake). Simulate a fresh follower instead.
        follower._version = 0.0
        assert follower.sync_dfx_config() is True
    assert follower.dump_max_times() == 7
