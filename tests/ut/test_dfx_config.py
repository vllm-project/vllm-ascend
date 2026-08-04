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
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.dfx.runtime_config import (
    DfxRuntimeConfig,
    _dfx_config_sync_group_or_none,
    _is_json_writer,
    _leaf_changes,
)


def test_leaf_changes_reports_only_diffs():
    old = {"dump": {"max_times": 0, "enabled": True}, "ascend_log": {"level": "INFO"}}
    new = {"dump": {"max_times": 3, "enabled": True}, "ascend_log": {"level": "INFO"}}
    assert _leaf_changes(old, new) == ["dump.max_times: 0 -> 3"]


def test_dfx_config_hot_reload_and_defaults(tmp_path: Path):
    cfg_path = tmp_path / "dfx_config.json"
    report_dir = tmp_path / "report"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=report_dir,
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=5,
    )

    assert cfg_path.exists()
    assert cfg.hot_reload_enabled is True
    assert cfg.dump_max_times() == 0
    assert cfg.ascend_log_level() == "INFO"
    assert cfg.detector_get("spec_acceptance", "enabled") is False

    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["max_times"] = 3
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
def test_detector_bool_knob_defaults_and_validation(
    tmp_path: Path, payload, getter, default, expected, err_match
):
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
    payload["dump"]["max_times"] = 9
    cfg.config_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg._last_reload_ts = 0.0
    assert cfg.sync_dfx_config() is False
    assert cfg.dump_max_times() == 0



def test_dump_enabled_without_detector_allowed(tmp_path: Path):
    """dump.enabled and detectors are orthogonal; manual-only dump is valid."""
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {"enabled": True, "max_times": 3},
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
    assert "manual_dump_only" in cfg.interaction_mode_summary()
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["enabled"] is True


def test_dump_once_not_consumed_when_dump_disabled(tmp_path: Path):
    from vllm_ascend.dfx.detector.manual_dump import ManualDumpDetector

    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"enabled": False, "dump_once": True}}),
        encoding="utf-8",
    )
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="file",
        reload_interval_seconds=0,
    )
    # Bootstrap may rewrite; ensure flags.
    cfg._data["dump"]["enabled"] = False
    cfg._data["dump"]["dump_once"] = True
    det = ManualDumpDetector(dfx_config=cfg, runner=None)
    assert det.check_all() == []
    assert cfg.dump_once() is True




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
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {"enabled": True, "max_times": 9, "cooldown_seconds": 10},
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
    assert saved["dump"]["max_times"] == 9
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
                "dump": {"max_times": 9},
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
    assert saved["dump"]["max_times"] == 0
    assert saved["ascend_log"]["level"] == "INFO"
    assert saved["detector"]["spec_acceptance"]["window"] == 10



def test_bootstrap_and_save_skip_persist_on_non_leader(tmp_path: Path, monkeypatch):
    """Non-leader ranks keep in-memory merge but must not write JSON."""
    monkeypatch.setenv("RANK", "1")
    cfg_path = tmp_path / "dfx_config.json"
    prior = {
        "dump": {"enabled": True, "max_times": 2, "cooldown_seconds": 10, "dump_once": False},
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
    assert cfg.save({"dump": {"max_times": 1}}) is False
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 2



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
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 0
    # Idempotent.
    mtime = cfg_path.stat().st_mtime
    assert cfg.ensure_persisted() is True
    assert cfg_path.stat().st_mtime == mtime


def test_ensure_persisted_skips_rewrite_when_file_exists(tmp_path: Path, monkeypatch):
    """Existing JSON must not be rewritten on restart (mtime churn / clobber)."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(
        json.dumps({"dump": {"max_times": 7}, "ascend_log": {"level": "WARNING"}}),
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
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 7


def test_save_prefers_disk_over_stale_memory(tmp_path: Path, monkeypatch):
    """save() must not wipe hand-edits that landed on disk after bootstrap."""
    monkeypatch.setenv("RANK", "0")
    cfg_path = tmp_path / "dfx_config.json"
    cfg_path.write_text(json.dumps({"dump": {"max_times": 0, "dump_once": True}}), encoding="utf-8")
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
        json.dumps({"dump": {"max_times": 5, "dump_once": True}}),
        encoding="utf-8",
    )
    assert cfg.save({"dump": {"dump_once": False}}) is True
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["dump"]["max_times"] == 5
    assert saved["dump"]["dump_once"] is False
    assert cfg.dump_max_times() == 5


def test_overwrite_deferred_keeps_file_until_leader_persists(tmp_path: Path, monkeypatch):
    """API/non-persist bootstrap must not delete default JSON (race with leader)."""
    monkeypatch.delenv("RANK", raising=False)
    root = tmp_path / "cwd"
    root.mkdir()
    monkeypatch.chdir(root)
    cfg_path = root / "dfx" / "config" / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(json.dumps({"dump": {"max_times": 9}, "ascend_log": {"level": "DEBUG"}}), encoding="utf-8")

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
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 9
    monkeypatch.setenv("RANK", "0")
    assert cfg.ensure_persisted() is True
    assert cfg_path.exists()
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 0


def test_non_leader_bootstrap_does_not_delete_leader_json(tmp_path: Path, monkeypatch):
    """RANK!=0 must not unlink the default-path file the leader just wrote."""
    root = tmp_path / "cwd"
    root.mkdir()
    monkeypatch.chdir(root)
    cfg_path = root / "dfx" / "config" / "dfx_config.json"
    cfg_path.parent.mkdir(parents=True)
    # Leader already materialized defaults.
    cfg_path.write_text(json.dumps({"dump": {"max_times": 0}, "ascend_log": {"level": "INFO"}}), encoding="utf-8")

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
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["max_times"] == 0


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
    assert logging.getLogger("vllm_ascend").level == logging.WARNING

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
    cfg_path.write_text(
        json.dumps(
            {
                "dump": {"max_times": 1},
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
    saved = json.loads(cfg.config_path.read_text(encoding="utf-8"))
    assert saved["ascend_log"]["debug"] == []
    assert "enabled" not in saved["ascend_log"]


def test_multi_dp_without_inner_world_falls_back_to_file_poll(tmp_path: Path):
    """Full-world broadcast under multi-DP deadlocks on one-sided dummy; use file."""
    cfg_path = tmp_path / "dfx_config.json"
    cfg = DfxRuntimeConfig(
        cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        sync_mode="broadcast",
        reload_interval_seconds=5,
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["dump"]["max_times"] = 4
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
    assert leader.save({"dump": {"max_times": 7}})

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
        assert follower.sync_dfx_config() is True
    assert follower.dump_max_times() == 7
