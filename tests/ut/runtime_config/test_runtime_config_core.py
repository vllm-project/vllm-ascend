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
        hot_reload=False,
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
        hot_reload=False,
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
        hot_reload=True,
        startup_overlay={
            "detector": {"token_repeat": {"enabled": True, "window": 16}},
        },
    )
    cfg._reload_interval = 0.01  # UT: avoid waiting fixed 3s
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
        hot_reload=True,
    )
    cfg._reload_interval = 0.01  # UT: avoid waiting fixed 3s
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
        hot_reload=True,
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
        hot_reload=True,
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
    from vllm_ascend.observability.runtime_config.config import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["dump"]["manual_trigger"] = 2
    with pytest.raises(ValueError, match="unknown key"):
        validate_runtime_config(data)


def test_token_repeat_window_wrong_type_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.config import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["detector"]["token_repeat"]["window"] = "x"
    with pytest.raises(ValueError, match="detector.token_repeat.window"):
        validate_runtime_config(data)


def test_token_repeat_window_non_integer_float_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.config import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["detector"]["token_repeat"]["window"] = 2.7
    with pytest.raises(ValueError, match="must be an integer"):
        validate_runtime_config(data)


def test_report_save_sensitive_wrong_type_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.config import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["report"]["save_sensitive_info"] = "yes"
    with pytest.raises(ValueError, match="save_sensitive_info"):
        validate_runtime_config(data)


def test_ascend_log_enabled_unknown_key_rejected():
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.config import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["ascend_log"]["enabled"] = True
    with pytest.raises(ValueError, match="unknown key"):
        validate_runtime_config(data)


def test_validate_rejects_top_level_reload_interval() -> None:
    """Unknown top-level keys (incl. retired reload_interval_seconds) hard-fail."""
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.config import validate_runtime_config

    data = deepcopy(_DEFAULTS)
    data["reload_interval_seconds"] = 999
    with pytest.raises(ValueError, match="unknown top-level key"):
        validate_runtime_config(data)


def test_startup_overlay_unknown_top_level_reload_interval_falls_back(tmp_path: Path):
    """Bad overlay is rejected at bootstrap; service keeps defaults."""
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        hot_reload=True,
        startup_overlay={
            "reload_interval_seconds": 999,
            "detector": {"token_repeat": {"enabled": True}},
        },
    )
    assert "reload_interval_seconds" not in cfg._data
    # Entire overlay dropped when validate fails — detector overlay not applied.
    assert cfg.detector_get("token_repeat", "enabled") is False


def test_unknown_nested_keys_hard_fail() -> None:
    """Unknown nested keys (formerly soft-popped) must fail validation."""
    from copy import deepcopy

    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.config import validate_runtime_config

    cases = [
        ("actions", {"queue_max_size": 128}, "queue_max_size"),
        ("dump", {"free_headroom_bytes": 1}, "free_headroom_bytes"),
        ("report", {"decode_token_ids": False}, "decode_token_ids"),
        ("report", {"include_block_ids": False}, "include_block_ids"),
        ("detector.spec_acceptance", {"short_log_interval_seconds": 9.0}, "short_log_interval_seconds"),
        ("detector.logits_finite", {"deferred_queue_max": 1}, "deferred_queue_max"),
        ("detector", {"output_substring": {"enabled": True}}, "output_substring"),
    ]
    for path, patch, needle in cases:
        data = deepcopy(_DEFAULTS)
        if path == "detector":
            data["detector"].update(patch)
        elif path.startswith("detector."):
            section = path.split(".", 1)[1]
            data["detector"][section].update(patch)
        else:
            data[path].update(patch)
        with pytest.raises(ValueError, match=needle):
            validate_runtime_config(data)


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


def test_manual_dump_watermark_never_rewrites_json(tmp_path: Path):
    """``manual_dump: N`` is a watermark; consume bumps done, leaves JSON alone."""
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        hot_reload=True,
    )
    _write(cfg_path, {"dump": {"manual_dump": 3, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.manual_dump_target() == 3
    assert cfg.manual_dumps_done() == 0
    assert cfg.manual_trigger_count() == 3

    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_dumps_done() == 1
    assert cfg.manual_trigger_count() == 2
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["manual_dump"] == 3

    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_dumps_done() == 2
    assert cfg.manual_trigger_count() == 1
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["manual_dump"] == 3

    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_dumps_done() == 3
    assert cfg.manual_trigger_count() == 0
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["manual_dump"] == 3
    assert cfg.consume_manual_trigger() is False


def test_manual_dump_skip_when_target_leq_done(tmp_path: Path):
    """Disk value ≤ already-completed dumps → no dump; raise N to dump again."""
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        hot_reload=True,
    )
    _write(cfg_path, {"dump": {"manual_dump": 1, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_dumps_done() == 1
    assert cfg.manual_trigger_count() == 0

    # Same or lower target: skip.
    _write(cfg_path, {"dump": {"manual_dump": 1, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.manual_trigger_count() == 0
    assert cfg.consume_manual_trigger() is False

    _write(cfg_path, {"dump": {"manual_dump": 0, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.manual_trigger_count() == 0

    # Bump above done → one more dump.
    _write(cfg_path, {"dump": {"manual_dump": 2, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.manual_trigger_count() == 1
    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_dumps_done() == 2
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["manual_dump"] == 2


def test_manual_dump_hand_edit_raises_target_while_catching_up(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        hot_reload=True,
    )
    _write(cfg_path, {"dump": {"manual_dump": 3, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_dumps_done() == 1
    assert cfg.manual_trigger_count() == 2

    # Hand-edit raises target; done stays — remaining grows.
    _write(cfg_path, {"dump": {"manual_dump": 5, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.manual_dumps_done() == 1
    assert cfg.manual_dump_target() == 5
    assert cfg.manual_trigger_count() == 4


def test_bootstrap_overwrite_wipes_prestart_manual_dump_then_hot_reload_rearms(
    tmp_path: Path,
):
    """Pre-start hand-edit of manual_dump is overwritten; post-start hot-reload arms it."""
    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {"dump": {"manual_dump": 1, "auto_max_times": 0}})

    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=False,  # production: persist deferred to ensure_persisted
        hot_reload=True,
    )
    # Bootstrap ignores the pre-start file → default false.
    assert cfg.manual_dump_target() == 0
    assert cfg.manual_trigger_count() == 0
    assert cfg.ensure_persisted() is True
    on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert on_disk["dump"]["manual_dump"] is False

    # Ops path: start first, then raise N in the live file.
    _write(cfg_path, {"dump": {"manual_dump": 1, "auto_max_times": 0}})
    assert cfg.reload(force=True) is True
    assert cfg.manual_dump_target() == 1
    assert cfg.manual_trigger_count() == 1
    assert cfg.consume_manual_trigger() is True
    assert cfg.manual_dumps_done() == 1
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["dump"]["manual_dump"] == 1


def test_same_mtime_content_change_reloads_via_digest(tmp_path: Path):
    """Bug #11: same-second equal-size edits must reload via content digest."""
    import os

    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        hot_reload=True,
    )
    # Equal-length values so st_size can stay stable; pin mtime after rewrite.
    _write(cfg_path, {"detector": {"token_repeat": {"enabled": False, "window": 10}}})
    assert cfg.reload(force=True) is True
    assert cfg.detector_get("token_repeat", "window") == 10
    pinned = cfg._mtime
    assert pinned is not None

    _write(cfg_path, {"detector": {"token_repeat": {"enabled": False, "window": 33}}})
    os.utime(cfg_path, (pinned, pinned))
    assert cfg.reload(force=False) is True
    assert cfg.detector_get("token_repeat", "window") == 33


def test_touch_same_content_skips_reload(tmp_path: Path):
    """mtime bump with identical bytes must not re-apply."""
    import os
    import time as _time

    cfg_path = tmp_path / "runtime_config.json"
    _write(cfg_path, {})
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        hot_reload=True,
    )
    _write(cfg_path, {"detector": {"token_repeat": {"enabled": True, "window": 8}}})
    assert cfg.reload(force=True) is True
    assert cfg.detector_get("token_repeat", "window") == 8
    digest_before = cfg._content_digest
    # Bump mtime without changing bytes.
    now = _time.time() + 5
    os.utime(cfg_path, (now, now))
    assert cfg.reload(force=False) is False
    assert cfg._content_digest == digest_before
    assert cfg.detector_get("token_repeat", "window") == 8


# ---- JSONC (loads_jsonc lives in config.py) ---------------------------------


def test_loads_jsonc_comments_and_trailing_commas():
    from vllm_ascend.observability.runtime_config.config import loads_jsonc

    raw = """
    {
      // line comment
      "a": 1,
      "b": [2, 3,],
      /* block
         comment */
      "c": "keep // inside",
    }
    """
    assert loads_jsonc(raw) == {"a": 1, "b": [2, 3], "c": "keep // inside"}


def test_loads_jsonc_escaped_quote_in_string():
    from vllm_ascend.observability.runtime_config.config import loads_jsonc

    raw = r'{ "msg": "say \"hi\"", }'
    assert loads_jsonc(raw) == {"msg": 'say "hi"'}
