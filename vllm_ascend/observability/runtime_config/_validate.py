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

"""Validate / coerce runtime_config JSON payloads in place."""

from __future__ import annotations

from typing import Any

from vllm_ascend.observability.runtime_config._defaults import (
    _DEFAULTS,
    ACTIONS_KEYS,
    ASCEND_LOG_KEYS,
    DETECTOR_KEYS,
    DETECTOR_SECTIONS,
    DUMP_KEYS,
    LOG_KEYS,
    MANUAL_TRIGGER_SECTION_KEYS,
    REPORT_KEYS,
    TOP_LEVEL_KEYS,
)
from vllm_ascend.observability.runtime_config._dist import SYNC_BROADCAST, SYNC_FILE
from vllm_ascend.observability.runtime_config._merge import _normalize_config_sections_into


def _is_int_list(value: Any) -> bool:
    """True when ``value`` is a non-empty ``list[int]`` (bool excluded)."""
    return (
        isinstance(value, list) and bool(value) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    )


def normalize_raw_patterns(raw: Any) -> list[Any]:
    """Validate/filter ``detector.output_substring.patterns`` entries (no tokenizer)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("detector.output_substring.patterns must be a list of str or int lists")
    out: list[Any] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            if item:
                out.append(item)
            continue
        if _is_int_list(item):
            out.append([int(x) for x in item])
            continue
        raise ValueError(
            f"detector.output_substring.patterns[{i}] must be a non-empty str or "
            f"non-empty list[int], got {type(item).__name__}"
        )
    return out


def normalize_ignore_token_ids(raw: Any) -> list[int]:
    """Validate config ``ignore_token_ids`` as a flat list of ints."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"ignore_token_ids must be a list of ints, got {type(raw).__name__}")
    out: list[int] = []
    for i, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"ignore_token_ids[{i}] must be int, got {item!r}")
        out.append(int(item))
    return out


def int_field(value: Any, field: str, *, min_value: int | None = None) -> int:
    # C3: reject None/str/NaN and silently-truncated floats (2.7 → 2) with
    # an error that names the offending field.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise ValueError(f"{field} must be a number, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer, got {value!r}")
    iv = int(value)
    if min_value is not None and iv < min_value:
        raise ValueError(f"{field} must be >= {min_value}, got {iv}")
    return iv


def float_field(
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


def validate_dump_mutual_exclusive(dump: dict[str, Any]) -> None:
    try:
        auto_on = int(dump.get("auto_max_times", 0)) > 0
    except (TypeError, ValueError):
        auto_on = False
    manual_raw = dump.get("manual_dump", False)
    manual_on = manual_raw not in (False, 0)
    if auto_on and manual_on:
        raise ValueError("dump.auto_max_times>0 and dump.manual_dump active are mutually exclusive")


def validate_runtime_config(data: dict[str, Any]) -> None:
    """Validate / normalize ``data`` in place.

    Detect and dump are orthogonal: dump-only / detect-only / both are valid.
    Soft warnings for easy-to-misread combos live on ``RuntimeConfig``.

    S10 fix: defensively re-run ``_normalize_config_sections`` at entry so
    validation is safe regardless of whether the caller normalized first.
    """
    _normalize_config_sections_into(data)
    unknown_top = sorted(set(data) - TOP_LEVEL_KEYS)
    if unknown_top:
        raise ValueError(f"runtime config has unknown top-level key(s) {unknown_top}; allowed={sorted(TOP_LEVEL_KEYS)}")
    for section in (
        "dump",
        "ascend_log",
        "log",
        "detector",
        "report",
        "actions",
    ):
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"runtime config missing object section '{section}'")
    unknown_actions = sorted(set(data["actions"]) - ACTIONS_KEYS)
    if unknown_actions:
        raise ValueError(f"actions has unknown key(s) {unknown_actions}; allowed={sorted(ACTIONS_KEYS)}")
    data["actions"]["queue_max_size"] = int_field(
        data["actions"].get("queue_max_size", _DEFAULTS["actions"]["queue_max_size"]),
        "actions.queue_max_size",
        min_value=1,
    )
    interval = data.get("reload_interval_seconds", 0)
    if not isinstance(interval, (int, float)) or interval < 0:
        raise ValueError(f"reload_interval_seconds must be >= 0, got {interval}")
    sync_mode = str(data.get("sync_mode", SYNC_BROADCAST)).lower()
    if sync_mode not in (SYNC_BROADCAST, SYNC_FILE):
        raise ValueError(f"sync_mode must be '{SYNC_BROADCAST}' or '{SYNC_FILE}'")
    unknown_dump = sorted(set(data["dump"]) - DUMP_KEYS)
    if unknown_dump:
        raise ValueError(f"dump has unknown key(s) {unknown_dump}; allowed={sorted(DUMP_KEYS)}")
    auto_max_times = data["dump"].get("auto_max_times", 0)
    data["dump"]["auto_max_times"] = int_field(auto_max_times, "dump.auto_max_times", min_value=0)
    auto_cd = data["dump"].get("auto_cooldown_seconds", 300)
    data["dump"]["auto_cooldown_seconds"] = int_field(auto_cd, "dump.auto_cooldown_seconds", min_value=0)
    manual_dump = data["dump"].get("manual_dump")
    if manual_dump is not None and not isinstance(manual_dump, bool):
        if isinstance(manual_dump, int) and not isinstance(manual_dump, bool):
            if manual_dump < 0:
                raise ValueError("dump.manual_dump must be >= 0")
            if manual_dump == 0:
                data["dump"]["manual_dump"] = False
        else:
            raise ValueError("dump.manual_dump must be bool or non-negative int")
    dump_dir_raw = data["dump"].get("dump_dir")
    if dump_dir_raw is not None and not isinstance(dump_dir_raw, str):
        raise ValueError("dump.dump_dir must be a string path or null")
    data["dump"]["free_headroom_bytes"] = int_field(
        data["dump"].get("free_headroom_bytes", _DEFAULTS["dump"]["free_headroom_bytes"]),
        "dump.free_headroom_bytes",
        min_value=0,
    )
    validate_dump_mutual_exclusive(data["dump"])
    unknown_log = sorted(set(data["log"]) - LOG_KEYS)
    if unknown_log:
        raise ValueError(f"log has unknown key(s) {unknown_log}; allowed={sorted(LOG_KEYS)}")
    for log_key in ("print_output_on_finish",):
        log_val = data["log"].get(log_key)
        if log_val is not None and not isinstance(log_val, bool):
            if log_val in (0, 1):
                data["log"][log_key] = bool(log_val)
            else:
                raise ValueError(f"log.{log_key} must be bool")
    unknown_report = sorted(set(data["report"]) - REPORT_KEYS)
    if unknown_report:
        raise ValueError(f"report has unknown key(s) {unknown_report}; allowed={sorted(REPORT_KEYS)}")
    save_sensitive = data["report"].get("save_sensitive_info")
    if save_sensitive is not None and not isinstance(save_sensitive, bool):
        if save_sensitive in (0, 1):
            data["report"]["save_sensitive_info"] = bool(save_sensitive)
        else:
            raise ValueError("report.save_sensitive_info must be bool")
    decode_ids = data["report"].get("decode_token_ids")
    if decode_ids is not None and not isinstance(decode_ids, bool):
        if decode_ids in (0, 1):
            data["report"]["decode_token_ids"] = bool(decode_ids)
        else:
            raise ValueError("report.decode_token_ids must be bool")
    for max_key in ("max_prompt_token_ids", "max_output_token_ids"):
        max_val = data["report"].get(max_key)
        if max_val is None:
            continue
        if isinstance(max_val, bool) or not isinstance(max_val, (int, float)):
            raise ValueError(f"report.{max_key} must be an int >= 0")
        if int(max_val) < 0:
            raise ValueError(f"report.{max_key} must be >= 0")
        data["report"][max_key] = int(max_val)
    if "max_per_req" in data["report"] and data["report"]["max_per_req"] is not None:
        data["report"]["max_per_req"] = int_field(data["report"]["max_per_req"], "report.max_per_req", min_value=1)
    block_val = data["report"].get("include_block_ids")
    if block_val is not None and not isinstance(block_val, bool):
        if block_val in (0, 1):
            data["report"]["include_block_ids"] = bool(block_val)
        else:
            raise ValueError("report.include_block_ids must be bool")
    level = data["ascend_log"].get("level", "INFO")
    if not isinstance(level, str):
        raise ValueError("ascend_log.level must be str")
    unknown_ascend = sorted(set(data["ascend_log"]) - ASCEND_LOG_KEYS)
    if unknown_ascend:
        raise ValueError(f"ascend_log has unknown key(s) {unknown_ascend}; allowed={sorted(ASCEND_LOG_KEYS)}")
    debug = data["ascend_log"].get("debug", [])
    if not isinstance(debug, list):
        raise ValueError("ascend_log.debug must be a list of module name strings")
    for item in debug:
        if not isinstance(item, (str, int, float)):
            raise ValueError("ascend_log.debug entries must be strings")
    modules = data["ascend_log"].get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("ascend_log.modules must be an object")
    for key, val in modules.items():
        if not isinstance(key, str):
            raise ValueError("ascend_log.modules keys must be strings")
        if not isinstance(val, str):
            raise ValueError("ascend_log.modules values must be strings")
    detector = data["detector"]
    known = set(DETECTOR_SECTIONS)
    for key, value in detector.items():
        if key == "manual_trigger":
            # Control-plane overrides for incident_type=manual_trigger (not a detector).
            if not isinstance(value, dict):
                raise ValueError("detector.manual_trigger must be an object")
            unknown_sub = sorted(set(value) - MANUAL_TRIGGER_SECTION_KEYS)
            if unknown_sub:
                raise ValueError(
                    f"detector.manual_trigger has unknown key(s) {unknown_sub}; "
                    f"allowed={sorted(MANUAL_TRIGGER_SECTION_KEYS)}"
                )
            continue
        if key not in known:
            raise ValueError(
                f"detector.{key} is not a known detector section; "
                f"expected nested objects among {sorted(known)} "
                f"(e.g. detector.spec_acceptance.enabled)"
            )
        if not isinstance(value, dict):
            raise ValueError(f"detector.{key} must be an object")
        unknown_sub = sorted(set(value) - DETECTOR_KEYS[key])
        if unknown_sub:
            raise ValueError(f"detector.{key} has unknown key(s) {unknown_sub}; allowed={sorted(DETECTOR_KEYS[key])}")
    for name in DETECTOR_SECTIONS:
        sec = detector.setdefault(name, {})
        if not isinstance(sec, dict):
            raise ValueError(f"detector.{name} must be an object")
        enabled = sec.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            if enabled in (0, 1):
                sec["enabled"] = bool(enabled)
            else:
                raise ValueError(f"detector.{name}.enabled must be bool")

    out_sub = detector["output_substring"]
    out_sub["patterns"] = normalize_raw_patterns(out_sub.get("patterns", []))
    add_special = out_sub.get("add_special_tokens")
    if add_special is not None and not isinstance(add_special, bool):
        if add_special in (0, 1):
            out_sub["add_special_tokens"] = bool(add_special)
        else:
            raise ValueError("detector.output_substring.add_special_tokens must be bool")
    match_prefix = out_sub.get("match_prefix")
    if match_prefix is not None and not isinstance(match_prefix, bool):
        if match_prefix in (0, 1):
            out_sub["match_prefix"] = bool(match_prefix)
        else:
            raise ValueError("detector.output_substring.match_prefix must be bool")

    token_repeat = detector["token_repeat"]
    token_repeat["window"] = int_field(token_repeat.get("window", 32), "detector.token_repeat.window", min_value=1)
    token_repeat["repeat_sum_threshold"] = int_field(
        token_repeat.get("repeat_sum_threshold", 64),
        "detector.token_repeat.repeat_sum_threshold",
        min_value=0,
    )
    token_repeat["min_tokens"] = int_field(
        token_repeat.get("min_tokens", token_repeat["window"]),
        "detector.token_repeat.min_tokens",
        min_value=0,
    )
    token_repeat["consecutive_hits"] = int_field(
        token_repeat.get("consecutive_hits", 1),
        "detector.token_repeat.consecutive_hits",
        min_value=1,
    )
    token_repeat["ignore_token_ids"] = normalize_ignore_token_ids(token_repeat.get("ignore_token_ids", []))

    spec = detector["spec_acceptance"]
    spec["window"] = int_field(spec.get("window", 10), "detector.spec_acceptance.window", min_value=1)
    for rate_key in ("low_threshold", "high_threshold"):
        spec[rate_key] = float_field(
            spec.get(rate_key, _DEFAULTS["detector"]["spec_acceptance"][rate_key]),
            f"detector.spec_acceptance.{rate_key}",
            min_value=0.0,
            max_value=1.0,
        )
    for len_key in ("len_low_threshold", "len_high_threshold"):
        spec[len_key] = float_field(
            spec.get(len_key, _DEFAULTS["detector"]["spec_acceptance"][len_key]),
            f"detector.spec_acceptance.{len_key}",
            min_value=0.0,
        )
    spec["short_log_interval_seconds"] = float_field(
        spec.get(
            "short_log_interval_seconds",
            _DEFAULTS["detector"]["spec_acceptance"]["short_log_interval_seconds"],
        ),
        "detector.spec_acceptance.short_log_interval_seconds",
        min_value=0.0,
    )

    logits = detector["logits_finite"]
    logits["deferred_queue_max"] = int_field(
        logits.get(
            "deferred_queue_max",
            _DEFAULTS["detector"]["logits_finite"]["deferred_queue_max"],
        ),
        "detector.logits_finite.deferred_queue_max",
        min_value=1,
    )
