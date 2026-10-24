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

"""Runtime control-plane config (hot-reload JSON).

Design (multi-DP safe — avoid full-world / cross-PP collectives):

1. **One writer / monitor per EngineCore (per DP replica)**
   Materializes JSON via ``ensure_persisted`` at bind (overwrite).
   ``dump.manual_dump`` is a read-only watermark — never rewritten by consume.
   Prefer ``inner_dp_world`` first rank; else TP0∧PP0 when ``dp_size>1``; else
   global world rank0 / ``RANK==0``.

2. **Fixed sync transport** (no JSON ``sync_mode`` knob)
   - last-PP × ``tp_size>1``: wave-head submits TP0-sourced due
     ``broadcast([wave_idx, config_due, dump_due])`` on ``DueBitsBusWorker``
     (``sync_due_bits_from_src``); end-of-wave drains / applies.
   - everyone else (non-last PP, API, ``tp_size<=1``): **local file poll**.

3. **Cross-DP config is not synchronized**
   TP groups are per EngineCore. Edit each DP's JSON (or a shared path each
   replica can see). Do **not** use full EP ``world_size`` for hot-reload.

Production: ``AscendConfig`` uses ``ensure_file=False``; worker
:meth:`RuntimeConfig.ensure_persisted` materializes JSON on the writer.

Payload helpers (JSONC load, deep-merge, validate) live in this module.
Defaults stay in ``_defaults`` (detector-cycle leaf); distributed roles /
task-bus live in ``dist``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_config._defaults import (
    _DEFAULTS,
    _RETIRED_ACTIONS_KEYS,
    _RETIRED_DETECTOR_KEYS,
    _RETIRED_DUMP_KEYS,
    _RETIRED_REPORT_KEYS,
    ACTION_QUEUE_MAX_SIZE,
    ACTIONS_KEYS,
    ASCEND_LOG_KEYS,
    DETECTOR_KEYS,
    DUMP_KEYS,
    HOT_RELOAD_INTERVAL_SECONDS,
    MANUAL_TRIGGER_SECTION_KEYS,
    REPORT_KEYS,
    TOP_LEVEL_KEYS,
)
from vllm_ascend.observability.runtime_config.detector_catalog import (
    DETECTOR_SECTIONS as _CATALOG_DETECTOR_SECTIONS,
)
from vllm_ascend.observability.runtime_config.detector_catalog import (
    RETIRED_DETECTOR_SECTIONS as _RETIRED_DETECTOR_SECTIONS,
)
from vllm_ascend.observability.runtime_config.detector_catalog import (
    validate_registered_detectors,
)
from vllm_ascend.observability.runtime_config.dist import (
    _bg_reload_paths,
    _is_distributed_worker_process,
    _is_json_writer,
    _log_file_poll_fallback_once,
    _process_role_tag,
)

# ---- JSONC ------------------------------------------------------------


def _strip_jsonc(text: str) -> str:
    """String-aware removal of comments and commas directly preceding ``}`` / ``]``.

    Trailing commas may sit before a comment that itself precedes ``}`` / ``]``
    (e.g. ``{ "a": 1, /* note */ }``). Lookahead therefore skips both whitespace
    and comments before deciding whether to drop the comma.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False

    def _skip_ws_and_comments(start: int) -> int:
        j = start
        while j < n:
            if text[j] in " \t\r\n":
                j += 1
                continue
            if text[j] == "/" and j + 1 < n and text[j + 1] == "/":
                j += 2
                while j < n and text[j] != "\n":
                    j += 1
                continue
            if text[j] == "/" and j + 1 < n and text[j + 1] == "*":
                j += 2
                while j < n and not (text[j] == "*" and j + 1 < n and text[j + 1] == "/"):
                    j += 1
                j += 2
                continue
            break
        return j

    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c == ",":
            j = _skip_ws_and_comments(i + 1)
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def loads_jsonc(text: str) -> Any:
    """``json.loads`` that also accepts ``//`` / ``/* */`` comments and trailing commas.

    Keeps the shipped ``.jsonc`` example template loadable as-is.
    """
    return json.loads(_strip_jsonc(text))


# ---- merge / normalize ------------------------------------------------


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


def dump_auto_on(dump: dict[str, Any]) -> bool:
    """True when ``dump.auto_max_times > 0`` (invalid values → False)."""
    try:
        return int(dump.get("auto_max_times", 0)) > 0
    except (TypeError, ValueError):
        return False


def manual_dump_active(raw: Any) -> bool:
    """True when ``dump.manual_dump`` is armed (``true`` or positive int)."""
    return raw not in (False, 0)


def manual_dump_target(raw: Any) -> int:
    """Watermark target from JSON: ``false``/``0``→0, ``true``→0 (continuous), positive int→N.

    Continuous ``true`` is not a watermark; callers use ``isinstance(raw, bool) and raw``.
    """
    if isinstance(raw, bool):
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


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

    Despite the plural name this only touches ``ascend_log`` today; other
    sections are validated / coerced by :func:`validate_runtime_config`.
    """
    if not isinstance(data, dict):
        return
    ascend = data.get("ascend_log")
    if not isinstance(ascend, dict):
        ascend = {}
        data["ascend_log"] = ascend
    _normalize_ascend_log_section_into(ascend)


def _normalize_config_sections(data: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy top level, deep-copy ``ascend_log``, then normalize in place."""
    out = dict(data)
    ascend = out.get("ascend_log")
    out["ascend_log"] = dict(ascend) if isinstance(ascend, dict) else {}
    _normalize_config_sections_into(out)
    return out


# ---- validate --------------------------------------------------------


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


def coerce_bool_field(container: dict[str, Any], key: str, field: str) -> None:
    """In-place coerce ``0``/``1`` → bool; reject other non-bool values.

    Missing / already-bool values are left unchanged. ``None`` is ignored
    (callers that require a default should set it before calling).
    """
    val = container.get(key)
    if val is None or isinstance(val, bool):
        return
    if val in (0, 1):
        container[key] = bool(val)
        return
    raise ValueError(f"{field} must be bool")


def validate_dump_mutual_exclusive(dump: dict[str, Any]) -> None:
    if dump_auto_on(dump) and manual_dump_active(dump.get("manual_dump", False)):
        raise ValueError("dump.auto_max_times>0 and dump.manual_dump active are mutually exclusive")


def validate_runtime_config(data: dict[str, Any]) -> None:
    """Validate / normalize ``data`` in place.

    Detect and dump are orthogonal: dump-only / detect-only / both are valid.
    Soft warnings for easy-to-misread combos live on ``RuntimeConfig``.

    S10 fix: defensively re-run ``_normalize_config_sections_into`` at entry so
    validation is safe regardless of whether the caller normalized first.
    """
    _normalize_config_sections_into(data)
    unknown_top = sorted(set(data) - TOP_LEVEL_KEYS)
    if unknown_top:
        raise ValueError(f"runtime config has unknown top-level key(s) {unknown_top}; allowed={sorted(TOP_LEVEL_KEYS)}")
    for section in (
        "dump",
        "ascend_log",
        "detector",
        "report",
        "actions",
    ):
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"runtime config missing object section '{section}'")
    for key in _RETIRED_ACTIONS_KEYS:
        data["actions"].pop(key, None)
    unknown_actions = sorted(set(data["actions"]) - ACTIONS_KEYS)
    if unknown_actions:
        raise ValueError(f"actions has unknown key(s) {unknown_actions}; allowed={sorted(ACTIONS_KEYS)}")
    for key in _RETIRED_DUMP_KEYS:
        data["dump"].pop(key, None)
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
    validate_dump_mutual_exclusive(data["dump"])
    for key in _RETIRED_REPORT_KEYS:
        data["report"].pop(key, None)
    unknown_report = sorted(set(data["report"]) - REPORT_KEYS)
    if unknown_report:
        raise ValueError(f"report has unknown key(s) {unknown_report}; allowed={sorted(REPORT_KEYS)}")
    coerce_bool_field(data["report"], "save_sensitive_info", "report.save_sensitive_info")
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
    for key in _RETIRED_DETECTOR_SECTIONS:
        detector.pop(key, None)
    known = set(_CATALOG_DETECTOR_SECTIONS)
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
        for retired in _RETIRED_DETECTOR_KEYS.get(key, ()):
            value.pop(retired, None)
        unknown_sub = sorted(set(value) - DETECTOR_KEYS[key])
        if unknown_sub:
            raise ValueError(f"detector.{key} has unknown key(s) {unknown_sub}; allowed={sorted(DETECTOR_KEYS[key])}")
    for name in _CATALOG_DETECTOR_SECTIONS:
        sec = detector.setdefault(name, {})
        if not isinstance(sec, dict):
            raise ValueError(f"detector.{name} must be an object")
    validate_registered_detectors(detector)


logger = init_logger_ascend(__name__)


DEFAULT_CONFIG_FILENAME = "runtime_config.json"


def default_runtime_root() -> Path:
    """Execution-directory runtime root: ``<cwd>/runtime``."""
    return Path(os.getcwd()) / "runtime"


def default_config_dir() -> Path:
    return default_runtime_root() / "config"


def _reject_unsafe_path(path: Path, *, label: str) -> Path:
    """Resolve and reject NUL / empty paths (basic path hygiene)."""
    raw = str(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"invalid {label}: empty or contains NUL")
    resolved = path.expanduser().resolve()
    try:
        cwd = Path.cwd().resolve()
        if resolved != cwd and cwd not in resolved.parents:
            logger.warning(
                "[runtime_config] %s is outside process cwd (%s): %s",
                label,
                cwd,
                resolved,
            )
    except Exception:
        pass
    return resolved


def resolve_runtime_config_path(configured_path: str | None = None) -> Path:
    """Resolve config file path.

    Priority:
    1. Explicit ``runtime_config_path`` / ``runtime-config`` from additional_config
    2. Default ``<cwd>/runtime/config/runtime_config.json``
    """
    if configured_path:
        return _reject_unsafe_path(Path(configured_path), label="runtime_config_path")
    return _reject_unsafe_path(default_config_dir() / DEFAULT_CONFIG_FILENAME, label="runtime_config_path")


def resolve_runtime_report_dir(config_path: Path, configured_report_dir: str | None = None) -> Path:
    if configured_report_dir:
        return _reject_unsafe_path(Path(configured_report_dir), label="runtime_report_dir")
    runtime_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return _reject_unsafe_path(runtime_root / "report", label="runtime_report_dir")


class RuntimeConfig:
    """Runtime guard switches loaded from JSON (per-DP broadcast or file poll).

    Prefer this name over a bare ``config`` module: it is a live control plane,
    not static build/packaging config. See module docstring for multi-DP rules.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        report_dir: str | Path | None = None,
        ensure_file: bool = False,
        hot_reload: bool = False,
        dump_dir: str | Path | None = None,
        startup_overlay: dict[str, Any] | None = None,
    ) -> None:
        # None → default ``<cwd>/runtime/config/runtime_config.json`` (not an "explicit" path).
        self._explicit_config_path = config_path is not None
        self.config_path = resolve_runtime_config_path(str(config_path) if config_path is not None else None)
        self.report_dir = resolve_runtime_report_dir(
            self.config_path,
            str(report_dir) if report_dir is not None else None,
        )
        # Startup override: False → off; True → fixed HOT_RELOAD_INTERVAL_SECONDS.
        # Authoritative; JSON cannot re-enable after load.
        self._reload_interval = HOT_RELOAD_INTERVAL_SECONDS if bool(hot_reload) else 0.0
        self._mtime: float | None = None
        # Bug #11 / S8: same-second edits need a content fingerprint. File size
        # alone is insufficient (e.g. window 10→33, true→false keep st_size).
        self._content_digest: str | None = None
        self._version: float = 0.0
        self._last_reload_ts = 0.0
        self._initial_broadcast_done = False
        self._data = deepcopy(_DEFAULTS)
        # Lazily filled hot-path bools; cleared on every ``_data`` mutation.
        self._hot_path_gates: dict[str, Any] | None = None
        self._bootstrap_persisted = False
        # Watermark progress for dump.manual_dump (int N). Never decreases;
        # never written back to JSON. Continuous ``true`` does not use this.
        self._manual_dumps_done = 0
        self._bg_reloader_started = False
        self._bg_thread: threading.Thread | None = None
        # Same seeding contract for dump.dump_dir (startup arg always applied at bootstrap).
        self._startup_dump_dir = (str(dump_dir).strip() if dump_dir else None) or None
        # ``additional_config.runtime_config`` (same schema as JSON file); applied once
        # at bootstrap after defaults. Not re-applied on hot-reload. Existing JSON is
        # overwritten on persist with this effective startup config.
        if startup_overlay is not None and not isinstance(startup_overlay, dict):
            raise ValueError(f"additional_config.runtime_config must be a dict, got {type(startup_overlay).__name__}.")
        self._startup_overlay = deepcopy(startup_overlay) if startup_overlay else None

        # In-memory merge always. ``ensure_file=True`` persists immediately (tests /
        # rare callers). Production AscendConfig uses False; worker leader calls
        # :meth:`ensure_persisted` once from ``RuntimeGuardProcessor``.
        self._bootstrap(persist=ensure_file)
        logger.info(
            "[runtime_config] path=%s explicit_path=%s report_dir=%s hot_reload=%s persisted=%s",
            self.config_path,
            self._explicit_config_path,
            self.report_dir,
            self.hot_reload_enabled,
            self._bootstrap_persisted,
        )
        if self.hot_reload_enabled:
            logger.info_once(
                "[runtime_config] hot-reload enabled path=%s (last-PP TP bus when available, else file poll)",
                str(self.config_path),
            )
        else:
            logger.info_once(
                "[runtime_config] hot-reload disabled "
                "(set additional_config.runtime_config_hot_reload=true to enable; "
                "dump.manual_dump also requires hot-reload)"
            )

    def _read_json_object(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                loaded = loads_jsonc(f.read())
            if not isinstance(loaded, dict):
                logger.error(
                    "[runtime_config] root must be object, got %s; ignoring file",
                    type(loaded).__name__,
                )
                return {}
            return loaded
        except Exception as exc:
            logger.warning(
                "[runtime_config] failed to read path=%s error=%s; using defaults",
                self.config_path,
                exc,
            )
            return {}

    def _merge_bootstrap(self, *, use_overlay: bool = True) -> dict[str, Any]:
        """Build effective config for process start: defaults ← startup overlay.

        Existing JSON on disk is ignored at bootstrap and overwritten when
        persisting. Hot-reload is what re-reads the file after start.
        """
        merged = deepcopy(_DEFAULTS)
        if use_overlay and self._startup_overlay:
            overlay = deepcopy(self._startup_overlay)
            pre = deepcopy(merged)
            merged = _deep_merge(merged, overlay)
            overlay_changes = _leaf_changes(pre, merged)
            if overlay_changes:
                logger.info(
                    "[runtime_config] applied additional_config.runtime_config overlay (%d keys) %s",
                    len(overlay_changes),
                    "; ".join(overlay_changes[:12]) + (" ..." if len(overlay_changes) > 12 else ""),
                )
        if self._startup_dump_dir:
            merged.setdefault("dump", {})["dump_dir"] = self._startup_dump_dir
        # validate_runtime_config normalizes ascend_log in place.
        return merged

    def _write_data_unlocked(self, data: dict[str, Any]) -> None:
        """Atomic write; caller must hold config lock / own the path."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.config_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, self.config_path)

    def _bootstrap(self, *, persist: bool) -> None:
        """Materialize defaults (+ startup overlay) and optionally overwrite JSON.

        Disk write is leader-only (or single-process); other ranks keep in-memory.
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        try:
            merged = self._merge_bootstrap()
            validate_runtime_config(merged)
        except Exception as exc:
            # Bad overlay must not kill the service; retry with defaults + ctor seeds.
            logger.error(
                "[runtime_config] startup config rejected path=%s error=%s; using defaults",
                self.config_path,
                exc,
            )
            merged = self._merge_bootstrap(use_overlay=False)
            validate_runtime_config(merged)

        can_write = persist and _is_json_writer()
        logger.info(
            "[runtime_config] bootstrap defaults (+ overlay) path=%s will_persist=%s "
            "(overwrites existing file when persisting)",
            self.config_path,
            can_write,
        )

        if can_write:
            try:
                with self._lock_config():
                    self._write_data_unlocked(merged)
                    mtime = self.config_path.stat().st_mtime
                self._apply_loaded(
                    merged,
                    version=mtime,
                    content_digest=self._digest_path(self.config_path),
                    announce=False,
                )
                self._bootstrap_persisted = True
                logger.info(
                    "[runtime_config] bootstrap saved path=%s explicit_path=%s %s",
                    self.config_path,
                    self._explicit_config_path,
                    self.interaction_mode_summary(),
                )
                self._warn_interaction_quirks()
            except Exception as exc:
                logger.warning(
                    "[runtime_config] bootstrap save failed path=%s error=%s; using in-memory",
                    self.config_path,
                    exc,
                )
                self._data = merged
                self._invalidate_hot_path_gates()
                self._version = 0.0
        else:
            self._data = merged
            self._invalidate_hot_path_gates()
            if self.config_path.exists():
                try:
                    mtime = self.config_path.stat().st_mtime
                    self._mtime = mtime
                    self._version = float(mtime)
                except OSError:
                    self._version = 0.0
                # Mark the pre-bootstrap file as already-reflected so reload()
                # does not re-apply it before the writer overwrites it with the
                # effective startup config (otherwise a stale hand-edit leaks in
                # via the first hot-reload, then gets persisted back out).
                self._content_digest = self._digest_path(self.config_path)
            else:
                self._mtime = None
                self._version = 0.0
            if persist and not _is_json_writer():
                logger.debug(
                    "[runtime_config] bootstrap skip persist (non-leader) path=%s %s",
                    self.config_path,
                    self.interaction_mode_summary(),
                )
            self._warn_interaction_quirks()
        self._last_reload_ts = time.time()

    def ensure_persisted(self) -> bool:
        """Materialize bootstrap defaults to disk once (worker leader / single-process).

        Safe to call from every worker: non-leaders no-op; leaders act at most once
        per process. Call from ``RuntimeGuardProcessor`` so API/EngineCore never persist.

        Always overwrites any existing JSON with the effective startup config
        (defaults ← ``additional_config.runtime_config`` overlay ← startup seeds).
        """
        if self._bootstrap_persisted:
            return True
        if not _is_json_writer():
            logger.debug(
                "[runtime_config] ensure_persisted skip (non-leader) path=%s",
                self.config_path,
            )
            return False
        try:
            with self._lock_config():
                self._write_data_unlocked(self._data)
                mtime = self.config_path.stat().st_mtime
            self._mtime = mtime
            self._content_digest = None  # bootstrap; set on first reload/save
            self._version = float(mtime)
            self._bootstrap_persisted = True
            logger.info(
                "[runtime_config] worker leader persisted path=%s explicit_path=%s (overwrote)",
                self.config_path,
                self._explicit_config_path,
            )
            return True
        except Exception as exc:
            logger.warning(
                "[runtime_config] ensure_persisted failed path=%s error=%s",
                self.config_path,
                exc,
            )
            return False

    # ---- section accessors -------------------------------------------------

    @property
    def hot_reload_enabled(self) -> bool:
        """True when startup ``runtime_config_hot_reload`` is enabled."""
        return self._reload_interval > 0

    @property
    def reload_interval_seconds(self) -> float:
        """Internal poll period when hot-reload is on; 0 when disabled."""
        return self._reload_interval

    @property
    def dump(self) -> dict[str, Any]:
        return self._data["dump"]

    @property
    def ascend_log(self) -> dict[str, Any]:
        return self._data["ascend_log"]

    @property
    def detector(self) -> dict[str, Any]:
        return self._data["detector"]

    def _invalidate_hot_path_gates(self) -> None:
        """Drop cached idle/active gates after any in-memory config mutation."""
        self._hot_path_gates = None

    def _hot_path_gates_cached(self) -> dict[str, Any]:
        """Version-stamped hot-path bools (recomputed when ``_data`` changes)."""
        cached = self._hot_path_gates
        if cached is not None:
            return cached
        dump = self._data.get("dump") or {}
        det = self._data.get("detector") or {}
        report = self._data.get("report") or {}
        any_det = self.detectors_enabled_in(self._data)

        dump_on = dump_auto_on(dump) or manual_dump_active(dump.get("manual_dump", False))
        save_sensitive = bool(report.get("save_sensitive_info", False))
        tok_rep = bool((det.get("token_repeat") or {}).get("enabled", False))

        raw_manual = dump.get("manual_dump", False)
        if isinstance(raw_manual, bool) and raw_manual:
            # Continuous: always "armed" sentinel (does not use watermark).
            remaining = 1
        else:
            remaining = max(0, manual_dump_target(raw_manual) - int(self._manual_dumps_done))

        needs_io = tok_rep or (any_det and save_sensitive)
        needs_sample = any_det

        cached = {
            "any_detector": any_det,
            "dump_enabled": dump_on,
            "needs_cumulative_io": needs_io,
            "needs_sample_phase_hooks": needs_sample,
            "manual_trigger_count": remaining,
        }
        self._hot_path_gates = cached
        return cached

    def dump_enabled(self) -> bool:
        return bool(self._hot_path_gates_cached()["dump_enabled"])

    def auto_dump_on(self) -> bool:
        return dump_auto_on(self.dump)

    def any_detector_enabled(self) -> bool:
        """True if at least one auto anomaly detector is enabled."""
        return bool(self._hot_path_gates_cached()["any_detector"])

    def needs_cumulative_io(self) -> bool:
        """True when sampled tokens must be appended to the IO store.

        Consumers: token_repeat (and any detector with ``save_sensitive_info``)
        that persist cumulative ``output_token_ids``.
        """
        return bool(self._hot_path_gates_cached()["needs_cumulative_io"])

    def needs_sample_phase_hooks(self) -> bool:
        """True when post-sample runtime_guard hooks must run (not a pure sample fast-path).

        Covers detectors and finish-output logging.
        Dump-only / hot-reload-only does not need the sample-phase hook chain.
        """
        return bool(self._hot_path_gates_cached()["needs_sample_phase_hooks"])

    # Detector section order comes from ``detector_catalog``.
    DETECTOR_SECTIONS = _CATALOG_DETECTOR_SECTIONS

    @staticmethod
    def detectors_enabled_in(data: dict[str, Any]) -> bool:
        """Whether ``data['detector']`` has any auto anomaly detector on."""
        det = data.get("detector") or {}
        for name in RuntimeConfig.DETECTOR_SECTIONS:
            sec = det.get(name)
            if isinstance(sec, dict) and bool(sec.get("enabled", False)):
                return True
        return False

    def interaction_mode_summary(self) -> str:
        """Short ops-facing mode tag for logs (detect / dump axes)."""
        _DISPLAY = {
            "spec_acceptance": "spec",
            "token_repeat": "token_repeat",
            "logits_finite": "logits_finite",
        }
        names: list[str] = []
        for section in RuntimeConfig.DETECTOR_SECTIONS:
            if bool(self.detector_get(section, "enabled", False)):
                names.append(_DISPLAY.get(section, section))
        dump_on = self.dump_enabled()
        auto_on = self.auto_dump_on()
        max_times = self.dump_max_times()
        if names and dump_on and auto_on:
            mode = "detect+auto_dump"
        elif names and dump_on:
            mode = "detect+manual_dump"
        elif names:
            mode = "detect_only"
        elif dump_on and auto_on:
            mode = "auto_dump_only"
        elif dump_on:
            mode = "manual_dump_only"
        else:
            mode = "idle"
        return (
            f"mode={mode} detectors={names} dump.active={dump_on} "
            f"auto_max_times={max_times} manual_dump={self.dump.get('manual_dump', False)}"
        )

    def _warn_interaction_quirks(self) -> None:
        """Warn on valid-but-easy-to-misread dump/detect combinations."""
        if self.dump_enabled() and not self.any_detector_enabled():
            logger.warning(
                "[runtime_config] dump active with no auto detector — "
                "auto dump will not trigger; dump.manual_dump still works %s",
                _process_role_tag(),
            )
        elif self.dump_enabled() and self.any_detector_enabled() and not self.auto_dump_on():
            logger.info(
                "[runtime_config] dump active with auto_max_times=0 — "
                "detect runs; auto-arm off; dump.manual_dump still works %s",
                _process_role_tag(),
            )

    def dump_max_times(self) -> int:
        return int(self.dump.get("auto_max_times", 0))

    def dump_cooldown_seconds(self) -> int:
        return int(self.dump.get("auto_cooldown_seconds", 300))

    def manual_trigger_continuous(self) -> bool:
        """True when ``dump.manual_dump`` is bool ``true`` (always-on dump)."""
        return self.dump.get("manual_dump", False) is True

    def manual_dump_target(self) -> int:
        """Watermark target ``N`` from ``dump.manual_dump`` (0 when off or continuous)."""
        return manual_dump_target(self.dump.get("manual_dump", False))

    def manual_dumps_done(self) -> int:
        """How many watermark manual dumps this process has completed."""
        return int(self._manual_dumps_done)

    def manual_trigger_count(self) -> int:
        """Remaining catch-up dumps: ``max(0, target - done)``; continuous → 1.

        Int ``N`` is a watermark: fire while ``done < N``. Disk value is never
        mutated by the process — raise ``N`` (e.g. 1→2) for another dump.
        If the value on disk is ≤ ``done``, skip. Requires hot-reload.
        """
        return int(self._hot_path_gates_cached()["manual_trigger_count"])

    def manual_trigger(self) -> bool:
        """True when manual dump is armed (continuous or ``target > done``)."""
        return self.manual_trigger_count() > 0

    def consume_manual_trigger(self) -> bool:
        """Record one completed manual dump wave; return True if armed.

        - ``true`` (bool): continuous — always armed; do not bump watermark;
          never touch JSON.
        - positive int ``N``: if ``done < N``, bump ``done`` by one and return
          True; else False. JSON ``manual_dump`` is left unchanged.

        Multi-DP sharing one file: each replica tracks its own ``done``.
        """
        if self.manual_trigger_continuous():
            logger.debug(
                "[runtime_config] manual_trigger continuous (true); watermark unused %s",
                _process_role_tag(),
            )
            return True
        target = self.manual_dump_target()
        if self._manual_dumps_done >= target:
            return False

        self._manual_dumps_done += 1
        self._invalidate_hot_path_gates()
        logger.info(
            "[runtime_config] manual_dump watermark done=%d target=%d (JSON unchanged) %s",
            self._manual_dumps_done,
            target,
            _process_role_tag(),
        )
        return True

    def ascend_log_level(self) -> str:
        return str(self.ascend_log.get("level", "INFO")).upper()

    def ascend_log_debug_modules(self) -> list[str]:
        raw = self.ascend_log.get("debug", [])
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    def ascend_log_modules(self) -> dict[str, str]:
        raw = self.ascend_log.get("modules", {})
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v).upper() for k, v in raw.items() if str(k).strip()}

    def report_save_sensitive_info(self) -> bool:
        """Whether anomaly reports persist token ids and decode them to text.

        Default False: only lengths (``*_token_count``). ``true`` keeps
        ``prompt_token_ids`` / cumulative ``output_token_ids`` and decodes them.
        """
        report = self._data.get("report") or {}
        return bool(report.get("save_sensitive_info", False))

    def report_decode_token_ids(self) -> bool:
        """Decode ``*_token_ids`` to text when ``save_sensitive_info`` is on."""
        return self.report_save_sensitive_info()

    def report_max_prompt_token_ids(self) -> int:
        """Max ``prompt_token_ids`` length to persist (0 = unlimited). Default 100000."""
        report = self._data.get("report") or {}
        return int(report.get("max_prompt_token_ids", 100000))

    def report_max_output_token_ids(self) -> int:
        """Max output-like ``*_token_ids`` length to persist (0 = unlimited). Default 100000."""
        report = self._data.get("report") or {}
        return int(report.get("max_output_token_ids", 100000))

    def report_max_per_req(self) -> int:
        """Max report files per ``(incident_type, req_id)`` (default 1).

        After a successful write reaches this cap, detection for that ``req_id``
        stops (all detectors). Between writes, wave backoff applies (64, then
        doubles). Default ``actions.defaults.on_trigger`` includes ``report``.
        """
        report = self._data.get("report") or {}
        try:
            return max(1, int(report.get("max_per_req", 1)))
        except (TypeError, ValueError):
            return 1

    def dump_root(self) -> Path:
        """KV dump landing root; ``<incident_type>/<req_id>/`` under it.

        ``dump.dump_dir`` (JSON, hot-reloadable; seeded from the startup
        ``runtime_dump_dir`` arg when unset) wins over the derived default
        ``<report_dir>/kv_cache``.
        """
        raw = (self._data.get("dump") or {}).get("dump_dir")
        if isinstance(raw, str) and raw.strip():
            return _reject_unsafe_path(Path(raw.strip()), label="runtime_dump_dir")
        return self.report_dir / "kv_cache"

    def actions_default_on_trigger(self) -> list[str]:
        actions = self._data.get("actions") or {}
        defaults = actions.get("defaults") or {}
        raw = defaults.get("on_trigger", ["report"])
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(x) for x in raw]
        return ["report"]

    def action_queue_max_size(self) -> int:
        """ActionQueue capacity at bind (fixed internal constant)."""
        return ACTION_QUEUE_MAX_SIZE

    def detector_section(self, name: str) -> dict[str, Any]:
        """Return nested ``detector.<name>`` object (empty dict if missing)."""
        sec = self.detector.get(name)
        return sec if isinstance(sec, dict) else {}

    def detector_get(self, section: str, key: str, default: Any = None) -> Any:
        """Read ``detector.<section>.<key>``."""
        return self.detector_section(section).get(key, default)

    def apply_ascend_log_level(self) -> None:
        """Apply live ``ascend_log`` (root default + per-module overrides)."""
        from vllm_ascend.logger import apply_ascend_log_level as _apply

        _apply(
            self.ascend_log_level(),
            self.ascend_log_debug_modules(),
            self.ascend_log_modules(),
        )

    def start_non_worker_background_reload(self) -> bool:
        """Daemon thread: file-poll JSON and re-apply ``ascend_log`` (API / EngineCore).

        - No-op when hot-reload is off, or this process is a distributed Worker.
        - Uses **local file reload only** — never joins worker world broadcast.
        - Does not persist JSON.
        - Callers should invoke :meth:`apply_ascend_log_level` once at construction
          for the initial level; this thread only re-applies after file changes.
        Workers keep step-driven :meth:`sync_runtime_config` and must not run this.
        If AscendConfig starts the thread before Worker env/world is ready, the
        loop exits as soon as :func:`_is_distributed_worker_process` becomes true.
        """
        if not self.hot_reload_enabled:
            return False
        if _is_distributed_worker_process():
            logger.info(
                "[runtime_config] skip non-worker reloader (worker process) path=%s",
                self.config_path,
            )
            return False
        if self._bg_reloader_started:
            return False
        path_key = str(self.config_path.resolve()) if self.config_path.exists() else str(self.config_path)
        if path_key in _bg_reload_paths:
            self._bg_reloader_started = True
            logger.info(
                "[runtime_config] non-worker reloader already running for path=%s",
                self.config_path,
            )
            return False
        self._bg_reloader_started = True
        _bg_reload_paths.add(path_key)
        interval = self.reload_interval_seconds

        def _loop() -> None:
            while True:
                time.sleep(interval)
                # AscendConfig may start this thread before Worker sets RANK /
                # world group; stop as soon as we are clearly a Worker so we
                # do not dual-reload alongside sync_for_step broadcast.
                if _is_distributed_worker_process():
                    _bg_reload_paths.discard(path_key)
                    logger.info(
                        "[runtime_config] non-worker reloader exiting (process is worker) path=%s",
                        self.config_path,
                    )
                    return
                try:
                    # Wait for worker leader to materialize the file after
                    # overwrite+delete; avoid no-op thrashing when missing.
                    if not self.config_path.exists():
                        continue
                    # Force file poll path even if JSON says broadcast — this
                    # process is outside the worker world group.
                    # Content diffs are logged inside ``_apply_loaded``.
                    if self._maybe_reload_local():
                        self.apply_ascend_log_level()
                        if self.manual_trigger():
                            logger.info(
                                "[runtime_config] dump.manual_dump=true seen on "
                                "non-worker reload — dump arms only on worker "
                                "execute_model (send a request). path=%s",
                                self.config_path,
                            )
                except Exception as exc:
                    logger.warning(
                        "[runtime_config] non-worker reload error path=%s error=%s",
                        self.config_path,
                        exc,
                    )

        self._bg_thread = threading.Thread(
            target=_loop,
            name="rg-non-worker-reload",
            daemon=True,
        )
        self._bg_thread.start()
        logger.info(
            "[runtime_config] non-worker background reload started interval=%.3fs path=%s",
            interval,
            self.config_path,
        )
        return True

    def sync_runtime_config(self) -> bool:
        """Interval-gated local JSON poll.

        Last-PP TP workers apply config via the merged due bus in
        ``RuntimeGuardProcessor``; this entry is for file-poll ranks, the
        non-worker background reloader, and tests.
        """
        if not self.hot_reload_enabled:
            return False
        _log_file_poll_fallback_once(path=str(self.config_path), role=_process_role_tag())
        logger.debug(
            "[runtime_config sync] enter stage=config_local_reload %s",
            _process_role_tag(),
        )
        changed = self._maybe_reload_local()
        logger.debug(
            "[runtime_config sync] leave stage=config_local_reload changed=%s %s",
            changed,
            _process_role_tag(),
        )
        return changed

    def config_due_local(self) -> bool:
        """Whether this rank would vote config-due on the next task-bus AR."""
        if not self.hot_reload_enabled:
            return False
        now = time.time()
        return bool((not self._initial_broadcast_done) or (now - self._last_reload_ts >= self.reload_interval_seconds))

    def build_config_sync_payload(self) -> tuple[dict[str, Any], bool]:
        """Leader: reload JSON and pack broadcast payload. Returns (payload, changed)."""
        role = _process_role_tag()
        logger.debug("[runtime_config sync] enter stage=config_reload_file %s", role)
        changed = self.reload(force=False)
        logger.debug(
            "[runtime_config sync] leave stage=config_reload_file changed=%s version=%.6f %s",
            changed,
            float(self._version),
            role,
        )
        first_sync = not self._initial_broadcast_done
        return (
            {
                "version": float(self._version),
                "data": deepcopy(self._data) if (changed or first_sync) else None,
            },
            changed,
        )

    def apply_config_sync_payload(
        self,
        payload: dict[str, Any],
        *,
        is_leader: bool,
        leader_changed: bool = False,
    ) -> bool:
        """Apply a config broadcast payload; marks initial sync done.

        Always advances ``_last_reload_ts`` on every rank that finishes the
        config lane (including ``data is None`` no-ops). Otherwise followers
        keep voting ``config_due`` every wave after the first interval.
        """
        self._last_reload_ts = time.time()
        first_sync = not self._initial_broadcast_done
        self._initial_broadcast_done = True
        version = float(payload.get("version", 0.0))
        data = payload.get("data")
        if data is None:
            return bool(leader_changed or first_sync) if is_leader else False
        if not isinstance(data, dict):
            return False
        if not is_leader:
            if version != self._version:
                try:
                    return self._apply_loaded(data, version=version)
                except Exception as exc:
                    logger.error(
                        "[runtime_config] follower _apply_loaded failed "
                        "path=%s version=%.6f error=%s — keeping last-known-good %s",
                        self.config_path,
                        version,
                        exc,
                        _process_role_tag(),
                    )
                    return False
            return False
        return bool(leader_changed or first_sync)

    def _maybe_reload_local(self) -> bool:
        now = time.time()
        if now - self._last_reload_ts < self.reload_interval_seconds:
            return False
        return self.reload(force=False)

    @staticmethod
    def _digest_bytes(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def _digest_path(cls, path: Path) -> str | None:
        try:
            return cls._digest_bytes(path.read_bytes())
        except OSError:
            return None

    def reload(self, *, force: bool = False) -> bool:
        """Local file reload (leader / file mode / pre-dist bootstrap).

        Hot-reload follows JSON only: ``defaults ← JSON``.
        """
        self._last_reload_ts = time.time()
        if not self.config_path.exists():
            if force:
                self._data = deepcopy(_DEFAULTS)
                self._invalidate_hot_path_gates()
                self._version = 0.0
            return False

        try:
            stat = self.config_path.stat()
            mtime = stat.st_mtime
        except OSError as exc:
            logger.warning("[runtime_config] stat failed path=%s error=%s", self.config_path, exc)
            return False

        try:
            with self._lock_config():
                raw = self.config_path.read_bytes()
            digest = self._digest_bytes(raw)
        except OSError as exc:
            logger.warning("[runtime_config] read failed path=%s error=%s", self.config_path, exc)
            return False

        # Bug #11: NFS mtime is 1s-granular. Same-second edits that keep
        # ``st_size`` unchanged (10→33, true→false, INFO→WARN) must still
        # reload. Skip only when mtime is older, or mtime ties **and** the
        # content digest matches the last applied payload.
        if not force and self._mtime is not None:
            if mtime < self._mtime:
                return False
            if digest == self._content_digest:
                # Content unchanged (incl. touch / same-second no-op rewrite).
                self._mtime = mtime
                return False

        try:
            loaded = loads_jsonc(raw.decode("utf-8"))
            if not isinstance(loaded, dict):
                logger.error("[runtime_config] root must be object, got %s", type(loaded).__name__)
                return False
            merged = _deep_merge(_DEFAULTS, loaded)
            return self._apply_loaded(
                merged,
                version=mtime,
                content_digest=digest,
            )
        except Exception as exc:
            logger.error("[runtime_config] reload failed path=%s error=%s", self.config_path, exc)
            return False

    def _apply_loaded(
        self,
        merged: dict[str, Any],
        *,
        version: float,
        content_digest: str | None = None,
        announce: bool = True,
    ) -> bool:
        merged = _normalize_config_sections(merged)
        validate_runtime_config(merged)
        changes = _leaf_changes(self._data, merged)
        self._data = merged
        self._invalidate_hot_path_gates()
        self._mtime = version
        self._version = version
        if content_digest is not None:
            self._content_digest = content_digest
        if announce and changes:
            # Only print fields that actually changed (e.g. dump.max_times).
            logger.info(
                "[runtime_config] updated path=%s version=%.6f %s changes=[%s] %s",
                str(self.config_path),
                self._version,
                _process_role_tag(),
                "; ".join(changes),
                self.interaction_mode_summary(),
            )
            self._warn_interaction_quirks()
            if self.manual_trigger() and self.dump_enabled():
                logger.info(
                    "[runtime_config] dump.manual_dump active — next worker execute_model will consume and arm dump %s",
                    _process_role_tag(),
                )
        elif announce:
            logger.debug(
                "[runtime_config] apply no content change path=%s version=%.6f %s",
                str(self.config_path),
                self._version,
                _process_role_tag(),
            )
        return True

    def _lock_config(self):
        lock_path = Path(f"{self.config_path}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        class _LockCtx:
            def __enter__(self_inner):
                self_inner._fd = lock_path.open("w", encoding="utf-8")
                fcntl.flock(self_inner._fd, fcntl.LOCK_EX)
                return self_inner._fd

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    fcntl.flock(self_inner._fd, fcntl.LOCK_UN)
                finally:
                    self_inner._fd.close()

        return _LockCtx()
