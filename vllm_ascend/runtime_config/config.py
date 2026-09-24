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

Design (multi-DP safe — avoid full-world collectives):

1. **One writer / monitor per EngineCore (per DP replica)**
   Reads & writes the JSON (``ensure_persisted`` / ``save`` / ``manual_trigger`` clear).
   Prefer ``inner_dp_world`` first rank; else TP0∧PP0 when ``dp_size>1``; else
   global world rank0 / ``RANK==0``.

2. **Sync scope never spans idle-asymmetric DPs**
   - ``broadcast`` + ``dp_size==1`` + **PP==1**: sync group = ``get_world_group()``.
     End-of-wave shares one due ``all_reduce([config_due, dump_due])`` then
     separate ``broadcast_object`` per due lane (idle waves pay only the AR).
   - ``broadcast`` + ``dp_size>1``: sync group = ``inner_dp_world`` only
     (same end-of-wave gate inside that DP).
   - ``broadcast`` + multi-DP but no ``inner_dp_world``: **local file poll**
     (no collective). Each EngineCore needs a readable ``runtime_config_path``
     (per-node copy or shared FS).
   - **PP>1**: ``sync_mode`` is **forced to file** (cross-PP broadcast deadlocks).
     Dump uses a single-lane task bus on the last-PP TP group only.
   - ``sync_mode=file``: every rank polls the path (shared FS).

3. **Cross-DP config is not synchronized**
   Edit each DP's JSON (or the shared path each DP can see). Do **not** use
   full EP ``world_size`` (e.g. 32) for config hot-reload — one-sided
   ``execute_dummy_batch`` after a request would deadlock.

Production: ``AscendConfig`` uses ``ensure_file=False``; worker
:meth:`RuntimeConfig.ensure_persisted` materializes JSON on the writer.

Implementation is split across:
``_defaults`` (schema), ``_paths``, ``_dist``, ``_merge``, ``_validate``
(pattern/list normalizers live in ``_validate``); this module keeps the live
``RuntimeConfig`` control plane.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from vllm_ascend.logger import init_logger_ascend

# ``_rg_multi_dp_file_fallback_logged`` is mutated by sync_runtime_config.
from vllm_ascend.runtime_config import _dist as _dist_mod
from vllm_ascend.runtime_config._defaults import (
    _DEFAULTS,
)
from vllm_ascend.runtime_config._defaults import (
    DETECTOR_SECTIONS as _DETECTOR_SECTIONS,
)
from vllm_ascend.runtime_config._dist import (
    SYNC_BROADCAST,
    SYNC_FILE,
    _bg_reload_paths,
    _dp_world_size_or_one,
    _is_distributed_worker_process,
    _is_json_writer,
    _log_pp_force_file_once,
    _pp_forces_file_sync,
    _process_role_tag,
    _runtime_config_sync_group_or_none,
)
from vllm_ascend.runtime_config._merge import (
    _deep_merge,
    _leaf_changes,
    _normalize_config_sections,
)
from vllm_ascend.runtime_config._paths import (
    _reject_unsafe_path,
    resolve_runtime_config_path,
    resolve_runtime_report_dir,
)
from vllm_ascend.runtime_config._validate import validate_runtime_config
from vllm_ascend.runtime_config.jsonc_io import loads_jsonc

logger = init_logger_ascend(__name__)


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
        sync_mode: str | None = None,
        reload_interval_seconds: float | int | None = None,
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
        # Startup override: None → default 0 (off); >0 → every N seconds.
        # This is authoritative and is not re-enabled by JSON after load.
        if reload_interval_seconds is None:
            self._reload_interval = 0.0
        else:
            self._reload_interval = float(reload_interval_seconds)
        if self._reload_interval < 0:
            raise ValueError(f"runtime_config_reload_interval must be >= 0, got {self._reload_interval}")
        self._mtime: float | None = None
        # Bug #11 / S8: same-second edits need a content fingerprint. File size
        # alone is insufficient (e.g. window 10→33, true→false keep st_size).
        self._content_digest: str | None = None
        self._version: float = 0.0
        self._last_reload_ts = 0.0
        self._ctor_sync_mode = sync_mode
        # C1: sync_mode frozen at first apply — ranks of a DP group must see
        # the same value for the whole process lifetime; hot-reload may not
        # flip it (collective split → hang).
        self._sync_mode_frozen: str | None = str(sync_mode).lower() if sync_mode is not None else None
        self._initial_broadcast_done = False
        self._data = deepcopy(_DEFAULTS)
        # Lazily filled hot-path bools; cleared on every ``_data`` mutation.
        self._hot_path_gates: dict[str, Any] | None = None
        self._bootstrap_persisted = False
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
                "[runtime_config] hot-reload enabled interval=%.3fs sync_mode=%s path=%s",
                self.reload_interval_seconds,
                self.sync_mode,
                str(self.config_path),
            )
        else:
            logger.info_once(
                "[runtime_config] hot-reload disabled "
                "(set additional_config.runtime_config_reload_interval > 0 to enable; "
                "default is 0; dump.manual_dump also requires interval > 0)"
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
            # Startup path/interval still win over overlay copies of those keys.
            overlay.pop("reload_interval_seconds", None)
            pre = deepcopy(merged)
            merged = _deep_merge(merged, overlay)
            overlay_changes = _leaf_changes(pre, merged)
            if overlay_changes:
                logger.info(
                    "[runtime_config] applied additional_config.runtime_config overlay (%d keys) %s",
                    len(overlay_changes),
                    "; ".join(overlay_changes[:12]) + (" ..." if len(overlay_changes) > 12 else ""),
                )
        if self._ctor_sync_mode is not None:
            merged["sync_mode"] = self._ctor_sync_mode
        # Persist startup hot-reload interval for visibility (runtime gate is still
        # ``self._reload_interval`` only).
        merged["reload_interval_seconds"] = self._reload_interval
        if self._startup_dump_dir:
            merged.setdefault("dump", {})["dump_dir"] = self._startup_dump_dir
        # validate_runtime_config normalizes ascend_log in place.
        return merged

    @staticmethod
    def _auto_on_from_dump(dump: dict[str, Any]) -> bool:
        try:
            return int(dump.get("auto_max_times", 0)) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _manual_dump_active(raw: Any) -> bool:
        return raw not in (False, 0)

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
        """True when startup ``runtime_config_reload_interval`` > 0."""
        return self._reload_interval > 0

    @property
    def sync_mode(self) -> str:
        # PP>1 cannot safely join a cross-PP config collective → always file.
        if _pp_forces_file_sync():
            _log_pp_force_file_once()
            return SYNC_FILE
        mode = str(self._data.get("sync_mode", SYNC_BROADCAST)).lower()
        if self._sync_mode_frozen is not None:
            mode = self._sync_mode_frozen
        return mode if mode in (SYNC_BROADCAST, SYNC_FILE) else SYNC_BROADCAST

    @property
    def reload_interval_seconds(self) -> float:
        """Effective hot-reload period from startup; 0 means disabled."""
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
        log = self._data.get("log") or {}
        any_det = False
        for name in RuntimeConfig.DETECTOR_SECTIONS:
            sec = det.get(name)
            if isinstance(sec, dict) and bool(sec.get("enabled", False)):
                any_det = True
                break

        dump_on = self._auto_on_from_dump(dump) or self._manual_dump_active(dump.get("manual_dump", False))
        print_out = bool(log.get("print_output_on_finish", False))
        save_sensitive = bool(report.get("save_sensitive_info", False))
        out_sub = bool((det.get("output_substring") or {}).get("enabled", False))
        tok_rep = bool((det.get("token_repeat") or {}).get("enabled", False))

        manual_raw = dump.get("manual_dump", False)
        if isinstance(manual_raw, bool):
            manual_count = 1 if manual_raw else 0
        else:
            try:
                manual_count = max(0, int(manual_raw))
            except (TypeError, ValueError):
                manual_count = 0

        needs_io = print_out or out_sub or tok_rep or (any_det and save_sensitive)
        needs_sample = any_det or print_out

        cached = {
            "any_detector": any_det,
            "dump_enabled": dump_on,
            "needs_cumulative_io": needs_io,
            "needs_sample_phase_hooks": needs_sample,
            "manual_trigger_count": manual_count,
        }
        self._hot_path_gates = cached
        return cached

    def dump_enabled(self) -> bool:
        return bool(self._hot_path_gates_cached()["dump_enabled"])

    def auto_dump_on(self) -> bool:
        return self._auto_on_from_dump(self.dump)

    def manual_dump_on(self) -> bool:
        return self._manual_dump_active(self.dump.get("manual_dump", False))

    def any_detector_enabled(self) -> bool:
        """True if at least one auto anomaly detector is enabled."""
        return bool(self._hot_path_gates_cached()["any_detector"])

    def needs_cumulative_io(self) -> bool:
        """True when sampled tokens must be appended to the IO store.

        Consumers: finish-time output logging, substring/repeat detectors, and
        sensitive reports that persist cumulative ``output_token_ids``.
        """
        return bool(self._hot_path_gates_cached()["needs_cumulative_io"])

    def needs_sample_phase_hooks(self) -> bool:
        """True when post-sample runtime_guard hooks must run (not a pure sample fast-path).

        Covers detectors and finish-output logging.
        Dump-only / hot-reload-only does not need the sample-phase hook chain.
        """
        return bool(self._hot_path_gates_cached()["needs_sample_phase_hooks"])

    # Schema key sets live in ``_defaults``; ``DETECTOR_SECTIONS`` is re-exported
    # for hot-path gates / callers that scan enabled detectors.
    DETECTOR_SECTIONS = _DETECTOR_SECTIONS

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
        names: list[str] = []
        if bool(self.detector_get("spec_acceptance", "enabled", False)):
            names.append("spec")
        if bool(self.detector_get("output_substring", "enabled", False)):
            names.append("output_substring")
        if bool(self.detector_get("token_repeat", "enabled", False)):
            names.append("token_repeat")
        if bool(self.detector_get("logits_finite", "enabled", False)):
            names.append("logits_finite")
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

    def manual_trigger_count(self) -> int:
        """Remaining manual dump waves from ``dump.manual_dump``.

        ``false``/``0`` → 0; ``true`` (continuous) → 1 as a positive sentinel;
        positive int → N. Continuous mode does not decrement on consume.
        Only observed after a successful hot-reload; requires
        ``runtime_config_reload_interval > 0``.
        """
        return int(self._hot_path_gates_cached()["manual_trigger_count"])

    def manual_trigger(self) -> bool:
        """True when manual dump is armed (continuous or remaining count > 0)."""
        return self.manual_trigger_count() > 0

    def consume_manual_trigger(self) -> bool:
        """Arm one manual dump wave; return True if armed.

        - ``true`` (bool): continuous — leave value as ``true``, do not persist.
        - positive int: decrement **in memory** by one each armed wave.
          Persist to JSON **only when the count reaches 0** (``false``), not on
          every decrement. While the in-memory count is still >0, a content
          change on disk still hot-reloads into memory (normal reload path).

        Multi-DP sharing one ``runtime_config.json``: the file keeps the original
        ``N`` until some replica persists ``false``. Each DP may therefore dump
        up to ``N`` times independently (worst case roughly ``num_DP × N``).
        """
        if self.manual_trigger_continuous():
            logger.debug(
                "[runtime_config] manual_trigger continuous (true); not clearing %s",
                _process_role_tag(),
            )
            return True
        remaining = self.manual_trigger_count()
        if remaining <= 0:
            return False

        new_val: bool | int = False if remaining <= 1 else remaining - 1
        self.dump["manual_dump"] = new_val
        self._invalidate_hot_path_gates()

        if new_val is not False:
            # Mid-count: memory only. Disk stays at the original N so other DPs
            # that share the file can still see N; a user hand-edit still reloads.
            logger.info(
                "[runtime_config] manual_dump → %s in-memory (was %d; JSON unchanged until 0) %s",
                new_val,
                remaining,
                _process_role_tag(),
            )
            return True

        # Drained to 0: persist false once (JSON writer only).
        if _is_json_writer():
            if self.save(updates={"dump": {"manual_dump": False}}):
                logger.info(
                    "[runtime_config] manual_dump drained → false (persisted) path=%s was=%d %s",
                    self.config_path,
                    remaining,
                    _process_role_tag(),
                )
            else:
                logger.warning(
                    "[runtime_config] manual_dump drained → false in-memory but failed to persist path=%s was=%d %s",
                    self.config_path,
                    remaining,
                    _process_role_tag(),
                )
        else:
            logger.info(
                "[runtime_config] manual_dump drained → false in-memory "
                "(non-writer; JSON still shows prior N until a writer persists) was=%d %s",
                remaining,
                _process_role_tag(),
            )
        return True

    def disable_detector_unavailable(self, section: str, *, reason: str) -> bool:
        """Force ``detector.<section>.enabled=false`` when a hard dependency is missing.

        Returns True if the in-memory flag was changed.
        """
        sec = self.detector_section(section)
        if not bool(sec.get("enabled", False)):
            return False
        sec["enabled"] = False
        self._invalidate_hot_path_gates()
        logger.error(
            "[runtime_config] detector.%s.enabled forced false: %s %s",
            section,
            reason,
            _process_role_tag(),
        )
        if _is_json_writer():
            if self.save({"detector": {section: {"enabled": False}}}):
                logger.info(
                    "[runtime_config] detector.%s.enabled=false persisted path=%s %s",
                    section,
                    self.config_path,
                    _process_role_tag(),
                )
            else:
                logger.warning(
                    "[runtime_config] detector.%s.enabled cleared in-memory but failed to persist path=%s %s",
                    section,
                    self.config_path,
                    _process_role_tag(),
                )
        else:
            logger.info(
                "[runtime_config] detector.%s.enabled cleared in-memory (non-writer) %s",
                section,
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
        """Whether anomaly reports persist plaintext token-id lists.

        Default False: only lengths (``*_token_count``). ``true`` keeps full
        ``prompt_token_ids`` and cumulative ``output_token_ids``.
        """
        report = self._data.get("report") or {}
        return bool(report.get("save_sensitive_info", False))

    def log_print_output_on_finish(self) -> bool:
        """Whether to log output token ids + text when any request finishes.

        Default False. When True, TP0 logs on reap (after ``mark_finished``) for every
        finished request (independent of ``save_sensitive_info``). Can be large /
        sensitive — leave off in prod.

        Accumulation starts only while this flag is true on each sample step
        (no backfill of tokens produced before enable). Hot-enabling mid-request
        may yield a partial finish log, or an empty one if the request finishes
        with no further appends after enable. Enable before traffic for full
        output.
        """
        log_sec = self._data.get("log") or {}
        return bool(log_sec.get("print_output_on_finish", False))

    def report_decode_token_ids(self) -> bool:
        """Whether to decode ``*_token_ids`` into text in reports.

        Covers prompt/output and window/current evidence fields.
        Only applies when ``save_sensitive_info`` is true. Default True.
        """
        report = self._data.get("report") or {}
        return bool(report.get("decode_token_ids", True))

    def report_max_prompt_token_ids(self) -> int:
        """Max ``prompt_token_ids`` length to persist (0 = unlimited). Default 1000."""
        report = self._data.get("report") or {}
        return int(report.get("max_prompt_token_ids", 1000))

    def report_max_output_token_ids(self) -> int:
        """Max output-like ``*_token_ids`` length to persist (0 = unlimited). Default 1000."""
        report = self._data.get("report") or {}
        return int(report.get("max_output_token_ids", 1000))

    def report_include_block_ids(self) -> bool:
        """Whether reports include the request's current GPU ``block_ids``."""
        report = self._data.get("report") or {}
        return bool(report.get("include_block_ids", True))

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
        """``actions.queue_max_size`` (ActionQueue capacity at bind / construct)."""
        actions = self._data.get("actions") or {}
        try:
            return max(1, int(actions.get("queue_max_size", _DEFAULTS["actions"]["queue_max_size"])))
        except (TypeError, ValueError):
            return int(_DEFAULTS["actions"]["queue_max_size"])

    def dump_get(self, key: str, default: Any = None) -> Any:
        dump = self._data.get("dump") or {}
        return dump.get(key, default)

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
        """Interval-gated JSON sync (file / standalone broadcast).

        Worker end-of-wave prefers the merged due-vector gate in
        ``RuntimeGuardProcessor`` (config + dump). This entry remains for
        file-mode polls, unit tests, and non-merged call sites.
        """
        if not self.hot_reload_enabled:
            return False
        if self.sync_mode == SYNC_BROADCAST:
            group = _runtime_config_sync_group_or_none()
            if group is not None and group.world_size > 1:
                return self._maybe_reload_broadcast(group)
            if not _dist_mod._rg_multi_dp_file_fallback_logged:
                _dist_mod._rg_multi_dp_file_fallback_logged = True
                if _pp_forces_file_sync():
                    _log_pp_force_file_once()
                elif _dp_world_size_or_one() > 1:
                    logger.info(
                        "[runtime_config] multi-DP: per-DP broadcast "
                        "unavailable → local file poll (no cross-DP sync); "
                        "place a readable runtime_config_path on each EngineCore "
                        "(per-node copy ok). path=%s %s",
                        self.config_path,
                        _process_role_tag(),
                    )
                else:
                    logger.info(
                        "[runtime_config] config hot-reload uses local file poll (broadcast group size<=1). path=%s %s",
                        self.config_path,
                        _process_role_tag(),
                    )
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

    def _maybe_reload_broadcast(self, sync_group) -> bool:
        """Standalone config-only bus (tests / non-merged callers)."""
        from vllm_ascend.runtime_config._task_bus import sync_task_bus

        role = _process_role_tag()
        config_due_local = self.config_due_local()
        logger.debug(
            "[runtime_config sync] enter stage=config_task_bus due_local=%s initial_done=%s group_size=%s %s",
            config_due_local,
            self._initial_broadcast_done,
            getattr(sync_group, "world_size", "?"),
            role,
        )
        leader_changed = [False]

        def _build() -> dict[str, Any]:
            payload, changed = self.build_config_sync_payload()
            leader_changed[0] = changed
            return payload

        payload = sync_task_bus(
            sync_group,
            due_local=config_due_local,
            build_payload=_build if sync_group.is_first_rank else None,
            payload=None,
            src=0,
        )
        logger.debug(
            "[runtime_config sync] leave stage=config_task_bus due=%s %s",
            payload is not None,
            role,
        )
        if payload is None or not isinstance(payload, dict):
            return False
        return self.apply_config_sync_payload(
            payload,
            is_leader=bool(sync_group.is_first_rank),
            leader_changed=leader_changed[0],
        )

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
            merged = _deep_merge(_DEFAULTS, _normalize_config_sections(loaded))
            return self._apply_loaded(
                _normalize_config_sections(merged),
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
        # C1: freeze sync_mode at first apply; later reloads keep the frozen value.
        # PP>1 always freezes to file (cross-PP broadcast is unsafe).
        if self._sync_mode_frozen is not None:
            merged["sync_mode"] = self._sync_mode_frozen
        else:
            mode = str(merged.get("sync_mode", SYNC_BROADCAST)).lower()
            if mode not in (SYNC_BROADCAST, SYNC_FILE):
                mode = SYNC_BROADCAST
            if _pp_forces_file_sync():
                if mode == SYNC_BROADCAST:
                    _log_pp_force_file_once()
                mode = SYNC_FILE
            self._sync_mode_frozen = mode
            merged["sync_mode"] = mode
        # Re-assert PP>1 file even if frozen earlier as broadcast (pre-dist).
        if _pp_forces_file_sync():
            if self._sync_mode_frozen != SYNC_FILE:
                _log_pp_force_file_once()
                self._sync_mode_frozen = SYNC_FILE
            merged["sync_mode"] = SYNC_FILE
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
            if self.manual_trigger():
                if self.dump_enabled():
                    logger.info(
                        "[runtime_config] dump.manual_dump active — next "
                        "worker execute_model will consume and arm dump %s",
                        _process_role_tag(),
                    )
                else:
                    logger.warning(
                        "[runtime_config] dump.manual_dump set but dump inactive "
                        "(auto_max_times=0 and manual_dump off) — will not consume %s",
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

    def save(
        self,
        updates: dict[str, Any] | None = None,
        *,
        derive_from_disk: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> bool:
        """Merge ``updates`` and write JSON. Leader (or single-process) only.

        Under the config lock, re-read disk first so a stale in-memory snapshot
        cannot wipe concurrent hand-edits (e.g. ``dump.max_times``) when only
        flushing ``manual_trigger``.

        ``derive_from_disk``: optional callback invoked AFTER the in-lock disk
        re-read + ``updates`` merge. Use this when the caller needs to derive
        a value from current disk (e.g. ``consume_manual_trigger`` deriving
        ``new_val`` from disk rather than stale in-mem), so ``updates`` does
        not inject stale-derived values back over the disk read.
        """
        if not _is_json_writer():
            logger.debug(
                "[runtime_config] save ignored on non-leader path=%s",
                self.config_path,
            )
            return False
        try:
            with self._lock_config():
                on_disk = self._read_json_object()
                # Disk wins over stale memory; then apply intentional updates.
                data = _deep_merge(deepcopy(self._data), on_disk) if on_disk else deepcopy(self._data)
                if updates:
                    data = _deep_merge(data, updates)
                if derive_from_disk is not None:
                    data = derive_from_disk(data)
                data = _normalize_config_sections(data)
                validate_runtime_config(data)
                self._write_data_unlocked(data)
                self._data = data
                self._invalidate_hot_path_gates()
                stat = self.config_path.stat()
                self._mtime = stat.st_mtime
                self._content_digest = self._digest_path(self.config_path)
                self._version = float(stat.st_mtime)
            logger.info("[runtime_config] saved path=%s", self.config_path)
            return True
        except Exception as exc:
            logger.error("[runtime_config] save failed path=%s error=%s", self.config_path, exc)
            return False

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
