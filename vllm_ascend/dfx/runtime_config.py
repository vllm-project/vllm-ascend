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

"""DFX runtime config: per-engine JSON leader + in-DP broadcast (or file poll).

Design (multi-DP safe — avoid full-world collectives):

1. **One writer / monitor per EngineCore (per DP replica)**
   Reads & writes the JSON (``ensure_persisted`` / ``save`` / ``manual_trigger`` clear).
   Prefer ``inner_dp_world`` first rank; else TP0∧PP0 when ``dp_size>1``; else
   global world rank0 / ``RANK==0``.

2. **Sync scope never spans idle-asymmetric DPs**
   - ``broadcast`` + ``dp_size==1``: sync group = ``get_world_group()``.
   - ``broadcast`` + ``dp_size>1``: sync group = ``inner_dp_world`` only
     (leader reads file → broadcast inside that DP).
   - ``broadcast`` + multi-DP but no ``inner_dp_world``: **local file poll**
     (no collective). Each EngineCore needs a readable ``dfx_config_path``
     (per-node copy or shared FS).
   - ``sync_mode=file``: every rank polls the path (shared FS).

3. **Cross-DP config is not synchronized**
   Edit each DP's JSON (or the shared path each DP can see). Do **not** use
   full EP ``world_size`` (e.g. 32) for config hot-reload — one-sided
   ``execute_dummy_batch`` after a request would deadlock.

Production: ``AscendConfig`` uses ``ensure_file=False``; worker
:meth:`DfxRuntimeConfig.ensure_persisted` materializes JSON on the writer.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

DEFAULT_CONFIG_FILENAME = "dfx_config.json"

# sync_mode values
SYNC_BROADCAST = "broadcast"
SYNC_FILE = "file"


def default_dfx_root() -> Path:
    """Execution-directory DFX root: ``<cwd>/dfx``."""
    return Path(os.getcwd()) / "dfx"


def default_config_dir() -> Path:
    return default_dfx_root() / "config"


_DEFAULTS: dict[str, Any] = {
    # broadcast: EngineCore leader reads JSON, in-DP broadcast (or file poll);
    # file: each rank polls the path (shared FS / per-node copy).
    "sync_mode": SYNC_BROADCAST,
    # Kept in JSON for visibility; effective hot-reload interval is set at
    # process start via additional_config.dfx_config_reload_interval (default 0).
    # Set >0 at startup to enable. JSON field alone cannot re-enable after start.
    "reload_interval_seconds": 0,
    "dump": {
        # Dump arming sink (default off). Orthogonal to detectors: detect-only,
        # dump+detect, and manual-only (dump on, no detector) are all valid.
        "enabled": False,
        # Auto-arm quota only; 0 = no auto dump. Does not affect detect or manual_trigger.
        "max_times": 0,
        "cooldown_seconds": 5 * 60,
        # Manual dump arm: false/0 = off; true = keep dumping every nonempty
        # real execute_model wave until set false; positive int N = next N
        # waves then off. Skips max_times / cooldown / input filters.
        # Needs dump.enabled=true and reload > 0.
        "manual_trigger": False,
        # Effective msprobe JSON path (seeded from ascend dump_config_path at
        # bootstrap when null). Hot-change → recreate debugger.
        "msprobe_config_path": None,
        # One-shot: recreate msprobe debugger from current path, then clear.
        "reload_msprobe": False,
    },
    "ascend_log": {
        "level": "INFO",
        # Relative module paths under vllm_ascend forced to DEBUG, e.g. ["dfx"].
        "debug": [],
    },
    # Ops logging switches (not persisted into anomaly / dump_finish JSON files).
    "log": {
        # Log [SamplingMeta] for the anomalous req (TP0 + last PP only).
        "print_sampling_meta": False,
        # When a request finishes: log output_token_ids + decoded text (TP0 only).
        # Applies to every finished request. Independent of dump_finish sidecars
        # (those are dump-activated reqs only, under report/).
        "print_output_on_finish": False,
    },
    "report": {
        # Default False: anomaly / dump_finish reports store lengths only.
        # Set true to persist prompt_token_ids + cumulative output_token_ids.
        "save_sensitive_info": False,
        # When save_sensitive_info, decode prompt/output ids to text (lazy tokenizer).
        "decode_token_ids": True,
        # Cap persisted token-id list lengths (0 = unlimited). Counts stay full.
        "max_prompt_token_ids": 1000,
        "max_output_token_ids": 1000,
        # Persist each request's current GPU block_ids in report detail.
        "include_block_ids": True,
        # Track/report last write wave per physical block (see blocks[]).
        "block_last_write_wave": False,
        # Track/report last writer req_id per physical block (see blocks[]).
        "block_last_writer": False,
    },
    # Per-detector nested sections. Each has ``enabled`` (default false).
    "detector": {
        # Shared detect behavior (not a detector section): keep detecting a
        # request on every step, but once an anomaly is found for it, stop
        # detecting that request (prevents endless reports for the same req).
        "stop_after_alert": True,
        "spec_acceptance": {
            "enabled": False,
            "window": 10,
            "low_threshold": 0.3,
            "len_low_threshold": 1.4,
            "high_threshold": 0.96,
            "len_high_threshold": 2.8,
        },
        "token_logprob": {
            "enabled": False,
            "window": 64,
            "stride": 32,
            "topk": 20,
            "ill_nan_window_thresh": 1,
            "ill_rare_window_thresh": 1,
            "ill_garbled_window_thresh": 1,
            "ill_repet_window_thresh": 2,
        },
        "output_substring": {
            "enabled": False,
            "patterns": [],
            "add_special_tokens": False,
            # true: patterns match only at the start (prefix) of cumulative output;
            # false (default): match anywhere as a contiguous token-id subsequence.
            "match_prefix": False,
        },
        # Sliding-window token re-read detector (no logprobs). Per new token:
        # score = count of that id in the previous ``window`` content tokens;
        # alert when sum of the last ``window`` scores exceeds threshold.
        "token_repeat": {
            "enabled": False,
            "window": 32,
            "repeat_sum_threshold": 64,
            # Require this many content tokens before alerting (0 = no warmup).
            "min_tokens": 32,
            # Require this many consecutive over-threshold steps.
            "consecutive_hits": 1,
            # Token ids skipped for the content window (e.g. punctuation fillers).
            "ignore_token_ids": [],
        },
    },
    # Detect-time InputFilterManager (+ one-shot prompt print for authoring).
    "input_filter": {
        # [] = no filter. Use type input_token_id_prefix for prefix matching.
        "filters": [],
        # One-shot: next real execute_model with requests logs prompt token ids
        # and length, then cleared to false. Needs reload_interval > 0.
        "print_input_token_ids_once": False,
    },
}


def _reject_unsafe_path(path: Path, *, label: str) -> Path:
    """Resolve and reject NUL / empty paths (basic path hygiene)."""
    raw = str(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"invalid {label}: empty or contains NUL")
    resolved = path.expanduser().resolve()
    # Soft sandbox: warn when outside cwd (shared NFS paths are common).
    try:
        cwd = Path.cwd().resolve()
        if resolved != cwd and cwd not in resolved.parents:
            logger.warning(
                "[DFX runtime_config] %s is outside process cwd (%s): %s",
                label,
                cwd,
                resolved,
            )
    except Exception:
        pass
    return resolved


def resolve_dfx_config_path(configured_path: str | None = None) -> Path:
    """Resolve config file path.

    Priority:
    1. Explicit ``dfx_config_path`` / ``dfx-config`` from additional_config
    2. Default ``<cwd>/dfx/config/dfx_config.json``
    """
    if configured_path:
        return _reject_unsafe_path(Path(configured_path), label="dfx_config_path")
    return _reject_unsafe_path(default_config_dir() / DEFAULT_CONFIG_FILENAME, label="dfx_config_path")


def resolve_dfx_report_dir(config_path: Path, configured_report_dir: str | None = None) -> Path:
    if configured_report_dir:
        return _reject_unsafe_path(Path(configured_report_dir), label="dfx_report_dir")
    dfx_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return _reject_unsafe_path(dfx_root / "report", label="dfx_report_dir")


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


# Paths that already have a non-worker background reloader in this process.
_bg_reload_paths: set[str] = set()


def _normalize_config_sections(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize the ``ascend_log`` section (level + debug list).

    ``ascend_log`` has no ``enabled`` field (level controls logging); strip it
    if a user adds one so the section stays canonical.
    """
    out = dict(data)
    ascend = out.get("ascend_log")
    if not isinstance(ascend, dict):
        ascend = {}
    else:
        ascend = dict(ascend)
    ascend.pop("enabled", None)
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
    out["ascend_log"] = ascend
    return out


def _world_group_or_none():
    try:
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group()
    except Exception:
        return None


def _dp_world_size_or_one() -> int:
    try:
        from vllm.distributed.parallel_state import get_dp_group

        return int(get_dp_group().world_size)
    except Exception:
        return 1


def _inner_dp_world_or_none():
    try:
        from vllm.distributed.parallel_state import get_inner_dp_world_group

        return get_inner_dp_world_group()
    except Exception:
        return None


def _dfx_config_sync_group_or_none():
    """Process group for DFX config broadcast, or None → local file poll.

    Never return the full multi-DP ``get_world_group()`` when ``dp_size>1``:
    after a request, one EngineCore may still ``execute_dummy_batch`` while the
    peer has gone idle — a cross-DP collective deadlocks.

    - dp==1: ``get_world_group()``
    - dp>1 + ``inner_dp_world``: that per-DP group (leader monitors JSON)
    - dp>1 without ``inner_dp_world``: ``None`` → file poll per rank
    """
    world = _world_group_or_none()
    if world is None:
        return None
    if _dp_world_size_or_one() <= 1:
        return world
    return _inner_dp_world_or_none()


_dfx_multi_dp_file_fallback_logged = False


def _is_distributed_worker_process() -> bool:
    """True when this process is (or is becoming) a distributed Worker.

    Used to keep the non-worker file-poll reloader off Workers. Prefer env
    markers (``RANK`` / ``LOCAL_RANK`` / ``VLLM_DP_RANK``) and a live world
    group — AscendConfig may run before ``RANK`` is set, so the background
    loop must re-check and exit if the process later becomes a Worker.
    """
    if os.environ.get("RANK") is not None:
        return True
    if os.environ.get("LOCAL_RANK") is not None:
        return True
    if os.environ.get("VLLM_DP_RANK") is not None:
        return True
    return _world_group_or_none() is not None


def _process_role_tag() -> str:
    """Identify which process applied config (worker broadcast vs API file-poll)."""
    world = _world_group_or_none()
    if world is not None:
        return f"role=worker world_rank={world.rank}/{world.world_size}"
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        return f"role=worker RANK={rank_env} (world not ready)"
    if _is_distributed_worker_process():
        return "role=worker (pre-world)"
    return "role=non-worker"


def _is_json_writer() -> bool:
    """True if this process may write the DFX JSON (one leader per EngineCore).

    Order:
    1. ``inner_dp_world`` first rank (per-DP monitor when the group exists)
    2. Multi-DP without that group: TP0 ∧ PP0 (one writer on each DP replica)
    3. Full world first rank / ``RANK==0`` / single-process
    """
    inner = _inner_dp_world_or_none()
    if inner is not None and inner.world_size > 1:
        return bool(inner.is_first_rank)

    if _dp_world_size_or_one() > 1:
        try:
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group

            return bool(get_tp_group().is_first_rank and get_pp_group().is_first_rank)
        except Exception:
            pass

    world = _world_group_or_none()
    if world is not None and world.world_size > 1:
        return bool(world.is_first_rank)
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            pass
    return True


class DfxRuntimeConfig:
    """Runtime DFX switches loaded from JSON (per-DP broadcast or file poll).

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
        msprobe_config_path: str | None = None,
    ) -> None:
        # None → default ``<cwd>/dfx/config/dfx_config.json`` (not an "explicit" path).
        self._explicit_config_path = config_path is not None
        self.config_path = resolve_dfx_config_path(str(config_path) if config_path is not None else None)
        self.report_dir = resolve_dfx_report_dir(
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
            raise ValueError(f"dfx_config_reload_interval must be >= 0, got {self._reload_interval}")
        self._mtime: float | None = None
        self._version: float = 0.0
        self._last_reload_ts = 0.0
        self._ctor_sync_mode = sync_mode
        self._initial_broadcast_done = False
        self._data = deepcopy(_DEFAULTS)
        self._bootstrap_persisted = False
        self._bg_reloader_started = False
        self._bg_thread: threading.Thread | None = None
        # Seed into bootstrap merge when dump.msprobe_config_path is still null.
        self._startup_msprobe_config_path = (str(msprobe_config_path).strip() if msprobe_config_path else None) or None

        # In-memory merge always. ``ensure_file=True`` persists immediately (tests /
        # rare callers). Production AscendConfig uses False; worker leader calls
        # :meth:`ensure_persisted` once from ``DfxProcessor``.
        self._bootstrap(persist=ensure_file)
        logger.info(
            "[DFX runtime_config] path=%s explicit_path=%s report_dir=%s hot_reload=%s persisted=%s",
            self.config_path,
            self._explicit_config_path,
            self.report_dir,
            self.hot_reload_enabled,
            self._bootstrap_persisted,
        )
        if self.hot_reload_enabled:
            logger.info_once(
                "[DFX runtime_config] hot-reload enabled interval=%.3fs sync_mode=%s path=%s",
                self.reload_interval_seconds,
                self.sync_mode,
                str(self.config_path),
            )
        else:
            logger.info_once(
                "[DFX runtime_config] hot-reload disabled "
                "(set additional_config.dfx_config_reload_interval > 0 to enable; "
                "default is 0; dump.manual_trigger also requires interval > 0)"
            )

    def _read_json_object(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                logger.error(
                    "[DFX runtime_config] root must be object, got %s; ignoring file",
                    type(loaded).__name__,
                )
                return {}
            return loaded
        except Exception as exc:
            logger.warning(
                "[DFX runtime_config] failed to read path=%s error=%s; using defaults",
                self.config_path,
                exc,
            )
            return {}

    def _merge_bootstrap(self, loaded: dict[str, Any]) -> dict[str, Any]:
        """Build effective config for process start.

        - No explicit ``dfx_config_path``: **overwrite** default-path JSON
          with ``_DEFAULTS`` only (ignore prior file on that path).
        - Explicit path: ``_DEFAULTS ← JSON`` (user-owned file).
        Missing keys always come from ``_DEFAULTS``.
        """
        loaded = _normalize_config_sections(loaded) if loaded else loaded
        if not self._explicit_config_path:
            # Default path is runtime-owned: restart without dfx_config_path
            # must not keep hand-edited leftovers from a previous run.
            merged = deepcopy(_DEFAULTS)
        else:
            merged = _deep_merge(_DEFAULTS, loaded)
        if self._ctor_sync_mode is not None:
            merged["sync_mode"] = self._ctor_sync_mode
        # Persist startup hot-reload interval for visibility (runtime gate is still
        # ``self._reload_interval`` only).
        merged["reload_interval_seconds"] = self._reload_interval
        # Seed visible msprobe path when JSON left it null.
        dump = merged.setdefault("dump", {})
        cur = dump.get("msprobe_config_path")
        if (cur is None or (isinstance(cur, str) and not cur.strip())) and self._startup_msprobe_config_path:
            dump["msprobe_config_path"] = self._startup_msprobe_config_path
        return _normalize_config_sections(merged)

    def _write_data_unlocked(self, data: dict[str, Any]) -> None:
        """Atomic write; caller must hold config lock / own the path."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.config_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, self.config_path)

    def _bootstrap(self, *, persist: bool) -> None:
        """Load / merge / optionally save complete effective config at startup.

        Disk write is leader-only (or single-process); other ranks keep in-memory merge.
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        overwrite_default = not self._explicit_config_path
        # Explicit path: read user JSON. Default path: ignore prior file.
        if self._explicit_config_path:
            loaded = self._read_json_object()
        else:
            loaded = {}
        merged = self._merge_bootstrap(loaded)
        self._validate(merged)

        can_write = persist and _is_json_writer()
        if overwrite_default:
            disk_preview = self._read_json_object() if self.config_path.exists() else {}
            disk_ascend = disk_preview.get("ascend_log") if isinstance(disk_preview, dict) else None
            logger.info(
                "[DFX runtime_config] overwrite default json path=%s reason=no_dfx_config_path "
                "will_persist=%s (hand-edits on the default path are ignored at startup; "
                "set additional_config.dfx_config_path for durable config, or "
                "dfx_config_reload_interval>0 to pick up post-start edits)",
                self.config_path,
                can_write,
            )
            if isinstance(disk_ascend, dict) and (
                str(disk_ascend.get("level", "INFO")).upper() != "INFO" or bool(disk_ascend.get("debug"))
            ):
                logger.warning(
                    "[DFX runtime_config] default-path file has ascend_log=%s but startup "
                    "resets to defaults (level=INFO, debug=[]). Edits will not stick across "
                    "restart without dfx_config_path. For live apply set "
                    "dfx_config_reload_interval>0 and confirm log line "
                    "'[ascend_log] applied'. path=%s",
                    disk_ascend,
                    self.config_path,
                )

        if can_write:
            try:
                with self._lock_config():
                    self._write_data_unlocked(merged)
                    mtime = self.config_path.stat().st_mtime
                self._apply_loaded(merged, version=mtime, announce=False)
                self._bootstrap_persisted = True
                logger.info(
                    "[DFX runtime_config] bootstrap saved path=%s explicit_path=%s overwrite_default=%s %s",
                    self.config_path,
                    self._explicit_config_path,
                    overwrite_default,
                    self.interaction_mode_summary(),
                )
                self._warn_interaction_quirks()
            except Exception as exc:
                logger.warning(
                    "[DFX runtime_config] bootstrap save failed path=%s error=%s; using in-memory",
                    self.config_path,
                    exc,
                )
                self._data = merged
                self._version = 0.0
        else:
            self._data = merged
            # Do NOT delete the default-path JSON here. Non-leader workers used to
            # unlink "stale" files while awaiting leader persist, which raced and
            # removed the file the leader had just written — service ends with no
            # dfx_config.json. Leader ``ensure_persisted`` overwrites defaults.
            if self.config_path.exists():
                try:
                    self._mtime = self.config_path.stat().st_mtime
                    self._version = float(self._mtime)
                except OSError:
                    self._version = 0.0
            else:
                self._mtime = None
                self._version = 0.0
            if persist and not _is_json_writer():
                logger.debug(
                    "[DFX runtime_config] bootstrap skip persist (non-leader) path=%s %s",
                    self.config_path,
                    self.interaction_mode_summary(),
                )
            self._warn_interaction_quirks()
        self._last_reload_ts = time.time()

    def ensure_persisted(self) -> bool:
        """Materialize bootstrap merge to disk once (worker leader / single-process).

        Safe to call from every worker: non-leaders no-op; leaders act at most once
        per process. Call from ``DfxProcessor`` so API/EngineCore never persist.

        If the JSON already exists and this is an **explicit** ``dfx_config_path``,
        skip rewrite: disk is the source of truth. Default path (no explicit
        path) always materializes ``defaults←startup`` so a restart does not
        keep leftover fields from a previous run.
        """
        if self._bootstrap_persisted:
            return True
        if not _is_json_writer():
            logger.debug(
                "[DFX runtime_config] ensure_persisted skip (non-leader) path=%s",
                self.config_path,
            )
            return False
        overwrite_default = not self._explicit_config_path
        try:
            with self._lock_config():
                if self.config_path.exists() and not overwrite_default:
                    mtime = self.config_path.stat().st_mtime
                    self._mtime = mtime
                    self._version = float(mtime)
                    self._bootstrap_persisted = True
                    logger.info(
                        "[DFX runtime_config] ensure_persisted skip rewrite (explicit path, file exists) path=%s",
                        self.config_path,
                    )
                    return True
                self._write_data_unlocked(self._data)
                mtime = self.config_path.stat().st_mtime
            self._mtime = mtime
            self._version = float(mtime)
            self._bootstrap_persisted = True
            logger.info(
                "[DFX runtime_config] worker leader persisted path=%s explicit_path=%s overwrite_default=%s",
                self.config_path,
                self._explicit_config_path,
                overwrite_default,
            )
            return True
        except Exception as exc:
            logger.warning(
                "[DFX runtime_config] ensure_persisted failed path=%s error=%s",
                self.config_path,
                exc,
            )
            return False

    # ---- section accessors -------------------------------------------------

    @property
    def hot_reload_enabled(self) -> bool:
        """True when startup ``dfx_config_reload_interval`` > 0."""
        return self._reload_interval > 0

    @property
    def sync_mode(self) -> str:
        mode = str(self._data.get("sync_mode", SYNC_BROADCAST)).lower()
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

    @property
    def input_filter(self) -> dict[str, Any]:
        return self._data["input_filter"]

    def dump_enabled(self) -> bool:
        return bool(self.dump.get("enabled", False))

    def any_detector_enabled(self) -> bool:
        """True if at least one auto anomaly detector is enabled."""
        return self.detectors_enabled_in(self._data)

    # Known nested detector sections under ``detector``.
    DETECTOR_SECTIONS: tuple[str, ...] = (
        "spec_acceptance",
        "token_logprob",
        "output_substring",
        "token_repeat",
    )
    # Allowed keys under ``dump`` (reject typos such as legacy dump_once).
    DUMP_KEYS: frozenset[str] = frozenset(_DEFAULTS["dump"])
    LOG_KEYS: frozenset[str] = frozenset(_DEFAULTS["log"])
    REPORT_KEYS: frozenset[str] = frozenset(_DEFAULTS["report"])

    @staticmethod
    def detectors_enabled_in(data: dict[str, Any]) -> bool:
        """Whether ``data['detector']`` has any auto anomaly detector on."""
        det = data.get("detector") or {}
        for name in DfxRuntimeConfig.DETECTOR_SECTIONS:
            sec = det.get(name)
            if isinstance(sec, dict) and bool(sec.get("enabled", False)):
                return True
        return False

    def interaction_mode_summary(self) -> str:
        """Short ops-facing mode tag for logs (detect / dump axes)."""
        names: list[str] = []
        if bool(self.detector_get("spec_acceptance", "enabled", False)):
            names.append("spec")
        if bool(self.detector_get("token_logprob", "enabled", False)):
            names.append("token")
        if bool(self.detector_get("output_substring", "enabled", False)):
            names.append("output_substring")
        if bool(self.detector_get("token_repeat", "enabled", False)):
            names.append("token_repeat")
        dump_on = self.dump_enabled()
        max_times = self.dump_max_times()
        if names and dump_on and max_times > 0:
            mode = "detect+auto_dump"
        elif names and dump_on:
            mode = "detect+manual_dump"
        elif names:
            mode = "detect_only"
        elif dump_on:
            mode = "manual_dump_only"
        else:
            mode = "idle"
        return f"mode={mode} detectors={names} dump.enabled={dump_on} max_times={max_times}"

    def _warn_interaction_quirks(self) -> None:
        """Warn on valid-but-easy-to-misread dump/detect combinations."""
        if self.dump_enabled() and not self.any_detector_enabled():
            logger.warning(
                "[DFX runtime_config] dump.enabled=true with no auto detector — "
                "auto dump will not trigger; dump.manual_trigger still works %s",
                _process_role_tag(),
            )
        elif self.dump_enabled() and self.any_detector_enabled() and self.dump_max_times() <= 0:
            logger.info(
                "[DFX runtime_config] dump.enabled=true max_times=0 — "
                "detect runs; auto-arm off; dump.manual_trigger still works %s",
                _process_role_tag(),
            )

    def dump_max_times(self) -> int:
        return int(self.dump.get("max_times", 0))

    def dump_cooldown_seconds(self) -> int:
        return int(self.dump.get("cooldown_seconds", 300))

    def manual_trigger_continuous(self) -> bool:
        """True when ``dump.manual_trigger`` is bool ``true`` (always-on dump)."""
        return self.dump.get("manual_trigger", False) is True

    def manual_trigger_count(self) -> int:
        """Remaining manual dump waves from ``dump.manual_trigger``.

        ``false``/``0`` → 0; ``true`` (continuous) → 1 as a positive sentinel;
        positive int → N. Continuous mode does not decrement on consume.
        Only observed after a successful hot-reload; requires
        ``dfx_config_reload_interval > 0``.
        """
        raw = self.dump.get("manual_trigger", False)
        if isinstance(raw, bool):
            return 1 if raw else 0
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def manual_trigger(self) -> bool:
        """True when manual dump is armed (continuous or remaining count > 0)."""
        return self.manual_trigger_count() > 0

    def dump_msprobe_config_path(self) -> str | None:
        """Effective msprobe JSON path from ``dump.msprobe_config_path`` (or None)."""
        raw = self.dump.get("msprobe_config_path")
        if raw is None:
            return None
        if not isinstance(raw, str):
            return None
        path = raw.strip()
        return path or None

    def dump_reload_msprobe(self) -> bool:
        """One-shot flag to recreate the msprobe debugger."""
        return bool(self.dump.get("reload_msprobe", False))

    def consume_reload_msprobe(self) -> bool:
        """If ``reload_msprobe`` is true, clear it (persist on writer) and return True."""
        if not self.dump_reload_msprobe():
            return False
        self.dump["reload_msprobe"] = False
        if _is_json_writer():
            if self.save({"dump": {"reload_msprobe": False}}):
                logger.info(
                    "[DFX runtime_config] reload_msprobe consumed → false path=%s %s",
                    self.config_path,
                    _process_role_tag(),
                )
            else:
                logger.warning(
                    "[DFX runtime_config] reload_msprobe cleared in-memory but failed to persist path=%s %s",
                    self.config_path,
                    _process_role_tag(),
                )
        else:
            logger.debug(
                "[DFX runtime_config] reload_msprobe cleared in-memory (non-writer) %s",
                _process_role_tag(),
            )
        return True

    def input_filter_configs(self) -> list[dict[str, Any]]:
        """Normalized ``input_filter.filters`` for ``InputFilterManager``."""
        from vllm_ascend.dfx.input_filters import normalize_input_filter_configs

        raw = self.input_filter.get("filters", [])
        try:
            return normalize_input_filter_configs(raw)
        except ValueError:
            return []

    def print_input_token_ids_once(self) -> bool:
        """One-shot prompt token-id print for filter authoring.

        Requires ``dfx_config_reload_interval > 0``. Cleared after a real
        ``execute_model`` wave that has printable prompts.
        """
        return bool(self.input_filter.get("print_input_token_ids_once", False))

    def consume_print_input_token_ids_once(self) -> bool:
        """If ``print_input_token_ids_once`` is true, clear it and return True."""
        if not self.print_input_token_ids_once():
            return False
        self.input_filter["print_input_token_ids_once"] = False
        if _is_json_writer():
            if self.save({"input_filter": {"print_input_token_ids_once": False}}):
                logger.info(
                    "[DFX runtime_config] print_input_token_ids_once consumed → false path=%s %s",
                    self.config_path,
                    _process_role_tag(),
                )
            else:
                logger.warning(
                    "[DFX runtime_config] print_input_token_ids_once cleared in-memory "
                    "but failed to persist path=%s %s",
                    self.config_path,
                    _process_role_tag(),
                )
        else:
            logger.info(
                "[DFX runtime_config] print_input_token_ids_once cleared in-memory (non-writer) %s",
                _process_role_tag(),
            )
        return True

    def consume_manual_trigger(self) -> bool:
        """Arm one manual dump wave; return True if armed.

        - ``true`` (bool): continuous — leave value as ``true``, do not persist.
        - positive int: decrement by one (``false`` when drained) and persist.
        All ranks update in-memory for the int path; only the JSON writer persists.
        """
        if self.manual_trigger_continuous():
            logger.debug(
                "[DFX runtime_config] manual_trigger continuous (true); not clearing %s",
                _process_role_tag(),
            )
            return True
        remaining = self.manual_trigger_count()
        if remaining <= 0:
            return False
        new_val: bool | int = False if remaining <= 1 else remaining - 1
        self.dump["manual_trigger"] = new_val
        if _is_json_writer():
            if self.save({"dump": {"manual_trigger": new_val}}):
                logger.info(
                    "[DFX runtime_config] manual_trigger consumed → %s (was %d) path=%s %s",
                    new_val,
                    remaining,
                    self.config_path,
                    _process_role_tag(),
                )
            else:
                logger.warning(
                    "[DFX runtime_config] manual_trigger decremented in-memory but failed "
                    "to persist path=%s remaining_was=%d %s",
                    self.config_path,
                    remaining,
                    _process_role_tag(),
                )
        else:
            logger.info(
                "[DFX runtime_config] manual_trigger → %s in-memory (non-writer; was %d) %s",
                new_val,
                remaining,
                _process_role_tag(),
            )
        return True

    def disable_dump_unavailable(self, *, reason: str) -> bool:
        """Force ``dump.enabled=false`` when msprobe dump cannot run.

        Used at startup / reload when debugger init fails. Returns True if
        the in-memory flag was changed.
        """
        if not self.dump_enabled():
            return False
        self.dump["enabled"] = False
        logger.error(
            "[DFX runtime_config] dump.enabled forced false: %s %s",
            reason,
            _process_role_tag(),
        )
        if _is_json_writer():
            if self.save({"dump": {"enabled": False}}):
                logger.info(
                    "[DFX runtime_config] dump.enabled=false persisted path=%s %s",
                    self.config_path,
                    _process_role_tag(),
                )
            else:
                logger.warning(
                    "[DFX runtime_config] dump.enabled cleared in-memory but failed to persist path=%s %s",
                    self.config_path,
                    _process_role_tag(),
                )
        else:
            logger.info(
                "[DFX runtime_config] dump.enabled cleared in-memory (non-writer) %s",
                _process_role_tag(),
            )
        return True

    def disable_detector_unavailable(self, section: str, *, reason: str) -> bool:
        """Force ``detector.<section>.enabled=false`` when a hard dependency is missing.

        Used for ``token_logprob`` when msprobe ILLDetector cannot be imported.
        Returns True if the in-memory flag was changed.
        """
        sec = self.detector_section(section)
        if not bool(sec.get("enabled", False)):
            return False
        sec["enabled"] = False
        logger.error(
            "[DFX runtime_config] detector.%s.enabled forced false: %s %s",
            section,
            reason,
            _process_role_tag(),
        )
        if _is_json_writer():
            if self.save({"detector": {section: {"enabled": False}}}):
                logger.info(
                    "[DFX runtime_config] detector.%s.enabled=false persisted path=%s %s",
                    section,
                    self.config_path,
                    _process_role_tag(),
                )
            else:
                logger.warning(
                    "[DFX runtime_config] detector.%s.enabled cleared in-memory but failed to persist path=%s %s",
                    section,
                    self.config_path,
                    _process_role_tag(),
                )
        else:
            logger.info(
                "[DFX runtime_config] detector.%s.enabled cleared in-memory (non-writer) %s",
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

    def report_save_sensitive_info(self) -> bool:
        """Whether anomaly / dump_finish reports persist plaintext token-id lists.

        Default False: only lengths (``*_token_count``). ``true`` keeps full
        ``prompt_token_ids`` and cumulative ``output_token_ids``.
        """
        report = self._data.get("report") or {}
        return bool(report.get("save_sensitive_info", False))

    def log_print_sampling_meta(self) -> bool:
        """Whether to log ``[SamplingMeta]`` when writing an anomaly report.

        Default False. When True, only TP0 + last PP emit the log (detect-only
        or after successful dump arm — wherever ``write_report`` runs).
        """
        log_sec = self._data.get("log") or {}
        return bool(log_sec.get("print_sampling_meta", False))

    def log_print_output_on_finish(self) -> bool:
        """Whether to log output token ids + text when any request finishes.

        Default False. When True, TP0 logs on reap (after ``mark_finished``) for every
        finished request (independent of dump_finish sidecars and of
        ``save_sensitive_info``). Can be large / sensitive — leave off in prod.
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

    def report_block_last_write_wave(self) -> bool:
        """Track and report each block's last KV-write wave."""
        report = self._data.get("report") or {}
        return bool(report.get("block_last_write_wave", False))

    def report_block_last_writer(self) -> bool:
        """Track and report each block's last writer ``req_id``."""
        report = self._data.get("report") or {}
        return bool(report.get("block_last_writer", False))

    def report_block_meta_enabled(self) -> bool:
        """True when any per-block write metadata tracking is on."""
        return self.report_block_last_write_wave() or self.report_block_last_writer()

    def detector_section(self, name: str) -> dict[str, Any]:
        """Return nested ``detector.<name>`` object (empty dict if missing)."""
        sec = self.detector.get(name)
        return sec if isinstance(sec, dict) else {}

    def detector_get(self, section: str, key: str, default: Any = None) -> Any:
        """Read ``detector.<section>.<key>``."""
        return self.detector_section(section).get(key, default)

    def stop_after_alert(self) -> bool:
        """True: stop detecting a request once it has produced an anomaly.

        Shared ``detector.stop_after_alert`` flag (default ``True``). Each request
        keeps being checked on every step until it alerts; afterwards
        ``DetectorManager`` skips it entirely so the same anomaly does not write
        endless reports. Set ``False`` to keep checking (and re-alerting) forever.
        """
        return bool(self.detector.get("stop_after_alert", True))

    def apply_ascend_log_level(self) -> None:
        """Apply live ``ascend_log`` to the ``vllm_ascend`` logger tree."""
        from vllm_ascend.logger import apply_ascend_log_level as _apply

        _apply(self.ascend_log_level(), self.ascend_log_debug_modules())

    def start_non_worker_background_reload(self) -> bool:
        """Daemon thread: file-poll JSON and re-apply ``ascend_log`` (API / EngineCore).

        - No-op when hot-reload is off, or this process is a distributed Worker.
        - Uses **local file reload only** — never joins worker world broadcast.
        - Does not persist JSON.
        - Callers should invoke :meth:`apply_ascend_log_level` once at construction
          for the initial level; this thread only re-applies after file changes.
        Workers keep step-driven :meth:`sync_dfx_config` and must not run this.
        If AscendConfig starts the thread before Worker env/world is ready, the
        loop exits as soon as :func:`_is_distributed_worker_process` becomes true.
        """
        if not self.hot_reload_enabled:
            return False
        if _is_distributed_worker_process():
            logger.info(
                "[DFX runtime_config] skip non-worker reloader (worker process) path=%s",
                self.config_path,
            )
            return False
        if self._bg_reloader_started:
            return False
        path_key = str(self.config_path.resolve()) if self.config_path.exists() else str(self.config_path)
        if path_key in _bg_reload_paths:
            self._bg_reloader_started = True
            logger.info(
                "[DFX runtime_config] non-worker reloader already running for path=%s",
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
                        "[DFX runtime_config] non-worker reloader exiting (process is worker) path=%s",
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
                                "[DFX runtime_config] dump.manual_trigger=true seen on "
                                "non-worker reload — dump arms only on worker "
                                "execute_model (send a request). path=%s",
                                self.config_path,
                            )
                except Exception as exc:
                    logger.warning(
                        "[DFX runtime_config] non-worker reload error path=%s error=%s",
                        self.config_path,
                        exc,
                    )

        self._bg_thread = threading.Thread(
            target=_loop,
            name="dfx-non-worker-reload",
            daemon=True,
        )
        self._bg_thread.start()
        logger.info(
            "[DFX runtime_config] non-worker background reload started interval=%.3fs path=%s",
            interval,
            self.config_path,
        )
        return True

    def sync_dfx_config(self) -> bool:
        """Canonical step entry for interval-gated DFX JSON sync.

        No-op when hot-reload is disabled (``dfx_config_reload_interval<=0``).

        Broadcast: collective on the **per-DP** sync group (never full multi-DP
        world). Group leader monitors JSON and broadcasts; every rank in that
        group must call each step. Multi-DP without ``inner_dp_world`` falls
        back to local file poll. Call from ``DfxProcessor.refresh_config`` /
        ``sync_for_step`` — do not skip early-PP ranks when broadcast is used.
        """
        if not self.hot_reload_enabled:
            return False
        if self.sync_mode == SYNC_BROADCAST:
            group = _dfx_config_sync_group_or_none()
            if group is not None and group.world_size > 1:
                return self._maybe_reload_broadcast(group)
            global _dfx_multi_dp_file_fallback_logged
            if not _dfx_multi_dp_file_fallback_logged:
                _dfx_multi_dp_file_fallback_logged = True
                if _dp_world_size_or_one() > 1:
                    logger.info(
                        "[DFX runtime_config] multi-DP: per-DP broadcast "
                        "unavailable → local file poll (no cross-DP sync); "
                        "place a readable dfx_config_path on each EngineCore "
                        "(per-node copy ok). path=%s %s",
                        self.config_path,
                        _process_role_tag(),
                    )
                else:
                    logger.info(
                        "[DFX runtime_config] config hot-reload uses local "
                        "file poll (broadcast group size<=1). path=%s %s",
                        self.config_path,
                        _process_role_tag(),
                    )
        logger.debug(
            "[DFX sync] enter stage=config_local_reload %s",
            _process_role_tag(),
        )
        changed = self._maybe_reload_local()
        logger.debug(
            "[DFX sync] leave stage=config_local_reload changed=%s %s",
            changed,
            _process_role_tag(),
        )
        return changed

    def _maybe_reload_local(self) -> bool:
        now = time.time()
        if now - self._last_reload_ts < self.reload_interval_seconds:
            return False
        return self.reload(force=False)

    def _maybe_reload_broadcast(self, sync_group) -> bool:
        """Per-DP (or single-engine) leader monitors JSON; group syncs payload.

        ``sync_group`` must be one EngineCore's ranks only (never cross-DP world).
        Interval gate uses ``all_reduce(due)`` so followers enter broadcast in
        lockstep with the leader; file I/O stays on ``sync_group`` first rank.
        """
        import torch

        now = time.time()
        interval = self.reload_interval_seconds
        due_local = 1.0 if ((not self._initial_broadcast_done) or (now - self._last_reload_ts >= interval)) else 0.0
        role = _process_role_tag()
        logger.debug(
            "[DFX sync] enter stage=config_all_reduce due_local=%.0f initial_done=%s group_size=%s %s",
            due_local,
            self._initial_broadcast_done,
            getattr(sync_group, "world_size", "?"),
            role,
        )
        due_t = torch.tensor([due_local], dtype=torch.float32)
        torch.distributed.all_reduce(
            due_t,
            op=torch.distributed.ReduceOp.MAX,
            group=sync_group.cpu_group,
        )
        due = float(due_t.item())
        logger.debug("[DFX sync] leave stage=config_all_reduce due=%.0f %s", due, role)
        if due < 0.5:
            return False

        self._last_reload_ts = now
        changed = False
        payload: dict[str, Any] | None = None
        if sync_group.is_first_rank:
            logger.debug("[DFX sync] enter stage=config_reload_file %s", role)
            # Always re-stat; reload() no-ops when mtime unchanged.
            changed = self.reload(force=False)
            logger.debug(
                "[DFX sync] leave stage=config_reload_file changed=%s version=%.6f %s",
                changed,
                float(self._version),
                role,
            )
            first_sync = not self._initial_broadcast_done
            # Cheap path: when file unchanged after initial sync, omit full JSON
            # so followers only pay for a small version fingerprint.
            if changed or first_sync:
                payload = {
                    "version": float(self._version),
                    "data": deepcopy(self._data),
                }
            else:
                payload = {
                    "version": float(self._version),
                    "data": None,
                }
        logger.debug("[DFX sync] enter stage=config_broadcast %s", role)
        payload = sync_group.broadcast_object(payload, src=0)
        logger.debug("[DFX sync] leave stage=config_broadcast %s", role)
        first_sync = not self._initial_broadcast_done
        self._initial_broadcast_done = True
        if payload is None or not isinstance(payload, dict):
            return False
        version = float(payload.get("version", 0.0))
        data = payload.get("data")
        if data is None:
            # Fingerprint-only: no content change on leader.
            return False if not sync_group.is_first_rank else (changed or first_sync)
        if not isinstance(data, dict):
            return False
        if not sync_group.is_first_rank:
            if version != self._version:
                return self._apply_loaded(data, version=version)
            return False
        # Leader: true if file changed, or first sync (refresh callers once).
        return changed or first_sync

    def reload(self, *, force: bool = False) -> bool:
        """Local file reload (leader / file mode / pre-dist bootstrap).

        Hot-reload follows JSON only: ``defaults ← JSON``.
        """
        self._last_reload_ts = time.time()
        if not self.config_path.exists():
            if force:
                self._data = deepcopy(_DEFAULTS)
                self._version = 0.0
            return False

        try:
            mtime = self.config_path.stat().st_mtime
        except OSError as exc:
            logger.warning("[DFX runtime_config] stat failed path=%s error=%s", self.config_path, exc)
            return False

        if not force and self._mtime is not None and mtime <= self._mtime:
            return False

        try:
            with self._lock_config(), self.config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                logger.error("[DFX runtime_config] root must be object, got %s", type(loaded).__name__)
                return False
            merged = _deep_merge(_DEFAULTS, _normalize_config_sections(loaded))
            return self._apply_loaded(_normalize_config_sections(merged), version=mtime)
        except Exception as exc:
            logger.error("[DFX runtime_config] reload failed path=%s error=%s", self.config_path, exc)
            return False

    def _apply_loaded(
        self,
        merged: dict[str, Any],
        *,
        version: float,
        announce: bool = True,
    ) -> bool:
        merged = _normalize_config_sections(merged)
        self._validate(merged)
        changes = _leaf_changes(self._data, merged)
        self._data = merged
        self._mtime = version
        self._version = version
        if announce and changes:
            # Only print fields that actually changed (e.g. dump.max_times).
            logger.info(
                "[DFX runtime_config] updated path=%s version=%.6f %s changes=[%s] %s",
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
                        "[DFX runtime_config] dump.manual_trigger=true loaded — next "
                        "worker execute_model will consume and arm dump %s",
                        _process_role_tag(),
                    )
                else:
                    logger.warning(
                        "[DFX runtime_config] dump.manual_trigger=true but "
                        "dump.enabled=false — will not consume until dump.enabled=true %s",
                        _process_role_tag(),
                    )
        elif announce:
            logger.debug(
                "[DFX runtime_config] apply no content change path=%s version=%.6f %s",
                str(self.config_path),
                self._version,
                _process_role_tag(),
            )
        return True

    def save(self, updates: dict[str, Any] | None = None) -> bool:
        """Merge ``updates`` and write JSON. Leader (or single-process) only.

        Under the config lock, re-read disk first so a stale in-memory snapshot
        cannot wipe concurrent hand-edits (e.g. ``dump.max_times``) when only
        flushing ``manual_trigger``.
        """
        if not _is_json_writer():
            logger.debug(
                "[DFX runtime_config] save ignored on non-leader path=%s",
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
                data = _normalize_config_sections(data)
                self._validate(data)
                self._write_data_unlocked(data)
                self._data = data
                self._mtime = self.config_path.stat().st_mtime
                self._version = float(self._mtime)
            logger.info("[DFX runtime_config] saved path=%s", self.config_path)
            return True
        except Exception as exc:
            logger.error("[DFX runtime_config] save failed path=%s error=%s", self.config_path, exc)
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

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        """Validate / normalize ``data`` in place.

        Detect and dump are orthogonal: dump-only / detect-only / both are valid.
        Soft warnings for easy-to-misread combos live in ``_warn_interaction_quirks``.
        """
        for section in (
            "dump",
            "ascend_log",
            "log",
            "detector",
            "input_filter",
            "report",
        ):
            if section not in data or not isinstance(data[section], dict):
                raise ValueError(f"dfx config missing object section '{section}'")
        interval = data.get("reload_interval_seconds", 0)
        if not isinstance(interval, (int, float)) or interval < 0:
            raise ValueError(f"reload_interval_seconds must be >= 0, got {interval}")
        sync_mode = str(data.get("sync_mode", SYNC_BROADCAST)).lower()
        if sync_mode not in (SYNC_BROADCAST, SYNC_FILE):
            raise ValueError(f"sync_mode must be '{SYNC_BROADCAST}' or '{SYNC_FILE}'")
        enabled = data["dump"].get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("dump.enabled must be bool")
        unknown_dump = sorted(set(data["dump"]) - DfxRuntimeConfig.DUMP_KEYS)
        if unknown_dump:
            raise ValueError(f"dump has unknown key(s) {unknown_dump}; allowed={sorted(DfxRuntimeConfig.DUMP_KEYS)}")
        unknown_log = sorted(set(data["log"]) - DfxRuntimeConfig.LOG_KEYS)
        if unknown_log:
            raise ValueError(f"log has unknown key(s) {unknown_log}; allowed={sorted(DfxRuntimeConfig.LOG_KEYS)}")
        unknown_report = sorted(set(data["report"]) - DfxRuntimeConfig.REPORT_KEYS)
        if unknown_report:
            raise ValueError(
                f"report has unknown key(s) {unknown_report}; allowed={sorted(DfxRuntimeConfig.REPORT_KEYS)}"
            )
        manual_trigger = data["dump"].get("manual_trigger")
        if manual_trigger is not None and not isinstance(manual_trigger, bool):
            # bool kept as-is; int N = remaining waves; reject other types.
            if isinstance(manual_trigger, int) and not isinstance(manual_trigger, bool):
                if manual_trigger < 0:
                    raise ValueError("dump.manual_trigger must be >= 0")
                # 0 → false for compact defaults; keep positive ints as counts.
                if manual_trigger == 0:
                    data["dump"]["manual_trigger"] = False
            else:
                raise ValueError("dump.manual_trigger must be bool or non-negative int")
        msprobe_path = data["dump"].get("msprobe_config_path")
        if msprobe_path is not None and not isinstance(msprobe_path, str):
            raise ValueError("dump.msprobe_config_path must be str or null")
        if isinstance(msprobe_path, str) and not msprobe_path.strip():
            data["dump"]["msprobe_config_path"] = None
        reload_msprobe = data["dump"].get("reload_msprobe")
        if reload_msprobe is not None and not isinstance(reload_msprobe, bool):
            if reload_msprobe in (0, 1):
                data["dump"]["reload_msprobe"] = bool(reload_msprobe)
            else:
                raise ValueError("dump.reload_msprobe must be bool")
        save_sensitive = data["report"].get("save_sensitive_info")
        if save_sensitive is not None and not isinstance(save_sensitive, bool):
            if save_sensitive in (0, 1):
                data["report"]["save_sensitive_info"] = bool(save_sensitive)
            else:
                raise ValueError("report.save_sensitive_info must be bool")
        for log_key in ("print_sampling_meta", "print_output_on_finish"):
            log_val = data["log"].get(log_key)
            if log_val is not None and not isinstance(log_val, bool):
                if log_val in (0, 1):
                    data["log"][log_key] = bool(log_val)
                else:
                    raise ValueError(f"log.{log_key} must be bool")
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
        for block_key in (
            "include_block_ids",
            "block_last_write_wave",
            "block_last_writer",
        ):
            block_val = data["report"].get(block_key)
            if block_val is not None and not isinstance(block_val, bool):
                if block_val in (0, 1):
                    data["report"][block_key] = bool(block_val)
                else:
                    raise ValueError(f"report.{block_key} must be bool")
        print_once = data["input_filter"].get("print_input_token_ids_once")
        if print_once is not None and not isinstance(print_once, bool):
            if print_once in (0, 1):
                data["input_filter"]["print_input_token_ids_once"] = bool(print_once)
            else:
                raise ValueError("input_filter.print_input_token_ids_once must be bool")
        from vllm_ascend.dfx.input_filters import normalize_input_filter_configs

        data["input_filter"]["filters"] = normalize_input_filter_configs(data["input_filter"].get("filters", []))
        level = data["ascend_log"].get("level", "INFO")
        if not isinstance(level, str):
            raise ValueError("ascend_log.level must be str")
        debug = data["ascend_log"].get("debug", [])
        if debug is None:
            debug = []
        if isinstance(debug, str):
            debug = [debug]
        if not isinstance(debug, list):
            raise ValueError("ascend_log.debug must be a list of module name strings")
        for item in debug:
            if not isinstance(item, (str, int, float)):
                raise ValueError("ascend_log.debug entries must be strings")
        detector = data["detector"]
        known = set(DfxRuntimeConfig.DETECTOR_SECTIONS)
        for key, value in detector.items():
            if key == "stop_after_alert":
                # Shared detect-behavior flag, not a detector section.
                if not isinstance(value, bool):
                    if value in (0, 1):
                        detector["stop_after_alert"] = bool(value)
                    else:
                        raise ValueError("detector.stop_after_alert must be bool")
                continue
            if key not in known:
                raise ValueError(
                    f"detector.{key} is not a known detector section; "
                    f"expected nested objects among {sorted(known)} "
                    f"(e.g. detector.spec_acceptance.enabled)"
                )
            if not isinstance(value, dict):
                raise ValueError(f"detector.{key} must be an object")
        for name in DfxRuntimeConfig.DETECTOR_SECTIONS:
            sec = detector.setdefault(name, {})
            if not isinstance(sec, dict):
                raise ValueError(f"detector.{name} must be an object")
            enabled = sec.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                if enabled in (0, 1):
                    sec["enabled"] = bool(enabled)
                else:
                    raise ValueError(f"detector.{name}.enabled must be bool")

        token = detector["token_logprob"]
        window = token.get("window", 64)
        stride = token.get("stride", 32)
        if int(window) < int(stride):
            raise ValueError("detector.token_logprob.window must be >= detector.token_logprob.stride")

        from vllm_ascend.dfx.detector.output_substring import normalize_raw_patterns

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

        from vllm_ascend.dfx.detector.token_repeat import normalize_ignore_token_ids

        token_repeat = detector["token_repeat"]
        window = int(token_repeat.get("window", 32))
        if window < 1:
            raise ValueError("detector.token_repeat.window must be >= 1")
        token_repeat["window"] = window
        thresh = int(token_repeat.get("repeat_sum_threshold", 64))
        if thresh < 0:
            raise ValueError("detector.token_repeat.repeat_sum_threshold must be >= 0")
        token_repeat["repeat_sum_threshold"] = thresh
        min_tokens = int(token_repeat.get("min_tokens", window))
        if min_tokens < 0:
            raise ValueError("detector.token_repeat.min_tokens must be >= 0")
        token_repeat["min_tokens"] = min_tokens
        consecutive_hits = int(token_repeat.get("consecutive_hits", 1))
        if consecutive_hits < 1:
            raise ValueError("detector.token_repeat.consecutive_hits must be >= 1")
        token_repeat["consecutive_hits"] = consecutive_hits
        token_repeat["ignore_token_ids"] = normalize_ignore_token_ids(token_repeat.get("ignore_token_ids", []))
