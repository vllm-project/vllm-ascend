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

"""Default ``runtime_config.json`` schema (hot-reload control plane)."""

from __future__ import annotations

from typing import Any

from vllm_ascend.runtime_config._dist import SYNC_BROADCAST

_DEFAULTS: dict[str, Any] = {
    # broadcast: EngineCore leader reads JSON, in-DP broadcast (or file poll);
    # file: each rank polls the path (shared FS / per-node copy).
    "sync_mode": SYNC_BROADCAST,
    # Kept in JSON for visibility; effective hot-reload interval is set at
    # process start via additional_config.runtime_config_reload_interval (default 0).
    # Set >0 at startup to enable. JSON field alone cannot re-enable after start.
    "reload_interval_seconds": 0,
    "dump": {
        # Auto dump (detector anomaly arm): quota >0 enables; mutually exclusive
        # with manual_dump. dump.enabled is derived at runtime (auto || manual).
        "auto_max_times": 0,
        "auto_cooldown_seconds": 5 * 60,
        # Manual dump: false/0=off; positive int N = next N armed waves
        # (prefer N=1 — one shot is enough). true=continuous every wave until
        # hot-reload false (not recommended: little debug value, floods disk /
        # ActionQueue). Needs runtime_config_reload_interval>0. Skips auto
        # quota/cooldown/filters. Count is decremented in-memory each armed
        # wave; JSON is rewritten only when the count reaches 0 (false). While
        # the in-memory count is still >0, a hand-edit to this file still
        # hot-reloads into memory as usual. Multi-DP sharing one
        # runtime_config.json: each DP replica may dump up to N times (file
        # stays at N until some replica persists 0) — worst case about
        # num_DP × N dumps across the cluster.
        "manual_dump": False,
        # KV dump landing root (default derived: <report_dir>/kv_cache).
        # ``<incident_type>/<req_id>/`` is created under it per incident.
        # Settable at startup (additional_config.runtime_dump_dir) and via
        # this JSON key (hot-reload); startup value seeds JSON when unset.
        "dump_dir": None,
        # Extra free space required besides the estimated dump payload.
        # Check is: statvfs(free) >= estimate + free_headroom_bytes.
        "free_headroom_bytes": 5 * 1024 * 1024 * 1024,
    },
    "ascend_log": {
        "level": "INFO",
        # Relative module paths under vllm_ascend forced to DEBUG, e.g. ["runtime_guard"].
        "debug": [],
        # Per-logger overrides, e.g. {"vllm.worker": "WARNING", "runtime_guard": "DEBUG"}.
        "modules": {},
    },
    # Ops logging switches (not persisted into anomaly report JSON files).
    "log": {
        # When a request finishes: log output_token_ids + decoded text (TP0 only).
        # Applies to every finished request. Accumulate only while true
        # (no backfill); mid-request enable may be partial or empty.
        "print_output_on_finish": False,
    },
    "report": {
        # Default False: anomaly reports store lengths only.
        # Set true to persist prompt_token_ids + cumulative output_token_ids.
        "save_sensitive_info": False,
        # When save_sensitive_info, decode prompt/output ids to text (lazy tokenizer).
        "decode_token_ids": True,
        # Cap persisted token-id list lengths (0 = unlimited). Counts stay full.
        "max_prompt_token_ids": 1000,
        "max_output_token_ids": 1000,
        # Persist each request's current GPU block_ids in report detail.
        "include_block_ids": True,
        # Same (incident_type, req_id): max report files; at cap, stop detecting
        # that req (all detectors). Default 1 = one report then stop.
        "max_per_req": 1,
    },
    # Nested detector sections under ``detector`` (each has ``enabled``).
    "actions": {
        "defaults": {
            # Always includes report so max_per_req can stop-detect after writes.
            "on_trigger": ["report"],
        },
        # ActionQueue capacity (bind-time). Raise under bursty dump_kv / CPU detect.
        "queue_max_size": 64,
    },
    "detector": {
        # Stop-detect after report.max_per_req successful writes (see report).
        "spec_acceptance": {
            "enabled": False,
            "window": 10,
            "low_threshold": 0.3,
            "len_low_threshold": 1.4,
            "high_threshold": 0.96,
            "len_high_threshold": 2.8,
            # Throttle per-req INFO short logs (seconds).
            "short_log_interval_seconds": 2.0,
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
        # Pre-sample logits NaN/Inf on sampling rows (no msprobe; ill_type=nan).
        # Every step: device isfinite + one .item() gate; indices D2H only on hit.
        "logits_finite": {
            "enabled": False,
            # Cap in-flight deferred alert batches (async keeps ~1-2 steps).
            "deferred_queue_max": 256,
        },
    },
}


# Known nested detector sections under ``detector``.
DETECTOR_SECTIONS: tuple[str, ...] = (
    "spec_acceptance",
    "output_substring",
    "token_repeat",
    "logits_finite",
)
# Allowed top-level keys (typos like ``windw`` must fail validation loudly).
TOP_LEVEL_KEYS: frozenset[str] = frozenset(_DEFAULTS)
# Allowed keys under ``dump`` / ``log`` / ``report``.
DUMP_KEYS: frozenset[str] = frozenset(_DEFAULTS["dump"])
LOG_KEYS: frozenset[str] = frozenset(_DEFAULTS["log"])
REPORT_KEYS: frozenset[str] = frozenset(_DEFAULTS["report"])
ASCEND_LOG_KEYS: frozenset[str] = frozenset(_DEFAULTS["ascend_log"])
ACTIONS_KEYS: frozenset[str] = frozenset(_DEFAULTS["actions"])
# Per-detector action overrides (not in each detector's default dict).
_DETECTOR_ACTION_KEYS: frozenset[str] = frozenset(
    {
        "on_trigger",
        "dump_kv",
        "set_log_level",
        "report",
    }
)
# Allowed keys per detector section (params ∪ action overrides).
DETECTOR_KEYS: dict[str, frozenset[str]] = {
    name: frozenset(sec) | _DETECTOR_ACTION_KEYS for name, sec in _DEFAULTS["detector"].items() if isinstance(sec, dict)
}
# Control-plane section for incident_type=manual_trigger (not a detector).
MANUAL_TRIGGER_SECTION_KEYS: frozenset[str] = frozenset(_DETECTOR_ACTION_KEYS)
