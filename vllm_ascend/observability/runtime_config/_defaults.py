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

"""Default ``runtime_config.json`` schema (hot-reload control plane).

Detector sections are assembled from :mod:`detector_catalog` (each detector
declares ``schema``). Shared dump/report/actions/ascend_log stay here.
"""

from __future__ import annotations

from typing import Any

# Startup-only / internal knobs (not exposed in runtime_config.json).
HOT_RELOAD_INTERVAL_SECONDS: float = 3.0
ACTION_QUEUE_MAX_SIZE: int = 64
DUMP_FREE_HEADROOM_BYTES: int = 5 * 1024 * 1024 * 1024
SPEC_SHORT_LOG_INTERVAL_SECONDS: float = 2.0
LOGITS_FINITE_DEFERRED_QUEUE_MAX: int = 256

# Retired nested JSON keys: silently dropped on validate so old on-disk configs
# still load. Unknown *top-level* keys are rejected (no soft-pop).
_RETIRED_DUMP_KEYS: frozenset[str] = frozenset({"free_headroom_bytes"})
_RETIRED_REPORT_KEYS: frozenset[str] = frozenset({"decode_token_ids", "include_block_ids"})
_RETIRED_ACTIONS_KEYS: frozenset[str] = frozenset({"queue_max_size"})

# Detector catalog (imports detector classes; only needs constants above).
from vllm_ascend.observability.runtime_config.detector_catalog import (  # noqa: E402
    build_detector_defaults,
    detector_param_keys,
    retired_detector_keys,
)

_RETIRED_DETECTOR_KEYS: dict[str, frozenset[str]] = retired_detector_keys()

_DEFAULTS: dict[str, Any] = {
    # Hot-reload on/off is startup-only (additional_config.runtime_config_hot_reload);
    # poll period is an internal constant — not exposed in this JSON.
    "dump": {
        # Auto dump (detector anomaly arm): quota >0 enables; mutually exclusive
        # with manual_dump. dump.enabled is derived at runtime (auto || manual).
        "auto_max_times": 0,
        "auto_cooldown_seconds": 5 * 60,
        # Manual dump watermark (never written back by the process):
        # false/0=off; positive int N = dump while process_done < N (one wave
        # at a time). Prefer bumping N by 1 for another shot (1→2→3…). If the
        # value on disk is ≤ already-completed dumps, skip. true=continuous
        # every wave until hot-reload false (not recommended). Needs
        # runtime_config_hot_reload=true. Skips auto quota/cooldown/filters.
        # Reports/request_info carry manual_dump_count (= completed seq).
        "manual_dump": False,
        # KV dump landing root (default derived: <report_dir>/kv_cache).
        # ``<incident_type>/<req_id>/`` is created under it per incident.
        # Settable at startup (additional_config.runtime_dump_dir) and via
        # this JSON key (hot-reload); startup value seeds JSON when unset.
        "dump_dir": None,
    },
    "ascend_log": {
        "level": "INFO",
        # Relative module paths under vllm_ascend forced to DEBUG, e.g. ["runtime_guard"].
        "debug": [],
        # Per-logger overrides, e.g. {"vllm.worker": "WARNING", "runtime_guard": "DEBUG"}.
        "modules": {},
    },
    "report": {
        # Default False: anomaly reports store lengths only.
        # Set true to persist prompt_token_ids + cumulative output_token_ids
        # (always decoded to text when sensitive is on).
        "save_sensitive_info": False,
        # Cap persisted token-id list lengths (0 = unlimited). Counts stay full.
        "max_prompt_token_ids": 100000,
        "max_output_token_ids": 100000,
        # Same (incident_type, req_id): max report files; at cap, stop detecting
        # that req (all detectors). Default 1 = one report then stop.
        # GPU block_ids are always included in report detail.
        "max_per_req": 1,
    },
    # Nested detector sections under ``detector`` (each has ``enabled``).
    "actions": {
        "defaults": {
            # Always includes report so max_per_req can stop-detect after writes.
            "on_trigger": ["report"],
        },
    },
    "detector": build_detector_defaults(),
}

# Allowed top-level keys (typos like ``windw`` must fail validation loudly).
TOP_LEVEL_KEYS: frozenset[str] = frozenset(_DEFAULTS)
# Allowed keys under ``dump`` / ``report``.
DUMP_KEYS: frozenset[str] = frozenset(_DEFAULTS["dump"])
REPORT_KEYS: frozenset[str] = frozenset(_DEFAULTS["report"])
ASCEND_LOG_KEYS: frozenset[str] = frozenset(_DEFAULTS["ascend_log"])
ACTIONS_KEYS: frozenset[str] = frozenset(_DEFAULTS["actions"])
# Per-detector action overrides (not in each detector's default dict).
_DETECTOR_ACTION_KEYS: frozenset[str] = frozenset(
    {
        "on_trigger",
        "dump_kv",
        "report",
    }
)
# Allowed keys per detector section (params ∪ action overrides).
DETECTOR_KEYS: dict[str, frozenset[str]] = {
    name: keys | _DETECTOR_ACTION_KEYS for name, keys in detector_param_keys().items()
}
# Control-plane section for incident_type=manual_trigger (not a detector).
MANUAL_TRIGGER_SECTION_KEYS: frozenset[str] = frozenset(_DETECTOR_ACTION_KEYS)
