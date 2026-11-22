# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Opt-in request-boundary timing for PD disaggregation diagnostics.

The records are emitted as single ``[KVPD_TIMING]`` JSON lines so that Prefill
and Decode engine logs can be aligned by request id without changing the
scheduler. The switch lives in ``kv_connector_extra_config`` and defaults to
False; no timing work is done when it is disabled.
"""

import json
import os
import time

from vllm.logger import logger

KVPD_TIMING_PREFIX = "[KVPD_TIMING]"
KVPD_TIMING_CONFIG_KEY = "enable_timing"
KVPD_DIAGNOSTIC_OPERATIONS = ("load_get", "save_put")
# Keep the first-token bookkeeping bounded on long-lived engines.
KVPD_FIRST_TOKEN_TRACK_LIMIT = 65536


def sum_transfer_bytes(sizes) -> int:
    """Sum a backend size list whose entries may themselves be size lists."""
    total = 0
    for size in sizes:
        if isinstance(size, (list, tuple)):
            total += sum(int(value) for value in size)
        else:
            total += int(size)
    return total


def kvpd_timing_enabled(vllm_config) -> bool:
    """Return whether PD timing was explicitly enabled for this engine."""
    if vllm_config is None:
        return False
    transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    extra_config = getattr(transfer_config, "kv_connector_extra_config", None)
    if not isinstance(extra_config, dict):
        return False
    return bool(extra_config.get(KVPD_TIMING_CONFIG_KEY, False))


def log_kvpd_timing(enabled: bool, event: str, req_id: str | None = None, **fields: object) -> None:
    """Emit one timestamped boundary record when timing is enabled."""
    if not enabled:
        return
    record: dict[str, object] = {
        "event": event,
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
    }
    if req_id is not None:
        record["req"] = req_id
    record.update(fields)
    logger.info("%s %s", KVPD_TIMING_PREFIX, json.dumps(record, sort_keys=True, default=str))
