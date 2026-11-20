# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Device-time attribution for the DSV4.1 attention forward.

Enabled with ``additional_config.pcp_config.enable_timing``.  Regions are timed
with NPU events; events are only synchronized in batches so the measurement
itself does not
serialize every layer.  Every ``flush`` logs the accumulated per-region share,
which is what separates "prepare (writes + collectives)", "select (indexer)"
and "attention (kernel)" costs between the replicated and owner builds.
"""

from __future__ import annotations

import torch
from vllm.logger import logger

_PENDING: list[tuple[str, torch.npu.Event, torch.npu.Event]] = []
_STATS: dict[str, list[float]] = {}
# 3 regions per layer x 40 layers ~= 4 forward steps between synchronizations.
_FLUSH_EVERY = 512
_logged = 0


def enter(name: str, enabled: bool = False):
    """Start timing ``name`` when the typed config enables timing."""
    if not enabled:
        return None
    event = torch.npu.Event(enable_timing=True)
    event.record()
    return name, event


def exit(token) -> None:
    if token is None:
        return
    name, start = token
    end = torch.npu.Event(enable_timing=True)
    end.record()
    _PENDING.append((name, start, end))
    if len(_PENDING) >= _FLUSH_EVERY:
        _flush()


def _flush() -> None:
    global _logged
    if not _PENDING:
        return
    torch.npu.synchronize()
    for name, start, end in _PENDING:
        entry = _STATS.setdefault(name, [0.0, 0.0])
        entry[0] += float(start.elapsed_time(end))
        entry[1] += 1.0
    _PENDING.clear()
    _logged += 1
    if _logged % 8:
        return
    total = sum(entry[0] for entry in _STATS.values()) or 1.0
    # A region can be absent from the interval that just ended (for example
    # the prefill-only chunk staging during a decode step), and the previous
    # log reset every entry to [0.0, 0.0].  Skip those instead of dividing by
    # a zero sample count.
    parts = [
        f"{name}={entry[0] / entry[1]:.2f}ms/{100.0 * entry[0] / total:.1f}%"
        for name, entry in sorted(_STATS.items(), key=lambda kv: kv[1][0], reverse=True)
        if entry[1] > 0
    ]
    logger.info("PCP timing: %s", " ".join(parts))
    for entry in _STATS.values():
        entry[0] = 0.0
        entry[1] = 0.0
