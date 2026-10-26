# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local level for Ascend-owned MS Service Metrics.

One integer per process. Runtime config calls :func:`set_metric_level` when
``metrics.level`` is applied. Handlers read it with :func:`get_metric_level`.
"""

from typing import Any

METRIC_LEVEL_OFF = 0
METRIC_LEVEL_BASIC = 1
METRIC_LEVEL_DETAIL = 2
METRIC_LEVEL_DEBUG = 3

_LEVEL_NAMES = ("off", "basic", "detail", "debug")
_LEVEL_BY_NAME = {name: index for index, name in enumerate(_LEVEL_NAMES)}

# None until the first read or an explicit set. Not shared across processes.
_metric_level: int | None = None


def parse_metric_level(value: Any) -> int:
    """Return ``0``–``3`` for a level name or integer. Reject anything else."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"invalid metric level: {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"invalid metric level: {value!r}")
        value = int(value)
    if isinstance(value, int):
        if METRIC_LEVEL_OFF <= value <= METRIC_LEVEL_DEBUG:
            return value
        raise ValueError(f"invalid metric level: {value!r}")
    if isinstance(value, str):
        stripped = value.strip().lower()
        named = _LEVEL_BY_NAME.get(stripped)
        if named is not None:
            return named
        if stripped.isdigit():
            parsed = int(stripped)
            if METRIC_LEVEL_OFF <= parsed <= METRIC_LEVEL_DEBUG:
                return parsed
    raise ValueError(f"invalid metric level: {value!r}")


def canonical_metric_level(value: Any) -> str:
    """Normalize a level to ``off``, ``basic``, ``detail``, or ``debug``."""
    return _LEVEL_NAMES[parse_metric_level(value)]


def get_metric_level() -> int:
    """Return the active level for this process. Default is basic."""
    global _metric_level
    if _metric_level is None:
        _metric_level = METRIC_LEVEL_BASIC
    return _metric_level


def set_metric_level(level: int | str) -> None:
    """Set the active level for this process.

    Later registrations and samples in this process observe the new value.
    Other processes keep their own level.
    """
    global _metric_level
    _metric_level = parse_metric_level(level)
