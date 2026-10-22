from __future__ import annotations

import os

from py_monitor.parsing import parse_unit_or_default_to_bytes


def env_scaled_size(name: str, default_value: float, default_unit_bytes: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return int(default_value * default_unit_bytes)
    try:
        return parse_unit_or_default_to_bytes(raw, default_unit_bytes)
    except Exception as exc:
        raise SystemExit(f"Invalid env {name}='{raw}': {exc}")


def env_float(name: str, default_value: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default_value
    try:
        value = float(raw)
    except Exception as exc:
        raise SystemExit(f"Invalid env {name}='{raw}': {exc}")
    if value < 0:
        raise SystemExit(f"Invalid env {name}='{raw}': value must be >= 0")
    return value
