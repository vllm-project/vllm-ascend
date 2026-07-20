from __future__ import annotations

import argparse
from typing import Tuple

from py_monitor.models import PoolSpec
from py_monitor.units import GB, MB


def parse_size_bytes(text: str) -> int:
    """Parse a human-readable size into bytes.

    Supported examples: "1073741824", "2GB", "512MB", "64KB".
    KB/MB/GB are interpreted with 1024-based multipliers.
    """
    value = text.strip()
    if not value:
        raise ValueError("empty size")

    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)

    value_up = value.upper()

    units = [
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("B", 1),
    ]

    for unit, mult in units:
        if value_up.endswith(unit):
            number = value_up[: -len(unit)].strip()
            if not number:
                raise ValueError(f"missing number for unit {unit}")
            return int(float(number) * mult)

    try:
        return int(float(value_up))
    except ValueError as exc:
        raise ValueError(f"unrecognized size suffix in '{text}'") from exc


def parse_unit_or_default_to_bytes(text: str, default_unit_bytes: int) -> int:
    raw = text.strip()
    if not raw:
        raise ValueError("empty value")
    if any(c.isalpha() for c in raw):
        size = parse_size_bytes(raw)
    else:
        size = int(float(raw) * default_unit_bytes)
    if size <= 0:
        raise ValueError("value must be > 0")
    return size


def parse_gb_to_bytes(text: str) -> int:
    return parse_unit_or_default_to_bytes(text, GB)


def parse_mb_to_bytes(text: str) -> int:
    return parse_unit_or_default_to_bytes(text, MB)


def parse_pool_key(text: str) -> Tuple[int, int]:
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError("expected dev:gran format")
    dev = int(parts[0].strip())
    gran_mb = int(parts[1].strip())
    if dev < 0:
        raise ValueError("dev must be >= 0")
    if gran_mb <= 0:
        raise ValueError("gran must be > 0")
    return dev, gran_mb * MB


def parse_pool_spec(text: str) -> PoolSpec:
    """Parse: device_id:granularity_mb:total_gb[:cap_gb]"""
    parts = text.split(":")
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            f"Invalid --pool '{text}'. Expected format device:granularity_mb:total_gb[:cap_gb] (e.g. 0:16:4 or 0:16:4:10)."
        )
    try:
        device_id = int(parts[0])
        gran_mb = int(parts[1])
        total_bytes = parse_gb_to_bytes(parts[2])
        cap_bytes = parse_gb_to_bytes(parts[3]) if len(parts) == 4 else 0
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --pool '{text}': {exc}") from exc
    if device_id < 0:
        raise argparse.ArgumentTypeError("device_id must be >= 0")
    if gran_mb <= 0:
        raise argparse.ArgumentTypeError("granularity_mb must be > 0")
    if total_bytes <= 0:
        raise argparse.ArgumentTypeError("total_gb must be > 0")
    if len(parts) == 4 and cap_bytes <= 0:
        raise argparse.ArgumentTypeError("cap_gb must be > 0 when provided")
    if len(parts) == 4 and cap_bytes < total_bytes:
        raise argparse.ArgumentTypeError("cap_gb must be >= total_gb")

    granularity_bytes = gran_mb * MB
    return PoolSpec(
        device_id=device_id,
        granularity_bytes=granularity_bytes,
        total_bytes=total_bytes,
        cap_bytes=cap_bytes,
    )
