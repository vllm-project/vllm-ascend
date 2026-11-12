"""Helpers for benchmark reports and profiling artifacts."""

import re


def safe_name(name: str) -> str:
    """Convert a benchmark or job identity into a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "unknown"
