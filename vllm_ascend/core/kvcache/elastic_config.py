from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(min_value, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ElasticPolicyConfig:
    min_free_pages: int = 3
    grow_step_pages: int = 1
    shrink_step_pages: int = 1
    max_free_pages: int = 5
    control_interval_ms: int = 100
    auto_grow: bool = True
    auto_shrink: bool = True

    @staticmethod
    def from_env() -> "ElasticPolicyConfig":
        min_free_pages = _env_int("KVCACHE_MIN_FREE_PAGES", 3, min_value=0)
        grow_step_pages = _env_int("KVCACHE_GROW_STEP_PAGES", 1, min_value=1)
        shrink_step_pages = _env_int("KVCACHE_SHRINK_STEP_PAGES", 1, min_value=1)

        default_max_free = min_free_pages + max(grow_step_pages, shrink_step_pages) * 2
        max_free_pages = _env_int("KVCACHE_MAX_FREE_PAGES", default_max_free, min_value=min_free_pages)

        control_interval_ms = _env_int("KVCACHE_CONTROL_INTERVAL_MS", 100, min_value=10)
        auto_grow = _env_bool("KVCACHE_AUTO_GROW", True)
        auto_shrink = _env_bool("KVCACHE_AUTO_SHRINK", True)

        return ElasticPolicyConfig(
            min_free_pages=min_free_pages,
            grow_step_pages=grow_step_pages,
            shrink_step_pages=shrink_step_pages,
            max_free_pages=max_free_pages,
            control_interval_ms=control_interval_ms,
            auto_grow=auto_grow,
            auto_shrink=auto_shrink,
        )
