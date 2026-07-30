"""DSA 稀疏卸载的拉起期 trace 配置与热路径只读查询。

本文件定义 ``dsa_sparse_config.trace_points`` 的解析和查询逻辑。当前只保留
图外首 token 边界探针；已退役数据面的 layer-wise tensor 采样和显式 stream
sync 不再属于公共配置。配置在拉起时解析为不可变结构，推理路径只做只读
过滤。DSA 开启且省略 trace 配置时默认记录 TP rank 0 的首次采样边界。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

DSA_TRACE_CONFIG_KEY = "dsa_sparse_trace_points"
DSA_TRACE_PUBLIC_CONTAINER_KEY = "dsa_sparse_config"
DSA_TRACE_PUBLIC_CONFIG_KEY = "trace_points"
DSA_TRACE_POINT_FIRST_SAMPLE = "first_sample"
DSA_TRACE_PUBLIC_KEYS = frozenset({"enabled", "points", "ranks"})
DSA_TRACE_ALL_POINTS = frozenset({
    DSA_TRACE_POINT_FIRST_SAMPLE,
})
DSA_TRACE_DEFAULT_POINTS = (DSA_TRACE_POINT_FIRST_SAMPLE,)
DSA_TRACE_DEFAULT_RANKS = (0,)


@dataclass(frozen=True)
class DSATraceConfig:
    enabled: bool = False
    points: frozenset[str] = frozenset()
    ranks: frozenset[int] | None = None


_DSA_TRACE_CONFIG = DSATraceConfig()


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _parse_csv_or_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _parse_int_filter(value: Any) -> frozenset[int] | None:
    if value is None or value in ("*", "all"):
        return None
    parsed: set[int] = set()
    for item in _parse_csv_or_iterable(value):
        if isinstance(item, str) and item.strip().lower() in ("*", "all"):
            return None
        parsed.add(int(item))
    return frozenset(parsed)


def _parse_points(value: Any) -> frozenset[str]:
    if value is None or value in ("*", "all"):
        return DSA_TRACE_ALL_POINTS
    items = [str(item).strip() for item in _parse_csv_or_iterable(value)]
    if any(item.lower() in ("*", "all") for item in items):
        return DSA_TRACE_ALL_POINTS
    points = {item for item in items if item}
    if not points:
        return frozenset()
    unknown = sorted(points - DSA_TRACE_ALL_POINTS)
    if unknown:
        raise ValueError(
            f"Unknown DSA trace point(s): {unknown}. Supported points: "
            f"{sorted(DSA_TRACE_ALL_POINTS)}")
    return frozenset(points)


def configure_dsa_trace(trace_config: Any) -> DSATraceConfig:
    """Parse public DSA trace config once at model-runner initialization."""
    global _DSA_TRACE_CONFIG

    if trace_config is None:
        _DSA_TRACE_CONFIG = DSATraceConfig()
        return _DSA_TRACE_CONFIG

    if isinstance(trace_config, bool):
        config = {"enabled": trace_config}
    elif isinstance(trace_config, dict):
        config = dict(trace_config)
    else:
        raise TypeError(
            "DSA trace config must be a dict or bool, got "
            f"{type(trace_config)!r}")

    unknown = sorted(set(config) - DSA_TRACE_PUBLIC_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown DSA trace config key(s): {unknown}. Supported keys: "
            f"{sorted(DSA_TRACE_PUBLIC_KEYS)}")

    enabled = _as_bool(config.get("enabled"), default=False)
    if not enabled:
        _DSA_TRACE_CONFIG = DSATraceConfig()
        return _DSA_TRACE_CONFIG

    _DSA_TRACE_CONFIG = DSATraceConfig(
        enabled=True,
        points=_parse_points(config.get("points")),
        ranks=_parse_int_filter(config.get("ranks")),
    )
    # Trace is an explicit debugging mode. Use WARNING so the activation
    # message remains visible with the default/INFO worker log level.
    logger.warning("Configured DSA trace points: %s", _DSA_TRACE_CONFIG)
    return _DSA_TRACE_CONFIG


def configure_dsa_trace_from_additional_config(
    additional_config: Any,
) -> DSATraceConfig:
    """Resolve and configure trace directly from the public DSA config.

    ``attach_dsa_sparse_cache_attrs`` normally mirrors
    ``dsa_sparse_config.trace_points`` to ``DSA_TRACE_CONFIG_KEY``. Model
    runner diagnostics must not depend on that patch having run first, so this
    boundary also accepts the public nested form. The already-normalized
    flattened form takes precedence when both are present.
    """
    if not isinstance(additional_config, dict):
        return configure_dsa_trace(None)

    flattened = additional_config.get(DSA_TRACE_CONFIG_KEY)
    nested = None
    dsa_config = additional_config.get(DSA_TRACE_PUBLIC_CONTAINER_KEY)
    if isinstance(dsa_config, dict):
        nested = dsa_config.get(DSA_TRACE_PUBLIC_CONFIG_KEY)
        if flattened is None and nested is None:
            nested = {
                "enabled": _as_bool(dsa_config.get("enabled"), default=False),
                "points": list(DSA_TRACE_DEFAULT_POINTS),
                "ranks": list(DSA_TRACE_DEFAULT_RANKS),
            }

    return configure_dsa_trace(
        flattened if flattened is not None else nested)


def dsa_trace_enabled(
    point: str,
    *,
    tp_rank: int | None = None,
) -> bool:
    config = _DSA_TRACE_CONFIG
    if not config.enabled or point not in config.points:
        return False
    if config.ranks is not None:
        if tp_rank is None:
            raise RuntimeError(
                "DSA trace rank filtering requires an explicit TP rank. "
                "Resolve get_tp_group().rank_in_group after distributed "
                "initialization instead of silently disabling trace output."
            )
        if int(tp_rank) not in config.ranks:
            return False
    return True
