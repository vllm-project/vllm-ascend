# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Normalize the public physical-K interface without changing legacy defaults.

The scheduler resolves this once at initialization. Worker hot paths only read
the enable flag; they do not rebuild or validate the parameter dictionary.
"""

import math
from typing import Any

_PHYSICAL_FIELDS = (
    ("enabled", "adaptive_draft_k", True),
    ("min_k", "adaptive_draft_k_min", 1),
    ("slack", "adaptive_draft_k_slack", 0),
    ("percentile", "adaptive_draft_k_percentile", 0.5),
)
_HYBRID_FIELDS = (
    ("enabled", "hybrid_policy_enabled", True),
    ("min_batch_size", "hybrid_min_batch_size", 8),
    ("acceptance_threshold", "hybrid_acceptance_threshold", 0.6),
    ("low_steps", "hybrid_low_steps", 4),
    ("high_steps", "hybrid_high_steps", 2),
    ("probe_interval", "hybrid_probe_interval", 32),
)


def _validate_fields(values: dict, fields: tuple, path: str) -> dict[str, Any]:
    result = {}
    for name, legacy_name, default in fields:
        value = values.get(name, default)
        valid = False
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            minimum = 0 if name in ("slack", "probe_interval") else 1
            valid = type(value) is int and value >= minimum
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
        if not valid:
            raise ValueError(f"{path}.{name} has invalid value {value!r}")
        result[legacy_name] = value
    return result


def resolve_method_params(dynamic_config: dict[str, Any]) -> dict[str, Any]:
    """Return scheduler/worker parameters; never mutate the caller's config.

    ``physical_k`` is opt-in. Its defaults enable physical K, V2 variable-width
    support and hybrid together, with zero slack. Absent/None preserves every
    legacy default, including slack=1 and disabled hybrid/V2 switches.
    """
    legacy = dynamic_config.get("method_params", {})
    physical = dynamic_config.get("physical_k")
    if physical is None:
        return dict(legacy) if isinstance(legacy, dict) else {}
    if not isinstance(legacy, dict) or not isinstance(physical, dict):
        raise ValueError("physical_k and method_params must be objects")
    if dynamic_config.get("policy") != "hardware_aware" or dynamic_config.get("method") not in ("dspark", "dflash"):
        raise ValueError("physical_k requires hardware_aware policy and dspark/dflash method")
    unknown = set(physical) - {name for name, _, _ in _PHYSICAL_FIELDS} - {"capture_k", "hybrid"}
    if unknown:
        raise ValueError(f"Unknown physical_k fields: {sorted(unknown)}")
    hybrid = physical.get("hybrid", {})
    if not isinstance(hybrid, dict):
        raise ValueError("physical_k.hybrid must be an object")
    unknown = set(hybrid) - {name for name, _, _ in _HYBRID_FIELDS}
    if unknown:
        raise ValueError(f"Unknown physical_k.hybrid fields: {sorted(unknown)}")
    translated = _validate_fields(physical, _PHYSICAL_FIELDS, "physical_k")
    translated.update(_validate_fields(hybrid, _HYBRID_FIELDS, "physical_k.hybrid"))
    translated["v2_varlen_physical_k"] = translated["adaptive_draft_k"]
    capture = physical.get("capture_k")
    if capture is not None:
        if not isinstance(capture, (list, tuple)) or not capture or any(type(k) is not int or k < 1 for k in capture):
            raise ValueError("physical_k.capture_k must be a non-empty list of positive integers")
        translated["v2_varlen_capture_k"] = list(capture)
    legacy_keys = set(translated) | {"v2_varlen_capture_k", "adaptive_draft_k_v2"}
    conflict = legacy_keys.intersection(legacy)
    if conflict:
        raise ValueError(f"physical_k cannot be combined with legacy method_params fields: {sorted(conflict)}")
    return {**legacy, **translated}


def v2_physical_k_enabled(dynamic_config: dict[str, Any]) -> bool:
    """Constant-size hot-path lookup; full validation happens at config init."""
    physical = dynamic_config.get("physical_k")
    if physical is not None:
        return physical.get("enabled", True)
    params = dynamic_config.get("method_params", {})
    if not isinstance(params, dict):
        return False
    return bool(params.get("v2_varlen_physical_k", params.get("adaptive_draft_k_v2", False)))
