# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward timing configuration; importing this module never initializes NPU."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardTimeMetricsConfig:
    enabled: bool = False
    window_size: int = 1000
    target_batch_sizes: frozenset[int] = frozenset()
    collect_draft: bool = True

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("forward_time_metrics_config must be an object")
        unknown = set(value) - {"enabled", "window_size", "target_batch_sizes", "collect_draft"}
        if unknown:
            raise ValueError(f"Unknown forward timing fields: {sorted(unknown)}")
        enabled = value.get("enabled", False)
        draft = value.get("collect_draft", True)
        if not isinstance(enabled, bool) or not isinstance(draft, bool):
            raise ValueError("enabled and collect_draft must be JSON booleans")
        window = value.get("window_size", 1000)
        if type(window) is not int or window <= 0:
            raise ValueError("window_size must be a positive integer")
        batches = value.get("target_batch_sizes", [])
        if not isinstance(batches, list) or any(type(bs) is not int or bs <= 0 for bs in batches):
            raise ValueError("target_batch_sizes must be a list of positive integers")
        return cls(enabled, window, frozenset(batches), draft)


def forward_phase(attn_state, has_spec_tokens=False, *, draft=False):
    """Use enum names, not version-specific integer values."""
    name = getattr(attn_state, "name", None)
    if name == "SpecDecoding":
        return "verify" if has_spec_tokens and not draft else "decode"
    return {
        "PrefillNoCache": "prefill",
        "PrefillCacheHit": "prefill",
        "DecodeOnly": "decode",
        "ChunkedPrefill": "mixed",
    }.get(name, "mixed")
