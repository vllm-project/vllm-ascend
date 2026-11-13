# SPDX-License-Identifier: Apache-2.0
"""Sampled layer/selector-head routes, frozen before graph capture."""

import json
import math
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import regex as re


@lru_cache(maxsize=8)
def _load_policy(path: str, num_selector_heads: int) -> Mapping[str, float]:
    policy = json.loads(Path(path).read_bytes())
    if (
        not isinstance(policy, dict)
        or policy.get("valid") is not True
        or policy.get("routing") != "frozen-layer-selector-head"
        or policy.get("per_token_dynamic") is not False
        or policy.get("selector_modified") is not False
    ):
        raise ValueError("Union policy must contain sampled, frozen layer/selector-head D8 values")
    decisions = policy.get("decisions")
    if not isinstance(decisions, dict) or not decisions:
        raise ValueError("Union policy decisions must be nonempty")
    values = {}
    layers = set()
    for key, row in decisions.items():
        match = re.fullmatch(r"(0|[1-9]\d*):(0|[1-9]\d*)", key)
        d8 = row.get("D8") if isinstance(row, dict) else None
        if (
            match is None
            or int(match[2]) >= num_selector_heads
            or isinstance(d8, bool)
            or not isinstance(d8, (int, float))
            or not math.isfinite(d8)
            or not 0 <= d8 <= 1
        ):
            raise ValueError(f"Invalid Union calibration entry: {key}")
        layers.add(int(match[1]))
        values[key] = float(d8)
    if set(values) != {f"{layer}:{head}" for layer in layers for head in range(num_selector_heads)}:
        raise ValueError("Union policy must cover every selector head of each sampled sparse layer")
    # One immutable snapshot per worker/topology. Forward never reads this file
    # or computes D8; old precomputed 'path' fields are deliberately ignored.
    return MappingProxyType(values)


def use_union_for_layer(
    threshold_pct: float,
    policy_path: str | None,
    layer_name: str,
    tp_rank: int,
    tp_size: int,
    num_selector_heads: int,
) -> bool:
    if threshold_pct == 100:
        return True
    if not math.isfinite(threshold_pct) or not 0 <= threshold_pct <= 100:
        raise ValueError("minimax_m3_union_d8_threshold_pct must be in [0, 100]")
    if not policy_path:
        raise ValueError("A D8 threshold below 100 requires minimax_m3_union_policy_path")
    if num_selector_heads <= 0 or tp_size < num_selector_heads or tp_size % num_selector_heads:
        raise ValueError("Union policy requires one local selector head, possibly replicated across TP ranks")
    if not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid Union policy TP rank")
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        raise ValueError(f"Unrecognized MiniMax-M3 sparse layer: {layer_name}")
    selector_head = tp_rank // (tp_size // num_selector_heads)
    key = f"{int(match[1])}:{selector_head}"
    values = _load_policy(policy_path, num_selector_heads)
    if key not in values:
        raise ValueError(f"Union policy is missing sparse layer/selector head {key}")
    return values[key] * 100 <= threshold_pct
