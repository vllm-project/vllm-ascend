# SPDX-License-Identifier: Apache-2.0
"""Immutable, opt-in MiniMax M3 prefill routing, resolved before graph capture."""

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

UNION_SIZE = 8
UNION_PAGE_SIZE = 128
UNION_MAX_Q_HEADS = 16


@lru_cache(maxsize=8)
def _load_policy(path: str, sha256: str, num_layers: int, num_selector_heads: int) -> frozenset[str]:
    # One immutable snapshot per file/hash/topology in a worker. No device data
    # or mutable routing state is cached, and forward never reads a file.
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("MiniMax M3 Union policy SHA256 mismatch")
    policy = json.loads(content)
    if (
        not isinstance(policy, dict)
        or policy.get("valid") is not True
        or policy.get("routing") != "frozen-layer-selector-head"
        or policy.get("per_token_dynamic") is not False
        or policy.get("selector_modified") is not False
    ):
        raise ValueError("Union policy must use frozen layer/selector-head routing without modifying the selector")
    decisions = policy.get("decisions")
    expected = {f"{layer}:{head}" for layer in range(num_layers) for head in range(num_selector_heads)}
    if not isinstance(decisions, dict) or set(decisions) != expected:
        raise ValueError("Union policy must cover every model layer and selector head exactly once")
    if any(not isinstance(row, dict) or row.get("path") not in ("union", "q-gather-kv") for row in decisions.values()):
        raise ValueError("Union policy paths must be 'union' or 'q-gather-kv'")
    return frozenset(key for key, row in decisions.items() if row["path"] == "union")


@dataclass(frozen=True)
class UnionPrefillConfig:
    mode: str = "off"
    policy_path: str | None = None
    policy_sha256: str | None = None

    @classmethod
    def from_dict(cls, config: dict | None) -> "UnionPrefillConfig":
        if config is None:
            return cls()
        if not isinstance(config, dict) or set(config) - {"mode", "policy_path", "policy_sha256"}:
            raise ValueError("Invalid minimax_m3_union configuration")
        mode = config.get("mode", "off")
        if mode not in ("off", "union", "policy"):
            raise ValueError("minimax_m3_union.mode must be off, union or policy")
        path, sha256 = config.get("policy_path"), config.get("policy_sha256")
        if mode == "policy":
            if not isinstance(path, str) or not path:
                raise ValueError("Union policy mode requires policy_path")
            if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
                raise ValueError("Union policy mode requires a 64-digit policy_sha256")
            sha256 = sha256.lower()
        elif path is not None or sha256 is not None:
            raise ValueError("policy_path and policy_sha256 require policy mode")
        return cls(mode, path, sha256)

    def use_union(self, layer_name: str, selector_head: int, num_layers: int, num_selector_heads: int) -> bool:
        if self.mode != "policy":
            return self.mode == "union"
        if num_layers <= 0 or not 0 <= selector_head < num_selector_heads:
            raise ValueError("Invalid Union policy topology")
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match is None or not 0 <= int(match[1]) < num_layers:
            raise ValueError(f"Unrecognized MiniMax M3 sparse layer: {layer_name}")
        assert self.policy_path is not None and self.policy_sha256 is not None
        routes = _load_policy(self.policy_path, self.policy_sha256, num_layers, num_selector_heads)
        return f"{int(match[1])}:{selector_head}" in routes
