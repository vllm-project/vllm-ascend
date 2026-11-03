# SPDX-License-Identifier: Apache-2.0
"""Extend the engine's prefix identity only for declared state dependencies.

This hashes small input metadata, never KV payload bytes or a replayed prefix.
The same policy is used with committed source tokens and the reader's prompt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class StateKeyPolicy:
    name: str
    extra_input_tokens: int = 0

    def __post_init__(self):
        if not self.name or type(self.extra_input_tokens) is not int or self.extra_input_tokens < 0:
            raise ValueError("state key policy needs a name and nonnegative dependency length")

    def bind(self, prefix_hash: bytes, boundary: int, tokens: Sequence[int]) -> bytes | None:
        """Return None when the model's additional committed inputs are absent."""
        if not isinstance(prefix_hash, bytes) or not prefix_hash or boundary <= 0:
            raise ValueError("state keys require a native prefix hash and positive boundary")
        if boundary > len(tokens):
            return None
        if not self.extra_input_tokens:
            return prefix_hash
        end = boundary + self.extra_input_tokens
        if end > len(tokens):
            return None
        dependency = tuple(tokens[boundary:end])
        if any(type(token) is not int or token < 0 for token in dependency):
            raise ValueError("state dependency must contain actual committed token IDs")
        value = (self.name, prefix_hash.hex(), dependency)
        return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).digest()
