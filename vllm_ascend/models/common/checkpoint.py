# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint inspection helpers that read names without loading tensor data."""

import json
from collections.abc import Iterable
from pathlib import Path

from safetensors import safe_open


def checkpoint_contains_any_weight(model_path: Path | str, weight_names: Iterable[str]) -> bool:
    """Whether a local safetensors checkpoint ships any of ``weight_names``.

    Reads shard indexes when present and otherwise the shard headers, so no
    tensor data is loaded. Every name is matched against one pass over the
    checkpoint rather than re-reading it per name.

    Raises:
        ValueError: the path is not a local safetensors checkpoint, so presence
            cannot be decided. Callers use this to reproduce what weight loading
            observed; a caller that cannot see the checkpoint must say so rather
            than report a missing weight.
        OSError: a checkpoint file exists but cannot be read.
    """
    wanted = set(weight_names)
    if not wanted:
        raise ValueError("weight_names must not be empty.")

    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(f"{str(model_path)!r} is not a local directory; cannot inspect it for {sorted(wanted)}.")

    index_paths = sorted(root.glob("*.safetensors.index.json"))
    for index_path in index_paths:
        # Quantized checkpoints ship their own index alongside the base one, so check every index.
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        if not wanted.isdisjoint(weight_map):
            return True

    shard_paths = sorted(root.glob("*.safetensors"))
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            # safe_open exposes names only through keys(); it is not a container itself.
            shard_keys = set(shard.keys())
        if not wanted.isdisjoint(shard_keys):
            return True

    if not index_paths and not shard_paths:
        raise ValueError(f"No safetensors checkpoint under {root}; cannot decide whether it ships {sorted(wanted)}.")
    return False


def checkpoint_contains_weight(model_path: Path | str, weight_name: str) -> bool:
    """Whether a local safetensors checkpoint ships ``weight_name``.

    Raises the same way as :func:`checkpoint_contains_any_weight`.
    """
    return checkpoint_contains_any_weight(model_path, (weight_name,))
