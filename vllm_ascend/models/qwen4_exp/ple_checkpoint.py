# SPDX-License-Identifier: Apache-2.0
"""Read contiguous global PLE rows from a sharded safetensors checkpoint."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


def checkpoint_identity(model_path: str | Path, prefix: str) -> str:
    """Identify immutable checkpoint metadata and the selected PLE table."""
    root = Path(model_path)
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    digest.update(prefix.encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class PLECheckpointShard:
    path: Path
    name: str
    start: int
    end: int


class PLECheckpoint:
    """A BF16 PLE table in checkpoint row order, without materializing the table."""

    def __init__(self, model_path: str | Path, prefix: str, parts: int, width: int):
        if parts <= 0 or width <= 0:
            raise ValueError("PLE checkpoint parts and width must be positive")
        root = Path(model_path)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        pattern = re.compile(re.escape(prefix) + r"\.shard_(\d+)\.weight\Z")
        indexed = {}
        for name, filename in index.items():
            match = pattern.fullmatch(name)
            if match:
                number = int(match.group(1))
                if number in indexed:
                    raise ValueError(f"Duplicate PLE shard index: {number}")
                indexed[number] = (name, filename)
        if set(indexed) != set(range(parts)):
            raise ValueError("PLE checkpoint shard indexes do not match split_ngram_parts")

        shards = []
        start = 0
        shard_rows = None
        for number in range(parts):
            name, filename = indexed[number]
            path = root / filename
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor = handle.get_slice(name)
                shape = tensor.get_shape()
                if len(shape) != 2 or shape[0] <= 0 or shape[1] != width or tensor.get_dtype() != "BF16":
                    raise ValueError(f"Unsupported PLE shard layout for {name}: {shape}, {tensor.get_dtype()}")
                if shard_rows is None:
                    shard_rows = shape[0]
                elif shape[0] != shard_rows and (number != parts - 1 or shape[0] > shard_rows):
                    raise ValueError(f"Inconsistent PLE shard row count for {name}: {shape[0]}")
                end = start + shape[0]
                shards.append(PLECheckpointShard(path, name, start, end))
                start = end
        self.shards = tuple(shards)
        self.weight_names = frozenset(shard.name for shard in shards)
        self.rows = start
        self.width = width

    def copy_rows(self, destination: torch.Tensor, start: int) -> None:
        """Copy a global contiguous range into an existing CPU BF16 allocation."""
        if destination.device.type != "cpu" or destination.dtype != torch.bfloat16:
            raise ValueError("PLE destination must be a CPU BF16 tensor")
        if destination.ndim != 2 or destination.shape[1] != self.width:
            raise ValueError("PLE destination width does not match checkpoint")
        end = start + destination.shape[0]
        if start < 0 or end > self.rows:
            raise ValueError("PLE row range exceeds checkpoint")
        for shard in self.shards:
            left, right = max(start, shard.start), min(end, shard.end)
            if left < right:
                with safe_open(shard.path, framework="pt", device="cpu") as handle:
                    source = handle.get_slice(shard.name)[left - shard.start : right - shard.start]
                    destination[left - start : right - start].copy_(source)
