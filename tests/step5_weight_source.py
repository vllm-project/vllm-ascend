# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-native payloads: never derive the reference through a live loader."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]


def is_vision_weight(name: str) -> bool:
    return name.startswith(
        ("model.vision_model.", "model.vit_large_projector.", "vision_model.", "vit_large_projector.")
    )


class Step5CheckpointSource:
    """Re-iterable source preserving checkpoint names, dtypes and element order.

    Supports a single safetensors file or an indexed sharded checkpoint. Only
    one CPU tensor is materialized at a time; unpacked IPC itself retains the
    full device payload until FINISH, so the fixture must fit on one card.
    """

    def __init__(self, directory, device="cpu"):
        self.directory = Path(directory)
        self.device = device
        index = self.directory / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.directory / "model.safetensors", framework="pt") as handle:
                mapping = dict.fromkeys(handle.keys(), "model.safetensors")
        if not mapping:
            raise ValueError("Empty Step5 checkpoint")
        self.entries = sorted(mapping.items())

    def __iter__(self):
        for name, shard in self.entries:
            with safe_open(self.directory / shard, framework="pt") as handle:
                yield name, handle.get_tensor(name).to(self.device)

    def metadata(self):
        # Inspect headers, not device tensors. Keep exactly the iteration order.
        dtypes = {
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F16": torch.float16,
            "I64": torch.int64,
            "I32": torch.int32,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        result = []
        for name, shard in self.entries:
            with safe_open(self.directory / shard, framework="pt") as handle:
                tensor = handle.get_slice(name)
                result.append(TensorMetadata(name, dtypes[tensor.get_dtype()], tuple(tensor.get_shape())))
        return result
