# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline checkpoint metadata and bounded synthetic GLM-5.3-Flash weights.

The shapes come from published safetensors headers, not a served-model parameter
enumeration. This keeps the startup-load oracle independent of runtime fusion.
No checkpoint tensor data or Transformers remote model code is needed.
"""

import json
from pathlib import Path

import torch

ASSETS = Path(__file__).with_name("glm53flash_assets")


def load_parameter_metadata(path: Path) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    names = [entry["name"] for entry in entries]
    if not entries or len(names) != len(set(names)):
        raise ValueError("GLM-5.3-Flash checkpoint metadata must be nonempty with unique names")
    parameters = []
    for entry in entries:
        shape = tuple(entry["shape"])
        if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(f"Invalid checkpoint shape for {entry['name']}: {shape}")
        parameters.append((entry["name"], shape, torch.bfloat16))
    return parameters


def initialize_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Adjust the source's seeded random tensor for KDA/mHC numerical stability.

    Both startup loading and live updates use these exact tensors. These values
    exercise the real modules but make no statement about checkpoint accuracy.
    """
    if name.endswith("A_log"):
        return tensor.zero_()
    if name.endswith("dt_bias"):
        return tensor.fill_(-2.0)
    if "hc_" in name:
        if name.endswith("scale"):
            return tensor.fill_(1.0)
        if name.endswith("base"):
            return tensor.zero_()
        return tensor.mul_(0.05)
    if "norm" in name and name.endswith("weight"):
        return tensor.fill_(1.0)
    if name.endswith("bias"):
        return tensor.zero_()
    return tensor.mul_(0.5)
