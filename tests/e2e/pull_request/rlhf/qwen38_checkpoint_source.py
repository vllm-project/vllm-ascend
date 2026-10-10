# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real checkpoint source; preserve fused expert names, shapes and values."""

from pathlib import Path

import torch
from safetensors import safe_open
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

from tests.e2e.pull_request.rlhf.prepare_qwen38_checkpoint import PROFILE, validate_checkpoint


class Qwen38CheckpointSource(WeightSource):
    def __init__(self, checkpoint: Path, manifest_sha256: str, device: torch.device) -> None:
        self.checkpoint_directory = checkpoint.resolve()
        self.manifest = validate_checkpoint(self.checkpoint_directory, manifest_sha256)
        self._device = device
        self._names = tuple(sorted(self.manifest["tensors"]))

    @property
    def case_id(self) -> str:
        return PROFILE

    def metadata(self) -> list[ParamMeta]:
        return [ParamMeta(n, torch.bfloat16, tuple(self.manifest["tensors"][n]["shape"])) for n in self._names]

    def __iter__(self):
        with torch.no_grad():
            for name in self._names:
                entry = self.manifest["tensors"][name]
                with safe_open(str(self.checkpoint_directory / entry["file"]), framework="pt", device="cpu") as shard:
                    tensor = shard.get_tensor(name)
                    yield name, tensor.to(self._device)
                del tensor
