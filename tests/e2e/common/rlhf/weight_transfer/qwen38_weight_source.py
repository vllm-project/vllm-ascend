# SPDX-License-Identifier: Apache-2.0
"""Deterministic BF16 parameters and valid PLE hash buffers for IPC CI."""

import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.qwen38_manifest import ASSET_NAMES, validate_ci_config
from tests.e2e.common.rlhf.weight_transfer.weight_source import FixedRandomWeightSource, expand_fused_expert_params

CHECKPOINT_SHARD_BYTES = 2 * 2**30


class Qwen38RandomWeightSource(FixedRandomWeightSource):
    """Reuse the shared per-name seed; serialize exactly the IPC payload."""

    def __init__(self, case: WeightUpdateModelCase, device: torch.device) -> None:
        model_dir = Path(case.model)
        manifest = json.loads((model_dir / "weight_manifest.json").read_text())
        config = json.loads((model_dir / "config.json").read_text())
        validate_ci_config(config)
        if manifest.get("weight_values") != "deterministic_generated_not_pretrained" or not manifest.get("ple_enabled"):
            raise ValueError("Prepare the four-layer PLE/MRoPE generated-weight descriptor")
        self._case = case
        self._device = device
        self._num_experts = int(config["text_config"]["num_experts"])
        self._parameters = []
        self._buffer_values = {}
        for entry in manifest["tensors"]:
            name, shape = entry["name"], tuple(entry["shape"])
            if entry["dtype"] in ("BF16", "F16", "F32"):
                self._parameters.extend(
                    (expanded_name, expanded_shape, torch.bfloat16)
                    for expanded_name, expanded_shape in expand_fused_expert_params(name, shape)
                )
            elif entry["dtype"] == "I64" and "values" in entry:
                if len(shape) != 1 or len(entry["values"]) != shape[0]:
                    raise ValueError(f"Malformed PLE buffer: {name}")
                self._parameters.append((name, shape, torch.int64))
                self._buffer_values[name] = entry["values"]
            else:
                raise ValueError(f"Unhandled integer/buffer layout: {entry}")
        names = [name for name, _, _ in self._parameters]
        if not names or len(set(names)) != len(names):
            raise ValueError("Generated checkpoint names must be non-empty and unique")

    def _make_tensor(self, name, shape, dtype=torch.bfloat16):
        if name in self._buffer_values:
            return torch.tensor(self._buffer_values[name], dtype=dtype, device=self._device)
        return super()._make_tensor(name, shape, dtype)

    def write_checkpoint(self, directory: str | Path) -> None:
        """Bound CPU serialization by one shard; use the same generator as IPC."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        required_bytes = sum(math.prod(meta.shape) * meta.dtype.itemsize for meta in self.metadata)
        free_bytes = shutil.disk_usage(target).free
        if free_bytes < required_bytes + CHECKPOINT_SHARD_BYTES:
            raise OSError(f"Reference needs {required_bytes / 2**30:.2f} GiB plus shard headroom")
        for name in ("config.json", *ASSET_NAMES):
            asset = Path(self._case.model) / name
            if asset.is_file():
                shutil.copyfile(asset, target / name)
        state, weight_map = {}, {}
        total_bytes = shard_bytes = shard_index = 0

        def flush() -> None:
            nonlocal shard_index, shard_bytes
            if not state:
                return
            shard_index += 1
            filename = f"generated-{shard_index:05d}.safetensors"
            save_file(state, str(target / filename), metadata={"format": "pt"})
            print(f"Generated reference shard: {filename}, {shard_bytes / 2**30:.2f} GiB", flush=True)
            weight_map.update(dict.fromkeys(state, filename))
            state.clear()
            shard_bytes = 0

        for name, tensor in self.iter_full():
            size_bytes = tensor.numel() * tensor.element_size()
            if state and shard_bytes + size_bytes > CHECKPOINT_SHARD_BYTES:
                flush()
            state[name] = tensor.detach().to("cpu").contiguous()
            shard_bytes += size_bytes
            total_bytes += size_bytes
        flush()
        (target / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2) + "\n"
        )
