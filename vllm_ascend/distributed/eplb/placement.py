# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Portable committed EPLB deployments; deliberately independent of torch/NPU.

Slot ``rank * slots_per_rank + local_slot`` stores the logical expert given
by ``physical_to_logical[layer][slot]``. Derived routing tables are not saved.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLACEMENT_SCHEMA_VERSION = 1
_METADATA_FIELDS = frozenset(
    {
        "model",
        "revision",
        "architecture",
        "hf_config_sha256",
        "quantization",
        "dtype",
        "ep_size",
        "tp_size",
        "dp_size",
        "pp_size",
        "num_logical_experts",
        "num_redundant_experts",
        "layer_names",
    }
)
_PARALLEL_FIELDS = ("ep_size", "tp_size", "dp_size", "pp_size")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict) or metadata.keys() != _METADATA_FIELDS:
        raise ValueError(f"EPLB placement metadata must contain exactly {sorted(_METADATA_FIELDS)}")
    for name in ("model", "architecture", "hf_config_sha256", "dtype"):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ValueError(f"EPLB placement {name} must be a non-empty string")
    for name in ("revision", "quantization"):
        if metadata[name] is not None and not isinstance(metadata[name], str):
            raise ValueError(f"EPLB placement {name} must be a string or null")
    for name in _PARALLEL_FIELDS:
        _integer(metadata[name], name)
    if metadata["pp_size"] != 1:
        raise ValueError("EPLB deployment files currently require pipeline_parallel_size=1")
    _integer(metadata["num_logical_experts"], "num_logical_experts")
    _integer(metadata["num_redundant_experts"], "num_redundant_experts", 0)
    names = metadata["layer_names"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("EPLB placement layer_names must be unique non-empty strings")


@dataclass(frozen=True)
class EplbPlacement:
    metadata: dict[str, Any]
    physical_to_logical: tuple[tuple[int, ...], ...]
    provenance: dict[str, Any]

    def validate(self, expected_metadata: dict[str, Any] | None = None) -> None:
        validate_metadata(self.metadata)
        if expected_metadata is not None:
            validate_metadata(expected_metadata)
            for name in sorted(_METADATA_FIELDS):
                if self.metadata[name] != expected_metadata[name]:
                    raise ValueError(
                        f"EPLB placement {name} mismatch: file={self.metadata[name]!r}, "
                        f"runtime={expected_metadata[name]!r}"
                    )
        experts = self.metadata["num_logical_experts"]
        physical_count = experts + self.metadata["num_redundant_experts"]
        ep_size = self.metadata["ep_size"]
        if physical_count % ep_size:
            raise ValueError("EPLB placement physical expert count must be divisible by EP size")
        if len(self.physical_to_logical) != len(self.metadata["layer_names"]):
            raise ValueError("EPLB placement MoE layer count does not match layer_names")
        slots = physical_count // ep_size
        for layer_idx, layer in enumerate(self.physical_to_logical):
            if len(layer) != physical_count:
                raise ValueError(f"EPLB placement layer {layer_idx} has an invalid physical slot count")
            if any(type(expert) is not int or not 0 <= expert < experts for expert in layer):
                raise ValueError(f"EPLB placement layer {layer_idx} contains an invalid logical expert ID")
            if set(layer) != set(range(experts)):
                raise ValueError(f"EPLB placement layer {layer_idx} does not cover every logical expert")
        if not isinstance(self.provenance, dict):
            raise ValueError("EPLB placement provenance must be an object")
        _canonical_json(self.provenance)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": PLACEMENT_SCHEMA_VERSION,
            "metadata": self.metadata,
            "physical_to_logical": [list(layer) for layer in self.physical_to_logical],
            "provenance": self.provenance,
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Any) -> "EplbPlacement":
        fields = {"schema_version", "metadata", "physical_to_logical", "provenance"}
        if not isinstance(data, dict) or data.keys() != fields:
            raise ValueError(f"EPLB placement must contain exactly {sorted(fields)}")
        if type(data["schema_version"]) is not int or data["schema_version"] != PLACEMENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported EPLB placement schema_version: {data['schema_version']!r}")
        mapping = data["physical_to_logical"]
        if not isinstance(mapping, list) or any(not isinstance(layer, list) for layer in mapping):
            raise ValueError("EPLB placement physical_to_logical must be a list of layer lists")
        placement = cls(data["metadata"], tuple(tuple(layer) for layer in mapping), data["provenance"])
        placement.validate()
        return placement


def load_placement(path: str | os.PathLike[str]) -> EplbPlacement:
    with open(path, encoding="utf-8") as source:
        return EplbPlacement.from_dict(json.load(source))


def save_placement(path: str | os.PathLike[str], placement: EplbPlacement) -> None:
    """Publish a complete file; a failed write never replaces an existing one."""
    data = placement.to_dict()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as target:
            temporary_path = target.name
            json.dump(data, target, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def build_placement_metadata(model: Any, model_config: Any, parallel_config: Any, ep_size: int) -> dict[str, Any]:
    """Use module paths and model structure, excluding machine-local HF paths."""
    module_names = {id(module): name for name, module in model.named_modules()}
    names = [module_names[id(layer)] for layer in model.moe_layers]
    hf_config = dict(model_config.hf_config.to_dict())
    for name in ("_name_or_path", "transformers_version"):
        hf_config.pop(name, None)
    metadata = {
        "model": str(model_config.model),
        "revision": model_config.revision,
        "architecture": str(model_config.architecture),
        "hf_config_sha256": hashlib.sha256(_canonical_json(hf_config).encode("utf-8")).hexdigest(),
        "quantization": model_config.quantization,
        "dtype": str(model_config.dtype),
        "ep_size": ep_size,
        "tp_size": parallel_config.tensor_parallel_size,
        "dp_size": parallel_config.data_parallel_size,
        "pp_size": parallel_config.pipeline_parallel_size,
        "num_logical_experts": model.num_logical_experts,
        "num_redundant_experts": model.num_redundant_experts,
        "layer_names": names,
    }
    validate_metadata(metadata)
    return metadata


def plan_placement_sources(
    old: tuple[int, ...],
    new: tuple[int, ...],
    ep_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Choose one exact old source, preferring unchanged and rank-local slots."""
    if not old or len(old) != len(new) or ep_size < 1 or len(old) % ep_size:
        raise ValueError("EPLB source planning requires equally sized dense EP placements")
    slots = len(old) // ep_size
    owners: dict[int, list[int]] = {}
    for physical, expert in enumerate(old):
        owners.setdefault(expert, []).append(physical)
    source_ranks, source_slots = [], []
    for rank in range(ep_size):
        ranks, offsets = [], []
        for slot in range(slots):
            physical = rank * slots + slot
            candidates = owners.get(new[physical], [])
            if not candidates:
                raise ValueError(f"No source owns logical expert {new[physical]}")
            source = min(candidates, key=lambda index: (index != physical, index // slots != rank, index))
            ranks.append(source // slots)
            offsets.append(source % slots)
        source_ranks.append(ranks)
        source_slots.append(offsets)
    return source_ranks, source_slots
