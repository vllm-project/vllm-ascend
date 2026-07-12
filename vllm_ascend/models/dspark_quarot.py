from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def resolve_quarot_rotation_path(model_path: str | Path) -> Path | None:
    model_root = Path(model_path)
    description_path = model_root / "quant_model_description.json"
    if not description_path.is_file():
        return None

    with description_path.open(encoding="utf-8") as description_file:
        description = json.load(description_file)

    if not description.get("is_rot_used", False):
        return None

    try:
        relative_path = description["optional"]["quarot"]["rotation_map"][
            "global_rotation"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"QuaRot is enabled but no global rotation is configured in {description_path}"
        ) from exc

    rotation_path = model_root / relative_path
    if not rotation_path.is_file():
        raise FileNotFoundError(
            f"QuaRot global rotation does not exist: {rotation_path}"
        )
    return rotation_path


def load_quarot_rotation(rotation_path: str | Path) -> torch.Tensor:
    state = load_file(str(rotation_path), device="cpu")
    try:
        rotation = state["global_rotation"]
    except KeyError as exc:
        raise KeyError(
            f"QuaRot file {rotation_path} does not contain 'global_rotation'"
        ) from exc
    return rotation


@torch.inference_mode()
def transform_fc_weight_for_quarot(
    fc_weight: torch.Tensor,
    rotation: torch.Tensor,
    *,
    target_device: torch.device | str | None = None,
) -> torch.Tensor:
    if fc_weight.ndim != 2:
        raise ValueError(f"Expected a 2-D fc weight, got shape={tuple(fc_weight.shape)}")
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(
            f"Expected a square rotation matrix, got shape={tuple(rotation.shape)}"
        )

    hidden_size = rotation.shape[0]
    if fc_weight.shape[1] % hidden_size != 0:
        raise ValueError(
            "FC input width must be a multiple of the QuaRot hidden size: "
            f"fc={tuple(fc_weight.shape)}, rotation={tuple(rotation.shape)}"
        )

    device = (
        torch.device(target_device) if target_device is not None else fc_weight.device
    )
    transformed = torch.empty(
        fc_weight.shape,
        dtype=fc_weight.dtype,
        device=device,
    )
    rotation_device = rotation.to(device=device, dtype=torch.float32)

    for start in range(0, fc_weight.shape[1], hidden_size):
        weight_chunk = fc_weight[:, start : start + hidden_size].to(
            device=device,
            dtype=torch.float32,
        )
        transformed_chunk = torch.matmul(weight_chunk, rotation_device)
        transformed[:, start : start + hidden_size].copy_(
            transformed_chunk.to(dtype=fc_weight.dtype)
        )

    return transformed
