# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Image metadata for decode requests that reuse prefill KV."""

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.request import Request


def get_image_grids(request: "Request") -> list[dict[str, Any]] | None:
    """Read CPU grids restored by the engine's multimodal receiver cache.

    Shared-memory handles and missing metadata fall back to original images.
    Never copy device tensors to CPU in the scheduler's completion path.
    """
    grids = []
    for feature in request.mm_features:
        if feature.modality != "image" or feature.data is None:
            return None
        field = feature.data.get("image_grid_thw")
        tensor = field.data if field is not None else None
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.numel() != 3:
            return None
        grid = tensor.reshape(-1).tolist()
        if any(type(value) is not int or value <= 0 for value in grid):
            return None
        image_hash = feature.mm_hash or feature.identifier
        if not isinstance(image_hash, str) or not image_hash:
            return None
        grids.append({"hash": image_hash, "grid": grid})
    return grids or None
