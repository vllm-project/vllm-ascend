from typing import Any

import torch

from .base import AscendLinearScheme
from .registry import register_scheme


@register_scheme("FLOAT", "linear")
class AscendFloatLinearMethod(AscendLinearScheme):
    """Passthrough scheme for unquantized FLOAT linear layers.

    Some layers (e.g. DSpark MTP draft layers) have weights stored in
    float (not quantized) even when the rest of the model uses W8A8.
    This scheme simply stores the weight in the model dtype and applies
    a standard linear projection.
    """

    def __init__(self):
        pass

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        weight = layer.weight
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        return torch.nn.functional.linear(x, weight, bias)