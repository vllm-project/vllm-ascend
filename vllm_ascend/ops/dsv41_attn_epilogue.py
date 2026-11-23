# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared weight layouts for the local V4.1 output projection."""

from functools import cache
from importlib import import_module


@cache
def get_attn_epilogue_out():
    """Resolve the optional operator once, before weight loading and capture."""
    try:
        module = import_module("ops.attn_epilogue")
    except ModuleNotFoundError as exc:
        if exc.name not in ("ops", "ops.attn_epilogue", "cannbotdsl"):
            raise
        return None
    return getattr(module, "attn_epilogue_out_op", None)


def prepare_attn_epilogue_weights(layer, o_lora_rank: int) -> None:
    """Keep checkpoint NK storage and expose the views needed by both paths."""
    weight, scale = layer.weight.data, layer.weight_scale.data
    if layer.prefix.endswith("wo_a"):
        groups = weight.shape[0] // o_lora_rank
        weight = weight.reshape(groups, o_lora_rank, -1)
        scale = scale.reshape(groups, o_lora_rank, -1, 2)
    else:
        # The unfused WoB linear consumes KN, while the fused kernel views the
        # same storage as NK. Neither path materializes a second set of weights.
        weight = weight.t()
        scale = scale.transpose(0, 1)
    layer.weight.data = weight
    layer.weight_scale.data = scale
