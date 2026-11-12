# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time packing for the local V4.1 DSL output projection."""

from functools import cache
from importlib import import_module

import torch
import torch_npu


@cache
def get_attn_epilogue_out():
    """Import once during model construction, before any graph capture."""
    return import_module("ops.attn_epilogue").attn_epilogue_out_op


def prepare_attn_epilogue_weights(layer, layout: str, o_lora_rank: int) -> None:
    """Replace checkpoint NK storage with the chosen fixed epilogue layout.

    DS block scales have already been expanded to [N,K/64,2]. ND keeps
    checkpoint weight bytes, avoiding another full set of resident projections.
    NZ transposes bytes and packs format 29 once; forward needs no conversion.
    """
    weight, scale = layer.weight.data, layer.weight_scale.data
    if layer.prefix.endswith("wo_a"):
        groups = weight.shape[0] // o_lora_rank
        weight = weight.reshape(groups, o_lora_rank, -1)
        scale = scale.reshape(groups, o_lora_rank, -1, 2)
    if layout == "nz":
        weight = weight.view(torch.uint8).transpose(-2, -1).contiguous().view(torch.float8_e4m3fn)
        scale = scale.transpose(-3, -2).contiguous()
        weight = torch_npu.npu_format_cast(weight, 29)
    layer.weight.data = weight
    layer.weight_scale.data = scale
