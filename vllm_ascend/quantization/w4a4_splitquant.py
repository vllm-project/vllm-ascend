# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""SplitQuant loader for the single validated dense fused NPU operator."""

from typing import Any, Dict, Optional
from pathlib import Path
from functools import lru_cache

import torch
import torch_npu
import os


def splitquant_operator_name() -> str:
    """Return the only supported SplitQuant operator."""
    return "splitquant_dense_fused"


@lru_cache(maxsize=1)
def load_splitquant_operator():
    """Load the library bundled with vLLM-Ascend in each worker process."""
    library = Path(__file__).resolve().parents[1] / "lib" / "libvllm_ascend_splitquant_dense.so"
    if not library.is_file():
        raise FileNotFoundError(
            f"SplitQuant operator library is not installed: {library}. "
            "Build vLLM-Ascend with custom kernels on a supported Atlas A2 host."
        )
    torch.ops.load_library(str(library))
    return getattr(torch.ops.npu, splitquant_operator_name())


class AscendW4A4SplitQuantLinearMethod:
    """Load SplitQuant tensors and dispatch to the registered fused op."""

    def __init__(self):
        self.group_size = 128
        self.prefix = "<unknown>"

    def get_weight(self, input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size,
                                      dtype=torch.int8)}

    def get_pertensor_param(self, params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    def get_perchannel_param(self, output_size: int,
                             params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype,
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        groups = input_size // self.group_size
        if input_size % self.group_size:
            raise ValueError("SplitQuant W4A4 requires input_size divisible by 128")
        return {
            "weight_scale": torch.empty(output_size, groups, dtype=torch.float32),
            "weight_offset": torch.empty(output_size, groups, dtype=torch.float32),
        }

    def get_extra_param(self, input_size: int, output_size: int,
                        params_dtype: torch.dtype, prefix: str,
                        names: tuple[str, ...]) -> Dict[str, Any]:
        """Allocate exported tensors by exact names in quant_model_description."""
        from vllm.config import get_current_vllm_config

        description = get_current_vllm_config().quant_config.quant_description
        groups = input_size // self.group_size
        params: Dict[str, Any] = {}
        for name in names:
            if prefix + "." + name not in description:
                continue
            if name in ("activation_clip_max", "activation_clip_min"):
                params[name] = torch.empty(groups, dtype=torch.float32)
            elif name == "group_trans":
                params[name] = torch.empty(groups, self.group_size,
                                           self.group_size, dtype=params_dtype)
            elif name == "right_trans":
                params[name] = torch.empty(input_size, input_size,
                                           dtype=params_dtype)
            elif name == "left_trans":
                heads = get_current_vllm_config().model_config.hf_config.num_attention_heads
                params[name] = torch.empty(heads, heads, dtype=params_dtype)
        return params

    def get_quantization_param(self, input_size: int, output_size: int,
                               params_dtype: torch.dtype,
                               prefix: str) -> Dict[str, Any]:
        return self.get_extra_param(
            input_size, output_size, params_dtype, prefix,
            ("activation_clip_max", "activation_clip_min"))

    def get_transform_param(self, input_size: int, output_size: int,
                            params_dtype: torch.dtype,
                            prefix: str) -> Dict[str, Any]:
        """Allocate only transform tensors present in this layer's description."""
        return self.get_extra_param(input_size, output_size, params_dtype,
                                    prefix, ("group_trans", "right_trans",
                                             "left_trans"))


    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None,
              tp_rank: Optional[int] = 0) -> torch.Tensor:
        if os.getenv("SPLITQUANT_REQUIRE_FUSED", "1") != "1":
            raise RuntimeError("SplitQuant deployment requires the validated fused AscendC operator")
        if not getattr(layer, "_splitquant_weight_packed", False):
            raise RuntimeError(f"SplitQuant weight is not packed at {self.prefix}")
        input_shape = x.shape
        x = x.reshape(-1, input_shape[-1]).contiguous()

        def transform_for_x(name: str):
            value = getattr(layer, name, None)
            if value is None:
                return None
            return value.to(device=x.device, dtype=x.dtype).contiguous()

        group_trans = transform_for_x("group_trans")
        right_trans = transform_for_x("right_trans")
        left_trans = transform_for_x("left_trans")
        try:
            op_name = splitquant_operator_name()
            splitquant_op = getattr(torch.ops.npu, op_name)
        except (AttributeError, RuntimeError):
            raise RuntimeError(
                "SplitQuant requires the validated npu::splitquant_dense_fused "
                "operator; the reference dequantize-and-matmul path is not used.")
        result = splitquant_op(
            x, layer.weight, layer.weight_scale, layer.weight_offset,
            layer.activation_clip_max, layer.activation_clip_min,
            bias, self.group_size, group_trans, right_trans, left_trans)
        self.fused_calls = getattr(self, "fused_calls", 0) + 1
        self.fused_operator_name = op_name
        return result.reshape(*input_shape[:-1], layer.weight.shape[0])

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Register the independently validated AIV implementation in every
        # vLLM worker, including workers started with multiprocessing spawn.
        load_splitquant_operator()
        if not hasattr(torch.ops.npu, "splitquant_dense_fused"):
            raise RuntimeError("Missing npu::splitquant_dense_fused operator")
        if getattr(layer, "_splitquant_weight_packed", False):
            raise RuntimeError(f"SplitQuant weight was already packed at {self.prefix}")
        if layer.weight.dtype != torch.int8:
            raise TypeError(f"Expected exported INT8-container W4 weight, got {layer.weight.dtype}")
        logical_k = layer.weight.shape[1]
        if logical_k <= 0 or logical_k % 2:
            raise ValueError(f"SplitQuant W4 K must be positive and even, got {logical_k}")
        codes = layer.weight.data.to(torch.int16)
        if bool(((codes < -8) | (codes > 7)).any()):
            raise ValueError(f"SplitQuant INT8 container contains values outside signed INT4 at {self.prefix}")
        packed = ((codes[:, 0::2] & 0xF) |
                  ((codes[:, 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()
        if os.getenv("SPLITQUANT_AUDIT_PACKED") == "1":
            raw = packed.to(torch.int16)
            unpacked = torch.stack((raw & 15, raw >> 4), dim=-1).reshape_as(codes)
            unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
            if not torch.equal(unpacked, codes):
                raise ValueError(f"Packed nibble round-trip failed at {self.prefix}")
            layer._splitquant_roundtrip_verified = True
        layer._splitquant_logical_shape = tuple(layer.weight.shape)
        layer.weight.data = packed.view(torch.int8)
        layer._splitquant_weight_packed = True
        layer.weight_scale.data = layer.weight_scale.data.float().contiguous()
        layer.weight_offset.data = layer.weight_offset.data.float().contiguous()
        layer.activation_clip_max.data = layer.activation_clip_max.data.float().contiguous()
        layer.activation_clip_min.data = layer.activation_clip_min.data.float().contiguous()
        for name in ("activation_clip_max", "activation_clip_min"):
            if not torch.isfinite(getattr(layer, name)).all():
                raise ValueError(f"Non-finite SplitQuant parameter {name} at {self.prefix}")
        if hasattr(layer, "group_trans"):
            if layer.group_trans.shape != (logical_k // self.group_size,
                                           self.group_size, self.group_size):
                raise ValueError(f"Invalid group_trans shape at {self.prefix}")
        if hasattr(layer, "left_trans"):
            heads = layer.left_trans.shape[0]
            if layer.left_trans.shape != (heads, heads) or logical_k % heads:
                raise ValueError(f"Invalid left_trans shape at {self.prefix}")
        if hasattr(layer, "right_trans"):
            if layer.right_trans.shape != (logical_k, logical_k):
                raise ValueError(f"Invalid right_trans shape at {self.prefix}")
        for name in ("group_trans", "left_trans", "right_trans"):
            value = getattr(layer, name, None)
            if value is not None:
                setattr(layer, name, torch.nn.Parameter(value.data.contiguous(),
                                                        requires_grad=False))

