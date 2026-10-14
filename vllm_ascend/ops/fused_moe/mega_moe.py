#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from inspect import Parameter, signature
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput
from vllm_ascend.quantization.quant_type import QuantType

_MEGA_MOE_SUPPORTED_SHARED_QUANTS = frozenset({QuantType.W4A8MXFP, QuantType.W8A8MXFP})
_FP4_PACK_FACTOR = 2
_MXFP_SCALE_BLOCK_SIZE = 64
_MXFP_SCALE_MULTIPLIER = 2
_FLOAT8_E4M3FN_WEIGHT_TYPE = 24
_TORCH_FLOAT8_E8M0FNU_DTYPE = getattr(torch, "float8_e8m0fnu", None)


def supports_mega_moe_shared_experts(mega_moe_op: Any) -> bool:
    """Require explicit shared-weight arguments from the installed CANN ABI."""
    required_parameters = {
        "shared_l1_weights",
        "shared_l2_weights",
        "shared_l1_weights_sf",
        "shared_l2_weights_sf",
        "shared_weight1_type",
        "shared_weight2_type",
    }
    try:
        parameters = signature(mega_moe_op).parameters
        keyword_parameters = {
            name
            for name, parameter in parameters.items()
            if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }
        supported = required_parameters <= keyword_parameters
    except (TypeError, ValueError):
        supported = False
    if not supported:
        logger.warning_once(
            "Installed MegaMoE operator does not expose shared weights and independent "
            "shared weight types; falling back to the standalone shared MLP."
        )
    return supported


def _as_tensor_list(tensor_or_list: torch.Tensor | list[torch.Tensor] | None, name: str) -> list[torch.Tensor]:
    if tensor_or_list is None:
        raise ValueError(f"{name} is required for A5 MegaMoE.")
    if isinstance(tensor_or_list, list):
        if not tensor_or_list:
            raise ValueError(f"{name} cannot be an empty list for A5 MegaMoE.")
        return tensor_or_list
    return [tensor_or_list]


def _view_mxfp_scales_as_e8m0(scales: list[torch.Tensor], name: str) -> list[torch.Tensor]:
    if _TORCH_FLOAT8_E8M0FNU_DTYPE is None:
        raise RuntimeError("A5 MegaMoE requires torch.float8_e8m0fnu to reinterpret MXFP weight scales.")

    normalized_scales: list[torch.Tensor] = []
    for idx, scale in enumerate(scales):
        if scale.dtype == _TORCH_FLOAT8_E8M0FNU_DTYPE:
            normalized_scales.append(scale)
        elif scale.dtype == torch.uint8:
            normalized_scales.append(scale.view(_TORCH_FLOAT8_E8M0FNU_DTYPE))
        else:
            raise RuntimeError(
                f"A5 MegaMoE requires {name}[{idx}] to be FLOAT8_E8M0 weight scale, "
                f"got dtype={scale.dtype}, shape={tuple(scale.shape)}."
            )
    return normalized_scales


def _validate_shared_mxfp_layout(
    fused_experts_input: MoEFusedExpertsInput,
    w1: list[torch.Tensor],
    w2: list[torch.Tensor],
    w1_scale: list[torch.Tensor],
    w2_scale: list[torch.Tensor],
    intermediate_hidden: int,
    quant_type: QuantType,
    routed_weight_type: int | None,
) -> Any:
    tensors = {
        "shared_w1": w1,
        "shared_w2": w2,
        "shared_w1_scale": w1_scale,
        "shared_w2_scale": w2_scale,
    }
    for name, values in tensors.items():
        if len(values) != 1:
            raise ValueError(f"A5 MegaMoE requires {name} to contain one stacked tensor.")
        if not values[0].is_contiguous():
            raise ValueError(f"A5 MegaMoE requires contiguous {name}, got stride={values[0].stride()}.")

    if quant_type not in _MEGA_MOE_SUPPORTED_SHARED_QUANTS:
        raise ValueError(f"A5 MegaMoE does not support shared quantization type {quant_type}.")

    hidden_size = int(fused_experts_input.hidden_states.shape[-1])
    weight_pack_factor = _FP4_PACK_FACTOR if quant_type == QuantType.W4A8MXFP else 1
    expected_shapes = {
        "shared_w1": (1, 2 * intermediate_hidden, hidden_size // weight_pack_factor),
        "shared_w2": (1, hidden_size, intermediate_hidden // weight_pack_factor),
        "shared_w1_scale": (
            1,
            2 * intermediate_hidden,
            (hidden_size + _MXFP_SCALE_BLOCK_SIZE - 1) // _MXFP_SCALE_BLOCK_SIZE,
            _MXFP_SCALE_MULTIPLIER,
        ),
        "shared_w2_scale": (
            1,
            hidden_size,
            (intermediate_hidden + _MXFP_SCALE_BLOCK_SIZE - 1) // _MXFP_SCALE_BLOCK_SIZE,
            _MXFP_SCALE_MULTIPLIER,
        ),
    }
    actual_shapes = {
        "shared_w1": tuple(w1[0].shape),
        "shared_w2": tuple(w2[0].shape),
        "shared_w1_scale": tuple(w1_scale[0].shape),
        "shared_w2_scale": tuple(w2_scale[0].shape),
    }
    if actual_shapes != expected_shapes:
        raise ValueError(
            "A5 MegaMoE received an incompatible shared-expert MXFP layout: "
            f"actual={actual_shapes}, expected={expected_shapes}."
        )
    if quant_type == QuantType.W8A8MXFP:
        return _FLOAT8_E4M3FN_WEIGHT_TYPE
    if routed_weight_type is None:
        raise ValueError("A5 MegaMoE requires the routed FP4 weight type for FP4 shared experts.")
    return routed_weight_type


def build_mega_moe_shared_kwargs(
    fused_experts_input: MoEFusedExpertsInput,
    intermediate_hidden: int,
    routed_weight_type: int | None,
) -> dict[str, Any]:
    """Validate shared MXFP weights before passing them to the fused operator."""
    shared_weights = fused_experts_input.shared_weights
    if shared_weights is None:
        return {}
    if fused_experts_input.quant.quant_type != QuantType.W4A8MXFP:
        raise ValueError("Shared MegaMoE fusion currently requires W4A8MXFP routed experts.")
    w1 = _as_tensor_list(shared_weights.w1, "shared_w1")
    w2 = _as_tensor_list(shared_weights.w2, "shared_w2")
    w1_scale = _view_mxfp_scales_as_e8m0(_as_tensor_list(shared_weights.w1_scale, "shared_w1_scale"), "shared_w1_scale")
    w2_scale = _view_mxfp_scales_as_e8m0(_as_tensor_list(shared_weights.w2_scale, "shared_w2_scale"), "shared_w2_scale")
    shared_quant_type = shared_weights.quant_type or fused_experts_input.quant.quant_type
    weight_type = _validate_shared_mxfp_layout(
        fused_experts_input,
        w1,
        w2,
        w1_scale,
        w2_scale,
        intermediate_hidden,
        shared_quant_type,
        routed_weight_type,
    )
    return {
        "shared_l1_weights": w1,
        "shared_l2_weights": w2,
        "shared_l1_weights_sf": w1_scale,
        "shared_l2_weights_sf": w2_scale,
        "shared_weight1_type": weight_type,
        "shared_weight2_type": weight_type,
    }
