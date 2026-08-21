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

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import get_a5_mega_moe_buffer_tokens_per_rank
from vllm_ascend.distributed.parallel_state import get_mega_moe_group
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEFusedExpertsInput
from vllm_ascend.quantization.quant_type import QuantType

_MEGA_MOE_SUPPORTED_QUANTS = {
    QuantType.MXFP8,
    QuantType.MXFP4,
    QuantType.W4A8MXFP,
}
_FP4_PACK_FACTOR = 2
_MXFP_SCALE_BLOCK_SIZE = 64
_MXFP_SCALE_MULTIPLIER = 2
# Tensor.view requires a torch.dtype; torch_npu's operator dtype enum is interpreted as a shape.
_TORCH_FLOAT8_E8M0FNU_DTYPE = getattr(torch, "float8_e8m0fnu", None)


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
            continue
        if scale.dtype == torch.uint8:
            normalized_scales.append(scale.view(_TORCH_FLOAT8_E8M0FNU_DTYPE))
            continue
        raise RuntimeError(
            f"A5 MegaMoE requires {name}[{idx}] to be FLOAT8_E8M0 weight scale, "
            f"got dtype={scale.dtype}, shape={tuple(scale.shape)}. "
            "Do not pass packed INT/Fused-MC2 UINT64 scales into MegaMoE."
        )
    return normalized_scales


def _get_mega_moe_ops():
    try:
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
    except ImportError as exc:
        raise RuntimeError(
            "A5 MegaMoE requires cann_ops_transformer.ops. Please install a CANN ops-transformer "
            "package that provides mega_moe and get_symm_buffer_for_mega_moe."
        ) from exc
    return get_symm_buffer_for_mega_moe, mega_moe


@dataclass(frozen=True, slots=True)
class _MegaMoEBufferKey:
    num_experts: int
    buffer_tokens_per_rank: int
    top_k: int
    hidden_size: int
    intermediate_hidden: int
    dispatch_quant_mode: int
    dispatch_quant_out_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _MegaMoESymmetricBufferState:
    key: _MegaMoEBufferKey
    buffer: Any


_MEGA_MOE_BUFFER_STATE_ATTR = "_mega_moe_symmetric_buffer_state"


class MegaMoEBackend:
    """A5 MegaMoE wrapper for the logical FUSED_MC2 MoE path."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

    @staticmethod
    def _validate_stacked_mxfp_layout(
        fused_experts_input: MoEFusedExpertsInput,
        w1: list[torch.Tensor],
        w2: list[torch.Tensor],
        w1_scale: list[torch.Tensor],
        w2_scale: list[torch.Tensor],
    ) -> int:
        tensors = {
            "w1": w1,
            "w2": w2,
            "w1_scale": w1_scale,
            "w2_scale": w2_scale,
        }
        for name, values in tensors.items():
            if len(values) != 1:
                raise ValueError(
                    f"A5 MegaMoE requires {name} to contain one stacked tensor, got {len(values)} tensors."
                )
            if not values[0].is_contiguous():
                raise ValueError(
                    f"A5 MegaMoE requires contiguous {name}, got shape={tuple(values[0].shape)} "
                    f"and stride={values[0].stride()}."
                )

        weight1 = w1[0]
        weight2 = w2[0]
        scale1 = w1_scale[0]
        scale2 = w2_scale[0]
        if weight1.ndim != 3 or weight2.ndim != 3:
            raise ValueError(
                "A5 MegaMoE expects stacked 3D expert weights, "
                f"got w1={tuple(weight1.shape)} and w2={tuple(weight2.shape)}."
            )
        if scale1.ndim != 4 or scale2.ndim != 4:
            raise ValueError(
                "A5 MegaMoE expects stacked 4D MXFP scales, "
                f"got w1_scale={tuple(scale1.shape)} and w2_scale={tuple(scale2.shape)}."
            )

        hidden_size = int(fused_experts_input.hidden_states.shape[-1])
        fp4_packed = fused_experts_input.quant.quant_type in {QuantType.MXFP4, QuantType.W4A8MXFP}
        pack_factor = _FP4_PACK_FACTOR if fp4_packed else 1
        projected_hidden = int(weight1.shape[1])
        if projected_hidden % 2 != 0:
            raise ValueError(f"A5 MegaMoE expects w1.shape[1] to be even, got {projected_hidden}.")
        intermediate_hidden = projected_hidden // 2
        if hidden_size % pack_factor != 0 or intermediate_hidden % pack_factor != 0:
            raise ValueError(
                "A5 MegaMoE FP4 dimensions must be divisible by the packing factor: "
                f"hidden_size={hidden_size}, intermediate_hidden={intermediate_hidden}, "
                f"pack_factor={pack_factor}."
            )
        expected_weight1_shape = (weight1.shape[0], projected_hidden, hidden_size // pack_factor)
        expected_weight2_shape = (weight1.shape[0], hidden_size, intermediate_hidden // pack_factor)
        expected_scale1_shape = (
            weight1.shape[0],
            projected_hidden,
            (hidden_size + _MXFP_SCALE_BLOCK_SIZE - 1) // _MXFP_SCALE_BLOCK_SIZE,
            _MXFP_SCALE_MULTIPLIER,
        )
        expected_scale2_shape = (
            weight1.shape[0],
            hidden_size,
            (intermediate_hidden + _MXFP_SCALE_BLOCK_SIZE - 1) // _MXFP_SCALE_BLOCK_SIZE,
            _MXFP_SCALE_MULTIPLIER,
        )
        actual_shapes = {
            "w1": tuple(weight1.shape),
            "w2": tuple(weight2.shape),
            "w1_scale": tuple(scale1.shape),
            "w2_scale": tuple(scale2.shape),
        }
        expected_shapes = {
            "w1": expected_weight1_shape,
            "w2": expected_weight2_shape,
            "w1_scale": expected_scale1_shape,
            "w2_scale": expected_scale2_shape,
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                "A5 MegaMoE received an incompatible stacked MXFP layout: "
                f"actual={actual_shapes}, expected={expected_shapes}."
            )
        return projected_hidden

    def _make_buffer_key(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        projected_hidden: int | None = None,
        *,
        buffer_tokens_per_rank: int,
    ) -> _MegaMoEBufferKey:
        hidden_size = int(fused_experts_input.hidden_states.shape[-1])
        if projected_hidden is None:
            w1 = _as_tensor_list(fused_experts_input.weights.w1, "w1")
            w2 = _as_tensor_list(fused_experts_input.weights.w2, "w2")
            w1_scale = _as_tensor_list(fused_experts_input.weights.w1_scale, "w1_scale")
            w2_scale = _as_tensor_list(fused_experts_input.weights.w2_scale, "w2_scale")
            projected_hidden = self._validate_stacked_mxfp_layout(
                fused_experts_input,
                w1,
                w2,
                w1_scale,
                w2_scale,
            )
        if fused_experts_input.hidden_states.shape[0] > buffer_tokens_per_rank:
            raise ValueError(
                "A5 MegaMoE input exceeds the symmetric buffer token capacity: "
                f"num_tokens={fused_experts_input.hidden_states.shape[0]}, "
                f"mega_moe_buffer_tokens_per_rank={buffer_tokens_per_rank}."
            )
        act_dtype = torch.float8_e4m3fn
        if fused_experts_input.quant.mxfp is not None and fused_experts_input.quant.mxfp.act_quant_type is not None:
            act_dtype = fused_experts_input.quant.mxfp.act_quant_type
        return _MegaMoEBufferKey(
            num_experts=self.moe_config.num_experts,
            buffer_tokens_per_rank=buffer_tokens_per_rank,
            top_k=self.moe_config.experts_per_token,
            hidden_size=hidden_size,
            intermediate_hidden=projected_hidden,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=act_dtype,
        )

    def _get_sym_buffer(self, fused_experts_input: MoEFusedExpertsInput, projected_hidden: int):
        mega_moe_group = get_mega_moe_group()
        ascend_config = get_ascend_config()
        buffer_tokens_per_rank = get_a5_mega_moe_buffer_tokens_per_rank(ascend_config.vllm_config)
        key = self._make_buffer_key(
            fused_experts_input,
            projected_hidden,
            buffer_tokens_per_rank=buffer_tokens_per_rank,
        )
        state: _MegaMoESymmetricBufferState | None = getattr(
            mega_moe_group,
            _MEGA_MOE_BUFFER_STATE_ATTR,
            None,
        )
        if state is not None:
            if state.key != key:
                raise RuntimeError(
                    "A5 MegaMoE symmetric buffer is shared by the main and draft models and cannot be replaced "
                    f"during inference: initialized={state.key}, requested={key}."
                )
            logger.debug("A5 MegaMoE reuses the process-wide symmetric buffer: %s", key)
            return state.buffer

        get_symm_buffer_for_mega_moe, _ = _get_mega_moe_ops()
        logger.debug("A5 MegaMoE creates the process-wide symmetric buffer: %s", key)
        buffer = get_symm_buffer_for_mega_moe(
            mega_moe_group.device_group,
            num_experts=key.num_experts,
            num_max_tokens_per_rank=key.buffer_tokens_per_rank,
            num_topk=key.top_k,
            hidden=key.hidden_size,
            intermediate_hidden=key.intermediate_hidden,
            dispatch_quant_mode=key.dispatch_quant_mode,
            dispatch_quant_out_dtype=key.dispatch_quant_out_dtype,
        )
        setattr(
            mega_moe_group,
            _MEGA_MOE_BUFFER_STATE_ATTR,
            _MegaMoESymmetricBufferState(key=key, buffer=buffer),
        )
        return buffer

    @staticmethod
    def _normalize_activation(activation: Any) -> str:
        activation_name = activation if isinstance(activation, str) else getattr(activation, "name", str(activation))
        activation_lower = activation_name.lower().removeprefix("moeactivation.")
        if activation_lower in ("silu", "swiglu"):
            return "swiglu"
        raise ValueError(
            f"A5 MegaMoE does not support activation={activation!r} without changing its semantics. "
            "Only SILU/SwiGLU is currently supported."
        )

    @staticmethod
    def _get_activation_clamp(swiglu_limit: float) -> float | None:
        return None if swiglu_limit <= 0 else float(swiglu_limit)

    def fused_experts(self, fused_experts_input: MoEFusedExpertsInput) -> tuple[torch.Tensor, torch.Tensor | None]:
        if fused_experts_input.quant.quant_type not in _MEGA_MOE_SUPPORTED_QUANTS:
            raise RuntimeError(
                f"A5 MegaMoE only supports MXFP MoE quantization in the first stage, "
                f"got {fused_experts_input.quant.quant_type}."
            )
        if not fused_experts_input.quant.is_mxfp:
            raise RuntimeError("A5 MegaMoE requires MXFP quant parameters.")
        if fused_experts_input.dynamic_eplb:
            raise RuntimeError("A5 MegaMoE does not support dynamic EPLB expert weight lists.")
        if fused_experts_input.routing.global_redundant_expert_num:
            raise RuntimeError("A5 MegaMoE does not support redundant physical experts.")

        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        w1 = _as_tensor_list(fused_experts_input.weights.w1, "w1")
        w2 = _as_tensor_list(fused_experts_input.weights.w2, "w2")
        w1_scale = _view_mxfp_scales_as_e8m0(
            _as_tensor_list(fused_experts_input.weights.w1_scale, "w1_scale"),
            "w1_scale",
        )
        w2_scale = _view_mxfp_scales_as_e8m0(
            _as_tensor_list(fused_experts_input.weights.w2_scale, "w2_scale"),
            "w2_scale",
        )
        projected_hidden = self._validate_stacked_mxfp_layout(
            fused_experts_input,
            w1,
            w2,
            w1_scale,
            w2_scale,
        )
        activation = self._normalize_activation(fused_experts_input.activation)
        activation_clamp = self._get_activation_clamp(fused_experts_input.swiglu_limit)
        mxfp = fused_experts_input.quant.mxfp
        weight_type = None if mxfp is None else mxfp.weight_quant_type

        _, mega_moe = _get_mega_moe_ops()
        logger.debug(
            "A5 MegaMoE call: hidden_states_shape=%s, topk_ids_shape=%s, topk_weights_shape=%s, "
            "w1_shapes=%s, w2_shapes=%s, w1_scale_shapes=%s, w2_scale_shapes=%s, "
            "w1_scale_dtypes=%s, w2_scale_dtypes=%s, quant_type=%s, "
            "is_mxfp=%s, mxfp_act_quant_type=%s, mxfp_weight_quant_type=%s, mxfp_scale_dtype=%s, "
            "mxfp_per_token_scale_dtype=%s, mxfp_use_bf16=%s, comm_quant_mode=%s, activation=%s, "
            "activation_clamp=%s, has_log2phy=%s, dynamic_eplb=%s",
            tuple(fused_experts_input.hidden_states.shape),
            tuple(topk_ids.shape),
            tuple(fused_experts_input.topk_weights.shape),
            [tuple(weight.shape) for weight in w1],
            [tuple(weight.shape) for weight in w2],
            [tuple(scale.shape) for scale in w1_scale],
            [tuple(scale.shape) for scale in w2_scale],
            [scale.dtype for scale in w1_scale],
            [scale.dtype for scale in w2_scale],
            fused_experts_input.quant.quant_type,
            fused_experts_input.quant.is_mxfp,
            None if mxfp is None else mxfp.act_quant_type,
            None if mxfp is None else mxfp.weight_quant_type,
            None if mxfp is None else mxfp.scale_dtype,
            None if mxfp is None else mxfp.per_token_scale_dtype,
            None if mxfp is None else mxfp.use_bf16,
            fused_experts_input.quant.comm_quant_mode,
            activation,
            activation_clamp,
            fused_experts_input.routing.log2phy is not None,
            fused_experts_input.dynamic_eplb,
        )
        mega_moe_kwargs = {
            "x": fused_experts_input.hidden_states,
            "topk_ids": topk_ids.to(torch.int32),
            "topk_weights": fused_experts_input.topk_weights,
            "l1_weights": w1,
            "l1_weights_sf": w1_scale,
            "l2_weights": w2,
            "l2_weights_sf": w2_scale,
            "sym_buffer": self._get_sym_buffer(fused_experts_input, projected_hidden),
            "activation": activation,
            "activation_clamp": activation_clamp,
        }
        if weight_type is not None:
            mega_moe_kwargs["weight1_type"] = weight_type
            mega_moe_kwargs["weight2_type"] = weight_type

        output, expert_tokens = mega_moe(**mega_moe_kwargs)
        logger.debug(
            "A5 MegaMoE output: output_shape=%s, expert_tokens_shape=%s",
            tuple(output.shape),
            None if expert_tokens is None else tuple(expert_tokens.shape),
        )
        return output, expert_tokens


__all__ = ["MegaMoEBackend"]
