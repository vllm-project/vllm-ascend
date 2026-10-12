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
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_forward_context import get_a5_mega_moe_buffer_tokens_per_rank
from vllm_ascend.distributed.parallel_state import get_mega_moe_group
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput
from vllm_ascend.quantization.quant_type import QuantType

_MEGA_MOE_SUPPORTED_QUANTS = {
    QuantType.W4A8MXFP,
}
_MEGA_MOE_MXFP_GROUP_SIZE = 32
_FP4_PACK_FACTOR = 2
_MXFP_SCALE_BLOCK_SIZE = 64
_MXFP_SCALE_MULTIPLIER = 2
_TORCH_FLOAT8_E8M0FNU_DTYPE = getattr(torch, "float8_e8m0fnu", None)
_TORCH_NPU_FLOAT4_E2M1FN_X2_DTYPE = getattr(torch_npu, "float4_e2m1fn_x2", None)


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


def _get_mega_moe_ops():
    try:
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
    except ImportError as exc:
        raise RuntimeError(
            "A5 MegaMoE requires cann_ops_transformer.ops. Install a package "
            "that provides mega_moe and get_symm_buffer_for_mega_moe."
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

    def _validate_stacked_mxfp_layout(
        self,
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
        if weight1.element_size() != 1 or weight2.element_size() != 1:
            raise ValueError(
                "A5 MegaMoE requires packed FP4 weights backed by one-byte storage, "
                f"got w1={weight1.dtype} and w2={weight2.dtype}."
            )
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

        hidden_states = fused_experts_input.hidden_states
        if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
            raise ValueError(
                "A5 MegaMoE requires contiguous 2D hidden states, "
                f"got shape={tuple(hidden_states.shape)} and stride={hidden_states.stride()}."
            )
        hidden_size = int(hidden_states.shape[-1])
        pack_factor = _FP4_PACK_FACTOR
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
        if hidden_size % _MXFP_SCALE_BLOCK_SIZE != 0 or intermediate_hidden % _MXFP_SCALE_BLOCK_SIZE != 0:
            raise ValueError(
                "A5 MegaMoE MXFP dimensions must be divisible by the scale block size: "
                f"hidden_size={hidden_size}, intermediate_hidden={intermediate_hidden}, "
                f"scale_block_size={_MXFP_SCALE_BLOCK_SIZE}."
            )

        num_local_experts = int(weight1.shape[0])
        configured_local_experts = getattr(self.moe_config, "num_local_experts", None)
        if configured_local_experts is not None and num_local_experts != configured_local_experts:
            raise ValueError(
                "A5 MegaMoE local expert count does not match the MoE config: "
                f"weights={num_local_experts}, configured={configured_local_experts}."
            )

        expected_shapes = {
            "w1": (weight1.shape[0], projected_hidden, hidden_size // pack_factor),
            "w2": (weight1.shape[0], hidden_size, intermediate_hidden // pack_factor),
            "w1_scale": (
                weight1.shape[0],
                projected_hidden,
                (hidden_size + _MXFP_SCALE_BLOCK_SIZE - 1) // _MXFP_SCALE_BLOCK_SIZE,
                _MXFP_SCALE_MULTIPLIER,
            ),
            "w2_scale": (
                weight1.shape[0],
                hidden_size,
                (intermediate_hidden + _MXFP_SCALE_BLOCK_SIZE - 1) // _MXFP_SCALE_BLOCK_SIZE,
                _MXFP_SCALE_MULTIPLIER,
            ),
        }
        actual_shapes = {
            "w1": tuple(weight1.shape),
            "w2": tuple(weight2.shape),
            "w1_scale": tuple(scale1.shape),
            "w2_scale": tuple(scale2.shape),
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                "A5 MegaMoE received an incompatible stacked MXFP layout: "
                f"actual={actual_shapes}, expected={expected_shapes}."
            )
        # CANN allocates the first grouped matmul from the full gate/up
        # projection width, which is 6144 for Kimi K3 (2 * 3072).
        return projected_hidden

    def _make_buffer_key(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        intermediate_hidden: int,
        *,
        buffer_tokens_per_rank: int,
    ) -> _MegaMoEBufferKey:
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
            hidden_size=int(fused_experts_input.hidden_states.shape[-1]),
            intermediate_hidden=intermediate_hidden,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=act_dtype,
        )

    def _get_sym_buffer(self, fused_experts_input: MoEFusedExpertsInput, intermediate_hidden: int):
        vllm_config = get_current_vllm_config()
        if vllm_config is None:
            raise RuntimeError("A5 MegaMoE requires a current vLLM configuration.")
        buffer_tokens_per_rank = get_a5_mega_moe_buffer_tokens_per_rank(vllm_config)
        key = self._make_buffer_key(
            fused_experts_input,
            intermediate_hidden,
            buffer_tokens_per_rank=buffer_tokens_per_rank,
        )
        mega_moe_group = get_mega_moe_group()
        state: _MegaMoESymmetricBufferState | None = getattr(
            mega_moe_group,
            _MEGA_MOE_BUFFER_STATE_ATTR,
            None,
        )
        if state is not None:
            if state.key != key:
                raise RuntimeError(
                    "A5 MegaMoE symmetric buffer is process-wide and cannot be replaced during inference: "
                    f"initialized={state.key}, requested={key}."
                )
            return state.buffer

        get_symm_buffer_for_mega_moe, _ = _get_mega_moe_ops()
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
        if activation_lower in ("situ", "situglu"):
            return "situglu"
        raise ValueError(
            f"A5 MegaMoE does not support activation={activation!r} without changing its semantics. "
            "Only SILU/SwiGLU and SiTU are currently supported."
        )

    def _resolve_activation(
        self,
        activation: Any,
    ) -> tuple[str, dict[str, float] | None, float | None]:
        activation_name = self._normalize_activation(activation)
        if activation_name == "situglu":
            beta = getattr(self.moe_config, "activation_situ_beta", None)
            linear_beta = getattr(self.moe_config, "activation_situ_linear_beta", None)
            if beta is None or linear_beta is None:
                raise ValueError(
                    "A5 MegaMoE SiTU requires activation_situ_beta and activation_situ_linear_beta in the MoE config."
                )
            return (
                activation_name,
                {
                    "beta": float(beta),
                    "linear_beta": float(linear_beta),
                },
                None,
            )

        swiglu_limit = getattr(self.moe_config, "swiglu_limit", None)
        activation_clamp = None if swiglu_limit is None or swiglu_limit <= 0 else float(swiglu_limit)
        return activation_name, None, activation_clamp

    def _validate_runtime_contract(self, fused_experts_input: MoEFusedExpertsInput) -> None:
        if fused_experts_input.quant.quant_type not in _MEGA_MOE_SUPPORTED_QUANTS:
            raise RuntimeError(
                f"A5 MegaMoE only supports W4A8MXFP routed experts, got {fused_experts_input.quant.quant_type}."
            )
        mxfp = fused_experts_input.quant.mxfp
        if mxfp is None:
            raise RuntimeError("A5 MegaMoE requires explicit MXFP runtime parameters.")
        if mxfp.group_size != _MEGA_MOE_MXFP_GROUP_SIZE:
            raise RuntimeError(
                f"A5 MegaMoE requires MXFP group_size={_MEGA_MOE_MXFP_GROUP_SIZE}, got {mxfp.group_size}."
            )
        if mxfp.act_quant_type != torch.float8_e4m3fn:
            raise RuntimeError(f"A5 MegaMoE requires FP8 E4M3 dispatch activations, got {mxfp.act_quant_type}.")
        if _TORCH_NPU_FLOAT4_E2M1FN_X2_DTYPE is None or (mxfp.weight_quant_type != _TORCH_NPU_FLOAT4_E2M1FN_X2_DTYPE):
            raise RuntimeError(f"A5 MegaMoE requires packed FP4 E2M1 expert weights, got {mxfp.weight_quant_type}.")
        if fused_experts_input.dynamic_eplb:
            raise RuntimeError("A5 MegaMoE does not support dynamic EPLB expert weight lists.")
        if fused_experts_input.routing.global_redundant_expert_num:
            raise RuntimeError("A5 MegaMoE does not support redundant physical experts.")
        if fused_experts_input.lora_context is not None:
            raise RuntimeError("A5 MegaMoE does not support MoE LoRA.")
        if fused_experts_input.topk_ids.ndim != 2:
            raise ValueError(f"A5 MegaMoE requires 2D top-k tensors, got {tuple(fused_experts_input.topk_ids.shape)}.")
        if fused_experts_input.topk_ids.shape[1] != self.moe_config.experts_per_token:
            raise ValueError(
                "A5 MegaMoE top-k width does not match the MoE config: "
                f"ids={fused_experts_input.topk_ids.shape[1]}, configured={self.moe_config.experts_per_token}."
            )
        if fused_experts_input.topk_ids.shape[0] != fused_experts_input.hidden_states.shape[0]:
            raise ValueError(
                "A5 MegaMoE top-k rows must match hidden-state rows: "
                f"topk={fused_experts_input.topk_ids.shape[0]}, "
                f"hidden={fused_experts_input.hidden_states.shape[0]}."
            )
        if fused_experts_input.topk_ids.shape != fused_experts_input.topk_weights.shape:
            raise ValueError(
                "A5 MegaMoE requires matching top-k id and weight shapes, "
                f"got ids={tuple(fused_experts_input.topk_ids.shape)}, "
                f"weights={tuple(fused_experts_input.topk_weights.shape)}."
            )

    def fused_experts(self, fused_experts_input: MoEFusedExpertsInput) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_runtime_contract(fused_experts_input)

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
        activation, activation_params, activation_clamp = self._resolve_activation(fused_experts_input.activation)
        mxfp = fused_experts_input.quant.mxfp
        weight_type = None if mxfp is None else mxfp.weight_quant_type

        _, mega_moe = _get_mega_moe_ops()
        mega_moe_kwargs = {
            "x": fused_experts_input.hidden_states,
            "topk_ids": fused_experts_input.topk_ids.to(torch.int32).contiguous(),
            "topk_weights": fused_experts_input.topk_weights.to(torch.float32).contiguous(),
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
        if activation_params is not None:
            mega_moe_kwargs["activation_params"] = activation_params

        try:
            output, expert_tokens = mega_moe(**mega_moe_kwargs)
        except TypeError as exc:
            if activation_params is not None:
                raise RuntimeError(
                    "The installed CANN MegaMoE API does not accept Kimi K3 SiTU activation_params; "
                    "install a compatible ops-transformer build."
                ) from exc
            raise
        return output, expert_tokens


__all__ = ["MegaMoEBackend"]
