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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, get_a5_mega_moe_buffer_tokens_per_rank
from vllm_ascend.device.mxfp_compat import FLOAT4_E2M1FN_X2_DTYPE
from vllm_ascend.distributed.parallel_state import get_mega_moe_group
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEFusedExpertsInput
from vllm_ascend.quantization.quant_type import QuantType

_A5_MEGA_MOE_QUANT_TYPE = QuantType.W4A8MXFP
_A5_MEGA_MOE_GROUP_SIZE = 32
_FP4_PACK_FACTOR = 2
_MXFP_SCALE_BLOCK_SIZE = 64
_MXFP_SCALE_MULTIPLIER = 2
_DISPATCH_QUANT_MODE_MXFP8 = 4
_TORCH_FLOAT8_E8M0FNU_DTYPE = getattr(torch, "float8_e8m0fnu", None)
_MEGA_MOE_BUFFER_STATE_ATTR = "_mega_moe_symmetric_buffer_state"


def _as_tensor_list(
    tensor_or_list: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    name: str,
) -> list[torch.Tensor]:
    if tensor_or_list is None:
        raise ValueError(f"{name} is required for A5 MegaMoE.")
    if isinstance(tensor_or_list, (list, tuple)):
        if not tensor_or_list:
            raise ValueError(f"{name} cannot be empty for A5 MegaMoE.")
        values = list(tensor_or_list)
    else:
        values = [tensor_or_list]
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"{name} must contain only tensors for A5 MegaMoE.")
    return values


def _view_mxfp_scales_as_e8m0(scales: list[torch.Tensor], name: str) -> list[torch.Tensor]:
    if _TORCH_FLOAT8_E8M0FNU_DTYPE is None:
        raise RuntimeError("A5 MegaMoE requires torch.float8_e8m0fnu to reinterpret MXFP weight scales.")

    normalized_scales: list[torch.Tensor] = []
    for index, scale in enumerate(scales):
        if scale.dtype == _TORCH_FLOAT8_E8M0FNU_DTYPE:
            normalized_scales.append(scale)
        elif scale.dtype == torch.uint8:
            normalized_scales.append(scale.view(_TORCH_FLOAT8_E8M0FNU_DTYPE))
        else:
            raise RuntimeError(
                f"A5 MegaMoE requires {name}[{index}] to use FLOAT8_E8M0 scales, "
                f"got dtype={scale.dtype}, shape={tuple(scale.shape)}."
            )
    return normalized_scales


def _get_mega_moe_ops():
    try:
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "A5 MegaMoE requires a CANN ops-transformer package that exports mega_moe and get_symm_buffer_for_mega_moe."
        ) from exc
    return get_symm_buffer_for_mega_moe, mega_moe


@dataclass(frozen=True, slots=True)
class MegaMoEBufferKey:
    ep_ranks: tuple[int, ...]
    num_experts: int
    tokens_per_rank: int
    top_k: int
    hidden: int
    intermediate_hidden: int
    dispatch_quant_mode: int
    dispatch_quant_out_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class _MegaMoESymmetricBufferState:
    key: MegaMoEBufferKey
    buffer: Any


class MegaMoEBackend:
    """A5 W4A8 MXFP backend for the logical FUSED_MC2 MoE path."""

    def __init__(self, moe_config: FusedMoEConfig, ops=None):
        self.moe_config = moe_config
        self._get_symm_buffer_op, self._mega_moe_op = ops or _get_mega_moe_ops()
        logger.info_once("A5 MegaMoE backend initialized for eligible fused routed experts.")

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
                    f"A5 MegaMoE requires contiguous {name}, got "
                    f"shape={tuple(values[0].shape)}, stride={values[0].stride()}."
                )

        weight1, weight2 = w1[0], w2[0]
        scale1, scale2 = w1_scale[0], w2_scale[0]
        if weight1.element_size() != 1 or weight2.element_size() != 1:
            raise ValueError(
                "A5 MegaMoE requires packed FP4 weights backed by one-byte storage, "
                f"got w1={weight1.dtype}, w2={weight2.dtype}."
            )
        if weight1.ndim != 3 or weight2.ndim != 3:
            raise ValueError(
                "A5 MegaMoE expects stacked 3D expert weights, "
                f"got w1={tuple(weight1.shape)}, w2={tuple(weight2.shape)}."
            )
        if scale1.ndim != 4 or scale2.ndim != 4:
            raise ValueError(
                "A5 MegaMoE expects stacked 4D MXFP scales, "
                f"got w1_scale={tuple(scale1.shape)}, w2_scale={tuple(scale2.shape)}."
            )

        hidden_states = fused_experts_input.hidden_states
        if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
            raise ValueError(
                "A5 MegaMoE requires contiguous 2D hidden states, "
                f"got shape={tuple(hidden_states.shape)}, stride={hidden_states.stride()}."
            )
        hidden = int(hidden_states.shape[-1])
        projected_hidden = int(weight1.shape[1])
        if projected_hidden % 2 != 0:
            raise ValueError(f"A5 MegaMoE expects w1.shape[1] to be even, got {projected_hidden}.")
        intermediate = projected_hidden // 2
        if hidden % _MXFP_SCALE_BLOCK_SIZE or intermediate % _MXFP_SCALE_BLOCK_SIZE:
            raise ValueError(
                "A5 MegaMoE requires hidden and intermediate dimensions to be "
                f"multiples of {_MXFP_SCALE_BLOCK_SIZE}, got hidden={hidden}, intermediate={intermediate}."
            )

        num_local_experts = int(weight1.shape[0])
        configured_local_experts = getattr(self.moe_config, "num_local_experts", None)
        if configured_local_experts is not None and num_local_experts != configured_local_experts:
            raise ValueError(
                "A5 MegaMoE local expert count does not match the MoE config: "
                f"weights={num_local_experts}, configured={configured_local_experts}."
            )
        expected_shapes = {
            "w1": (num_local_experts, projected_hidden, hidden // _FP4_PACK_FACTOR),
            "w2": (num_local_experts, hidden, intermediate // _FP4_PACK_FACTOR),
            "w1_scale": (
                num_local_experts,
                projected_hidden,
                hidden // _MXFP_SCALE_BLOCK_SIZE,
                _MXFP_SCALE_MULTIPLIER,
            ),
            "w2_scale": (
                num_local_experts,
                hidden,
                intermediate // _MXFP_SCALE_BLOCK_SIZE,
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
        return projected_hidden

    @staticmethod
    def _resolve_activation(
        activation: Any,
        swiglu_limit: float,
    ) -> tuple[str, dict[str, float | None] | None, float | None]:
        if isinstance(activation, SituActivationConfig):
            return (
                "situglu",
                {
                    "beta": activation.beta,
                    "linear_beta": activation.linear_beta,
                },
                None,
            )

        activation_name = activation if isinstance(activation, str) else getattr(activation, "value", None)
        if not isinstance(activation_name, str):
            activation_name = getattr(activation, "name", str(activation))
        normalized = activation_name.lower().removeprefix("moeactivation.")
        if normalized not in {"silu", "swiglu"}:
            raise ValueError(f"A5 MegaMoE does not support activation={activation!r} without changing its semantics.")
        activation_clamp = float(swiglu_limit) if swiglu_limit > 0 else None
        return "swiglu", None, activation_clamp

    def _validate_runtime_contract(self, fused_experts_input: MoEFusedExpertsInput) -> None:
        if fused_experts_input.quant.quant_type != _A5_MEGA_MOE_QUANT_TYPE:
            raise RuntimeError(
                "A5 MegaMoE currently supports only W4A8MXFP routed experts, "
                f"got {fused_experts_input.quant.quant_type}."
            )
        mxfp = fused_experts_input.quant.mxfp
        if mxfp is None:
            raise RuntimeError("A5 MegaMoE requires explicit MXFP runtime parameters.")
        if mxfp.group_size != _A5_MEGA_MOE_GROUP_SIZE:
            raise RuntimeError(f"A5 MegaMoE requires MXFP group_size={_A5_MEGA_MOE_GROUP_SIZE}, got {mxfp.group_size}.")
        if FLOAT4_E2M1FN_X2_DTYPE is None or mxfp.weight_quant_type != FLOAT4_E2M1FN_X2_DTYPE:
            raise RuntimeError(
                f"A5 MegaMoE requires packed float4_e2m1fn_x2 expert weights, got {mxfp.weight_quant_type}."
            )
        if fused_experts_input.dynamic_eplb:
            raise RuntimeError("A5 MegaMoE does not support dynamic EPLB.")
        if fused_experts_input.routing.global_redundant_expert_num:
            raise RuntimeError("A5 MegaMoE does not support redundant physical experts.")
        if fused_experts_input.lora_context is not None:
            raise RuntimeError("A5 MegaMoE does not support MoE LoRA.")
        if fused_experts_input.topk_ids.ndim != 2:
            raise ValueError(f"A5 MegaMoE requires 2D top-k tensors, got {tuple(fused_experts_input.topk_ids.shape)}.")
        if fused_experts_input.topk_ids.shape[1] != self.moe_config.experts_per_token:
            raise ValueError(
                "A5 MegaMoE top-k width does not match the MoE config: "
                f"ids={fused_experts_input.topk_ids.shape[1]}, "
                f"configured={self.moe_config.experts_per_token}."
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

    @staticmethod
    def _get_ep_ranks(mega_moe_group) -> tuple[int, ...]:
        ranks = getattr(mega_moe_group, "ranks", None)
        if ranks is None:
            ranks = range(mega_moe_group.world_size)
        return tuple(int(rank) for rank in ranks)

    def _make_buffer_key(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        projected_hidden: int,
        *,
        buffer_tokens_per_rank: int,
        mega_moe_group,
    ) -> MegaMoEBufferKey:
        num_tokens = int(fused_experts_input.hidden_states.shape[0])
        if num_tokens > buffer_tokens_per_rank:
            raise ValueError(
                "A5 MegaMoE input exceeds the symmetric buffer token capacity: "
                f"num_tokens={num_tokens}, capacity={buffer_tokens_per_rank}."
            )
        return MegaMoEBufferKey(
            ep_ranks=self._get_ep_ranks(mega_moe_group),
            num_experts=self.moe_config.num_experts,
            tokens_per_rank=buffer_tokens_per_rank,
            top_k=self.moe_config.experts_per_token,
            hidden=int(fused_experts_input.hidden_states.shape[-1]),
            intermediate_hidden=projected_hidden,
            dispatch_quant_mode=_DISPATCH_QUANT_MODE_MXFP8,
            dispatch_quant_out_dtype=torch.float8_e4m3fn,
        )

    def _get_sym_buffer(self, fused_experts_input: MoEFusedExpertsInput, projected_hidden: int):
        mega_moe_group = get_mega_moe_group()
        ascend_config = get_ascend_config()
        buffer_tokens_per_rank = get_a5_mega_moe_buffer_tokens_per_rank(ascend_config.vllm_config)
        key = self._make_buffer_key(
            fused_experts_input,
            projected_hidden,
            buffer_tokens_per_rank=buffer_tokens_per_rank,
            mega_moe_group=mega_moe_group,
        )
        state: _MegaMoESymmetricBufferState | None = getattr(
            mega_moe_group,
            _MEGA_MOE_BUFFER_STATE_ATTR,
            None,
        )
        if state is not None:
            if state.key != key:
                raise RuntimeError(
                    "A5 MegaMoE symmetric buffer is process-wide and immutable: "
                    f"initialized={state.key}, requested={key}."
                )
            logger.debug("A5 MegaMoE reuses the process-wide symmetric buffer: %s", key)
            return state.buffer

        try:
            capturing = bool(_EXTRA_CTX.capturing)
        except (AttributeError, AssertionError, RuntimeError):
            capturing = False
        if capturing:
            raise RuntimeError("A5 MegaMoE symmetric buffer must be initialized during warmup before ACLGraph capture.")

        logger.info_once("A5 MegaMoE creates the process-wide symmetric buffer: %s", key)
        buffer = self._get_symm_buffer_op(
            mega_moe_group.device_group,
            num_experts=key.num_experts,
            num_max_tokens_per_rank=key.tokens_per_rank,
            num_topk=key.top_k,
            hidden=key.hidden,
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

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_runtime_contract(fused_experts_input)

        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]
        topk_ids = topk_ids.to(torch.int32).contiguous()
        topk_weights = fused_experts_input.topk_weights.contiguous()

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
        activation, activation_params, activation_clamp = self._resolve_activation(
            fused_experts_input.activation,
            fused_experts_input.swiglu_limit,
        )
        mxfp = fused_experts_input.quant.mxfp
        assert mxfp is not None

        kwargs = {
            "x": fused_experts_input.hidden_states,
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "l1_weights": w1,
            "l1_weights_sf": w1_scale,
            "l2_weights": w2,
            "l2_weights_sf": w2_scale,
            "weight1_type": mxfp.weight_quant_type,
            "weight2_type": mxfp.weight_quant_type,
            "sym_buffer": self._get_sym_buffer(fused_experts_input, projected_hidden),
            "activation": activation,
            "activation_clamp": activation_clamp,
        }
        if activation_params is not None:
            kwargs["activation_params"] = activation_params
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "A5 MegaMoE call: x=%s, topk=%s, w1=%s, w2=%s, "
                "w1_scale=%s, w2_scale=%s, activation=%s, activation_params=%s.",
                tuple(fused_experts_input.hidden_states.shape),
                tuple(topk_ids.shape),
                tuple(w1[0].shape),
                tuple(w2[0].shape),
                tuple(w1_scale[0].shape),
                tuple(w2_scale[0].shape),
                activation,
                activation_params,
            )
        try:
            output, expert_tokens = self._mega_moe_op(**kwargs)
        except TypeError as exc:
            if activation_params is not None:
                raise RuntimeError(
                    "The installed CANN MegaMoE API does not accept SiTU "
                    "activation_params; install the Kimi K3 compatible ops-transformer build."
                ) from exc
            raise
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "A5 MegaMoE output: output=%s, expert_tokens=%s.",
                tuple(output.shape),
                None if expert_tokens is None else tuple(expert_tokens.shape),
            )
        return output, expert_tokens


__all__ = ["MegaMoEBackend", "MegaMoEBufferKey"]
