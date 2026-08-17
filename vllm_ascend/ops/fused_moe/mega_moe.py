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

import torch
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEFusedExpertsInput
from vllm_ascend.quantization.quant_type import QuantType


_MEGA_MOE_SUPPORTED_QUANTS = {
    QuantType.MXFP8,
    QuantType.MXFP4,
    QuantType.W4A8MXFP,
}


def _as_tensor_list(tensor_or_list: torch.Tensor | list[torch.Tensor] | None, name: str) -> list[torch.Tensor]:
    if tensor_or_list is None:
        raise ValueError(f"{name} is required for A5 MegaMoE.")
    if isinstance(tensor_or_list, list):
        if not tensor_or_list:
            raise ValueError(f"{name} cannot be an empty list for A5 MegaMoE.")
        return tensor_or_list
    return [tensor_or_list]


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
    max_tokens_per_rank: int
    top_k: int
    hidden_size: int
    intermediate_hidden: int
    dispatch_quant_mode: int
    dispatch_quant_out_dtype: torch.dtype


class MegaMoEBackend:
    """A5 MegaMoE wrapper for the logical FUSED_MC2 MoE path."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config
        self._sym_buffer = None
        self._sym_buffer_key: _MegaMoEBufferKey | None = None

    def _make_buffer_key(self, fused_experts_input: MoEFusedExpertsInput) -> _MegaMoEBufferKey:
        hidden_size = fused_experts_input.hidden_states.shape[-1]
        w1 = _as_tensor_list(fused_experts_input.weights.w1, "w1")[0]
        if w1.ndim != 3:
            raise ValueError(f"A5 MegaMoE expects expert weight w1 to be 3D, got shape {tuple(w1.shape)}.")
        if w1.shape[-1] != hidden_size:
            raise ValueError(
                f"A5 MegaMoE expects w1 shape (num_experts_per_rank, 2 * intermediate_hidden, hidden), "
                f"got w1 shape {tuple(w1.shape)} and hidden size {hidden_size}."
            )
        projected_hidden = int(w1.shape[-2])
        if projected_hidden % 2 != 0:
            raise ValueError(f"A5 MegaMoE expects w1.shape[-2] to be even, got {projected_hidden}.")
        intermediate_hidden = projected_hidden // 2
        max_tokens_per_rank = get_ascend_config().mega_moe_max_tokens
        act_dtype = torch.float8_e4m3fn
        if fused_experts_input.quant.mxfp is not None and fused_experts_input.quant.mxfp.act_quant_type is not None:
            act_dtype = fused_experts_input.quant.mxfp.act_quant_type
        return _MegaMoEBufferKey(
            num_experts=self.moe_config.num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            top_k=self.moe_config.experts_per_token,
            hidden_size=hidden_size,
            intermediate_hidden=intermediate_hidden,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=act_dtype,
        )

    def _get_sym_buffer(self, fused_experts_input: MoEFusedExpertsInput):
        key = self._make_buffer_key(fused_experts_input)
        if self._sym_buffer is not None and self._sym_buffer_key == key:
            logger.info("A5 MegaMoE reuses sym buffer: %s", key)
            return self._sym_buffer

        get_symm_buffer_for_mega_moe, _ = _get_mega_moe_ops()
        logger.info("A5 MegaMoE creates sym buffer: %s", key)
        self._sym_buffer = get_symm_buffer_for_mega_moe(
            get_ep_group().device_group,
            num_experts=key.num_experts,
            num_max_tokens_per_rank=key.max_tokens_per_rank,
            num_topk=key.top_k,
            hidden=key.hidden_size,
            intermediate_hidden=key.intermediate_hidden,
            dispatch_quant_mode=key.dispatch_quant_mode,
            dispatch_quant_out_dtype=key.dispatch_quant_out_dtype,
        )
        self._sym_buffer_key = key
        return self._sym_buffer

    @staticmethod
    def _normalize_activation(activation: str) -> str:
        activation_lower = activation.lower()
        if activation_lower in ("silu", "swiglu"):
            return "swiglu"
        if activation_lower in ("situ", "situglu"):
            return "situglu"
        raise ValueError(f"A5 MegaMoE does not support activation={activation!r}.")

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

        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        w1 = _as_tensor_list(fused_experts_input.weights.w1, "w1")
        w2 = _as_tensor_list(fused_experts_input.weights.w2, "w2")
        w1_scale = _as_tensor_list(fused_experts_input.weights.w1_scale, "w1_scale")
        w2_scale = _as_tensor_list(fused_experts_input.weights.w2_scale, "w2_scale")
        activation = self._normalize_activation(fused_experts_input.activation)
        activation_clamp = self._get_activation_clamp(fused_experts_input.swiglu_limit)

        _, mega_moe = _get_mega_moe_ops()
        logger.info(
            "A5 MegaMoE call: hidden_states_shape=%s, topk_ids_shape=%s, topk_weights_shape=%s, "
            "w1_shapes=%s, w2_shapes=%s, w1_scale_shapes=%s, w2_scale_shapes=%s, quant_type=%s, "
            "activation=%s, activation_clamp=%s, has_log2phy=%s, dynamic_eplb=%s",
            tuple(fused_experts_input.hidden_states.shape),
            tuple(topk_ids.shape),
            tuple(fused_experts_input.topk_weights.shape),
            [tuple(weight.shape) for weight in w1],
            [tuple(weight.shape) for weight in w2],
            [tuple(scale.shape) for scale in w1_scale],
            [tuple(scale.shape) for scale in w2_scale],
            fused_experts_input.quant.quant_type,
            activation,
            activation_clamp,
            fused_experts_input.routing.log2phy is not None,
            fused_experts_input.dynamic_eplb,
        )
        output, expert_tokens = mega_moe(
            x=fused_experts_input.hidden_states,
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=fused_experts_input.topk_weights,
            l1_weights=w1,
            l1_weights_sf=w1_scale,
            l2_weights=w2,
            l2_weights_sf=w2_scale,
            sym_buffer=self._get_sym_buffer(fused_experts_input),
            activation=activation,
            activation_clamp=activation_clamp,
        )
        logger.info(
            "A5 MegaMoE output: output_shape=%s, expert_tokens_shape=%s",
            tuple(output.shape),
            None if expert_tokens is None else tuple(expert_tokens.shape),
        )
        return output, expert_tokens


__all__ = ["MegaMoEBackend"]
