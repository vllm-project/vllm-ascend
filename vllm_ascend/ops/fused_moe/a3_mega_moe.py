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

import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.ascend_config import A3_MEGA_MOE_TOKENS_PER_RANK_LIMIT, get_ascend_config
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig

    from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEFusedExpertsInput

_A3_MEGA_MOE_SUPPORTED_EP_SIZES = frozenset({2, 4, 8, 16, 32, 48, 64, 96, 128})
_A3_MEGA_MOE_HIDDEN_SIZE_MIN = 1024
_A3_MEGA_MOE_HIDDEN_SIZE_MAX = 8192
_A3_MEGA_MOE_INTERMEDIATE_SIZE_MIN = 512
_A3_MEGA_MOE_INTERMEDIATE_SIZE_MAX = 3072
_A3_MEGA_MOE_TILE_SIZE = 512
_A3_MEGA_MOE_TOP_K_LIMIT = 16
_A3_MEGA_MOE_EXPERTS_LIMIT = 1024
_A3_MEGA_MOE_EXPERTS_PER_RANK_LIMIT = 128
_A3_MEGA_MOE_SUPPORTED_QUANT_NAMES = {
    "w8a8",
    "w4a8",
    "w8a8_dynamic",
    "w4a8_dynamic",
    "quanttype.w8a8",
    "quanttype.w4a8",
}

_CANN_MEGA_MOE_QUANT_MODE_NONE = 0
_CANN_MEGA_MOE_QUANT_MODE_INT8 = 2


@lru_cache(maxsize=1)
def is_a3_mega_moe_package_available() -> bool:
    try:
        return importlib.util.find_spec("cann_ops_transformer.ops") is not None
    except (ImportError, ValueError):
        return False


def _get_model_dimension(vllm_config: Any, name: str) -> int | None:
    model_config = getattr(vllm_config, "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    value = getattr(hf_text_config, name, None)
    if value is None and name == "hidden_size" and hasattr(model_config, "get_hidden_size"):
        value = model_config.get_hidden_size()
    return None if value is None else int(value)


def _get_num_experts(vllm_config: Any) -> int | None:
    model_config = getattr(vllm_config, "model_config", None)
    get_num_experts = getattr(model_config, "get_num_experts", None)
    if callable(get_num_experts):
        num_experts = get_num_experts()
        if num_experts is not None:
            return int(num_experts)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    num_experts = getattr(hf_text_config, "num_experts", getattr(hf_text_config, "n_routed_experts", None))
    return None if num_experts is None else int(num_experts)


def _is_a3_mega_moe_model_config_supported(vllm_config: Any, ep_size: int) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    if getattr(model_config, "dtype", None) != torch.bfloat16:
        return False

    hidden_size = _get_model_dimension(vllm_config, "hidden_size")
    if (
        hidden_size is None
        or hidden_size < _A3_MEGA_MOE_HIDDEN_SIZE_MIN
        or hidden_size > _A3_MEGA_MOE_HIDDEN_SIZE_MAX
        or hidden_size % _A3_MEGA_MOE_TILE_SIZE != 0
    ):
        return False

    intermediate_size = _get_model_dimension(vllm_config, "moe_intermediate_size")
    if (
        intermediate_size is None
        or intermediate_size < _A3_MEGA_MOE_INTERMEDIATE_SIZE_MIN
        or intermediate_size > _A3_MEGA_MOE_INTERMEDIATE_SIZE_MAX
        or intermediate_size % _A3_MEGA_MOE_TILE_SIZE != 0
    ):
        return False

    hf_text_config = vllm_config.model_config.hf_text_config
    top_k = getattr(hf_text_config, "num_experts_per_tok", getattr(hf_text_config, "top_k_experts", None))
    if top_k is None or not 1 <= int(top_k) <= _A3_MEGA_MOE_TOP_K_LIMIT:
        return False

    num_experts = _get_num_experts(vllm_config)
    if (
        num_experts is None
        or not ep_size <= num_experts <= _A3_MEGA_MOE_EXPERTS_LIMIT
        or num_experts % ep_size != 0
        or not 1 <= num_experts // ep_size <= _A3_MEGA_MOE_EXPERTS_PER_RANK_LIMIT
    ):
        return False

    activation = getattr(hf_text_config, "hidden_act", getattr(hf_text_config, "hidden_activation", "silu"))
    activation_name = str(getattr(activation, "name", activation)).lower().removeprefix("moeactivation.")
    if activation_name not in {"silu", "swiglu", "swiglustep"}:
        return False

    quant_config = getattr(vllm_config, "quant_config", None)
    quant_description = getattr(quant_config, "quant_description", None)
    quant_type = getattr(hf_text_config, "moe_quantize", getattr(hf_text_config, "quantize", None))
    if quant_type is None and isinstance(quant_description, dict):
        for key in ("moe_quantize", "quantize", "moe_quant_type", "quant_type", "model_quant_type"):
            quant_type = quant_description.get(key)
            if quant_type is not None:
                break
    if quant_type is None:
        return quant_config is None
    quant_name = str(getattr(quant_type, "name", quant_type)).lower()
    if quant_name in {"w4a8", "w4a8_dynamic", "quanttype.w4a8"}:
        group_size = quant_description.get("group_size") if isinstance(quant_description, dict) else None
        if group_size != 0:
            return False
    return quant_name in _A3_MEGA_MOE_SUPPORTED_QUANT_NAMES


def is_a3_mega_moe_enabled(vllm_config: Any | None = None) -> bool:
    if get_ascend_device_type() != AscendDeviceType.A3 or not is_a3_mega_moe_package_available():
        return False

    ascend_config = get_ascend_config()
    if ascend_config.enable_fused_mc2 != 1:
        return False
    if vllm_config is None:
        vllm_config = getattr(ascend_config, "vllm_config", None)
    if vllm_config is None:
        return False

    parallel_config = vllm_config.parallel_config
    if not parallel_config.enable_expert_parallel:
        return False
    ep_size = parallel_config.world_size_across_dp // parallel_config.pipeline_parallel_size
    if ep_size not in _A3_MEGA_MOE_SUPPORTED_EP_SIZES:
        return False

    eplb_config = getattr(ascend_config, "eplb_config", None)
    if bool(getattr(eplb_config, "dynamic_eplb", False)):
        return False
    if int(getattr(eplb_config, "num_redundant_experts", 0)) != 0:
        return False
    if bool(getattr(ascend_config, "mix_placement", False)):
        return False
    return _is_a3_mega_moe_model_config_supported(vllm_config, ep_size)


def _get_a3_mega_moe_ops():
    try:
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
    except ImportError as exc:
        raise RuntimeError(
            "A3 MegaMoE requires cann_ops_transformer.ops. Install a CANN ops-transformer "
            "package that provides mega_moe and get_symm_buffer_for_mega_moe."
        ) from exc
    return get_symm_buffer_for_mega_moe, mega_moe


def _get_a3_mega_moe_quant_settings(quant_type: QuantType) -> tuple[int, torch.dtype | None]:
    if quant_type in {QuantType.W8A8, QuantType.W4A8}:
        return _CANN_MEGA_MOE_QUANT_MODE_INT8, torch.int8
    if quant_type == QuantType.NONE:
        return _CANN_MEGA_MOE_QUANT_MODE_NONE, None
    raise RuntimeError(f"A3 MegaMoE supports BF16, W8A8, and W4A8 MoE weights, got {quant_type}.")


def _as_tensor_list(
    value: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...], name: str
) -> list[torch.Tensor]:
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"{name} cannot be empty for A3 MegaMoE.")
    return values


class A3MegaMoEBackend:
    """A3-only wrapper around the CANN MegaMoE operator."""

    def __init__(self, moe_config: FusedMoEConfig, token_dispatcher: Any):
        self.moe_config = moe_config
        self.token_dispatcher = token_dispatcher
        self._symmetric_buffer = None
        self._get_symmetric_buffer, self._mega_moe = _get_a3_mega_moe_ops()

    def _init_symmetric_buffer(
        self,
        dispatch_quant_mode: int,
        dispatch_quant_out_dtype: int | None,
    ):
        from vllm_ascend.distributed.parallel_state import get_mc2_group

        ep_world_size = int(self.token_dispatcher.ep_world_size)
        buffer_tokens_per_rank = max(1, int(self.token_dispatcher.max_num_tokens_per_rank))
        top_k = self.moe_config.experts_per_token
        experts_per_rank = max(1, self.moe_config.num_experts // ep_world_size)
        max_recv_token_num = buffer_tokens_per_rank * ep_world_size * min(top_k, experts_per_rank)
        return self._get_symmetric_buffer(
            get_mc2_group().device_group,
            self.moe_config.num_experts,
            buffer_tokens_per_rank,
            top_k,
            hidden=self.moe_config.hidden_dim,
            intermediate_hidden=self.moe_config.intermediate_size_per_partition,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=dispatch_quant_mode,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        )

    def fused_experts(self, fused_experts_input: MoEFusedExpertsInput) -> tuple[torch.Tensor, torch.Tensor | None]:
        if fused_experts_input.dynamic_eplb or fused_experts_input.routing.global_redundant_expert_num:
            raise RuntimeError("A3 MegaMoE does not support dynamic or redundant EPLB experts.")
        if self.token_dispatcher.global_bs != 0:
            raise RuntimeError("A3 MegaMoE requires uniform token counts across participating ranks.")
        if fused_experts_input.hidden_states.shape[0] > self.token_dispatcher.max_num_tokens_per_rank:
            raise ValueError(
                "A3 MegaMoE input exceeds its symmetric buffer capacity: "
                f"num_tokens={fused_experts_input.hidden_states.shape[0]}, "
                f"capacity={self.token_dispatcher.max_num_tokens_per_rank}."
            )

        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        dispatch_quant_mode, dispatch_quant_out_dtype = _get_a3_mega_moe_quant_settings(
            fused_experts_input.quant.quant_type
        )
        if self._symmetric_buffer is None:
            self._symmetric_buffer = self._init_symmetric_buffer(
                dispatch_quant_mode,
                dispatch_quant_out_dtype,
            )
        else:
            self._symmetric_buffer.dispatch_quant_mode = dispatch_quant_mode
            self._symmetric_buffer.dispatch_quant_out_dtype = dispatch_quant_out_dtype

        active_mask = None
        raw_mask = fused_experts_input.routing.mc2_mask
        if raw_mask is not None:
            active_mask = (
                raw_mask.to(torch.int8).contiguous() if raw_mask.dtype != torch.int8 else raw_mask.contiguous()
            )
            if active_mask.shape != (fused_experts_input.hidden_states.shape[0],):
                raise ValueError(
                    "A3 MegaMoE active mask must match the local token count: "
                    f"mask_shape={tuple(active_mask.shape)}, num_tokens={fused_experts_input.hidden_states.shape[0]}."
                )

        activation_name = str(fused_experts_input.activation).lower().removeprefix("moeactivation.")
        activation = "swiglu" if activation_name == "silu" else activation_name
        if activation not in {"swiglu", "swiglustep"}:
            raise RuntimeError(f"A3 MegaMoE does not support activation {fused_experts_input.activation!r}.")
        activation_clamp = fused_experts_input.swiglu_limit if fused_experts_input.swiglu_limit > 0 else None
        output, expert_tokens = self._mega_moe(
            fused_experts_input.hidden_states,
            topk_ids.to(torch.int32),
            fused_experts_input.topk_weights.to(torch.float32),
            _as_tensor_list(fused_experts_input.weights.w1, "w1"),
            _as_tensor_list(fused_experts_input.weights.w2, "w2"),
            self._symmetric_buffer,
            l1_weights_sf=fused_experts_input.weights.w1_scale,
            l2_weights_sf=fused_experts_input.weights.w2_scale,
            l1_bias=fused_experts_input.weights.w1_scale_bias,
            l2_bias=fused_experts_input.weights.w2_scale_bias,
            x_active_mask=active_mask,
            activation=activation,
            activation_clamp=activation_clamp,
        )
        return output, expert_tokens


__all__ = [
    "A3_MEGA_MOE_TOKENS_PER_RANK_LIMIT",
    "A3MegaMoEBackend",
    "is_a3_mega_moe_enabled",
    "is_a3_mega_moe_package_available",
]
