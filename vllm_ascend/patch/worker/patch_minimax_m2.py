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
# MiniMax-M2 on Ascend: MoE all_reduce, k_norm weight sharding, fp8 load dequant.
#

from collections.abc import Iterable

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.mamba.linear_attn import MiniMaxText01RMSNormTP
from vllm.model_executor.models.minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2ForCausalLM,
    MiniMaxM2Model,
    MiniMaxM2MoE,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice

FP8_DTYPES = tuple(
    getattr(torch, dtype_name)
    for dtype_name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, dtype_name)
)


# ---------------------------------------------------------------------------
# MiniMaxM2MoE.forward: use maybe_all_reduce_tensor_model_parallel
# ---------------------------------------------------------------------------
def _patched_moe_forward(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # router_logits: (num_tokens, n_experts)
    router_logits, _ = self.gate(hidden_states.to(torch.float32))
    final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)
    return final_hidden_states.view(num_tokens, hidden_dim)


MiniMaxM2MoE.forward = _patched_moe_forward


# ---------------------------------------------------------------------------
# MiniMaxM2Attention: num_kv_head_replicas and k_norm weight sharding
# ---------------------------------------------------------------------------
_original_attention_init = MiniMaxM2Attention.__init__


def _patched_attention_init(self, *args, **kwargs) -> None:
    _original_attention_init(self, *args, **kwargs)
    tp_size = get_tensor_model_parallel_world_size()
    self.num_kv_head_replicas = max(1, tp_size // self.total_num_kv_heads)
    if self.total_num_kv_heads < tp_size:
        rms_norm_eps = getattr(getattr(self, "q_norm", None), "variance_epsilon", 1e-6)
        self.k_norm = MiniMaxText01RMSNormTP(
            self.head_dim * self.total_num_kv_heads,
            eps=rms_norm_eps,
            weight_shard_world_size=self.total_num_kv_heads,
            weight_shard_rank=get_tensor_model_parallel_rank() // self.num_kv_head_replicas,
        )


MiniMaxM2Attention.__init__ = _patched_attention_init


def _patch_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    cos, sin = get_cos_and_sin_slice()
    q, k, v = torch.ops.vllm.split_qkv_tp_rmsnorm_rope(
        input=qkv,
        q_weight=self.q_norm.weight,
        k_weight=self.k_norm.weight,
        q_hidden_size=self.q_size,
        kv_hidden_size=self.kv_size,
        head_dim=self.head_dim,
        rotary_dim=getattr(self.rotary_emb, "rotary_dim", self.head_dim),
        eps=self.q_norm.variance_epsilon,
        tp_world=self.q_norm.tp_world,
        cos=cos,
        sin=sin,
    )
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


MiniMaxM2Attention.forward = _patch_forward


# ---------------------------------------------------------------------------
# MiniMaxM2Model: fp8 dequant helpers and load_weights wrapper
# ---------------------------------------------------------------------------
def _need_dequantize_fp8_weights(self) -> bool:
    quant_cfg = getattr(self.config, "quantization_config", None)
    return (
        isinstance(quant_cfg, dict) and quant_cfg.get("quant_method") == "fp8" and current_platform.device_name == "npu"
    )


def _dequantize_fp8_block_weight(
    fp8_weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    block_n, block_k = block_size
    n, k = fp8_weight.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    if tuple(weight_scale_inv.shape) != (n_tiles, k_tiles):
        raise ValueError(
            "Unexpected fp8 scale shape: "
            f"weight={tuple(fp8_weight.shape)}, "
            f"scale={tuple(weight_scale_inv.shape)}, "
            f"block_size={block_size}"
        )
    expanded_scale = weight_scale_inv.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1)
    expanded_scale = expanded_scale[:n, :k].to(dtype=torch.bfloat16)
    return fp8_weight.to(dtype=torch.bfloat16) * expanded_scale


def _fp8_dequant_weight_iter(
    self: "MiniMaxM2Model",
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    quant_cfg = getattr(self.config, "quantization_config", {})
    block_cfg = quant_cfg.get("weight_block_size", [128, 128])
    weight_block_size: tuple[int, int] = (128, 128)
    if isinstance(block_cfg, list) and len(block_cfg) == 2:
        weight_block_size = (int(block_cfg[0]), int(block_cfg[1]))

    pending_fp8_weights: dict[str, torch.Tensor] = {}
    pending_fp8_scales: dict[str, torch.Tensor] = {}

    for name, loaded_weight in weights:
        if name.endswith(".weight_scale_inv"):
            paired_weight_name = name[: -len("_scale_inv")]
            pending_weight = pending_fp8_weights.pop(paired_weight_name, None)
            if pending_weight is None:
                pending_fp8_scales[name] = loaded_weight
                continue
            loaded_weight = self._dequantize_fp8_block_weight(pending_weight, loaded_weight, weight_block_size)
            name = paired_weight_name
        elif loaded_weight.dtype in FP8_DTYPES and name.endswith(".weight"):
            scale_name = f"{name}_scale_inv"
            pending_scale = pending_fp8_scales.pop(scale_name, None)
            if pending_scale is None:
                pending_fp8_weights[name] = loaded_weight
                continue
            loaded_weight = self._dequantize_fp8_block_weight(loaded_weight, pending_scale, weight_block_size)
        yield name, loaded_weight

    if pending_fp8_weights or pending_fp8_scales:
        raise ValueError(
            "Unpaired fp8 MiniMax-M2 weight/scale tensors detected: "
            f"pending_weights={len(pending_fp8_weights)}, "
            f"pending_scales={len(pending_fp8_scales)}"
        )


MiniMaxM2Model._need_dequantize_fp8_weights = _need_dequantize_fp8_weights
MiniMaxM2Model._dequantize_fp8_block_weight = staticmethod(_dequantize_fp8_block_weight)
MiniMaxM2Model._fp8_dequant_weight_iter = _fp8_dequant_weight_iter

_original_load_weights = MiniMaxM2Model.load_weights


def _patched_load_weights(
    self: "MiniMaxM2Model",
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    if self._need_dequantize_fp8_weights():
        weights = self._fp8_dequant_weight_iter(weights)
    return _original_load_weights(self, weights)


MiniMaxM2Model.load_weights = _patched_load_weights


# ---------------------------------------------------------------------------
# MiniMaxM2Model / MiniMaxM2ForCausalLM: Eagle3 aux hidden states support
# ---------------------------------------------------------------------------
_original_minimax_m2_forward = MiniMaxM2Model.forward


def _patched_minimax_m2_forward(
    self: "MiniMaxM2Model",
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    aux_layers: tuple[int, ...] = getattr(self, "aux_hidden_state_layers", ()) or ()
    if not aux_layers:
        return _original_minimax_m2_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states: list[torch.Tensor] = []
    for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
        layer_idx = self.start_layer + idx
        if layer_idx in aux_layers:
            aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
        hidden_states, residual = layer(positions, hidden_states, residual)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


if not getattr(_original_minimax_m2_forward, "_vllm_ascend_minimax_eagle3_patched", False):
    MiniMaxM2Model.forward = _patched_minimax_m2_forward  # type: ignore[assignment]
    MiniMaxM2Model.forward._vllm_ascend_minimax_eagle3_patched = True  # type: ignore[attr-defined]


def _set_aux_hidden_state_layers(self: "MiniMaxM2ForCausalLM", layers: tuple[int, ...]) -> None:
    self.model.aux_hidden_state_layers = tuple(int(x) for x in layers)


def _get_eagle3_default_aux_hidden_state_layers(self: "MiniMaxM2ForCausalLM") -> tuple[int, ...]:
    num_layers = len(self.model.layers)
    return (2, num_layers // 2, num_layers - 3)


def _get_eagle3_aux_hidden_state_layers(self: "MiniMaxM2ForCausalLM") -> tuple[int, ...]:
    return _get_eagle3_default_aux_hidden_state_layers(self)


# vLLM 0.18+: `supports_eagle3(model)` is `isinstance(model, SupportsEagle3)` (see
# `vllm.model_executor.models.interfaces`). `SupportsEagle3` extends `SupportsEagleBase`;
# runtime protocol checks require class attributes below (not only Eagle3 methods), or
# isinstance fails and model_runner_v1 raises:
# "Model does not support EAGLE3 interface but aux_hidden_state_outputs was requested".
MiniMaxM2ForCausalLM.has_own_lm_head = False  # type: ignore[misc]
MiniMaxM2ForCausalLM.has_own_embed_tokens = False  # type: ignore[misc]
MiniMaxM2ForCausalLM.supports_eagle3 = True  # type: ignore[misc]

MiniMaxM2ForCausalLM.set_aux_hidden_state_layers = _set_aux_hidden_state_layers  # type: ignore[attr-defined]
MiniMaxM2ForCausalLM.get_eagle3_default_aux_hidden_state_layers = (  # type: ignore[attr-defined]
    _get_eagle3_default_aux_hidden_state_layers
)
MiniMaxM2ForCausalLM.get_eagle3_aux_hidden_state_layers = _get_eagle3_aux_hidden_state_layers  # type: ignore[attr-defined]
