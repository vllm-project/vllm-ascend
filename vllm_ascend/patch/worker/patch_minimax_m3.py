# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Adapt vLLM's native MiniMax-M3 model to the Ascend backend."""

import sys
from collections.abc import Iterable
from types import ModuleType
from typing import Any, cast

import torch
from transformers import PretrainedConfig
from vllm.config import CacheConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter


def _install_fused_allreduce_norm_fallback() -> None:
    """Avoid importing vLLM's FlashInfer-only fusion module on Ascend."""
    module_name = "vllm.model_executor.layers.fused_allreduce_gemma_rms_norm"
    if module_name in sys.modules:
        return

    fallback_module = ModuleType(module_name)

    def fused_allreduce_gemma_rms_norm(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm: GemmaRMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.distributed import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() > 1:
            hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)
        return norm(hidden_states, residual)

    cast(Any, fallback_module).fused_allreduce_gemma_rms_norm = fused_allreduce_gemma_rms_norm
    sys.modules[module_name] = fallback_module


_install_fused_allreduce_norm_fallback()

from vllm.models.minimax_m3.common.vision_tower import (  # noqa: E402
    MiniMaxVLAttention,
)
from vllm.models.minimax_m3.nvidia import model as minimax_m3_model  # noqa: E402

from vllm_ascend.attention.msa_m3 import (  # noqa: E402
    MiniMaxM3SparseAttention as AscendMiniMaxM3SparseAttentionBase,
)

_ORIGINAL_MINIMAX_M3_MAYBE_ADD_HIDDEN_STATE = (
    minimax_m3_model.MiniMaxM3Model._maybe_add_hidden_state
)


class AscendMiniMaxM3SparseAttention(AscendMiniMaxM3SparseAttentionBase):
    """Translate vLLM 0.24's MiniMax-M3 constructor to the Ascend backend."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: CacheConfig | None = None,
        topk_indices_buffer: Any | None = None,
    ) -> None:
        del topk_indices_buffer
        sparse_cfg = config.sparse_attention_config
        disable_index_value = sparse_cfg["sparse_disable_index_value"][layer_id] == 1
        rope_parameters = {
            "rope_theta": getattr(config, "rope_theta", 10000),
            "partial_rotary_factor": getattr(config, "partial_rotary_factor", 1.0),
        }
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=config.rotary_dim,
            rope_parameters=rope_parameters,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            sparse_cfg=sparse_cfg,
            disable_index_value=disable_index_value,
            # The native decoder performs the tensor-parallel all-reduce
            # together with the following Gemma RMSNorm.
            reduce_results=False,
        )


def _forward_minimax_m3_attention(
    self: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Run dense attention without vLLM's CUDA-only fused QK/RoPE op."""
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q_shape = q.shape
    k_shape = k.shape
    q = self.q_norm(q.reshape(-1, self.head_dim).contiguous()).reshape(q_shape)
    k = self.k_norm(k.reshape(-1, self.head_dim).contiguous()).reshape(k_shape)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v.contiguous())
    output, _ = self.o_proj(attn_output)
    return output


def _maybe_add_minimax_m3_hidden_state(
    self: Any,
    aux_hidden_states: list[torch.Tensor],
    layer_idx: int,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> list[torch.Tensor]:
    if (
        residual is not None
        and layer_idx in self.aux_hidden_state_layers
        and hidden_states.size(0) != residual.size(0)
    ):
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)
    return _ORIGINAL_MINIMAX_M3_MAYBE_ADD_HIDDEN_STATE(
        self, aux_hidden_states, layer_idx, hidden_states, residual
    )


def _apply_minimax_m3_vision_rotary_emb(
    self: Any,
    qk_reshaped: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    seq_len: int,
    rotary_segment_lengths: list[int] | None,
) -> torch.Tensor:
    """Apply MiniMax vision's partial RoPE with the Ascend rotary op."""
    del seq_len, rotary_segment_lengths
    rotary_dim = rotary_cos.shape[-1] * 2
    qk_rot = self.apply_rotary_emb(qk_reshaped[..., :rotary_dim], rotary_cos, rotary_sin)
    return torch.cat((qk_rot, qk_reshaped[..., rotary_dim:]), dim=-1)


def _load_minimax_m3_weights(
    self: Any,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load both fused and split Ascend sparse-indexer projections."""
    stacked_params_mapping: list[tuple[str, str, int | str]] = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        # W8A8 can require a separately quantized indexer projection. Try it
        # before the fused QKV-indexer layout used by the native CUDA model.
        (".indexer_proj", ".index_q_proj", "index_q"),
        (".indexer_proj", ".index_k_proj", "index_k"),
        (".qkv_proj", ".index_q_proj", "index_q"),
        (".qkv_proj", ".index_k_proj", "index_k"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    expert_params_mapping = self.get_expert_mapping()
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    for name, loaded_weight in weights:
        if name.startswith("model."):
            name = name[len("model.") :]
        if "mtp." in name or "rotary_emb.inv_freq" in name:
            continue
        if "weight_scale_inv" in name:
            name = name.replace("weight_scale_inv", "weight_scale")
        elif "scale_inv" in name:
            name = name.replace("scale_inv", "scale")

        if ".index_" in name and ".index_q_proj" not in name and ".index_k_proj" not in name:
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self) or name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if "block_sparse_moe.experts." in name and name not in params_dict:
                continue
            mapped_name = name.replace(weight_name, param_name)
            if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                continue
            if is_pp_missing_parameter(mapped_name, self):
                continue
            if mapped_name.endswith((".k_scale", ".v_scale")):
                remapped_name = maybe_remap_kv_scale_name(mapped_name, params_dict)
                if remapped_name is not None and remapped_name in params_dict:
                    param = params_dict[remapped_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(remapped_name)
                    break
            if mapped_name not in params_dict:
                continue
            param = params_dict[mapped_name]
            param.weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(mapped_name)
            break
        else:
            is_expert_weight = False
            for (
                param_name,
                weight_name,
                expert_id,
                shard_id,
            ) in expert_params_mapping:
                if weight_name not in name:
                    continue
                is_expert_weight = True
                mapped_name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped_name, self):
                    break
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                success = param.weight_loader(
                    param,
                    loaded_weight,
                    mapped_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(mapped_name)
                    break
            else:
                if is_expert_weight:
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self) or name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if getattr(weight_loader, "supports_moe_loading", False):
                    if loaded_weight.shape == param.shape:
                        default_weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                        continue
                    raise ValueError(
                        f"FusedMoE parameter {name!r} has incompatible "
                        f"checkpoint shape {tuple(loaded_weight.shape)} for "
                        f"parameter shape {tuple(param.shape)}"
                    )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

    return loaded_params


minimax_m3_model.MiniMAXGemmaRMSNorm = GemmaRMSNorm
minimax_m3_model.MiniMaxM3Attention.forward = _forward_minimax_m3_attention
minimax_m3_model.MiniMaxM3Model._maybe_add_hidden_state = (
    _maybe_add_minimax_m3_hidden_state
)
minimax_m3_model.MiniMaxM3SparseAttention = AscendMiniMaxM3SparseAttention
minimax_m3_model.MiniMaxM3Model.load_weights = _load_minimax_m3_weights
MiniMaxVLAttention._apply_rotary_emb = _apply_minimax_m3_vision_rotary_emb