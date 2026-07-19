#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from inspect import signature

import torch

from vllm_ascend import envs

try:
    from vllm.model_executor.models.gemma4 import (
        Gemma4DecoderLayer,
        Gemma4Router,
    )
except ImportError:
    Gemma4DecoderLayer = None
    Gemma4Router = None


def _cached_to_dtype(module: torch.nn.Module, name: str, tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == x.dtype and tensor.device == x.device:
        return tensor

    cache_key = (
        x.device,
        x.dtype,
        tensor.data_ptr(),
        getattr(tensor, "_version", None),
    )
    key_name = f"_ascend_{name}_cache_key"
    cache_name = f"_ascend_{name}_cache"
    if getattr(module, key_name, None) != cache_key:
        setattr(module, cache_name, tensor.to(device=x.device, dtype=x.dtype))
        setattr(module, key_name, cache_key)
    return getattr(module, cache_name)


def _get_config(init_fn, args: tuple[object, ...], kwargs: dict[str, object]):
    try:
        bound = signature(init_fn).bind_partial(None, *args, **kwargs)
        return bound.arguments.get("config")
    except Exception:
        return args[0] if args else kwargs.get("config")


def _router_forward_topk(
    self: Gemma4Router,
    x: torch.Tensor,
    per_expert_scale: torch.Tensor,
    top_k: int,
    sync_base: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = x.shape[0]
    num_experts = self.proj.weight.shape[0]

    norm_scratch = getattr(self, "_dgemma_router_norm_scratch", None)
    if norm_scratch is None or norm_scratch.shape[0] < seq_len:
        norm_scratch = torch.empty((max(seq_len, 256), self.hidden_size), dtype=x.dtype, device=x.device)
        self._dgemma_router_norm_scratch = norm_scratch

    logits_scratch = getattr(self, "_dgemma_router_logits_scratch", None)
    if logits_scratch is None or logits_scratch.shape[0] < seq_len:
        logits_scratch = torch.empty((max(seq_len, 256), num_experts), dtype=torch.float32, device=x.device)
        self._dgemma_router_logits_scratch = logits_scratch

    sync_scratch = getattr(self, "_dgemma_router_sync_scratch", None)
    if sync_scratch is None:
        sync_scratch = torch.empty((128,), dtype=torch.int32, device=x.device)
        self._dgemma_router_sync_scratch = sync_scratch

    return torch.ops._C_ascend.npu_dgemma_fused_router_front(
        x.contiguous(),
        _cached_to_dtype(self, "scale", self.scale, x),
        self.proj.weight,
        norm_scratch[:seq_len],
        logits_scratch[:seq_len],
        per_expert_scale.to(torch.float32),
        sync_scratch,
        self.hidden_size,
        num_experts,
        top_k,
        sync_base,
        self.norm.variance_epsilon,
    )


if Gemma4Router is not None:

    def _forward(self: Gemma4Router, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x * _cached_to_dtype(self, "root_size", self.root_size, x)
        x = x * _cached_to_dtype(self, "scale", self.scale, x)
        router_logits, _ = self.proj(x)
        return router_logits

    Gemma4Router.forward = _forward
    Gemma4Router.forward_topk = _router_forward_topk


if Gemma4DecoderLayer is not None and not hasattr(Gemma4DecoderLayer, "_ascend_dgemma_original_forward"):
    Gemma4DecoderLayer._ascend_dgemma_original_init = Gemma4DecoderLayer.__init__
    Gemma4DecoderLayer._ascend_dgemma_original_forward = Gemma4DecoderLayer.forward

    def _decoder_init(self: Gemma4DecoderLayer, *args, **kwargs) -> None:
        original_init = self.__class__._ascend_dgemma_original_init
        config = _get_config(original_init, args, kwargs)
        original_init(self, *args, **kwargs)
        self._dgemma_router_sync_base = 1
        self._dgemma_top_k_experts = getattr(config, "top_k_experts", None)

    def _decoder_forward(
        self: Gemma4DecoderLayer,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            if envs.VLLM_ASCEND_DGEMMA_FUSE_ROUTER_FRONT_ASCENDC and residual.shape[0] == 1:
                top_k = self._dgemma_top_k_experts
                if top_k is None:
                    top_k = self.moe.experts.experts_per_token
                router_logits = self.router.forward_topk(
                    residual,
                    self.moe.per_expert_scale,
                    top_k,
                    self._dgemma_router_sync_base,
                )
            else:
                router_logits = self.router(residual)
            hidden_states_2 = self.moe(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            per_layer_contribution = self.post_per_layer_input_norm(per_layer_contribution)
            hidden_states = hidden_states + per_layer_contribution

        hidden_states = hidden_states * self.layer_scalar

        return hidden_states, residual

    Gemma4DecoderLayer.__init__ = _decoder_init
    Gemma4DecoderLayer.forward = _decoder_forward
