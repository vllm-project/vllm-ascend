# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Backport Qwen3 MoE's model-level SP until vllm-project/vllm#57337 is available."""

from functools import wraps
from itertools import islice

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeModel, Qwen3MoeSparseMoeBlock
from vllm.model_executor.models.utils import sequence_parallel_chunk


def _should_use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_text_config
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.use_sequence_parallel_moe
        and parallel_config.pipeline_parallel_size == 1
        and not vllm_config.model_config.is_multimodal_model
        and getattr(config, "num_experts", 0) > 0
        and not getattr(config, "mlp_only_layers", [])
        and getattr(config, "decoder_sparse_step", 1) == 1
    )


def _use_sequence_parallel(self) -> bool:
    return getattr(self.layers[self.start_layer], "use_attn_reduce_scatter_for_moe", False)


def _model_forward_sp(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    # Only PP=1 is eligible; multimodal subclasses retain their original layout.
    hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    full_num_tokens = positions.shape[-1]
    hidden_states = sequence_parallel_chunk(hidden_states)
    residual = None
    aux_hidden_states = self._maybe_add_hidden_state([], self.start_layer, hidden_states, residual)
    for layer_idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer), start=self.start_layer):
        hidden_states, residual = layer(positions, hidden_states, residual)
        self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)
    hidden_states, _ = self.norm(hidden_states, residual)

    hidden_size = hidden_states.shape[-1]
    if aux_hidden_states:
        hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
    hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)[:full_num_tokens]
    if aux_hidden_states:
        hidden_states, *aux_hidden_states = hidden_states.split(hidden_size, dim=-1)
        return hidden_states, aux_hidden_states
    return hidden_states


def _apply_patch() -> None:
    # Also makes repeated application a no-op. Native upstream SP owns all three
    # boundaries; mixing its model forward with this backport would double-shard.
    if hasattr(Qwen3MoeModel, "use_sequence_parallel"):
        return

    original_init = Qwen3MoeDecoderLayer.__init__
    original_decoder_forward = Qwen3MoeDecoderLayer.forward
    original_moe_forward = Qwen3MoeSparseMoeBlock.forward
    original_model_forward = Qwen3MoeModel.forward

    @wraps(original_init)
    def decoder_init(self, vllm_config, prefix="", is_fused_checkpoint_transposed=False):
        original_init(self, vllm_config, prefix, is_fused_checkpoint_transposed)
        self.use_attn_reduce_scatter_for_moe = _should_use_sequence_parallel(vllm_config)
        # Fine-grained o_proj custom ops own a different communication layout.
        # Only the standard RowParallelLinear reads reduce_results at forward.
        if getattr(self.self_attn.o_proj, "custom_op", None) is not None:
            self.use_attn_reduce_scatter_for_moe = False
        if self.use_attn_reduce_scatter_for_moe:
            self.self_attn.o_proj.reduce_results = False

    @wraps(original_decoder_forward)
    def decoder_forward(self, positions, hidden_states, residual):
        if not getattr(self, "use_attn_reduce_scatter_for_moe", False):
            return original_decoder_forward(self, positions, hidden_states, residual)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)[: positions.shape[-1]]
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        sp_pad = (-hidden_states.shape[0]) % get_tensor_model_parallel_world_size()
        hidden_states = F.pad(hidden_states, (0, 0, 0, sp_pad))
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        return hidden_states, residual

    def moe_forward(self, hidden_states, already_sequence_parallel=False):
        if not already_sequence_parallel:
            return original_moe_forward(self, hidden_states)
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)

    @wraps(original_model_forward)
    def model_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if not self.use_sequence_parallel:
            return original_model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        return _model_forward_sp(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    Qwen3MoeDecoderLayer.__init__ = decoder_init
    Qwen3MoeDecoderLayer.forward = decoder_forward
    Qwen3MoeSparseMoeBlock.forward = moe_forward
    Qwen3MoeModel.use_sequence_parallel = property(_use_sequence_parallel)
    Qwen3MoeModel.forward = model_forward


_apply_patch()
