# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""Cross-layer refill loop for eager Qwen3-MoE token cutting."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
from vllm.sequence import IntermediateTensors

_original_qwen3_moe_forward = Qwen3MoeModel.forward


@contextmanager
def _moe_token_cut_mc2_context(forward_context, hidden_states: torch.Tensor):
    """Refresh valid rows without changing this forward's MC2 shard capacity."""
    num_tokens = hidden_states.shape[0]
    padded_num_tokens = forward_context.padded_num_tokens
    if num_tokens > padded_num_tokens:
        raise RuntimeError("MoE token-cut rows exceed the MC2 padding capacity")
    original_mask = forward_context.mc2_mask
    original_num_tokens = forward_context.num_tokens
    # Keep the padded capacity fixed across layers. Only the compacted prefix
    # is real; do not mutate the reserved mask or masks held by earlier layers.
    forward_context.mc2_mask = torch.arange(padded_num_tokens, device=hidden_states.device) < num_tokens
    forward_context.num_tokens = num_tokens
    try:
        yield
    finally:
        forward_context.mc2_mask = original_mask
        forward_context.num_tokens = original_num_tokens


@wraps(_original_qwen3_moe_forward)
def _qwen3_moe_forward_with_token_refill(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    forward_context = get_forward_context()
    token_cut_state = getattr(forward_context, "moe_token_cut_state", None)
    frame = getattr(token_cut_state, "model_graph_frame", None)
    if frame is not None:
        return frame.forward(self, forward_context, input_ids, positions)
    if token_cut_state is None or not token_cut_state.cross_layer_active:
        return _original_qwen3_moe_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)
    if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
        raise RuntimeError("MoE token-cut cross-layer refill does not support PP")
    is_dp = token_cut_state.data_parallel_size > 1
    mc2_available = getattr(getattr(forward_context, "moe_comm_type", None), "name", None) == "MC2"
    if is_dp and not mc2_available:
        # Another DP rank may be in a large prefill using AllToAll. Retain local
        # checkpoints until MC2 is available instead of re-routing saved rows.
        token_cut_state.allow_restore_this_forward = False

    hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    hidden_states, positions = token_cut_state.prepare_model_rows(hidden_states, positions)
    residual = None
    aux_hidden_states = self._maybe_add_hidden_state([], self.start_layer, hidden_states, residual)
    if aux_hidden_states:
        raise RuntimeError("MoE token-cut cross-layer refill does not support auxiliary hidden states")

    original_attn_metadata = forward_context.attn_metadata
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        is_sparse_layer = hasattr(getattr(layer, "mlp", None), "experts")
        if is_sparse_layer and token_cut_state.native_combined:
            token_cut_state.register_native_attention_source(
                layer_idx, original_attn_metadata, layer.self_attn.attn.layer_name
            )
        if hidden_states.shape[0] > 0:
            row_indices = token_cut_state.original_row_indices(positions.device)
            layer_name = layer.self_attn.attn.layer_name
            if is_sparse_layer:
                token_cut_state.capture_layer_attention_metadata(layer_idx, original_attn_metadata, layer_name)
            forward_context.attn_metadata = token_cut_state.project_attention_metadata(
                original_attn_metadata, layer_name, row_indices.device
            )
            try:
                if residual is None:
                    residual = hidden_states
                    hidden_states = layer.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = layer.input_layernorm(hidden_states, residual)
                hidden_states = layer.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                )
            finally:
                forward_context.attn_metadata = original_attn_metadata
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        elif residual is None:
            residual = hidden_states

        hidden_states, residual, positions = token_cut_state.restore_before_moe(
            layer_idx, hidden_states, residual, positions
        )
        if hidden_states.shape[0] == 0 and not (is_dp and is_sparse_layer):
            continue

        if is_sparse_layer:
            token_cut_state.set_layer_inputs(layer_idx, hidden_states, residual, positions)
        pre_moe_hidden_states = hidden_states
        model_row_count = hidden_states.shape[0]
        mlp_input = hidden_states
        if is_dp and is_sparse_layer and model_row_count == 0 and not mc2_available:
            # AllToAll has no active-row mask. Match the existing idle-worker
            # convention: send one dummy row, then discard its local output.
            mlp_input = hidden_states.new_zeros((1, hidden_states.shape[1]))
        with _moe_token_cut_mc2_context(forward_context, hidden_states):
            hidden_states = layer.mlp(mlp_input)[:model_row_count]
        if is_sparse_layer:
            hidden_states, residual, positions = token_cut_state.compact_after_moe(
                layer_idx,
                hidden_states,
                pre_moe_hidden_states,
                residual,
                positions,
            )
        self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)
        if aux_hidden_states:
            raise RuntimeError("MoE token-cut cross-layer refill does not support auxiliary hidden states")

    forward_context.attn_metadata = original_attn_metadata
    if hidden_states.shape[0] > 0:
        hidden_states, _ = self.norm(hidden_states, residual)
    hidden_states = token_cut_state.finalize_model_rows(hidden_states)
    return hidden_states


Qwen3MoeModel.forward = _qwen3_moe_forward_with_token_refill
