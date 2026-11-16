# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-shape adapter for the Kimi-K3 text tower; no training kernels.

Names and dimensions follow moonshotai/Kimi-K3's modeling_kimi_linear.py.
The served architecture is KimiLinearForCausalLM, so checkpoint names start at
model.* / lm_head.*, without the multimodal language_model.* wrapper. Keep the
three checkpoint convolution weights separate: the real loader packs them and
rebuilds Ascend's derived convolution tensor after each load.
"""

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn


def kimi_k3_parameter_shapes(config: Any) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Enumerate the unquantized checkpoint, including learned router state."""
    hidden = config.hidden_size
    linear = config.linear_attn_config
    projection = linear["num_heads"] * linear["head_dim"]
    yield "model.embed_tokens.weight", (config.vocab_size, hidden)
    yield "lm_head.weight", (config.vocab_size, hidden)
    yield "model.norm.weight", (hidden,)
    if config.attn_res_block_size is not None:
        yield "model.output_attn_res_norm.weight", (hidden,)
        yield "model.output_attn_res_proj.weight", (1, hidden)

    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        attention = prefix + "self_attn."
        if index + 1 in linear["kda_layers"]:
            for name in ("q_proj", "k_proj", "v_proj"):
                yield attention + name + ".weight", (projection, hidden)
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                yield attention + name + ".weight", (projection, 1, linear["short_conv_kernel_size"])
            yield attention + "A_log", (linear["num_heads"],)
            yield attention + "dt_bias", (projection,)
            yield attention + "f_a_proj.weight", (linear["head_dim"], hidden)
            yield attention + "f_b_proj.weight", (projection, linear["head_dim"])
            yield attention + "b_proj.weight", (linear["num_heads"], hidden)
            if linear["use_full_rank_gate"]:
                yield attention + "g_proj.weight", (projection, hidden)
            else:
                yield attention + "g_a_proj.weight", (linear["head_dim"], hidden)
                yield attention + "g_b_proj.weight", (projection, linear["head_dim"])
            yield attention + "o_norm.weight", (linear["head_dim"],)
            yield attention + "o_proj.weight", (hidden, projection)
        else:
            assert index + 1 in linear["full_attn_layers"], f"Unspecified Kimi attention layer: {index + 1}"
            q_head = config.qk_nope_head_dim + config.qk_rope_head_dim
            yield attention + "q_a_proj.weight", (config.q_lora_rank, hidden)
            yield attention + "q_a_layernorm.weight", (config.q_lora_rank,)
            yield attention + "q_b_proj.weight", (config.num_attention_heads * q_head, config.q_lora_rank)
            yield attention + "kv_a_proj_with_mqa.weight", (config.kv_lora_rank + config.qk_rope_head_dim, hidden)
            yield attention + "kv_a_layernorm.weight", (config.kv_lora_rank,)
            yield (
                attention + "kv_b_proj.weight",
                (
                    config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                    config.kv_lora_rank,
                ),
            )
            yield attention + "o_proj.weight", (hidden, config.num_attention_heads * config.v_head_dim)
            if config.mla_use_output_gate:
                yield attention + "g_proj.weight", (config.num_attention_heads * config.v_head_dim, hidden)

        if index >= config.first_k_dense_replace and index % config.moe_layer_freq == 0:
            moe = prefix + "block_sparse_moe."
            routed_hidden = config.routed_expert_hidden_size
            yield moe + "gate.weight", (config.num_experts, hidden)
            yield moe + "gate.e_score_correction_bias", (config.num_experts,)
            for expert in range(config.num_experts):
                for name, shape in (
                    ("w1", (config.moe_intermediate_size, routed_hidden)),
                    ("w2", (routed_hidden, config.moe_intermediate_size)),
                    ("w3", (config.moe_intermediate_size, routed_hidden)),
                ):
                    yield moe + f"experts.{expert}.{name}.weight", shape
            shared = config.moe_intermediate_size * config.num_shared_experts
            yield moe + "shared_experts.gate_proj.weight", (shared, hidden)
            yield moe + "shared_experts.up_proj.weight", (shared, hidden)
            yield moe + "shared_experts.down_proj.weight", (hidden, shared)
            yield moe + "routed_expert_down_proj.weight", (routed_hidden, hidden)
            yield moe + "routed_expert_up_proj.weight", (hidden, routed_hidden)
            if config.latent_moe_use_norm:
                yield moe + "routed_expert_norm.weight", (routed_hidden,)
        else:
            yield prefix + "mlp.gate_proj.weight", (config.intermediate_size, hidden)
            yield prefix + "mlp.up_proj.weight", (config.intermediate_size, hidden)
            yield prefix + "mlp.down_proj.weight", (hidden, config.intermediate_size)

        for name in ("input_layernorm", "post_attention_layernorm"):
            yield prefix + name + ".weight", (hidden,)
        if config.attn_res_block_size is not None:
            for name in ("self_attention_res_norm", "mlp_res_norm"):
                yield prefix + name + ".weight", (hidden,)
            for name in ("self_attention_res_proj", "mlp_res_proj"):
                yield prefix + name + ".weight", (1, hidden)


def kimi_k3_meta_model(config: Any) -> nn.Module:
    """Build only meta parameters, independent of Hub training-code imports."""
    model = nn.Module()
    for name, shape in kimi_k3_parameter_shapes(config):
        parts = name.split(".")
        module = model
        for part in parts[:-1]:
            if part not in module._modules:
                module.add_module(part, nn.Module())
            module = module.get_submodule(part)
        module.register_parameter(parts[-1], nn.Parameter(torch.empty(shape, device="meta"), requires_grad=False))
    return model
