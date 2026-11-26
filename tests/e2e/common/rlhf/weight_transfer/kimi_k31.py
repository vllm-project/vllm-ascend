# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint geometry of the reduced BF16 Kimi K3.1 Pro language model."""

from collections.abc import Iterator
from typing import Any


def checkpoint_shapes(config: Any) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Emit unsharded checkpoint names before Ascend packs KDA or fuses the indexer."""
    prefix = "language_model.model."
    hidden = config.hidden_size
    yield prefix + "embed_tokens.weight", (config.vocab_size, hidden)
    yield "language_model.lm_head.weight", (config.vocab_size, hidden)
    yield prefix + "norm.weight", (hidden,)
    yield prefix + "output_attn_res_norm.weight", (hidden,)
    yield prefix + "output_attn_res_proj.weight", (1, hidden)

    linear = config.linear_attn_config
    heads, head_dim = linear["num_heads"], linear["head_dim"]
    projection = heads * head_dim
    for index in range(config.num_hidden_layers):
        layer = f"{prefix}layers.{index}."
        attention = layer + "self_attn."
        if index + 1 in linear["kda_layers"]:
            for name in ("q", "k", "v"):
                yield attention + name + "_proj.weight", (projection, hidden)
                yield attention + name + "_conv1d.weight", (projection, 1, linear["short_conv_kernel_size"])
            # The published checkpoint stores 128 entries for the 96 KDA heads.
            yield attention + "A_log", (128,)
            yield attention + "dt_bias", (projection,)
            yield attention + "f_a_proj.weight", (head_dim, hidden)
            yield attention + "f_b_proj.weight", (projection, head_dim)
            yield attention + "b_proj.weight", (heads, hidden)
            yield attention + "g_proj.weight", (projection, hidden)
            yield attention + "o_norm.weight", (head_dim,)
            yield attention + "o_proj.weight", (hidden, projection)
        else:
            dsa = config.dsa_config
            assert index + 1 in dsa["dsa_layers"]
            q_rank, kv_rank = config.q_lora_rank, config.kv_lora_rank
            n_heads = config.num_attention_heads
            nope, rope, value = config.qk_nope_head_dim, config.qk_rope_head_dim, config.v_head_dim
            yield attention + "q_a_proj.weight", (q_rank, hidden)
            yield attention + "q_a_layernorm.weight", (q_rank,)
            yield attention + "q_b_proj.weight", (n_heads * (nope + rope), q_rank)
            yield attention + "kv_a_proj_with_mqa.weight", (kv_rank + rope, hidden)
            yield attention + "kv_a_layernorm.weight", (kv_rank,)
            yield attention + "kv_b_proj.weight", (n_heads * (nope + value), kv_rank)
            yield attention + "o_proj.weight", (hidden, n_heads * value)
            yield attention + "g_proj.weight", (n_heads * value, hidden)
            index_heads, index_dim = dsa["index_num_heads"], dsa["index_head_dim"]
            yield attention + "indexer.wq_b.weight", (index_heads * index_dim, q_rank)
            yield attention + "indexer.wk.weight", (index_dim, hidden)
            yield attention + "indexer.k_norm.weight", (index_dim,)
            yield attention + "indexer.k_norm.bias", (index_dim,)
            yield attention + "indexer.weights_proj.weight", (index_heads, hidden)

        if index < config.first_k_dense_replace:
            yield layer + "mlp.gate_proj.weight", (config.intermediate_size, hidden)
            yield layer + "mlp.up_proj.weight", (config.intermediate_size, hidden)
            yield layer + "mlp.down_proj.weight", (hidden, config.intermediate_size)
        else:
            moe = layer + "block_sparse_moe."
            routed, inter = config.routed_expert_hidden_size, config.moe_intermediate_size
            shared = inter * config.num_shared_experts
            yield moe + "gate.weight", (config.num_experts, hidden)
            yield moe + "gate.e_score_correction_bias", (config.num_experts,)
            for expert in range(config.num_experts):
                for name, shape in (("w1", (inter, routed)), ("w2", (routed, inter)), ("w3", (inter, routed))):
                    yield moe + f"experts.{expert}.{name}.weight", shape
            yield moe + "shared_experts.gate_proj.weight", (shared, hidden)
            yield moe + "shared_experts.up_proj.weight", (shared, hidden)
            yield moe + "shared_experts.down_proj.weight", (hidden, shared)
            yield moe + "routed_expert_down_proj.weight", (routed, hidden)
            yield moe + "routed_expert_up_proj.weight", (hidden, routed)
            yield moe + "routed_expert_norm.weight", (routed,)

        for name in ("input_layernorm", "post_attention_layernorm", "self_attention_res_norm", "mlp_res_norm"):
            yield layer + name + ".weight", (hidden,)
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            yield layer + name + ".weight", (1, hidden)
