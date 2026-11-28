# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-specific checkpoint name conversions, tensor shapes and meta-model factories."""

from collections.abc import Iterator
from typing import Any

import regex as re
import torch
from torch import nn

_DSV4_SELF_ATTN_RENAMES = (
    # Descriptive HF names -> the compact names the checkpoint/loader expect.
    ("self_attn.compressor.position_bias", "self_attn.compressor.ape"),
    ("self_attn.compressor.kv_proj.weight", "self_attn.compressor.wkv.weight"),
    ("self_attn.compressor.gate_proj.weight", "self_attn.compressor.wgate.weight"),
    ("self_attn.compressor.kv_norm.weight", "self_attn.compressor.norm.weight"),
    ("self_attn.q_a_proj.weight", "self_attn.wq_a.weight"),
    ("self_attn.q_a_norm.weight", "self_attn.q_norm.weight"),
    ("self_attn.q_b_proj.weight", "self_attn.wq_b.weight"),
    ("self_attn.kv_proj.weight", "self_attn.wkv.weight"),
    ("self_attn.o_a_proj.weight", "self_attn.wo_a.weight"),
    ("self_attn.o_b_proj.weight", "self_attn.wo_b.weight"),
)

_DSV4_EXPERT_SHARDS = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def deepseek_v4_checkpoint_name(name: str) -> str:
    """Translate an HF DeepSeek-V4 parameter name into its served-model form.

    ``DeepSeek-V4-Flash`` differs from the HF ``from_config`` model in ways that
    ``load_weights`` cannot bridge on its own:

    * routed experts ship as per-expert ``w1/w3/w2`` (HF: fused ``gate_up_proj``
      / ``down_proj``); the loader fuses them into ``w13``/``w2``;
    * the compressor lives *under* the indexer (HF: the indexer sits under the
      compressor) and its norm is called ``norm`` (HF/checkpoint: ``kv_norm``);
    * ``wq_b``/``weights_proj`` belong to the indexer itself, and the attention
      projections use ``wq_a``/``wkv``/``wo_a``/``wo_b``.

    Without this mapping ``load_weights`` raises ``KeyError`` on the first
    unmatched tensor.
    """
    if name == "model.embed_tokens.weight":
        return "embed.weight"
    if name == "model.norm.weight":
        return "norm.weight"
    if name == "lm_head.weight":
        return "head.weight"

    m = re.match(r"^model\.hc_head\.hc_(base|fn|scale)$", name)
    if m:
        return f"hc_head_{m.group(1)}"
    m = re.match(r"^model\.layers\.(\d+)\.(attn|ffn)_hc\.(base|fn|scale)$", name)
    if m:
        return f"layers.{m.group(1)}.hc_{m.group(2)}_{m.group(3)}"
    m = re.match(r"^model\.layers\.(\d+)\.self_attn\.sinks$", name)
    if m:
        return f"layers.{m.group(1)}.attn.attn_sink"

    m = re.match(r"^(.*\.mlp\.experts\.\d+)\.(gate_proj|down_proj|up_proj)\.weight$", name)
    if m:
        return f"{m.group(1)}.{_DSV4_EXPERT_SHARDS[m.group(2)]}.weight"

    m = re.match(r"^(.*\.self_attn)\.compressor\.indexer\.(.+)$", name)
    if m:
        base = m.group(1)
        tail = {
            "position_bias": "ape",
            "kv_proj.weight": "wkv.weight",
            "gate_proj.weight": "wgate.weight",
            "kv_norm.weight": "norm.weight",
            "q_b_proj.weight": "wq_b.weight",
            "scorer.weights_proj.weight": "weights_proj.weight",
        }.get(m.group(2), m.group(2))
        if tail.startswith(("wq_b.", "weights_proj.")):
            return f"{base}.indexer.{tail}"
        return f"{base}.indexer.compressor.{tail}"

    for old, new in _DSV4_SELF_ATTN_RENAMES:
        if name.endswith(old):
            return name[: -len(old)] + new
    return name


def deepseek_v4_vision_params(config: Any) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
    """Describe vision and router parameters absent from the HF text meta model.

    The shared weight source enumerates the text backbone, expands expert
    shards and applies ``deepseek_v4_checkpoint_name``. This adapter adds only
    missing checkpoint entries; it does not allocate or generate weights.

    Shapes follow the vision checkpoint: fused ViT gate/up projections,
    downsampled aligner input and four image-marker vectors. Vision-only router
    biases remain FP32; vision and marker payloads use BF16, including markers whose
    served parameters are FP32 and are cast by the loader on copy.
    """
    dim = int(config.vision_dim)
    inter = int(config.vision_inter_dim)
    patch = int(config.vision_patch_size)
    ratio = int(config.vision_downsample_ratio)
    hidden = int(config.hidden_size)
    params: list[tuple[str, tuple[int, ...], torch.dtype]] = [
        ("vision.patch_embed.proj.weight", (dim, 3 * patch * patch), torch.bfloat16),
        ("vision.patch_embed.proj.bias", (dim,), torch.bfloat16),
        ("vision.norm.weight", (dim,), torch.bfloat16),
    ]
    for layer in range(int(config.vision_n_layers)):
        prefix = f"vision.blocks.{layer}"
        params += [
            (f"{prefix}.norm1.weight", (dim,), torch.bfloat16),
            (f"{prefix}.attn.wqkv.weight", (3 * dim, dim), torch.bfloat16),
            (f"{prefix}.attn.wqkv.bias", (3 * dim,), torch.bfloat16),
            (f"{prefix}.attn.wo.weight", (dim, dim), torch.bfloat16),
            (f"{prefix}.attn.wo.bias", (dim,), torch.bfloat16),
            (f"{prefix}.norm2.weight", (dim,), torch.bfloat16),
            (f"{prefix}.mlp.w1.weight", (2 * inter, dim), torch.bfloat16),
            (f"{prefix}.mlp.w2.weight", (dim, inter), torch.bfloat16),
        ]
    params += [
        ("aligner.w1.weight", (hidden, dim * ratio * ratio), torch.bfloat16),
        ("aligner.w1.bias", (hidden,), torch.bfloat16),
        ("aligner.w2.weight", (hidden, hidden), torch.bfloat16),
        ("aligner.w2.bias", (hidden,), torch.bfloat16),
    ]
    params += [(name, (hidden,), torch.bfloat16) for name in ("image_start", "image_end", "image_newline", "image_pad")]
    # HF's text meta model omits the vision-only router bias. Its persistent
    # e_score_correction_bias buffer is already enumerated by the shared source.
    expert_shape = (int(config.n_routed_experts),)
    for layer in range(int(config.num_hidden_layers)):
        prefix = f"model.layers.{layer}.mlp.gate"
        params.append((f"{prefix}.bias_vl", expert_shape, torch.float32))
    return params


def minimax_m3_checkpoint_name(name: str) -> str:
    """Map native HF text tensors to MiniMax's published checkpoint namespace."""
    if name.startswith("lm_head."):
        return f"language_model.{name}"
    name = name.replace(".self_attn.indexer.", ".self_attn.index_")
    name = name.replace(".mlp.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias")
    name = re.sub(r"\.mlp\.(experts|shared_experts|gate)\.", r".block_sparse_moe.\1.", name)
    match = re.match(r"^(.*\.block_sparse_moe\.experts\.\d+)\.(gate_proj|down_proj|up_proj)\.weight$", name)
    if match:
        shard = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[match.group(2)]
        return f"{match.group(1)}.{shard}.weight"
    return name


def expand_fused_expert_params(name: str, shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Map HF fused routed-expert tensors onto checkpoint-facing per-expert names.

    ``GlmMoeDsaForCausalLM`` keeps routed experts fused as
    ``mlp.experts.gate_up_proj`` ``[E, 2I, H]`` and ``mlp.experts.down_proj``
    ``[E, H, I]``, while the published checkpoint - and therefore vLLM's
    ``expert_params_mapping`` - stores one tensor per expert
    (``mlp.experts.{e}.gate_proj.weight`` etc.).  Without this expansion the
    fused names reach ``load_weights`` unmatched and raise
    ``KeyError: 'layers.N.mlp.experts.gate_up_proj'``.
    """
    for suffix, kind in ((".mlp.experts.gate_up_proj", "gate_up"), (".mlp.experts.down_proj", "down")):
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix.rsplit(".", 1)[-1])]  # keeps the trailing dot
        if kind == "gate_up":
            num_experts, fused_inter, hidden = shape
            inter = fused_inter // 2
            out: list[tuple[str, tuple[int, ...]]] = []
            for expert in range(num_experts):
                out.append((f"{prefix}{expert}.gate_proj.weight", (inter, hidden)))
                out.append((f"{prefix}{expert}.up_proj.weight", (inter, hidden)))
            return out
        num_experts, hidden, inter = shape
        return [(f"{prefix}{expert}.down_proj.weight", (hidden, inter)) for expert in range(num_experts)]
    return [(name, shape)]


def expand_minimax_m3_params(name: str, shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Expand MiniMax's dense/shared gate-up and routed-expert tensors."""
    if name.endswith((".mlp.gate_up_proj.weight", ".mlp.shared_experts.gate_up_proj.weight")):
        fused_intermediate, hidden = shape
        assert fused_intermediate % 2 == 0, f"{name}: odd fused gate/up dimension"
        shard_shape = (fused_intermediate // 2, hidden)
        prefix = name.removesuffix(".gate_up_proj.weight")
        return [(f"{prefix}.{projection}.weight", shard_shape) for projection in ("gate_proj", "up_proj")]
    return expand_fused_expert_params(name, shape)


def kimi_k3_parameter_shapes(config: Any) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Enumerate Kimi-K3 text checkpoint shapes from modeling_kimi_linear.py.

    Text-only loading uses model.* and lm_head.* names. Keep the three source
    convolutions separate so the serving loader performs its normal packing.
    """
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
