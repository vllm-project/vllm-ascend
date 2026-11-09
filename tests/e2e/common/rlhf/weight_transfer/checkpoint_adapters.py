# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Checkpoint-facing name and shape conversions."""

from typing import Any

import regex as re
import torch

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
    """Checkpoint-facing parameter set of the DeepSeek-V4 vision tower.

    ``transformers``' ``DeepseekV4ForCausalLM`` builds the text backbone only,
    while vLLM and vLLM-Ascend serve the multimodal wrapper
    (``DeepseekV4ForConditionalGeneration`` /
    ``AscendDeepseekV4ForConditionalGeneration``) whose ``vision`` ViT, ``aligner``
    and four ``image_*`` vectors are not part of any HF meta model. The names are
    the ones the published checkpoint ships and the wrapper resolves
    (``vision.*`` / ``aligner.*`` / ``image_*``, see ``_vision_parameter_name``),
    and every shape was verified against the real safetensors header of
    ``deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`` (``vision_n_layers=32`` -> 267
    tensors, e.g. ``vision.blocks.0.mlp.w1.weight`` == [5632, 1024] and
    ``aligner.w1.weight`` == [4096, 9216]).
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
    # The wrapper allocates these four vectors in float32, but the checkpoint -
    # and therefore the payload - ships them in BF16, and ``default_weight_loader``
    # casts on copy, so the payload stays faithful to the checkpoint.
    params += [(name, (hidden,), torch.bfloat16) for name in ("image_start", "image_end", "image_newline", "image_pad")]
    # HF's text meta model omits these real router parameters. A full update
    # must carry them rather than relying on the previous device allocation.
    for layer in range(int(config.num_hidden_layers)):
        params.append((f"model.layers.{layer}.mlp.gate.bias_vl", (int(config.n_routed_experts),), torch.float32))
        if config.mlp_layer_types[layer] != "hash_moe":
            params.append(
                (
                    f"model.layers.{layer}.mlp.gate.e_score_correction_bias",
                    (int(config.n_routed_experts),),
                    torch.float32,
                )
            )
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
