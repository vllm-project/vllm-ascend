# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-specific checkpoint-name conversions."""

import regex as re

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


def minimax_m3_checkpoint_name(name: str) -> str:
    """Translate a native HF MiniMax-M3 name into its published checkpoint form.

    The native Transformers model differs from the MiniMaxAI/MiniMax-M3
    checkpoint in the following names:

    * ``lm_head`` lives under ``language_model``;
    * attention ``indexer.q_proj/k_proj/q_norm/k_norm`` become
      ``index_q_proj/index_k_proj/index_q_norm/index_k_norm``;
    * routed experts, shared experts and the router live under
      ``block_sparse_moe`` (HF: ``mlp``), with the routing correction bias
      directly under ``block_sparse_moe`` (HF: ``mlp.gate``);
    * routed expert ``gate_proj/down_proj/up_proj`` become ``w1/w2/w3``;
      dense and shared expert projections keep their original names.

    This prepares checkpoint-style names for the existing vLLM loader, for
    both startup weights and IPC/HCCL updates. Fused tensor expansion and the
    ``language_model.model`` prefix are handled separately by the source.
    """
    m = re.match(r"^lm_head\.(.+)$", name)
    if m:
        return f"language_model.lm_head.{m.group(1)}"

    m = re.match(r"^(.*\.self_attn)\.indexer\.(.+)$", name)
    if m:
        return f"{m.group(1)}.index_{m.group(2)}"

    m = re.match(r"^(.*)\.mlp\.gate\.e_score_correction_bias$", name)
    if m:
        return f"{m.group(1)}.block_sparse_moe.e_score_correction_bias"

    m = re.match(r"^(.*)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.weight$", name)
    if m:
        shard = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[m.group(3)]
        return f"{m.group(1)}.block_sparse_moe.experts.{m.group(2)}.{shard}.weight"

    m = re.match(r"^(.*)\.mlp\.(experts|shared_experts|gate)\.(.+)$", name)
    if m:
        return f"{m.group(1)}.block_sparse_moe.{m.group(2)}.{m.group(3)}"
    return name
