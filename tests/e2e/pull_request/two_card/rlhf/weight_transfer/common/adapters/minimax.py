# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MiniMax checkpoint-name conversion."""

import regex as re


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
        # Only the published direct projection/norm weights have index_* names.
        # Reject new/nested HF state here rather than emitting a guessed key.
        if re.fullmatch(r"(?:q_proj|k_proj|q_norm|k_norm)\.weight", m.group(2)) is None:
            raise ValueError(f"Unsupported MiniMax-M3 indexer parameter: {name}")
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
