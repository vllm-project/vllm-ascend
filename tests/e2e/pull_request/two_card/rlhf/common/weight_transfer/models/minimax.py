# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MiniMax model configurations and checkpoint-name conversion."""

import regex as re

from tests.e2e.pull_request.two_card.rlhf.common.weight_transfer.models.base import WeightUpdateModelCase


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


# Reserve HBM for M3's FIA graph workspace rather than sizing the KV cache
# from the memory-utilization fraction. The four-layer, 1024-token test only
# needs a small cache; keep startup and live-update servers identical.
MINIMAX_M3_KV_CACHE_BYTES = 128 * 2**20


MINIMAX_M3_CASE = WeightUpdateModelCase(
    id="minimax-m3-sparse-moe-layout",
    model="MiniMax/MiniMax-M3",
    hf_overrides={
        "architectures": ["MiniMaxM3SparseForCausalLM"],
        "text_config": {
            "num_hidden_layers": 4,
            "num_local_experts": 8,
            # Explicit schedules for the fixed four-layer test model;
            # native config derives these before overrides are applied.
            "layer_types": ["full_attention", "full_attention", "full_attention", "minimax_m3_sparse"],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
        },
    },
    meta_config_attribute="text_config",
    checkpoint_model_prefix="language_model.model.",
    checkpoint_name_map=minimax_m3_checkpoint_name,
    # Disable the checkpoint's auto_map remote config and use Transformers'
    # native MiniMaxM3VLConfig, whose text_config is MiniMaxM3VLTextConfig.
    # AutoModelForCausalLM resolves that text config to MiniMaxM3VLForCausalLM
    # for meta-model construction. The legacy remote config instead yields
    # a generic text PreTrainedConfig that AutoModel cannot resolve.
    trust_remote_code=False,
    split_fused_mlp=True,
    extra_server_args=("--block-size", "128", "--kv-cache-memory-bytes", str(MINIMAX_M3_KV_CACHE_BYTES)),
)


MINIMAX_CASES = (MINIMAX_M3_CASE,)
