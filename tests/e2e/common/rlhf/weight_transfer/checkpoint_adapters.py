# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Checkpoint-facing name and shape conversions."""

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


def deepseek_v4_dspark_parameters(config, parameters):
    """Use the reduced V4 decoder schema in the DSpark mtp checkpoint namespace."""
    result = []
    for name, shape, dtype in parameters:
        name = name.removeprefix("model.")
        if not name.startswith("layers.0."):
            continue
        name = "mtp.0." + name.removeprefix("layers.0.")
        name = name.replace(".self_attn.", ".attn.").replace(".mlp.", ".ffn.")
        name = name.replace(".input_layernorm.", ".attn_norm.").replace(".post_attention_layernorm.", ".ffn_norm.")
        result.append((name, shape, dtype))
    hidden, hc, rank = config.hidden_size, config.hc_mult, config.dspark_markov_rank
    result.extend(
        [
            ("mtp.0.main_proj.weight", (hidden, hidden * len(config.dspark_target_layer_ids)), torch.bfloat16),
            ("mtp.0.main_norm.weight", (hidden,), torch.bfloat16),
            ("mtp.0.norm.weight", (hidden,), torch.bfloat16),
            ("mtp.0.hc_head_fn", (hc, hc * hidden), torch.float32),
            ("mtp.0.hc_head_base", (hc,), torch.float32),
            ("mtp.0.hc_head_scale", (1,), torch.float32),
            ("mtp.0.markov_head.markov_w1.weight", (config.vocab_size, rank), torch.bfloat16),
            ("mtp.0.markov_head.markov_w2.weight", (config.vocab_size, rank), torch.bfloat16),
            ("mtp.0.confidence_head.proj.weight", (1, hidden + rank), torch.float32),
        ]
    )
    return result


def qwen3_draft_parameters(config, parameters):
    """Adapt a meta Qwen3 schema to upstream's EAGLE3/DFlash loaders.

    These loaders prepend ``model.`` themselves. Embedding and output weights
    are omitted intentionally: the proposer shares the target's copies.
    """
    architecture = config.architectures[0]
    eagle = architecture == "Eagle3Qwen3ForCausalLM"
    result = []
    for name, shape, dtype in parameters:
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            continue
        name = name.removeprefix("model.")
        if (
            eagle
            and name.startswith("layers.0.self_attn.")
            and name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight"))
        ):
            shape = (shape[0], 2 * shape[1])
        result.append((name, shape, dtype))
    hidden = config.hidden_size
    result.append(("fc.weight", (hidden, 3 * hidden), torch.bfloat16))
    if eagle:
        result.extend(
            (f"layers.{i}.hidden_norm.weight", (hidden,), torch.bfloat16) for i in range(config.num_hidden_layers)
        )
    else:
        result.append(("hidden_norm.weight", (hidden,), torch.bfloat16))
    return result


def deepseek_v4_mtp_parameters(config, parameters):
    """Use one V4 decoder and the classic MTP projections in ``mtp.0.*``.

    Embedding and output weights are shared with the target and omitted.
    """
    result = []
    for name, shape, dtype in parameters:
        name = name.removeprefix("model.")
        if not name.startswith("layers.0."):
            continue
        name = "mtp.0." + name.removeprefix("layers.0.")
        name = name.replace(".self_attn.", ".attn.").replace(".mlp.", ".ffn.")
        name = name.replace(".input_layernorm.", ".attn_norm.").replace(".post_attention_layernorm.", ".ffn_norm.")
        result.append((name, shape, dtype))
    hidden, hc = config.hidden_size, config.hc_mult
    result.extend(
        [
            ("mtp.0.e_proj.weight", (hidden, hidden), torch.bfloat16),
            ("mtp.0.h_proj.weight", (hidden, hidden), torch.bfloat16),
            ("mtp.0.enorm.weight", (hidden,), torch.bfloat16),
            ("mtp.0.hnorm.weight", (hidden,), torch.bfloat16),
            ("mtp.0.norm.weight", (hidden,), torch.bfloat16),
            ("mtp.0.hc_head_fn", (hc, hc * hidden), torch.float32),
            ("mtp.0.hc_head_base", (hc,), torch.float32),
            ("mtp.0.hc_head_scale", (1,), torch.float32),
        ]
    )
    return result


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
