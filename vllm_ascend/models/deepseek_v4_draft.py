# SPDX-License-Identifier: Apache-2.0

from typing import Any

import regex as re

_MTP_WEIGHT_RE = re.compile(r"^mtp\.(\d+)\.(.+)$")


def is_deepseek_v4_dspark_config(config: Any) -> bool:
    return config.model_type == "deepseek_v4" and int(getattr(config, "dspark_block_size", 0) or 0) > 0


def remap_dspark_mtp_weight_name(name: str, num_hidden_layers: int) -> str | None:
    """Map a DeepSeek V4 checkpoint MTP name to the draft runtime layer."""
    match = _MTP_WEIGHT_RE.match(name)
    if match is None:
        return None

    stage_idx = int(match.group(1))
    rest = match.group(2)
    if rest.startswith("confidence_head."):
        return None

    name = f"model.layers.{num_hidden_layers + stage_idx}.{rest}"
    replacements = (
        (".attn.", ".self_attn."),
        (".ffn_norm.", ".post_attention_layernorm."),
        (".attn_norm.", ".input_layernorm."),
        (".ffn.", ".mlp."),
        (".w1.", ".gate_proj."),
        (".w2.", ".down_proj."),
        (".w3.", ".up_proj."),
        (".mlp.gate.bias", ".mlp.gate.e_score_correction_bias"),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return name
