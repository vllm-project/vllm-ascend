# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Only the full-attention layers of these hybrid Qwen targets use FA3.
_FA3_TARGET_ARCHITECTURES = frozenset(
    {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    }
)
# For MiniMax M3, only the paired DSpark draft can use FA3.
MINIMAX_M3_ARCHITECTURES = frozenset({"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"})
_FA3_DRAFT_ARCHITECTURES = {
    ("dflash_head", "dflash"): frozenset({"DFlash2DraftModel"}),
    ("dspark_head", "dspark"): frozenset(
        {"DSparkMiniMaxDraftModel", "MiniMaxM31FlashA3DSparkModel", "Qwen3DSparkModel", "Qwen3OmniDSparkModel"}
    ),
}


def supports_fa3_model(config: VllmConfig) -> bool:
    spec = config.speculative_config
    v2 = getattr(config, "use_v2_model_runner", False)
    # V2 retains the target config while loading a tagged draft. The legacy
    # MiniMax route instead receives the currently loaded model's own config.
    if v2:
        from vllm.compilation import backends

        tag = backends.model_tag
    else:
        tag = "dspark_head"
    if tag == "backbone":
        return getattr(config.model_config, "architecture", None) in _FA3_TARGET_ARCHITECTURES and not getattr(
            config.quant_config, "enable_c8_quant", False
        )
    if spec is None or spec.draft_model_config is None:
        return False
    allowed = _FA3_DRAFT_ARCHITECTURES.get((tag, spec.method))
    if allowed is None:
        return False
    architectures = spec.draft_model_config.architectures if v2 else config.model_config.hf_config.architectures
    if allowed.isdisjoint(architectures or ()):
        return False

    target_hf = getattr(spec.target_model_config, "hf_config", None)
    minimax = not MINIMAX_M3_ARCHITECTURES.isdisjoint(getattr(target_hf, "architectures", None) or ())
    if tag == "dspark_head":
        if not minimax or config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
            return False
        quant_config = config.quant_config
    else:
        if minimax:
            return False
        # DFlash keeps the target's quant_config in the loader context, but
        # constructs its attention layers with the draft checkpoint's recipe.
        from vllm.model_executor.models.utils import get_draft_quant_config

        quant_config = get_draft_quant_config(config)
    return not getattr(quant_config, "enable_c8_quant", False)
