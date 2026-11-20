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
    }
)
MINIMAX_M3_ARCHITECTURES = frozenset({"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"})
# Each target family opts in only its listed DSpark drafts.
_FA3_DSPARK_DRAFT_ARCHITECTURES = {
    MINIMAX_M3_ARCHITECTURES: frozenset({"DSparkMiniMaxDraftModel", "MiniMaxM31FlashA3DSparkModel"}),
}
_FA3_DFLASH_DRAFT_ARCHITECTURES = frozenset({"DFlash2DraftModel"})


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
    target_hf = getattr(spec.target_model_config, "hf_config", None)
    target_architectures = getattr(target_hf, "architectures", None) or ()
    draft = spec.draft_model_config if v2 else config.model_config.hf_config
    draft_architectures = draft.architectures or ()
    for targets, drafts in _FA3_DSPARK_DRAFT_ARCHITECTURES.items():
        if not targets.isdisjoint(target_architectures):
            return (
                (tag, spec.method) == ("dspark_head", "dspark")
                and not drafts.isdisjoint(draft_architectures)
                and config.cache_config.cache_dtype in ("auto", "float16", "bfloat16")
                and not getattr(config.quant_config, "enable_c8_quant", False)
            )
    if (tag, spec.method) != ("dflash_head", "dflash") or _FA3_DFLASH_DRAFT_ARCHITECTURES.isdisjoint(
        draft_architectures
    ):
        return False

    # DFlash retains the target's quant_config in the loader context.
    from vllm.model_executor.models.utils import get_draft_quant_config

    return not getattr(get_draft_quant_config(config), "enable_c8_quant", False)
