# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


# These targets may use FA3 on full-attention layers when their draft uses FA3.
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


def supports_fa3_model(spec: SpeculativeConfig, model_tag: str) -> bool:
    """Match the FA3 target/draft architecture pair for the active loader."""
    target = spec.target_model_config
    target_hf = getattr(target, "hf_config", None)
    target_architectures = getattr(target_hf, "architectures", None) or ()
    draft_architectures = spec.draft_model_config.architectures or ()
    for targets, drafts in _FA3_DSPARK_DRAFT_ARCHITECTURES.items():
        if not targets.isdisjoint(target_architectures):
            return model_tag == "dspark_head" and spec.method == "dspark" and not drafts.isdisjoint(draft_architectures)

    if spec.method != "dflash" or _FA3_DFLASH_DRAFT_ARCHITECTURES.isdisjoint(draft_architectures):
        return False
    return model_tag == "dflash_head" or (
        model_tag == "backbone" and getattr(target, "architecture", None) in _FA3_TARGET_ARCHITECTURES
    )
