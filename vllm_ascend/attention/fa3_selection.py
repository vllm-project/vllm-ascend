# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


# DSpark matches the target family and its draft allowlist below. DFlash matches the draft architecture.
_FA3_TARGET_ARCHITECTURES = {
    "backbone": frozenset(
        {
            "Qwen3NextForCausalLM",
            "Qwen3_5ForCausalLM",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        }
    ),
    "dspark_head": frozenset({"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}),
    "dflash_head": frozenset({"DFlash2DraftModel"}),
}
_FA3_DSPARK_DRAFT_ARCHITECTURES = frozenset({"DSparkMiniMaxDraftModel", "MiniMaxM31FlashA3DSparkModel"})


def supports_fa3_model(spec: SpeculativeConfig, model_tag: str) -> bool:
    """Match the FA3 target/draft architecture pair for the active loader."""
    target = spec.target_model_config
    target_hf = getattr(target, "hf_config", None)
    target_architectures = getattr(target_hf, "architectures", None) or ()
    draft_architectures = spec.draft_model_config.architectures or ()
    is_minimax_target = not _FA3_TARGET_ARCHITECTURES["dspark_head"].isdisjoint(target_architectures)
    if (
        is_minimax_target
        and model_tag == "dspark_head"
        and spec.method == "dspark"
        and not _FA3_DSPARK_DRAFT_ARCHITECTURES.isdisjoint(draft_architectures)
    ):
        return True

    if (
        is_minimax_target
        or spec.method != "dflash"
        or _FA3_TARGET_ARCHITECTURES["dflash_head"].isdisjoint(draft_architectures)
    ):
        return False
    return model_tag == "dflash_head" or (
        model_tag == "backbone" and getattr(target, "architecture", None) in _FA3_TARGET_ARCHITECTURES["backbone"]
    )
