# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


# Each loader tag lists the architectures it can run with FA3.
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
    "dspark_head": frozenset({"DSparkMiniMaxDraftModel", "MiniMaxM31FlashA3DSparkModel"}),
    "dflash_head": frozenset({"DFlash2DraftModel"}),
}


def supports_fa3_model(spec: SpeculativeConfig, model_tag: str, minimax_m3_architectures: frozenset[str]) -> bool:
    """Match the FA3 target/draft architecture pair for the active loader."""
    target = spec.target_model_config
    target_hf = getattr(target, "hf_config", None)
    is_minimax_target = not minimax_m3_architectures.isdisjoint(getattr(target_hf, "architectures", None) or ())
    draft_architectures = spec.draft_model_config.architectures or ()
    if (
        model_tag == "dspark_head"
        and spec.method == "dspark"
        and is_minimax_target
        and not _FA3_TARGET_ARCHITECTURES[model_tag].isdisjoint(draft_architectures)
    ):
        return True

    # MiniMax targets only support FA3 on a DSpark draft.
    if spec.method != "dflash" or is_minimax_target:
        return False
    if model_tag == "backbone":
        draft_supports_fa3 = not _FA3_TARGET_ARCHITECTURES["dflash_head"].isdisjoint(draft_architectures)
        return draft_supports_fa3 and getattr(target, "architecture", None) in _FA3_TARGET_ARCHITECTURES[model_tag]
    return model_tag == "dflash_head" and not _FA3_TARGET_ARCHITECTURES[model_tag].isdisjoint(draft_architectures)
