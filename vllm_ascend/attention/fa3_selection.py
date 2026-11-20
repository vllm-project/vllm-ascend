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
    if (
        model_tag == "dspark_head"
        and spec.method == "dspark"
        and not _FA3_TARGET_ARCHITECTURES[model_tag].isdisjoint(target_architectures)
        and not _FA3_DSPARK_DRAFT_ARCHITECTURES.isdisjoint(draft_architectures)
    ):
        return True

    # MiniMax targets only support FA3 on a DSpark draft.
    if spec.method != "dflash" or not _FA3_TARGET_ARCHITECTURES["dspark_head"].isdisjoint(target_architectures):
        return False
    if model_tag == "backbone":
        draft_supports_fa3 = not _FA3_TARGET_ARCHITECTURES["dflash_head"].isdisjoint(draft_architectures)
        return draft_supports_fa3 and getattr(target, "architecture", None) in _FA3_TARGET_ARCHITECTURES[model_tag]
    return model_tag == "dflash_head" and not _FA3_TARGET_ARCHITECTURES[model_tag].isdisjoint(draft_architectures)
