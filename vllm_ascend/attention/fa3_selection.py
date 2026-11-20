# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


# Each draft loader maps to (target architectures, draft architectures).
# A matching DFlash pair enables FA3 on both loaders. A matching DSpark pair enables it only on the draft.
_FA3_ARCHITECTURE_PAIRS = {
    "dspark_head": (
        frozenset({"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}),
        frozenset({"DSparkMiniMaxDraftModel", "MiniMaxM31FlashA3DSparkModel"}),
    ),
    "dflash_head": (
        frozenset(
            {
                "Qwen3NextForCausalLM",
                "Qwen3_5ForCausalLM",
                "Qwen3_5MoeForCausalLM",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
            }
        ),
        frozenset({"DFlash2DraftModel"}),
    ),
}


def supports_fa3_model(spec: SpeculativeConfig, model_tag: str) -> bool:
    """Match the FA3 target/draft architecture pair for the active loader."""
    if model_tag != "backbone" and model_tag not in _FA3_ARCHITECTURE_PAIRS:
        return False

    target = spec.target_model_config
    target_hf = getattr(target, "hf_config", None)
    dspark_targets, _ = _FA3_ARCHITECTURE_PAIRS["dspark_head"]
    is_dspark_target = not dspark_targets.isdisjoint(getattr(target_hf, "architectures", None) or ())
    supported_method = "dspark" if is_dspark_target else "dflash"
    if spec.method != supported_method:
        return False

    draft_tag = f"{supported_method}_head"
    supported_targets, supported_drafts = _FA3_ARCHITECTURE_PAIRS[draft_tag]
    if supported_drafts.isdisjoint(spec.draft_model_config.architectures or ()):
        return False

    if is_dspark_target:
        return model_tag == draft_tag
    return model_tag in ("backbone", draft_tag) and getattr(target, "architecture", None) in supported_targets
