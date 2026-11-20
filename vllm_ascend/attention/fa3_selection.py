# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SpeculativeConfig


_QWEN_FA3_TARGETS = frozenset(
    {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)

# The backbone lists eligible target architectures. Each draft loader lists (target architectures, draft architectures).
_FA3_ARCHITECTURE_PAIRS: dict[str, frozenset[str] | tuple[frozenset[str], frozenset[str]]] = {
    "backbone": _QWEN_FA3_TARGETS,
    "dspark_head": (
        frozenset({"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}),
        frozenset({"DSparkMiniMaxDraftModel", "MiniMaxM31FlashA3DSparkModel"}),
    ),
    "dflash_head": (_QWEN_FA3_TARGETS, frozenset({"DFlash2DraftModel"})),
}


def _matches_target(target: ModelConfig, supported_targets: frozenset[str]) -> bool:
    target_architecture = getattr(target, "architecture", None)
    if target_architecture is not None:
        return target_architecture in supported_targets
    target_hf = getattr(target, "hf_config", None)
    return not supported_targets.isdisjoint(getattr(target_hf, "architectures", None) or ())


def supports_fa3_model(spec: SpeculativeConfig, model_tag: str) -> bool:
    """Match the backbone architecture or the active draft loader's model pair."""
    rule = _FA3_ARCHITECTURE_PAIRS.get(model_tag)
    if rule is None:
        return False
    if isinstance(rule, frozenset):
        return _matches_target(spec.target_model_config, rule)

    if spec.draft_model_config is None or spec.method != model_tag.removesuffix("_head"):
        return False
    supported_targets, supported_drafts = rule
    return _matches_target(spec.target_model_config, supported_targets) and not supported_drafts.isdisjoint(
        spec.draft_model_config.architectures or ()
    )
