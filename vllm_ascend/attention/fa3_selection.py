# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


_QWEN_FA3_TARGETS = frozenset(
    {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
_MINIMAX_DSPARK_TARGETS = frozenset({"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"})

# The backbone lists eligible target architectures. Each draft loader lists (target architectures, draft architectures).
_FA3_ARCHITECTURE_PAIRS: dict[str, frozenset[str] | tuple[frozenset[str], frozenset[str]]] = {
    "backbone": _QWEN_FA3_TARGETS,
    "dspark_head": (
        _MINIMAX_DSPARK_TARGETS,
        frozenset({"DSparkMiniMaxDraftModel", "MiniMaxM31FlashA3DSparkModel"}),
    ),
    "dflash_head": (_QWEN_FA3_TARGETS, frozenset({"DFlash2DraftModel"})),
}


def supports_fa3_model(spec: SpeculativeConfig, model_tag: str) -> bool:
    """Match the FA3 target/draft architecture pair for the active loader."""
    if model_tag not in _FA3_ARCHITECTURE_PAIRS:
        return False

    draft_tag = f"{spec.method}_head"
    draft_pair = _FA3_ARCHITECTURE_PAIRS.get(draft_tag)
    if not isinstance(draft_pair, tuple):
        return False
    supported_targets, supported_drafts = draft_pair
    if model_tag == "backbone":
        supported_targets = supported_targets.intersection(_FA3_ARCHITECTURE_PAIRS[model_tag])
    elif model_tag != draft_tag:
        return False
    if supported_drafts.isdisjoint(spec.draft_model_config.architectures or ()):
        return False

    target = spec.target_model_config
    target_architecture = getattr(target, "architecture", None)
    target_hf = getattr(target, "hf_config", None)
    # MiniMax is identified by HF architecture aliases. Qwen uses the architecture resolved by vLLM.
    minimax_aliases = _MINIMAX_DSPARK_TARGETS.intersection(getattr(target_hf, "architectures", None) or ())
    if minimax_aliases or target_architecture in _MINIMAX_DSPARK_TARGETS:
        return not supported_targets.isdisjoint(minimax_aliases)
    return target_architecture in supported_targets
