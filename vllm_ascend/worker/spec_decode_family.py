# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from enum import Enum
from typing import Any


class SpecDecodeFamily(str, Enum):
    """Runner-independent family selected from a speculative config."""

    NGRAM = "ngram"
    NGRAM_GPU = "ngram_gpu"
    SUFFIX = "suffix"
    MEDUSA = "medusa"
    DSPARK = "dspark"
    GEMMA4_MTP = "gemma4_mtp"
    STEP3P5_MTP = "step3p5_mtp"
    MULTI_KV_CACHE_GROUP_MTP = "multi_kv_cache_group_mtp"
    EAGLE = "eagle"
    DFLASH2 = "dflash2"
    DFLASH = "dflash"
    DRAFT_MODEL = "draft_model"
    EXTRACT_HIDDEN_STATES = "extract_hidden_states"
    MTP = "mtp"
    UNKNOWN = "unknown"


_MULTI_KV_CACHE_GROUP_MTP_MODEL_TYPES = {"glm5_next_mtp"}
_MULTI_KV_CACHE_GROUP_MTP_ARCHITECTURES = {"Glm5NextMTPModel"}


def _call_bool(config: Any, name: str) -> bool:
    value = getattr(config, name, False)
    return bool(value()) if callable(value) else bool(value)


def _architecture_names(owner: Any) -> set[str]:
    if owner is None:
        return set()
    names: set[str] = set()
    architecture = getattr(owner, "architecture", None)
    if isinstance(architecture, str):
        names.add(architecture)
    architectures = getattr(owner, "architectures", None)
    if isinstance(architectures, str):
        names.add(architectures)
    elif architectures is not None:
        names.update(name for name in architectures if isinstance(name, str))
    return names


def is_dflash2_draft(speculative_config: Any, method: str | None = None) -> bool:
    """Return whether a DFlash config selects the DFlash2 draft architecture."""
    method = method or getattr(speculative_config, "method", None)
    if method != "dflash":
        return False
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    hf_text_config = getattr(draft_model_config, "hf_text_config", None)
    model_arch_config = getattr(draft_model_config, "model_arch_config", None)
    architectures: set[str] = set()
    for owner in (draft_model_config, hf_config, hf_text_config, model_arch_config):
        architectures.update(_architecture_names(owner))
    return "DFlash2DraftModel" in architectures


def is_multi_kv_cache_group_mtp(vllm_config: Any) -> bool:
    """Return whether the MTP draft model uses more than one physical KV group."""
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    architectures = set(getattr(hf_config, "architectures", None) or ())
    return model_type in _MULTI_KV_CACHE_GROUP_MTP_MODEL_TYPES or bool(
        architectures & _MULTI_KV_CACHE_GROUP_MTP_ARCHITECTURES
    )


def uses_ascend_mtp_speculator(family: SpecDecodeFamily) -> bool:
    """Return whether MRV2 should route the family through its generic MTP speculator.

    MRV1 has a dedicated multi-KV-group proposer, while MRV2 currently shares
    the generic Ascend MTP speculator for both MTP families.
    """
    return family in (
        SpecDecodeFamily.MTP,
        SpecDecodeFamily.MULTI_KV_CACHE_GROUP_MTP,
    )


def resolve_spec_decode_family(vllm_config: Any, method: str | None = None) -> SpecDecodeFamily:
    """Resolve the execution family shared by MRV1 and MRV2 dispatch.

    Keeping this classification independent from either runner prevents the
    two entry points from disagreeing about special draft architectures and
    MTP variants. Runner modules remain responsible for constructing their own
    proposer or speculator implementation.
    """
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return SpecDecodeFamily.UNKNOWN

    method = method or getattr(speculative_config, "method", None)
    if method == "ngram":
        return SpecDecodeFamily.NGRAM
    if method == "ngram_gpu":
        return SpecDecodeFamily.NGRAM_GPU
    if method == "suffix":
        return SpecDecodeFamily.SUFFIX
    if method == "medusa":
        return SpecDecodeFamily.MEDUSA
    if method == "dspark":
        return SpecDecodeFamily.DSPARK
    if method == "draft_model":
        return SpecDecodeFamily.DRAFT_MODEL
    if method == "extract_hidden_states":
        return SpecDecodeFamily.EXTRACT_HIDDEN_STATES
    if method == "dflash":
        if is_dflash2_draft(speculative_config, method=method):
            return SpecDecodeFamily.DFLASH2
        return SpecDecodeFamily.DFLASH
    if method in ("eagle", "eagle3", "mtp"):
        if method == "mtp" and _call_bool(speculative_config, "use_gemma4_mtp"):
            return SpecDecodeFamily.GEMMA4_MTP
        if _call_bool(speculative_config, "use_step3p5_mtp"):
            return SpecDecodeFamily.STEP3P5_MTP
        if method == "mtp" and is_multi_kv_cache_group_mtp(vllm_config):
            return SpecDecodeFamily.MULTI_KV_CACHE_GROUP_MTP
        return SpecDecodeFamily.MTP if method == "mtp" else SpecDecodeFamily.EAGLE
    return SpecDecodeFamily.UNKNOWN
