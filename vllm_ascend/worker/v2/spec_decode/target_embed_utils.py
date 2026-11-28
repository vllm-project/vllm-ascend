# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
from vllm.distributed import GroupCoordinator
from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.models.utils import PPMissingLayer


def _unwrap_embed_base(module: Any) -> Any:
    return module.base_layer if isinstance(module, BaseLayerWithLoRA) else module


def _references_embed(candidate: Any, target_embed: Any, target_base: Any) -> bool:
    if candidate is None or isinstance(candidate, PPMissingLayer):
        return False
    candidate_base = _unwrap_embed_base(candidate)
    if candidate is target_embed or candidate is target_base or candidate_base is target_base:
        return True
    target_weight = getattr(target_base, "weight", None)
    return target_weight is not None and getattr(candidate_base, "weight", None) is target_weight


def _config_ties_word_embeddings(config: Any) -> bool:
    if config is None:
        return False
    get_text_config = getattr(config, "get_text_config", None)
    text_config = get_text_config() if callable(get_text_config) else config
    return bool(getattr(text_config, "tie_word_embeddings", False) or getattr(config, "tie_word_embeddings", False))


def _target_needs_embed_on_last_stage(
    target_model: Any,
    target_language_model: Any,
    target_inner: Any,
    target_embed: Any,
    target_base: Any,
    vllm_config: Any,
) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    configs = (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
        getattr(target_inner, "config", None),
        getattr(target_language_model, "config", None),
        getattr(target_model, "config", None),
    )
    if any(_config_ties_word_embeddings(cfg) for cfg in configs):
        return True
    lm_heads = (getattr(target_language_model, "lm_head", None), getattr(target_model, "lm_head", None))
    return any(_references_embed(lm_head, target_embed, target_base) for lm_head in lm_heads)


def release_target_embed(
    speculator: Any,
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module | None,
    pp_group: GroupCoordinator | None = None,
) -> None:
    """Release an unused target input embedding from the final PP stage."""
    if draft_model is None:
        return
    if pp_group is None or pp_group.world_size == 1:
        pp_group = speculator._target_pp_group
    if pp_group.world_size <= 1 or not pp_group.is_last_rank:
        return

    get_language_model = getattr(target_model, "get_language_model", None)
    language_model: Any = get_language_model() if callable(get_language_model) else target_model
    target_inner = getattr(language_model, "model", language_model)
    target_embed = getattr(target_inner, "embed_tokens", None)
    if target_embed is None or isinstance(target_embed, PPMissingLayer):
        return

    target_base = _unwrap_embed_base(target_embed)
    if _target_needs_embed_on_last_stage(
        target_model,
        language_model,
        target_inner,
        target_embed,
        target_base,
        speculator.vllm_config,
    ):
        return
    if _references_embed(getattr(speculator, "target_embeddings", None), target_embed, target_base):
        return

    draft_inner = getattr(draft_model, "model", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if _references_embed(draft_embed, target_embed, target_base):
        return

    target_inner.embed_tokens = PPMissingLayer()
