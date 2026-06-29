# SPDX-License-Identifier: Apache-2.0
"""Load helper for the DSpark draft model on Ascend NPU.

Mirrors the upstream ``vllm/v1/worker/gpu/spec_decode/dspark/utils.py:load_dspark_model``
(vllm PR #46995). DSpark V4 checkpoints ship the draft weights inside the
target's ``mtp.*`` namespace and do NOT carry a private copy of ``embed_tokens``
or ``lm_head`` — we have to alias both from the loaded target model. The
self-contained dense-target variants (e.g. ``deepseek-ai/dspark_qwen3_8b_block7``)
flag ``dspark_shares_target_embeddings = False`` to skip the aliasing.

Constraints (raised explicitly so we don't silently produce wrong outputs):

* No pipeline parallelism (PP=1)
* No prefill / decode context parallelism (PCP/DCP = 1)
"""

import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None, "load_dspark_model needs a speculative_config."
    draft_model_config = speculative_config.draft_model_config

    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism (PP > 1).")
    parallel_cfg = vllm_config.parallel_config
    if getattr(parallel_cfg, "prefill_context_parallel_size", 1) > 1:
        raise NotImplementedError("DSpark does not support prefill context parallelism (PCP > 1).")
    if getattr(parallel_cfg, "decode_context_parallel_size", 1) > 1:
        raise NotImplementedError("DSpark does not support decode context parallelism (DCP > 1).")

    draft_model = get_model(vllm_config=vllm_config, model_config=draft_model_config)

    # Self-contained dense DSpark drafts (e.g. Qwen3 variant) ship their own
    # embed_tokens / lm_head — aliasing the target's would overwrite freshly
    # loaded draft weights. Only the DeepSeek-V4 family (weights live in the
    # target checkpoint's mtp.* namespace) wants the alias.
    if not getattr(draft_model, "dspark_shares_target_embeddings", True):
        logger.info_once("DSpark draft owns its own embed_tokens / lm_head; skipping target alias.")
        return draft_model

    target_language_model = (
        target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
    )
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    if target_embed is not None:
        if getattr(draft_inner, "embed_tokens", None) is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = getattr(target_model, "lm_head", None)
    if target_lm_head is not None:
        if getattr(draft_model, "lm_head", None) is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    logger.info_once("DSpark draft aliased target embed_tokens + lm_head.")
    return draft_model
