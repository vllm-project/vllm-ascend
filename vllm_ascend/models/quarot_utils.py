# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared QuaRot weight-alignment helpers for draft models.

This module must stay importable independently of any DSpark/DFlash
model module; do not import model classes here.
"""

import torch
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

from vllm_ascend.models.llama_eagle3 import load_quarot_target_layer
from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
)

TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)


# Process the first linear weight with rotation matrix, if the target model uses rotary quantization
def process_weight(linear_weight: torch.Tensor, rotation_weight: torch.Tensor):
    assert linear_weight.shape[1] % rotation_weight.shape[0] == 0, (
        f"Linear weight shape[1] must be a multiple of rotation weight shape[0],"
        f" but get {linear_weight.shape[1]=} and {rotation_weight.shape[0]=}"
    )
    rotation_weight = rotation_weight.to(device=linear_weight.device, dtype=torch.float32)
    hidden_size = rotation_weight.shape[0]
    ori_dtype = linear_weight.dtype
    processed_weight = torch.empty(linear_weight.shape, dtype=torch.float32, device=linear_weight.device)
    for start_pos in range(0, linear_weight.shape[1], hidden_size):
        linear_weight_chunked = linear_weight[:, start_pos : start_pos + hidden_size].to(torch.float32)
        processed_weight[:, start_pos : start_pos + hidden_size].copy_(
            torch.matmul(linear_weight_chunked, rotation_weight)
        )
    return processed_weight.to(ori_dtype)


@torch.no_grad()
def align_draft_weights(model, projection, vllm_config):
    """Align draft inputs with the rotated target without modifying shared weights."""
    rotation_path = get_rotation_path(vllm_config)
    if rotation_path is not None:
        rotation = get_rotation_matrix(rotation_path).cpu()
        weight = projection.weight
        weight.copy_(process_weight(weight.cpu(), rotation).to(weight.device))
    else:
        if get_pp_group().world_size == 1:
            return
        # PP stages past the first cannot alias the target's stage-0
        # embedding; load this rank's vocab shard from the target
        # checkpoint unrotated so the draft owns its copy.
        rotation = None
        weight = projection.weight
    target_config = vllm_config.model_config.hf_text_config
    for owner, name, layer_cls, weight_names, own_flag in (
        (model.model, "embed_tokens", VocabParallelEmbedding, TARGET_EMBED_WEIGHT_NAMES, "has_own_embed_tokens"),
        (model, "lm_head", ParallelLMHead, TARGET_LM_HEAD_WEIGHT_NAMES, "has_own_lm_head"),
    ):
        if getattr(model, own_flag, False):
            continue
        if rotation is None and name != "embed_tokens":
            # The target lm_head lives on the last PP stage and can still
            # be aliased; only the embedding needs its own copy.
            continue
        with torch.device(weight.device):
            layer = layer_cls(target_config.vocab_size, target_config.hidden_size, params_dtype=weight.dtype)
        load_quarot_target_layer(layer, vllm_config.model_config.model, weight_names, rotation, f"draft {name}.weight")
        layer.quant_method.process_weights_after_loading(layer)
        setattr(owner, name, layer)
        setattr(model, own_flag, True)
