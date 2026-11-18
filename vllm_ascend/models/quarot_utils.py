# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared QuaRot weight-alignment helpers for draft models.

This module must stay importable independently of any DSpark/DFlash
model module; do not import model classes here.
"""

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
)

logger = logging.getLogger(__name__)

TARGET_EMBED_WEIGHT_NAMES = (
    "language_model.model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
TARGET_LM_HEAD_WEIGHT_NAMES = (
    "language_model.lm_head.weight",
    "lm_head.weight",
)


def _find_safetensors_weight(
    model_path: Path,
    weight_names: tuple[str, ...],
) -> tuple[Path, str]:
    """Locate one target tensor without loading unrelated checkpoint shards."""
    for index_path in sorted(model_path.glob("*.safetensors.index.json")):
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})
        for weight_name in weight_names:
            if shard_name := weight_map.get(weight_name):
                return model_path / shard_name, weight_name

    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
        for weight_name in weight_names:
            if weight_name in shard_keys:
                return shard_path, weight_name

    raise KeyError(f"None of {weight_names!r} was found in the target checkpoint at {model_path}.")


@torch.inference_mode()
def load_quarot_target_layer(
    layer: nn.Module,
    target_model_path: Path | str,
    weight_names: tuple[str, ...],
    rotation: torch.Tensor | None,
    label: str,
) -> None:
    """Load one target vocab shard into the draft's unrotated hidden basis.

    ``rotation=None`` loads the shard unchanged (no QuaRot alignment)."""
    target_model_path = Path(target_model_path)
    shard_path, weight_name = _find_safetensors_weight(
        target_model_path,
        weight_names,
    )
    shard_indices = getattr(layer, "shard_indices", None)
    if shard_indices is None:
        start_index = 0
        end_index = layer.weight.shape[0]
    else:
        start_index = shard_indices.org_vocab_start_index
        end_index = shard_indices.org_vocab_end_index

    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        target_weight = shard.get_slice(weight_name)[start_index:end_index]

    if rotation is not None:
        rotation = rotation.to(
            device=layer.weight.device,
            dtype=torch.float32,
        )
        target_weight = target_weight.to(
            device=layer.weight.device,
            dtype=torch.float32,
        )
        aligned_weight = torch.matmul(target_weight, rotation.T)
    else:
        # No rotation: plain shard copy for drafts that cannot alias the
        # target layer (e.g. the target embedding on a later PP stage).
        aligned_weight = target_weight.to(
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
    loaded_rows = aligned_weight.shape[0]
    layer.weight.data[:loaded_rows].copy_(aligned_weight.to(layer.weight.dtype))
    layer.weight.data[loaded_rows:].zero_()
    logger.info(
        "[spec_decode/quarot] Loaded and aligned %s from %s (%s).",
        label,
        shard_path.name,
        tuple(layer.weight.shape),
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
