import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

from vllm_ascend.utils import (
    get_rotation_matrix,
    get_rotation_path,
)

logger = logging.getLogger(__name__)


def get_embedding_tensor(directory_path):
    """Scans the directory and returns the first tensor found that contains 'embed' in its key."""
    if not os.path.isdir(directory_path):
        return None
    for filename in os.listdir(directory_path):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory_path, filename)
            state_dict = load_file(file_path)
            for key, tensor in state_dict.items():
                if "embed" in key.lower():
                    return tensor
    return None


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
    rotation: torch.Tensor,
    label: str,
) -> None:
    """Load one target vocab shard into the draft's unrotated hidden basis."""
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

    rotation = rotation.to(
        device=layer.weight.device,
        dtype=torch.float32,
    )
    target_weight = target_weight.to(
        device=layer.weight.device,
        dtype=torch.float32,
    )
    aligned_weight = torch.matmul(target_weight, rotation.T)
    loaded_rows = aligned_weight.shape[0]
    layer.weight.data[:loaded_rows].copy_(aligned_weight.to(layer.weight.dtype))
    layer.weight.data[loaded_rows:].zero_()
    logger.info(
        "[spec_decode/quarot] Loaded and aligned %s from %s (%s).",
        label,
        shard_path.name,
        tuple(layer.weight.shape),
    )


def compute_rotation_matrix3(Q: torch.Tensor) -> torch.Tensor:
    """Anti-rotate matrix for 3 layers of hidden_states."""
    return torch.block_diag(Q, Q, Q)


class AscendEagle3LlamaForCausalLM(Eagle3LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.target_model_path = Path(vllm_config.model_config.model)
        self.rotation_path = get_rotation_path(vllm_config)
        self.is_quarot_used = self.rotation_path is not None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if self.is_quarot_used:
            Q = get_rotation_matrix(self.rotation_path)
            Q3 = compute_rotation_matrix3(Q)
            if isinstance(self.config.dtype, str):
                embed_dtype = getattr(torch, self.config.dtype)
            else:
                embed_dtype = self.config.dtype
            processed_weights: list[tuple[str, torch.Tensor]] = []
            includes_embed_tokens = False
            for name, loaded_weight in weights:
                if "fc." in name:
                    dtype = loaded_weight.dtype
                    loaded_weight = (loaded_weight.to(torch.float32) @ Q3.to(torch.float32)).to(dtype)
                if "embed_tokens" in name:
                    includes_embed_tokens = True
                processed_weights.append((name, loaded_weight))

            if not includes_embed_tokens:
                embed_weight = (
                    get_embedding_tensor(self.target_model_path).to(torch.float32) @ Q.T.to(torch.float32)
                ).to(embed_dtype)
                processed_weights.append(("embed_tokens.weight", embed_weight))
            super().load_weights(processed_weights)
        else:
            super().load_weights(weights)
