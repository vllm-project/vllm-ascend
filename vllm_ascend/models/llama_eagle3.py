import logging
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.config import VllmConfig
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

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


def get_rotation_path(vllm_config: VllmConfig) -> Path | None:
    quant_config = vllm_config.quant_config
    if quant_config is None:
        return None
    target_model_path = vllm_config.model_config.model
    try:
        quant_description = quant_config.quant_description
        rotation_relative_path = quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    except KeyError:
        return None
    return Path(target_model_path) / rotation_relative_path


def get_rotation_matrix(rotation_path: Path | None) -> torch.Tensor:
    """Load the global rotation matrix."""
    try:
        safetensor_data = load_file(rotation_path)
        Q = safetensor_data["global_rotation"]
        return Q
    except Exception as e:
        logger.error(
            "Failed to load rotation weight from '%s'. If you want to use quarot model with eagle3, take a check.",
            rotation_path,
        )
        raise e


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
