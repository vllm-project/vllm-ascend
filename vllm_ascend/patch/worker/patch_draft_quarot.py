import logging
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    process_eagle_weight,
)

logger = logging.getLogger(__name__)


def get_embedding_tensor(directory_path):
    """
    Scans the directory and returns the first tensor found that contains 'embed' in its key.
    Returns the tensor if found, otherwise None.
    """
    if not os.path.isdir(directory_path):
        return None

    # List files and filter for .safetensors
    for filename in os.listdir(directory_path):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory_path, filename)

            # Load the file
            state_dict = load_file(file_path)

            # Search for the first matching key
            for key, tensor in state_dict.items():
                if "embed" in key.lower():
                    # Return immediately once found
                    return tensor

    return None


def get_rotation_path(target_vllm_config):
    """
    Gets the path of the rotation matrix, returns None if the target model is not a quarot model.
    """
    target_model_path = target_vllm_config.model_config.model
    try:
        quant_description = target_vllm_config.quant_config.quant_description
        rotation_relative_path = quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    except KeyError:
        return None

    return Path(target_model_path) / rotation_relative_path


def get_rotataion_matrix(rotation_path):
    """
    Anti-rotate maxtrix.
    """
    try:
        safetensor_data = load_file(rotation_path)
        Q = safetensor_data["global_rotation"]

        return Q
    except Exception as e:
        logger.error(
            f"Failed to load rotation weight from '{rotation_path}'. "
            "If you want to use quarot model with eagle3, take a check."
        )
        raise e


def compute_rotataion_matrix3(Q):
    """
    Anti-rotate matrix for 3 layers of hidden_states.
    """
    return torch.block_diag(Q, Q, Q)


def patch_load_weights(target_vllm_config):
    target_model_path = Path(target_vllm_config.model_config.model)
    rotation_path = get_rotation_path(target_vllm_config)

    # if rotation path is not found, then quarot is not in use.
    if rotation_path is None:
        return

    Eagle3LlamaForCausalLM.load_weights = make_load_weights(target_model_path, rotation_path)


def make_load_weights(target_model_path, rotation_path):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        Q = get_rotataion_matrix(rotation_path)
        Q3 = compute_rotataion_matrix3(Q)
        if isinstance(self.config.dtype, str):
            embed_dtype = getattr(torch, self.config.dtype)
        else:
            embed_dtype = self.config.dtype

        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "fc." in name:
                # anti-rotate fc
                dtype = loaded_weight.dtype
                loaded_weight = loaded_weight @ Q3.to(dtype)
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # process embedding if drafter does not have embedding
        if not includes_embed_tokens:
            name = "model.embed_tokens.weight"
            loaded_weight = get_embedding_tensor(target_model_path).to(embed_dtype) @ Q.T.to(embed_dtype)
            model_weights[name] = loaded_weight

            includes_embed_tokens = True
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())

    return load_weights
