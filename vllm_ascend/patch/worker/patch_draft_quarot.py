import logging
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
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
            "Failed to load rotation weight from '%s'. If you want to use quarot model with eagle3, take a check.",
            rotation_path,
        )
        raise e


def compute_rotataion_matrix3(Q):
    """
    Anti-rotate matrix for 3 layers of hidden_states.
    """
    return torch.block_diag(Q, Q, Q)


def transform_quarot_linear_weight(weight, rotation):
    if weight.ndim != 2 or rotation.ndim != 2:
        raise ValueError(
            f"Expected 2D weight and rotation, got {weight.shape=} {rotation.shape=}"
        )
    hidden_size = rotation.shape[0]
    if rotation.shape[1] != hidden_size or weight.shape[1] % hidden_size != 0:
        raise ValueError(
            "DFlash fc input width must be a multiple of the QuaRot hidden size: "
            f"weight={tuple(weight.shape)}, rotation={tuple(rotation.shape)}"
        )

    output = torch.empty_like(weight)
    rotation_fp32 = rotation.to(device=weight.device, dtype=torch.float32)
    for start in range(0, weight.shape[1], hidden_size):
        end = start + hidden_size
        output[:, start:end] = torch.matmul(
            weight[:, start:end].to(torch.float32), rotation_fp32
        ).to(weight.dtype)
    return output


def patch_load_weights(target_vllm_config):
    target_model_path = Path(target_vllm_config.model_config.model)
    rotation_path = get_rotation_path(target_vllm_config)

    # if rotation path is not found, then quarot is not in use.
    if rotation_path is None:
        return

    Eagle3LlamaForCausalLM.load_weights = make_load_weights(target_model_path, rotation_path)
    original_dflash_load_weights = DFlashQwen3ForCausalLM.load_weights
    if not getattr(original_dflash_load_weights, "_vllm_ascend_quarot_wrapper", False):
        DFlashQwen3ForCausalLM.load_weights = make_dflash_load_weights(
            rotation_path, original_dflash_load_weights
        )


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
                loaded_weight = (loaded_weight.to(torch.float32) @ Q3.to(torch.float32)).to(dtype)
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # process embedding if drafter does not have embedding
        if not includes_embed_tokens:
            name = "model.embed_tokens.weight"
            loaded_weight = (get_embedding_tensor(target_model_path).to(torch.float32) @ Q.T.to(torch.float32)).to(
                embed_dtype
            )
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


def make_dflash_load_weights(rotation_path, original_load_weights):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        rotation = get_rotataion_matrix(rotation_path)
        transformed_fc = False

        def transformed_weights():
            nonlocal transformed_fc
            for name, loaded_weight in weights:
                if name == "fc.weight" or name.endswith(".fc.weight"):
                    loaded_weight = transform_quarot_linear_weight(
                        loaded_weight, rotation
                    )
                    transformed_fc = True
                    logger.info(
                        "Applied target QuaRot rotation to DFlash fc.weight: "
                        "shape=%s, rotation=%s",
                        tuple(loaded_weight.shape),
                        rotation_path,
                    )
                yield name, loaded_weight

        result = original_load_weights(self, transformed_weights())
        if not transformed_fc:
            raise RuntimeError(
                "DFlash checkpoint did not provide fc.weight; target QuaRot "
                "rotation was not applied"
            )
        return result

    load_weights._vllm_ascend_quarot_wrapper = True
    return load_weights
