import json
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.logger import logger
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    process_eagle_weight,
)

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
    target_model_path = Path(target_vllm_config.model_config.model)
    quant_config = getattr(target_vllm_config, "quant_config", None)
    quant_description = getattr(quant_config, "quant_description", None)

    if quant_description is None:
        description_path = target_model_path / "quant_model_description.json"
        if not description_path.is_file():
            return None
        with description_path.open(encoding="utf-8") as description_file:
            quant_description = json.load(description_file)

    if not quant_description.get("is_rot_used", True):
        return None

    try:
        rotation_relative_path = quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    except (KeyError, TypeError):
        return None

    return target_model_path / rotation_relative_path


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


@torch.inference_mode()
def transform_quarot_linear_weight(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    target_device: torch.device | str | None = None,
):
    """Rotate each hidden-size input block of a draft linear weight."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape={tuple(weight.shape)}")
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"Expected a square rotation matrix, got shape={tuple(rotation.shape)}")

    hidden_size = rotation.shape[0]
    if weight.shape[1] % hidden_size != 0:
        raise ValueError(
            "Linear input width must be a multiple of the QuaRot hidden size: "
            f"weight={tuple(weight.shape)}, rotation={tuple(rotation.shape)}"
        )

    device = torch.device(target_device) if target_device is not None else weight.device
    transformed = torch.empty(weight.shape, dtype=weight.dtype, device=device)
    rotation_device = rotation.to(device=device, dtype=torch.float32)
    for start in range(0, weight.shape[1], hidden_size):
        weight_chunk = weight[:, start : start + hidden_size].to(
            device=device,
            dtype=torch.float32,
        )
        transformed[:, start : start + hidden_size].copy_(
            torch.matmul(weight_chunk, rotation_device).to(dtype=weight.dtype)
        )
    return transformed


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
        logger.info("Target model does not use QuaRot; draft weight patch is not needed")
        return

    Eagle3LlamaForCausalLM.load_weights = make_load_weights(target_model_path, rotation_path)
    try:
        from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
    except ImportError:
        logger.info("Qwen3 DSpark model is unavailable; only Eagle3 QuaRot loading was patched")
        return

    current_load_weights = Qwen3DSparkForCausalLM.load_weights
    if getattr(current_load_weights, "_vllm_ascend_quarot_wrapper", False):
        logger.info("Qwen3 DSpark QuaRot weight loader is already patched")
        return

    Qwen3DSparkForCausalLM.load_weights = make_qwen3_dspark_load_weights(
        rotation_path,
        current_load_weights,
    )
    logger.warning(
        "Patched Qwen3 DSpark weight loading for target QuaRot rotation: %s",
        rotation_path,
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


def make_qwen3_dspark_load_weights(rotation_path, original_load_weights):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        rotation = get_rotataion_matrix(rotation_path)
        transformed_fc = False

        def transformed_weights():
            nonlocal transformed_fc
            for name, loaded_weight in weights:
                if name == "fc.weight":
                    loaded_weight = transform_quarot_linear_weight(
                        loaded_weight,
                        rotation,
                        target_device=self.model.fc.weight.device,
                    )
                    transformed_fc = True
                yield name, loaded_weight

        result = original_load_weights(self, transformed_weights())
        if transformed_fc:
            logger.warning(
                "Applied target QuaRot rotation to Qwen3 DSpark fc.weight: "
                "shape=%s, rotation=%s",
                tuple(self.model.fc.weight.shape),
                rotation_path,
            )
        else:
            logger.warning(
                "Qwen3 DSpark checkpoint did not provide the expected fc.weight; "
                "target QuaRot rotation was not applied"
            )
        return result

    load_weights._vllm_ascend_quarot_wrapper = True
    return load_weights
