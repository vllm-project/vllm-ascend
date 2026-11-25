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

DEFAULT_QUAROT_ROTATION_PATH = Path("optional/quarot.safetensors")


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
    if not isinstance(quant_description, dict):
        return None

    optional = quant_description.get("optional", {})
    if optional is None:
        optional = {}
    if not isinstance(optional, dict):
        raise ValueError("QuaRot metadata field 'optional' must be a dictionary.")
    has_quarot_metadata = "quarot" in optional
    quarot = optional.get("quarot", {})
    if quarot is None:
        quarot = {}
    if not isinstance(quarot, dict):
        raise ValueError("QuaRot metadata field 'optional.quarot' must be a dictionary.")
    rotation_map = quarot.get("rotation_map", {})
    if rotation_map is None:
        rotation_map = {}
    if not isinstance(rotation_map, dict):
        raise ValueError("QuaRot metadata field 'optional.quarot.rotation_map' must be a dictionary.")
    rotation_relative_path = rotation_map.get("global_rotation")

    if rotation_relative_path is None or rotation_relative_path == "":
        # Older ModelSlim descriptions may omit the optional QuaRot mapping
        # even though the rotation is packaged at the conventional location.
        # Use the same fallback for every DSpark/QuaRot loading path so FC and
        # shared embed/lm_head weights cannot end up in different bases.
        rotation_path = target_model_path / DEFAULT_QUAROT_ROTATION_PATH
        if rotation_path.is_file():
            logger.info(
                "QuaRot rotation mapping is missing; using the default file at %s.",
                rotation_path,
            )
            return rotation_path
        if has_quarot_metadata:
            raise FileNotFoundError(
                "QuaRot metadata is present, but no global rotation file was configured "
                f"and the default file does not exist: {rotation_path}"
            )
        return None

    if not isinstance(rotation_relative_path, (str, os.PathLike)):
        raise ValueError("QuaRot global_rotation must be a filesystem path.")
    rotation_path = Path(rotation_relative_path)
    if not rotation_path.is_absolute():
        rotation_path = target_model_path / rotation_path
    if not rotation_path.is_file():
        raise FileNotFoundError(f"Configured QuaRot rotation file does not exist: {rotation_path}")
    return rotation_path


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
            "Failed to load QuaRot rotation weight from '%s'. Check the target and draft model configuration.",
            rotation_path,
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
