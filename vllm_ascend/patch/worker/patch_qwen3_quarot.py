import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    process_eagle_weight,
)


def patch_load_weights(target_config):
    Eagle3LlamaForCausalLM.load_weights = make_load_weights(target_config)


def make_load_weights(target_config):
    logger = logging.getLogger(__name__)
    quant_cfg = target_config.quant_config
    rotation_matrix3 = None

    model_path = target_config.model_config.model
    try:
        rotation_rel_path = quant_cfg.quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    except KeyError as e:
        logger.error(
            "Invalid quant_config: missing key "
            "quant_description['optional']['quarot']['rotation_map']['global_rotation']. "
            "If you don't use quarot model, please ignore it. "
            f"Error: {e}"
        )
    else:
        rotation_path = Path(model_path) / rotation_rel_path
        try:
            safetensor_data = load_file(rotation_path)
            Q = safetensor_data["global_rotation"]
            rotation_matrix3 = torch.block_diag(Q, Q, Q)
        except Exception as e:
            logger.error(
                f"Failed to load rotation weight from '{rotation_path}'. "
                "If you don't use quarot model, please ignore it. "
                f"Error: {e}"
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
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
            if "fc." in name and rotation_matrix3 is not None:
                loaded_weight = loaded_weight @ rotation_matrix3.to(loaded_weight.dtype)
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
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
