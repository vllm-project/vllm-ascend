#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model detection and configuration utilities for vLLM Ascend.

This module provides functionality for:
- Detecting model types (MoE, VL, drafter MoE)
- Checking model capabilities (RoPE, layer indexing)
- Getting model configuration
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

_IS_MOE_MODEL = None
_IS_DRAFTER_MOE_MODEL = None
_IS_VL_MODEL = None
_HAS_LAYER_IDX = None
_HAS_ROPE = None


def get_max_hidden_layers(hf_config) -> int:
    cfg_dict = hf_config.to_dict()
    layer_counts = []

    def _rec_find(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "num_hidden_layers" and isinstance(v, int):
                    layer_counts.append(v)
                else:
                    _rec_find(v)

    _rec_find(cfg_dict)
    if not layer_counts:
        raise ValueError("Not found num_hidden_layers in model config.")
    return max(layer_counts)


def is_moe_model(vllm_config: VllmConfig):
    """Checks if the model is a MoE model by config"""
    global _IS_MOE_MODEL
    if _IS_MOE_MODEL is None:
        model_configs = vllm_config.model_config.hf_text_config.to_dict()
        _IS_MOE_MODEL = _is_contain_expert(model_configs)
    return _IS_MOE_MODEL


def is_drafter_moe_model(vllm_config: VllmConfig):
    """Checks if the drafter model is a MoE model by config"""
    global _IS_DRAFTER_MOE_MODEL
    if _IS_DRAFTER_MOE_MODEL is None:
        model_configs = vllm_config.speculative_config.draft_model_config.hf_text_config \
            .to_dict()
        _IS_DRAFTER_MOE_MODEL = _is_contain_expert(model_configs)
    return _IS_DRAFTER_MOE_MODEL


def speculative_enable_dispatch_gmm_combine_decode(
        vllm_config: VllmConfig) -> bool:
    if vllm_config.speculative_config is None:
        return True
    speculative_method = getattr(vllm_config.speculative_config, "method",
                                 None)
    if speculative_method in [None, "ngram", "suffix"]:
        return True
    if speculative_method in ["eagle", "eagle3"]:
        return False
    if speculative_method == "mtp":
        mtp_quant_type = getattr(vllm_config.model_config.hf_text_config,
                                 "mtp_quantize", None)
        return mtp_quant_type == "w8a8_dynamic"
    return False


def _is_contain_expert(config: Any):
    if isinstance(config, dict):
        for k, v in config.items():
            if "expert" in str(k):
                return True
            if _is_contain_expert(v):
                return True
    return False


def is_vl_model(vllm_config: VllmConfig):
    """Checks if the model is a VL model by config"""
    global _IS_VL_MODEL
    if _IS_VL_MODEL is None and vllm_config and vllm_config.model_config:
        hf_config = vllm_config.model_config.hf_config.to_dict()
        if "thinker_config" in hf_config:
            # Qwen-Omni-thinker models
            _IS_VL_MODEL = True
        else:
            _IS_VL_MODEL = "vision_config" in hf_config
    return _IS_VL_MODEL


def has_rope(vllm_config: VllmConfig):
    """Checks if the model uses rope."""
    global _HAS_ROPE
    if _HAS_ROPE is None and vllm_config and vllm_config.model_config:
        hf_config = vllm_config.model_config.hf_text_config.to_dict()
        _HAS_ROPE = "rope_parameters" in hf_config
    return _HAS_ROPE


def has_layer_idx(model_instance: Any) -> bool:
    if model_instance is None:
        return False

    global _HAS_LAYER_IDX
    if _HAS_LAYER_IDX is None:
        _HAS_LAYER_IDX = hasattr(model_instance, "model") and \
            hasattr(model_instance.model, "start_layer")
    return _HAS_LAYER_IDX
