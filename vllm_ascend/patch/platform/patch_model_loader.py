#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Patch for model loader to handle Ascend quantization computed parameters.

This patch extends the weight loading tracking to handle Ascend-specific
quantization parameters that are computed rather than loaded from checkpoint.
"""

import torch.nn as nn
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


def _patched_track_weights_loading(self, model: nn.Module, loaded_weights: set[str] | None) -> None:
    """Extended weight loading tracking for Ascend quantization.

    This method extends the original track_weights_loading to also handle:
    1. Ascend quantization computed parameters (weight_offset, quant_bias,
       deq_scale, weight_scale) that are computed during process_weights_after_loading
    2. Optional layer norm biases that may not exist in some checkpoints
    """
    weights_to_load = {name for name, _ in model.named_parameters()}
    if loaded_weights is not None:
        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            # Ignore kv_cache scale, which can be missing in checkpoints
            if isinstance(quant_method, BaseKVCacheMethod):
                for param_name, _ in module.named_parameters():
                    full_name = f"{name}.{param_name}" if name else param_name
                    loaded_weights.add(full_name)
            # Handle computed parameters from quantization methods
            # These are computed during process_weights_after_loading
            # rather than loaded from checkpoint
            if quant_method is not None:
                computed_params = getattr(quant_method, "get_computed_params", lambda: set())()
                if computed_params:
                    for param_name, _ in module.named_parameters():
                        if any(c in param_name for c in computed_params):
                            full_name = f"{name}.{param_name}" if name else param_name
                            loaded_weights.add(full_name)
        weights_not_loaded = weights_to_load - loaded_weights
        # Filter out optional bias parameters that may not exist in checkpoint
        # Some models create bias parameters but checkpoints don't have them
        # (e.g., LayerNorm biases in MLA attention)
        weights_not_loaded = {w for w in weights_not_loaded if not w.endswith(".bias")}
        if weights_not_loaded:
            raise ValueError(f"Following weights were not initialized from checkpoint: {weights_not_loaded}")


from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.16.0"):
    DefaultModelLoader.track_weights_loading = _patched_track_weights_loading
