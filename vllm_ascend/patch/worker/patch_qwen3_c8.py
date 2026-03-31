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

from collections.abc import Iterable

import torch
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

_orig_qwen3_causal_lm_load_weights = Qwen3ForCausalLM.load_weights


def _patched_qwen3_causal_lm_load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    quant_config = self.quant_config
    if quant_config is None or not callable(getattr(quant_config, "get_cache_scale", None)):
        return _orig_qwen3_causal_lm_load_weights(self, weights)

    params_dict = dict(self.named_parameters())
    c8_loaded_params: set[str] = set()

    def _intercept_c8_scales(
        raw_weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, loaded_weight in raw_weights:
            scale_name = quant_config.get_cache_scale(name)
            if scale_name is not None:
                if scale_name in params_dict:
                    param = params_dict[scale_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight.squeeze())
                    c8_loaded_params.add(scale_name)
            else:
                yield name, loaded_weight

    loaded_params = _orig_qwen3_causal_lm_load_weights(self, _intercept_c8_scales(weights))
    loaded_params.update(c8_loaded_params)
    return loaded_params


Qwen3ForCausalLM.load_weights = _patched_qwen3_causal_lm_load_weights
