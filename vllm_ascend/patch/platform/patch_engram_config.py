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

"""Allow Engram on Ascend until the vLLM pin includes removal of the CUDA gate."""

import importlib.util

# Older vLLM builds have no Engram config to patch.
if importlib.util.find_spec("vllm.config.engram") is not None:
    from vllm.config.engram import EngramConfig, model_has_engram_layers

    def verify_model_config(self, model_config) -> None:
        # Keep upstream's model/layer checks; only its CUDA requirement is lifted.
        if not model_has_engram_layers(model_config):
            raise ValueError("EngramConfig requires a supported model with non-empty n-gram layer ids.")

    EngramConfig.verify_model_config = verify_model_config
