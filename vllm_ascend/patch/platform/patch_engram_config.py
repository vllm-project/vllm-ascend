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

"""Allow Engram on Ascend and host table sharing across PCP replicas."""

import importlib.util
from copy import copy

from vllm_ascend.utils import vllm_version_is

# Older vLLM builds have no Engram config to patch.
if importlib.util.find_spec("vllm.config.engram") is not None:
    from vllm.config.engram import EngramConfig, model_has_engram_layers

    if vllm_version_is("0.31.0"):

        def verify_model_config(self, model_config) -> None:
            # vLLM main already removed the CUDA requirement in #59171.
            if not model_has_engram_layers(model_config):
                raise ValueError("EngramConfig requires a supported model with non-empty n-gram layer ids.")

        EngramConfig.verify_model_config = verify_model_config

    _original_verify_parallel_config = EngramConfig.verify_parallel_config

    # TODO: Remove this override once all supported vLLM versions natively
    # support Engram PCP, including DP x PCP groups and parallel validation.
    def verify_parallel_config(self, parallel_config) -> None:
        """Count PCP peers for host sharing, retaining upstream validation."""
        if self.dp_shared_memory and parallel_config.data_parallel_size == 1:
            parallel_config = copy(parallel_config)
            parallel_config.data_parallel_size = parallel_config.prefill_context_parallel_size
        _original_verify_parallel_config(self, parallel_config)

    EngramConfig.verify_parallel_config = verify_parallel_config
