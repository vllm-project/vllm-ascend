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

"""Allow Ascend Engram and validate PCP peers with the native config type."""

import importlib.util
from copy import copy

# Older vLLM builds have no Engram config to patch.
if importlib.util.find_spec("vllm.config.engram") is not None:
    from vllm.config.engram import EngramConfig, model_has_engram_layers

    def verify_model_config(self, model_config) -> None:
        # Keep upstream's model/layer checks; only its CUDA requirement is lifted.
        if not model_has_engram_layers(model_config):
            raise ValueError("EngramConfig requires a supported model with non-empty n-gram layer ids.")

    EngramConfig.verify_model_config = verify_model_config

    _verify_parallel_config = EngramConfig.verify_parallel_config

    def verify_parallel_config(self, parallel_config) -> None:
        storage_parallel_config = parallel_config
        pcp = parallel_config.prefill_context_parallel_size
        if self.dp_shared_memory and pcp > 1:
            # Native validation counts DP-only replicas. Include PCP storage
            # peers in a copy without changing runtime DP or native config types.
            storage_parallel_config = copy(parallel_config)
            storage_parallel_config.data_parallel_size *= pcp
        _verify_parallel_config(self, storage_parallel_config)
        if pcp <= 1:
            return
        tp = parallel_config.tensor_parallel_size
        dp = parallel_config.data_parallel_size
        if (
            parallel_config.enable_elastic_ep
            or tp not in (1, 2, 4, 8)
            or min(dp, pcp) < 1
            or tp * dp * pcp > 16
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.decode_context_parallel_size != 1
            or parallel_config.nnodes != 1
            or (not parallel_config.data_parallel_external_lb and parallel_config.data_parallel_size_local != dp)
        ):
            raise ValueError(
                "Ascend Engram PCP requires single-node TP=1/2/4/8 with at most 16 ranks, "
                "with all DP replicas local and PP=DCP=1."
            )

    EngramConfig.verify_parallel_config = verify_parallel_config
