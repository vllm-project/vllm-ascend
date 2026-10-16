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

"""Allow Engram on Ascend with compatible node-local table groups."""

import importlib.util
from functools import wraps
from math import gcd

# Older vLLM builds have no Engram config to patch.
if importlib.util.find_spec("vllm.config.engram") is not None:
    from vllm.config import get_current_vllm_config
    from vllm.config.engram import EngramConfig, model_has_engram_layers
    from vllm.distributed import parallel_state

    def verify_model_config(self, model_config) -> None:
        # Keep upstream's model/layer checks; only its CUDA requirement is lifted.
        if not model_has_engram_layers(model_config):
            raise ValueError("EngramConfig requires a supported model with non-empty n-gram layer ids.")

    # TODO: Remove this override once all supported vLLM versions natively
    # support Engram PCP, including DP x PCP groups and parallel validation.
    def verify_parallel_config(self, parallel_config) -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1 and parallel_config.prefill_context_parallel_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires data_parallel_size > 1 or prefill_context_parallel_size > 1."
                )
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1 and parallel_config.enable_elastic_ep:
            raise ValueError("Engram embedding_across_dp is not supported with elastic EP yet.")

    EngramConfig.verify_model_config = verify_model_config
    EngramConfig.verify_parallel_config = verify_parallel_config

    _original_engram_dp_shard_size = parallel_state._engram_dp_shard_size

    @wraps(_original_engram_dp_shard_size)
    def _engram_dp_shard_size(world_size: int, data_parallel_size: int, replica_size: int) -> int:
        # Keep upstream's node/replica checks, including its EDP=1 fallback.
        shard_size = _original_engram_dp_shard_size(world_size, data_parallel_size, replica_size)
        config = get_current_vllm_config()
        engram = config.engram_config
        model = config.model_config
        if engram is None or engram.dp_shared_memory or model is None or model.architecture != "DeepseekV41ForCausalLM":
            return shard_size

        # Ascend uses uniform TP x EDP head shards. A shared host table only
        # shards heads over TP, so its sharing group must not be reduced here.
        text_config = model.hf_text_config
        num_heads = text_config.engram_n_heads * (text_config.engram_max_ngram_size - 1)
        tp_size = config.parallel_config.tensor_parallel_size
        if num_heads <= 0 or num_heads % tp_size:
            raise ValueError(
                f"Engram requires a positive hash-head count divisible by TP: {num_heads} heads, TP={tp_size}."
            )
        # This is the largest divisor of the node-local candidate that also
        # divides the heads remaining after TP. Never enlarge the node domain.
        return gcd(shard_size, num_heads // tp_size)

    parallel_state._engram_dp_shard_size = _engram_dp_shard_size
