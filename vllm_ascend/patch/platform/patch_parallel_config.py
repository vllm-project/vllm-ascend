#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# Allow sequence-parallel MoE with a single data-parallel rank.
#
# Upstream vLLM gates ParallelConfig.use_sequence_parallel_moe on
# data_parallel_size > 1. Ascend's SP dispatch (EP-group all-gather /
# reduce-scatter in vllm_ascend.ops.fused_moe with dp_size guards, TP-based
# sp_shard in models, DP-tolerant DPMetadata in forward_context) is valid
# with DP=1, so this patch drops only the DP requirement. Every reader
# (models, enable_sp, ascend_config gates, use_all2all) keys off this one
# property, keeping the switch consistent.

from vllm.config.parallel import ParallelConfig


@property  # type: ignore[misc]
def _ascend_use_sequence_parallel_moe(self) -> bool:
    return (
        self.all2all_backend
        in (
            "allgather_reducescatter",
            "deepep_high_throughput",
            "deepep_low_latency",
            "mori_high_throughput",
            "mori_low_latency",
            "nixl_ep",
        )
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
    )


ParallelConfig.use_sequence_parallel_moe = _ascend_use_sequence_parallel_moe
