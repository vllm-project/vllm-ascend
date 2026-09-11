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
# Step3.5/3.7 on Ascend: allow sequence-parallel MoE at DP=1.

from vllm.config.parallel import ParallelConfig

# Module-level flags caching the architecture decision. They are only set once
# a config context is active (i.e. during worker init / model construction);
# accesses before that (e.g. forward-time reads) return False without caching,
# so the decision can be re-resolved on a later access.


# Replicates vllm.config.parallel.ParallelConfig.use_sequence_parallel_moe
# (vllm main). Upstream requires a DP group because the
# allgather_reducescatter EP backend uses it for the expert dispatch
# all-to-all; Ascend handles EP communication itself, so the TP-only (DP=1)
# layout is valid for Step3.5/3.7. Keep the whitelist in sync with upstream.
def _use_sequence_parallel_moe(self) -> bool:
    return (
        self.all2all_backend
        in (
            "allgather_reducescatter",
            "deepep_high_throughput",
            "deepep_low_latency",
            "flashinfer_nvlink_one_sided",
            "mori_high_throughput",
            "mori_low_latency",
            "nixl_ep",
        )
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
    )


ParallelConfig.use_sequence_parallel_moe = property(_use_sequence_parallel_moe)
