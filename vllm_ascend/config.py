# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/kernels/test_moe.py
# Copyright 2023 The vLLM team.
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

import torch
from torch.distributed import ProcessGroup, ReduceOp
from vllm.config import ParallelConfig


def stateless_init_dp_group(self) -> "ProcessGroup":
    from vllm.distributed.utils import \
        stateless_init_torch_distributed_process_group

    dp_group = stateless_init_torch_distributed_process_group(
        self.data_parallel_master_ip,
        self.get_next_dp_init_port(),
        self.data_parallel_rank,
        self.data_parallel_size,
        backend="hccl")

    return dp_group

def has_unfinished_dp(dp_group: "ProcessGroup",
                      has_unfinished: bool) -> bool:

    tensor = torch.tensor([has_unfinished],
                          dtype=torch.int32,
                          device="npu")
    torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
    aggregated_has_unfinished = bool(tensor.item())
    return aggregated_has_unfinished



ParallelConfig.stateless_init_dp_group = stateless_init_dp_group
ParallelConfig.has_unfinished_dp = has_unfinished_dp
