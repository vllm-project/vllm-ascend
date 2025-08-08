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
# This file is a part of the vllm-ascend project.
#

import random

import torch


def generate_log2phy_map(expert_map):
    num_local_experts = expert_map.max() + 1
    log2phy_map = expert_map.clone()
    num_ranks, num_global_expert = log2phy_map.shape

    row_indices = torch.arange(num_ranks).view(-1, 1).expand(num_ranks,\
        num_global_expert) * num_local_experts
    log2phy_map[log2phy_map != -1] += row_indices[log2phy_map != -1]

    for idx in range(num_global_expert):
        positive_rank_idx = torch.where(log2phy_map[:, idx] != -1)[0]
        negative_rank_idx = torch.where(log2phy_map[:, idx] == -1)[0]
        num_rank_holding_expert = positive_rank_idx.size(0)

        if num_rank_holding_expert == 1:
            log2phy_map[negative_rank_idx, idx] = torch.full(
                (num_ranks - 1, ),
                log2phy_map[positive_rank_idx, idx].item(),
                dtype=log2phy_map.dtype)
        else:
            random_list = [
                random.choice(log2phy_map[positive_rank_idx, idx])
                for _ in range(num_ranks - num_rank_holding_expert)
            ]
            log2phy_map[negative_rank_idx, idx] = torch.tensor(random_list, \
                                                               dtype=log2phy_map.dtype)

    return log2phy_map


def determine_default_log2phy_map(global_expert_num, world_size, rank_id):
    local_num_experts = global_expert_num // world_size

    expert_map_all = torch.full((world_size, global_expert_num),
                                -1,
                                dtype=torch.int32)

    for r in range(world_size):
        if r < world_size - 1:
            start = r * local_num_experts
            end = (r + 1) * local_num_experts
            local_count = local_num_experts
        else:
            start = r * local_num_experts
            end = global_expert_num
            local_count = global_expert_num - r * local_num_experts

        local_ids = torch.arange(local_count, dtype=torch.int32)
        expert_map_all[r, start:end] = local_ids

    log2phy_map_all = generate_log2phy_map(expert_map_all)

    return log2phy_map_all[rank_id]
