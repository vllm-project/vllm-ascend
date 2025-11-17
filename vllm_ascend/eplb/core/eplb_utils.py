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
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove eplb utils.
import os.path
import random
import sys

import torch
from vllm.logger import logger


def determine_default_expert_map(global_expert_num, world_size, rank_id,
                                 global_redundant_expert_num):
    if world_size == 1:
        local_ids = torch.arange(global_expert_num, dtype=torch.int32)
        return (global_expert_num, local_ids)

    local_num_experts = global_expert_num // world_size

    expert_map = torch.full((global_expert_num, ), -1, dtype=torch.int32)

    if rank_id < world_size - 1:
        start = rank_id * local_num_experts
        end = (rank_id + 1) * local_num_experts
        local_count = local_num_experts
    else:
        start = rank_id * local_num_experts
        end = global_expert_num
        local_count = global_expert_num - rank_id * local_num_experts

    if isinstance(local_count, int):
        local_ids = torch.arange(local_count, dtype=torch.int32)
        expert_map[start:end] = local_ids

    return (local_count, expert_map)


def generate_log2phy_map(expert_map):
    num_local_experts = expert_map.max() + 1
    log2phy_map = expert_map.clone()
    num_ranks, num_global_expert = log2phy_map.shape

    row_indices = torch.arange(num_ranks).view(-1, 1).expand(num_ranks, \
                                                             num_global_expert) * num_local_experts
    log2phy_map[log2phy_map != -1] += row_indices[log2phy_map != -1]

    for idx in range(num_global_expert):
        positive_rank_idx = torch.where(log2phy_map[:, idx] != -1)[0]
        negative_rank_idx = torch.where(log2phy_map[:, idx] == -1)[0]
        num_rank_holding_expert = positive_rank_idx.size(0)

        if num_rank_holding_expert == 0:
            log2phy_map[:, idx] = torch.full((num_ranks, ),
                                             0,
                                             dtype=log2phy_map.dtype)

        if num_rank_holding_expert == 1:
            log2phy_map[negative_rank_idx, idx] = torch.full(
                (num_ranks - 1, ),
                log2phy_map[positive_rank_idx, idx].item(),
                dtype=log2phy_map.dtype)
        else:
            try:
                random_list = [
                    random.choice(log2phy_map[positive_rank_idx, idx])
                    for _ in range(num_ranks - num_rank_holding_expert)
                ]
                log2phy_map[negative_rank_idx,
                            idx] = torch.tensor(random_list,
                                                dtype=log2phy_map.dtype)
            except Exception as e:
                logger.error(f"Fail to get log2phy_map: {str(e)}")

    return log2phy_map


def determine_default_log2phy_map(global_expert_num, world_size, rank_id):
    if world_size == 1:
        local_ids = torch.arange(global_expert_num, dtype=torch.int32)
        expert_map_all = local_ids.unsqueeze(0).expand(world_size, -1)
        log2phy_map_all = generate_log2phy_map(expert_map_all)
        return log2phy_map_all[rank_id]

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

        if isinstance(local_count, int):
            local_ids = torch.arange(local_count, dtype=torch.int32)
            expert_map_all[r, start:end] = local_ids

    log2phy_map_all = generate_log2phy_map(expert_map_all)

    return log2phy_map_all[rank_id]


class EPLBParamUtils:

    @staticmethod
    def check_iterations(iterations):
        if not isinstance(iterations, int):
            raise TypeError(f"The {iterations} is not int.")
        if iterations <= 0:
            raise ValueError(
                f"The {iterations} can not less than or equal to 0.")
        if iterations > sys.maxsize:
            raise ValueError(
                f"The {iterations} can not large than {sys.maxsize}")

    @staticmethod
    def check_dynamic_eplb(dynamic_eplb):
        if dynamic_eplb is None:
            return
        if not isinstance(dynamic_eplb, bool):
            raise TypeError("The dynamic_eplb is not bool.")
        if dynamic_eplb and os.getenv("DYNAMIC_EPLB", "false") != "true":
            raise ValueError(
                'Can not enable dynamic_eplb when not export DYNAMIC_EPLB="true".'
            )

    @staticmethod
    def check_expert_map_path(expert_map):
        if expert_map is None:
            return
        if not isinstance(expert_map, str):
            raise TypeError("The expert_map is not str.")
        if not expert_map.strip():
            raise ValueError("The expert_map is not empty.")
        _, ext = os.path.splitext(expert_map)
        if ext.lower() != ".json":
            raise TypeError("The expert_map is not json.")
        if not os.path.exists(expert_map):
            raise ValueError("The expert_map is not exist.")
        try:
            with open(expert_map, "w", encoding='utf-8') as f:
                f.read()
        except Exception as e:
            raise IOError(
                f"Fail read expert info from {expert_map}, please check the reading permission of {expert_map} : {e}"
            )

    @staticmethod
    def check_expert_map_record_path(expert_map_record_path):
        if expert_map_record_path is None:
            return
        if not isinstance(expert_map_record_path, str):
            raise TypeError("The expert_map_record_path is not str.")
        if not expert_map_record_path.strip():
            raise ValueError("The expert_map_record_path is empty.")
        _, ext = os.path.splitext(expert_map_record_path)
        if ext.lower() != ".json":
            raise TypeError("The expert_map_record_path is not json.")
        if os.getenv("EXPERT_MAP_RECORD", "false") != "true":
            raise ValueError(
                'Can not enable expert_map_record_path when not export EXPERT_MAP_RECORD="true".'
            )
        try:
            with open(expert_map_record_path, "w", encoding='utf-8') as f:
                f.write("")
        except Exception as e:
            raise IOError(
                f"Fail write expert info to {expert_map_record_path}, please check the writing permission of {expert_map_record_path} : {e}"
            )
