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

import torch
import random

class ExpertMapUtils():

#     @classmethod
#     def generate_index_dicts(cls, tensor_2d):
#         dict_list = []
#         current_idx = 0
#
#         for row in tensor_2d:
#             value_to_index = {}
#             for i in range(row.size(0)):
#                 value = row[i].item()
#                 value_to_index[value] = current_idx + i
#             dict_list.append(value_to_index)
#             current_idx += row.size(0)
#
#         return dict_list

    @classmethod
    def generate_log2phy_map(cls, expert_map):
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
                log2phy_map[negative_rank_idx, idx] = torch.full((num_ranks - 1,),
                                    log2phy_map[positive_rank_idx, idx].item(),
                                    dtype=log2phy_map.dtype)
            else:
                random_list = [random.choice(log2phy_map[positive_rank_idx, idx])
                    for _ in range(num_ranks - num_rank_holding_expert)]
                log2phy_map[negative_rank_idx, idx] = torch.tensor(random_list,\
                    dtype=log2phy_map.dtype)

        return log2phy_map

    @classmethod
    def global2local(cls,
        placement: torch.Tensor,
        E_local: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        G, _ = placement.shape
        device = placement.device

        pt_local = torch.full(( G, E_local),
                              fill_value=-1,
                              dtype=torch.long,
                              device=device)

        valid = placement >= 0
        g_idx, k_idx = valid.nonzero(as_tuple=True)
        slot_idx = placement[g_idx, k_idx]

        pt_local[g_idx, slot_idx] = k_idx

        return pt_local
