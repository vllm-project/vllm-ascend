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

class ExpertMapUtils():

    @classmethod
    def generate_index_dicts(cls, tensor_2d):
        dict_list = []
        current_idx = 0

        for row in tensor_2d:
            value_to_index = {}
            for i in range(row.size(0)):
                value = row[i].item()
                value_to_index[value] = current_idx + i
            dict_list.append(value_to_index)
            current_idx += row.size(0)

        return dict_list

    @classmethod
    def generate_log2phy_map(cls, expert_map):
        num_local_experts = expert_map.max() + 1
        expert_map = cls.global2local(expert_map, num_local_experts)
        ranks_num, global_expert_num = expert_map.shape
        concatenated = torch.flatten(expert_map)
        rank_expert_to_global = cls.generate_index_dicts(expert_map)
        result_dict: Dict[int, List[int]] = {}
        for idx, value in enumerate(concatenated):
            key = value.item()
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(idx)

        log2phy_map = torch.full((ranks_num, self.global_expert_num),
                                    -1,
                                    dtype=torch.int32)
        for rank in range(ranks_num):
            for key in result_dict:
                indices_in_concat = result_dict[key]
                if key in rank_expert_to_global[rank]:
                    log2phy_map[rank][key] = rank_expert_to_global[rank][key]
                else:
                    chosen_index = random.choice(indices_in_concat)
                    log2phy_map[rank][key] = chosen_index
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
