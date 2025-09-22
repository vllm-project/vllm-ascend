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
import torch


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

    if isinstance(global_redundant_expert_num,
                  int) and rank_id < global_redundant_expert_num:
        local_count += 1
        if end < global_expert_num:
            end += 1
        else:
            start -= 1

    if isinstance(local_count, int):
        local_ids = torch.arange(local_count, dtype=torch.int32)
        expert_map[start:end] = local_ids

    return (local_count, expert_map)


def generate_log2phy_map(expert_map):
    """
    Generate a log-to-physical map for experts in a fully vectorized manner.

    Args:
        expert_map: Tensor of shape [num_ranks, num_global_expert], with -1 indicating
                    rank does not hold the expert.

    Returns:
        log2phy_map: Tensor of same shape, mapping logical experts to physical IDs.
    """
    num_ranks, _ = expert_map.shape
    num_local_experts = expert_map.max() + 1
    device = expert_map.device

    # Step 1: linear mapping based on rank
    log2phy_map = expert_map.clone()
    row_indices = torch.arange(num_ranks, device=device).view(
        -1, 1) * num_local_experts
    mask = log2phy_map != -1
    # broadcast addition
    log2phy_map = log2phy_map + row_indices * mask.long()

    # Step 2: find positive/negative positions
    positive_mask = log2phy_map != -1
    negative_mask = ~positive_mask

    # Count number of ranks holding each global expert
    num_positive_per_col = positive_mask.sum(dim=0)  # [num_global_expert]

    # Step 3: handle columns with only one rank holding the expert
    single_pos_mask = num_positive_per_col == 1
    if single_pos_mask.any():
        # get row indices for the positive element in these columns
        # pos_idx = torch.nonzero(positive_mask[:, single_pos_mask], as_tuple=True)
        # broadcast to fill negative positions
        for col_idx in torch.nonzero(single_pos_mask, as_tuple=True)[0]:
            pos_row = torch.nonzero(positive_mask[:, col_idx])[0]
            neg_rows = torch.nonzero(negative_mask[:, col_idx])[:, 0]
            log2phy_map[neg_rows, col_idx] = log2phy_map[pos_row, col_idx]

    # Step 4: handle columns with multiple ranks holding the expert
    multi_pos_mask = num_positive_per_col > 1
    if multi_pos_mask.any():
        for col_idx in torch.nonzero(multi_pos_mask, as_tuple=True)[0]:
            pos_rows = torch.nonzero(positive_mask[:, col_idx])[:, 0]
            neg_rows = torch.nonzero(negative_mask[:, col_idx])[:, 0]
            if len(neg_rows) > 0:
                # random assignment from available positive ranks
                rand_idx = torch.randint(0,
                                         len(pos_rows), (len(neg_rows), ),
                                         device=device)
                log2phy_map[neg_rows,
                            col_idx] = log2phy_map[pos_rows[rand_idx], col_idx]

    return log2phy_map


def determine_default_log2phy_map(global_expert_num, world_size, rank_id,
                                  global_redundant_expert_num):
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

        if isinstance(global_redundant_expert_num,
                      int) and rank_id < global_redundant_expert_num:
            local_count += 1
            if end < global_expert_num:
                end += 1
            else:
                start -= 1

        if isinstance(local_count, int):
            local_ids = torch.arange(local_count, dtype=torch.int32)
            expert_map_all[r, start:end] = local_ids

    log2phy_map_all = generate_log2phy_map(expert_map_all)

    return log2phy_map_all[rank_id]
