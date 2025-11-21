# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import time

import torch

from ..utils import calculate_time
from .optimizers import Optimizer


class HEAT_ExpertsBalancer(Optimizer):
    """
    TokenBalance optimizer class inherits from Optimizer.

    This optimizer is designed to balance tokens across experts based on a specified capacity factor.
    It initializes necessary mappings based on cluster status and determines the expert-per-device count.
    """

    def __init__(self,
                 cluster_status,
                 rank=None,
                 is_global_maximum_offset=False,
                 num_device_per_host=8):
        """
        Initializes the TokenBalance optimizer instance.

        Parameters:
            cluster_status: Object containing the current status and mapping information of the cluster.

        """

        # Extract expert mapping information from the provided cluster status
        self.is_global_maximum_offset = is_global_maximum_offset
        self.placement_pattern = cluster_status.placement_pattern
        self.device = self.placement_pattern.device
        if rank is not None:
            self.rank = rank
        else:
            self.rank = cluster_status.rank

        # Calculate the number of experts assigned per device.
        # It sums the expert mapping tensor on the first row of the first dimension.
        # Adding 0.5 before converting to integer ensures proper rounding.

        expert_sums = torch.sum(self.placement_pattern, dim=-1)  # 对最后一维求和
        self.expert_per_device = int(torch.max(expert_sums).item() +
                                     0.5)  # 求所有和的最大值

        # Build a mapping from expert IDs back to their positions.
        # This is required for later optimization steps.

        self.num_device_per_host = num_device_per_host
        self._build_loacl_expert_mapping_super_()
        cluster_status.expert_mapping.local_expert_mapping += self.expert_mapping_
        self.local_expert_mapping = cluster_status.expert_mapping.local_expert_mapping
        self.optimize = self._DEVICE_SPECIFIC_optimize

    """Optimizes based solely on scores, ignoring loads."""

    def _DEVICE_SPECIFIC_optimize(
        self, layer_idx_moe: int, tokens: torch.Tensor,
        token_expert_id: torch.Tensor, token_scores: torch.Tensor,
        cluster_status: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimizes token distribution based on `expert_position` by directly mapping to global expert IDs.

        Args:
            layer_idx_moe (int):
                Index of the current Mixture-of-Experts (MoE) layer.
            tokens (torch.Tensor):
                Token data with shape [batch_size, seq_len, hidden_dim].
            token_expert_id (torch.Tensor):
                Initially assigned expert IDs for each token, with shape [batch_size, seq_len, top_k].
            token_scores (torch.Tensor):
                Scores corresponding to each token's assigned expert, with shape [batch_size, seq_len, top_k].
            cluster_status (list):
                Cluster status information (currently unused).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Optimized token distribution information:
                - tokens: The original token tensor, shape [batch_size, seq_len, hidden_dim].
                - token_expert_placement_ids: Updated expert IDs, shape [batch_size, seq_len, top_k].
                - token_scores: Original expert score tensor, shape [batch_size, seq_len, top_k].
        """
        # Return the updated token distribution
        return tokens, self.local_expert_mapping[layer_idx_moe,
                                                 token_expert_id], token_scores

    def _build_loacl_expert_mapping_super_(self):
        """
        混合优化版本：结合向量化和针对性循环，保留核心的 rank % count 分配逻辑。
        """
        num_devices, num_layers, num_epids = self.placement_pattern.shape

        # 1. 计算每个 expert 在多少个 device 上存在
        expert_counts = torch.sum(
            self.placement_pattern, dim=0,
            dtype=torch.int)  # shape: (num_layers, num_epids)

        # 初始化结果张量
        placement_pattern_super = torch.zeros_like(self.placement_pattern,
                                                   dtype=torch.int)

        # --- 处理 expert 只存在于一个 device 上的情况 (向量化) ---
        single_device_mask = (expert_counts == 1)
        placement_pattern_super = torch.where(single_device_mask.unsqueeze(0),
                                              self.placement_pattern,
                                              placement_pattern_super)

        # --- 处理 expert 存在于多个 (>=2) device 上的情况 ---
        multi_device_mask = (expert_counts >= 2)

        # --- 子情况 1: 当前 device (self.rank) 拥有这个 multi-device expert (向量化) ---
        current_device_has_it_mask = (self.placement_pattern[self.rank]
                                      == 1) & multi_device_mask
        placement_pattern_super[self.rank, current_device_has_it_mask] = 1

        # --- 子情况 2: 当前 device (self.rank) *不* 拥有这个 multi-device expert (识别 + 循环处理) ---
        other_devices_have_it_mask = (
            ~(self.placement_pattern[self.rank] == 1)) & multi_device_mask

        # 获取需要进行模数分配的专家的 layer 和 expert 索引
        l_indices, e_indices = torch.where(other_devices_have_it_mask)

        # 如果存在需要特殊处理的专家，则进行循环
        if len(l_indices) > 0:
            # 提取这些专家在所有设备上的映射情况，方便查找
            # relevant_mapping shape: (num_devices, num_relevant_experts)
            relevant_mapping = self.placement_pattern[:, l_indices, e_indices]
            # 提取这些专家的设备计数值
            # relevant_counts shape: (num_relevant_experts,)
            relevant_counts = expert_counts[l_indices, e_indices]

            # 计算目标索引 (在拥有该专家的设备列表中的索引)
            # target_indices_in_group shape: (num_relevant_experts,)
            target_indices_in_group = self.rank % relevant_counts

            # 循环处理每一个需要模数分配的专家
            for i in range(len(l_indices)):
                layer_idx = l_indices[i]
                expert_idx = e_indices[i]
                target_idx_in_group = target_indices_in_group[i]

                # 找到拥有当前专家 (layer_idx, expert_idx) 的设备列表
                # devices_with_expert shape: (count,)
                devices_with_expert = torch.where(relevant_mapping[:,
                                                                   i] == 1)[0]

                # 为了确保一致性，最好对设备列表进行排序
                # (原始代码没有明确排序，但依赖列表顺序可能不稳定)
                sorted_devices = torch.sort(devices_with_expert)[0]

                # 根据计算出的模数索引选择目标设备
                target_device = sorted_devices[target_idx_in_group]

                # 在 placement_pattern_super 中标记目标设备
                placement_pattern_super[target_device, layer_idx,
                                        expert_idx] = 1

        # 调用后续处理函数
        if not self.is_global_maximum_offset:
            self.expert_mapping_ = self._construct_expert_mapping_from_placement_pattern(
                placement_pattern_super)
            self.placement_pattern_super = placement_pattern_super
        else:
            self.expert_mapping_ = self._construct_expert_mapping_from_placement_pattern_with_global_offset(
                placement_pattern_super)
            self.placement_pattern_super = placement_pattern_super

        return 0

    def _construct_expert_mapping_from_placement_pattern(
            self, expert_mappingX):
        """
        计算 `expert_position`，从 `expert_mappingX` 生成唯一的 global expert 索引 (向量化版本)。
        此版本支持每层每个设备的专家数量可以不同。

        Args:
            expert_mappingX (torch.Tensor): 形状为 (num_devices, num_layers, num_epids) 的 0/1 张量，
                                            表示专家是否放置在特定设备、层和 epid 位置。

        Returns:
            - torch.Tensor: 形状为 (num_layers, num_epids) 的张量，存储每个 expert 的唯一索引，或 -1。
        """

        placement_pattern = self.placement_pattern
        # 1. 计算本地累积和 (沿最后一个维度，即 num_epids)
        #    这决定了专家在其所在设备和层内的本地序号 (1-based)
        # cumsum_local shape: (num_devices, num_layers, num_epids)
        cumsum_local = torch.cumsum(placement_pattern, dim=2)

        # 2. 计算本地 0-based 索引
        # local_indices shape: (num_devices, num_layers, num_epids)
        local_indices = cumsum_local - 1

        # --- 新增：计算层级设备偏移量 ---
        # 3. 计算每个设备在每层实际拥有的专家数量
        # experts_per_device_layer shape: (num_devices, num_layers)
        experts_per_device_layer = torch.sum(placement_pattern,
                                             dim=2,
                                             dtype=torch.long)

        # 4. 计算每层内，设备专家数量的前缀和 (cumulative sum across devices)
        # cumulative_experts_per_layer shape: (num_devices, num_layers)
        # cumulative_experts_per_layer[d, l] = sum(experts on devices 0 to d in layer l)
        cumulative_experts_per_layer = torch.cumsum(experts_per_device_layer,
                                                    dim=0)

        # 5. 计算每个设备在每层的偏移量
        # 偏移量是其前面所有设备在该层上的专家总数
        # device_offset_layer shape: (num_devices, num_layers)
        device_offset_layer = torch.zeros_like(cumulative_experts_per_layer)
        # 将前缀和向下移动一位，设备 d 的偏移量等于设备 d-1 的前缀和
        device_offset_layer[1:, :] = cumulative_experts_per_layer[:-1, :]
        # 将偏移量扩展维度以进行广播
        # device_offset_layer shape: (num_devices, num_layers, 1) -> broadcast to (num_devices, num_layers, num_epids)
        device_offset_layer = device_offset_layer.unsqueeze(2)
        # --- 结束新增部分 ---

        # 6. 计算所有位置的潜在全局索引 (使用新的层级偏移量)
        # potential_global_indices shape: (num_devices, num_layers, num_epids)
        potential_global_indices = device_offset_layer + local_indices

        # 7. 创建全局掩码 (哪些位置实际有专家)
        # mask_all shape: (num_devices, num_layers, num_epids)
        mask_all = expert_mappingX == 1

        # 8. 初始化结果张量（使用一个中间态，形状与 potential_global_indices 相同）
        # 用 -1 填充，这样在后续取 max 时，没有专家的位置会保持 -1
        global_indices_masked = torch.full_like(potential_global_indices,
                                                -1,
                                                dtype=torch.long)

        # 9. 使用掩码，只在专家存在的位置填充计算出的全局索引
        global_indices_masked[mask_all] = potential_global_indices[mask_all]

        # 10. 沿着设备维度 (dim=0) 取最大值进行降维
        # 假设每个 (layer, epid) 最多只有一个设备拥有该专家。
        # 如果没有任何设备拥有该专家，max(-1, -1, ...) 结果是 -1。
        # expert_position shape: (num_layers, num_epids)
        # torch.max 返回 (values, indices)，我们只需要 values
        expert_position = torch.max(global_indices_masked, dim=0)[0]

        return expert_position.to(torch.int32)

    def _construct_expert_mapping_from_placement_pattern_with_global_offset(
            self, expert_mappingX):
        """
        计算 `expert_position`，从 `expert_mappingX` 生成唯一的 global expert 索引 (向量化版本)。
        此版本支持每层每个设备的专家数量可以不同。

        Args:
            expert_mappingX (torch.Tensor): 形状为 (num_devices, num_layers, num_epids) 的 0/1 张量，
                                            表示专家是否放置在特定设备、层和 epid 位置。

        Returns:
            - torch.Tensor: 形状为 (num_layers, num_epids) 的张量，存储每个 expert 的唯一索引，或 -1。
        """

        placement_pattern = self.placement_pattern
        # 1. 计算本地累积和 (沿最后一个维度，即 num_epids)
        #    这决定了专家在其所在设备和层内的本地序号 (1-based)
        # cumsum_local shape: (num_devices, num_layers, num_epids)
        cumsum_local = torch.cumsum(placement_pattern, dim=2)

        # 2. 计算本地 0-based 索引
        # local_indices shape: (num_devices, num_layers, num_epids)
        local_indices = cumsum_local - 1

        # --- 新增：计算层级设备偏移量 ---
        # 3. 计算每个设备在每层实际拥有的专家数量, 使用全局最大的作为offset
        # experts_per_device_layer shape: (num_devices, num_layers)
        # 先按原来的方式计算每个设备每层的专家数量
        experts_per_device_layer_original = torch.sum(placement_pattern,
                                                      dim=2,
                                                      dtype=torch.long)

        # 在num_layers维度上取最大值，确保每个设备在所有层中使用相同数量的专家
        experts_per_device_max = torch.max(experts_per_device_layer_original,
                                           dim=1,
                                           keepdim=True)[0]

        # 将最大值复制到每一层，形成最终的experts_per_device_layer
        experts_per_device_layer = experts_per_device_max.expand_as(
            experts_per_device_layer_original)
        # 4. 计算每层内，设备专家数量的前缀和 (cumulative sum across devices)
        # cumulative_experts_per_layer shape: (num_devices, num_layers)
        # cumulative_experts_per_layer[d, l] = sum(experts on devices 0 to d in layer l)
        cumulative_experts_per_layer = torch.cumsum(experts_per_device_layer,
                                                    dim=0)

        # 5. 计算每个设备在每层的偏移量
        # 偏移量是其前面所有设备在该层上的专家总数
        # device_offset_layer shape: (num_devices, num_layers)
        device_offset_layer = torch.zeros_like(cumulative_experts_per_layer)
        # 将前缀和向下移动一位，设备 d 的偏移量等于设备 d-1 的前缀和
        device_offset_layer[1:, :] = cumulative_experts_per_layer[:-1, :]
        # 将偏移量扩展维度以进行广播
        # device_offset_layer shape: (num_devices, num_layers, 1) -> broadcast to (num_devices, num_layers, num_epids)
        device_offset_layer = device_offset_layer.unsqueeze(2)
        # --- 结束新增部分 ---

        # 6. 计算所有位置的潜在全局索引 (使用新的层级偏移量)
        # potential_global_indices shape: (num_devices, num_layers, num_epids)
        potential_global_indices = device_offset_layer + local_indices

        # 7. 创建全局掩码 (哪些位置实际有专家)
        # mask_all shape: (num_devices, num_layers, num_epids)
        mask_all = expert_mappingX == 1

        # 8. 初始化结果张量（使用一个中间态，形状与 potential_global_indices 相同）
        # 用 -1 填充，这样在后续取 max 时，没有专家的位置会保持 -1
        global_indices_masked = torch.full_like(potential_global_indices,
                                                -1,
                                                dtype=torch.long)

        # 9. 使用掩码，只在专家存在的位置填充计算出的全局索引
        global_indices_masked[mask_all] = potential_global_indices[mask_all]

        # 10. 沿着设备维度 (dim=0) 取最大值进行降维
        # 假设每个 (layer, epid) 最多只有一个设备拥有该专家。
        # 如果没有任何设备拥有该专家，max(-1, -1, ...) 结果是 -1。
        # expert_position shape: (num_layers, num_epids)
        # torch.max 返回 (values, indices)，我们只需要 values
        expert_position = torch.max(global_indices_masked, dim=0)[0]

        return expert_position.to(torch.int32)
