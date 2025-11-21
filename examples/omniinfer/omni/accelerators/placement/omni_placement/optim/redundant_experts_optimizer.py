# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import time

import torch

from ..utils import calculate_time
from .optimizers import Optimizer


class Redundant2ExpertsBalancer(Optimizer):
    """
    TokenBalance optimizer class inherits from Optimizer.

    This optimizer is designed to balance tokens across experts based on a specified capacity factor.
    It initializes necessary mappings based on cluster status and determines the expert-per-device count.
    """
    def __init__(self, cluster_status, rank = None, capacity_factor=1.0, Device_specific = False, num_device_per_host = 8):
        """
        Initializes the TokenBalance optimizer instance.

        Parameters:
            cluster_status: Object containing the current status and mapping information of the cluster.

        """

        # Extract expert mapping information from the provided cluster status
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
        self.expert_per_device = int(torch.max(expert_sums).item() + 0.5)  # 求所有和的最大值

        # Build a mapping from expert IDs back to their original positions.
        # This is required for later optimization steps.
        if Device_specific:
            self.num_device_per_host = num_device_per_host
            self._build_loacl_expert_mapping_with_frozen_part()
            cluster_status.expert_mapping.local_expert_mapping += self.expert_mapping_frozen
            self.local_expert_mapping = cluster_status.expert_mapping.local_expert_mapping
            self.optimize = self._DEVICE_SPECIFIC_optimize

        else:
            self._build_expert_mapping()
            self.optimize = self._NOT_DEVICE_SPECIFIC_optimize



    """Optimizes based solely on scores, ignoring loads."""

    def _NOT_DEVICE_SPECIFIC_optimize(self,
                layer_idx_moe: int,
                tokens: torch.Tensor,
                token_expert_id: torch.Tensor,
                token_scores: torch.Tensor,
                cluster_status: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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

        # Determine the midpoint of token_expert_id
        midpoint = token_expert_id.shape[0] // 2

        # Map the first half of token_expert_id using expert_position_A
        token_expert_id[:midpoint] = self.expert_position_A[layer_idx_moe, token_expert_id[:midpoint]]

        # Map the second half of token_expert_id using expert_position_B
        token_expert_id[midpoint:] = self.expert_position_B[layer_idx_moe, token_expert_id[midpoint:]]


        # Return the updated token distribution
        return tokens, token_expert_id, token_scores


    """Optimizes based solely on scores, ignoring loads."""

    def _DEVICE_SPECIFIC_optimize(self,
                layer_idx_moe: int,
                tokens: torch.Tensor,
                token_expert_id: torch.Tensor,
                token_scores: torch.Tensor,
                cluster_status: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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
        return tokens, self.local_expert_mapping[layer_idx_moe, token_expert_id], token_scores



    def _build_loacl_expert_mapping_with_frozen_part(self):
        """
        向量化版本：计算用于机内冗余的 placement_pattern_frozen。

        Returns:
            - int: 0 (表示成功，与原函数一致)
            Sets self.expert_mapping_frozen internally.
        """
        num_devices, num_layers, num_epids = self.placement_pattern.shape
        placement_pattern = self.placement_pattern # 使用 self.placement_pattern

        # --- 1. 计算每个 (layer, epid) 的副本数 ---
        # expert_counts shape: (num_layers, num_epids)
        expert_counts = torch.sum(placement_pattern, dim=0)

        # 初始化 placement_pattern_frozen
        placement_pattern_frozen = torch.zeros_like(placement_pattern, dtype=torch.int)

        # --- 2. 处理唯一专家 (counts == 1)
        unique_mask_2d = (expert_counts == 1)      # unique_mask_2d shape: (num_layers, num_epids)
        # unique_mask_3d shape: (1, num_layers, num_epids) -> broadcast to (num_devices, num_layers, num_epids)
        unique_mask_3d = unique_mask_2d.unsqueeze(0)
        # 在 placement_pattern_frozen 中设置唯一专家的位置, 只有当原始 placement_pattern 中为 1 且计数为 1 时才设置
        placement_pattern_frozen = torch.where(unique_mask_3d & (placement_pattern == 1), 1, placement_pattern_frozen)

        # --- 3. 处理冗余专家 (counts >= 2) ---  redundant_mask_2d shape: (num_layers, num_epids)
        redundant_mask_2d = (expert_counts >= 2)
        # redundant_locations_3d: 标记原始 placement 中冗余专家的具体位置  shape: (num_devices, num_layers, num_epids)
        redundant_locations_3d = redundant_mask_2d.unsqueeze(0) & (placement_pattern == 1)

        # 如果没有任何冗余专家，可以提前结束冗余部分的计算
        if not redundant_locations_3d.any():
            self.expert_mapping_frozen = self._construct_expert_mapping_from_placement_pattern(placement_pattern_frozen)
            return 0

        # --- 3a. 向量化组检查 --- host_ids shape: (num_devices,)
        host_ids = torch.arange(num_devices, device=self.device) // self.num_device_per_host
        # current_host_id: scalar
        current_host_id = self.rank // self.num_device_per_host
        # is_on_current_host shape: (num_devices, 1, 1)
        is_on_current_host = (host_ids == current_host_id).view(-1, 1, 1)

        # group_has_it_map: 标记每个 (layer, epid) 是否在当前 host 上有副本  shape: (1, num_layers, num_epids)  计算时只考虑冗余专家所在的原始位置
        group_has_it_map = (torch.sum(placement_pattern * is_on_current_host, dim=0, keepdim=True) > 0)
        # 确保只在真正冗余的 (layer, epid) 上判断 group_has_it
        group_has_it_map = group_has_it_map & redundant_mask_2d.unsqueeze(0)


        # --- 3b. 向量化条件检查 ---
        # 确保 num_devices > 0 且 num_epids 可被 num_devices 整除（或处理好边界）
        assert num_devices > 0, "num_devices must be positive"
        experts_per_device_ideal = num_epids // num_devices
        if experts_per_device_ideal == 0:
            # 处理 num_epids < num_devices 的情况，可能需要调整逻辑或报错
            print(f"Warning/Error: num_epids ({num_epids}) < num_devices ({num_devices}). Partitioning logic might be invalid.")
            # 根据具体需求决定如何处理，这里可能直接返回或使用备用逻辑
            # 为了代码能运行，暂时假设 ideal 分区只对 epid < num_devices*1 有效
            experts_per_device_ideal = 1 # 示例处理，需根据实际情况调整

        expected_device_id = (torch.arange(num_epids, device=self.device) // experts_per_device_ideal).view(1, 1, -1) # expected_device_id shape: (1, 1, num_epids)
        # Clamp expected_device_id to be within valid device range [0, num_devices-1], 这在 num_epids 不是 num_devices 的精确倍数时很重要
        expected_device_id = torch.clamp(expected_device_id, max=num_devices - 1)


        # device_indices shape: (num_devices, 1, 1)
        device_indices = torch.arange(num_devices, device=self.device).view(-1, 1, 1)

        # redundancy_condition shape: (num_devices, 1, num_epids) -> broadcasts
        redundancy_condition = (expected_device_id != device_indices)
        # frozen_condition shape: (num_devices, 1, num_epids) -> broadcasts
        frozen_condition = (expected_device_id == device_indices)

        # --- 3c. 组合逻辑 ---
        # 条件1: 本地组有副本，且满足冗余条件  shape: (num_devices, num_layers, num_epids)
        set_when_group_has = group_has_it_map & redundancy_condition

        # 条件2: 本地组无副本，且满足 Frozen 条件  shape: (num_devices, num_layers, num_epids)
        set_when_group_has_not = (~group_has_it_map) & frozen_condition

        # 最终决定哪些冗余位置应该被设置 (需要是原始冗余位置之一)  shape: (num_devices, num_layers, num_epids)
        final_redundant_mask = redundant_locations_3d & (set_when_group_has | set_when_group_has_not)

        # 更新 placement_pattern_frozen 中冗余专家的位置
        placement_pattern_frozen = torch.where(final_redundant_mask, 1, placement_pattern_frozen)

        # --- 4. 生成最终映射 ---
        self.expert_mapping_frozen = self._construct_expert_mapping_from_placement_pattern(placement_pattern_frozen)

        return 0 # 保持与原函数一致的返回值


    def _build_expert_mapping(self):
        """
        Vectorized calculation of expert_position_A and expert_position_B.

        Returns:
            - int: 0 (for consistency with original)
            Sets self.expert_position_A and self.expert_position_B internally.
        """
        num_devices, num_layers, num_epids = self.placement_pattern.shape
        placement_pattern = self.placement_pattern # Use the class attribute

        # --- 1. Count devices per expert ---
        # expert_counts shape: (num_layers, num_epids)
        expert_counts = torch.sum(placement_pattern, dim=0)

        # --- 2. Create masks for counts 1 and 2 ---  mask_1_2d shape: (num_layers, num_epids)
        mask_1_2d = (expert_counts == 1)
        # mask_2_2d shape: (num_layers, num_epids)
        mask_2_2d = (expert_counts == 2)

        # --- 3. Find min and max device indices for experts present ---  device_indices_3d shape: (num_devices, 1, 1) -> broadcasts
        device_indices_3d = torch.arange(num_devices, device=self.device).view(-1, 1, 1)

        # Create masked indices for min/max calculation.
        # Replace 0s in placement_pattern with values that won't affect min/max.
        # For min: replace 0 with a large value (num_devices)
        # For max: replace 0 with a small value (-1)
        masked_indices_for_min = torch.where(placement_pattern == 1, device_indices_3d, num_devices)
        masked_indices_for_max = torch.where(placement_pattern == 1, device_indices_3d, -1)

        # min/max_dev_idx_2d shape: (num_layers, num_epids)
        # These hold the lowest/highest device index for *any* expert present at (j, epid)
        min_dev_idx_2d = torch.min(masked_indices_for_min, dim=0)[0]
        max_dev_idx_2d = torch.max(masked_indices_for_max, dim=0)[0]

        # --- 4. Construct expert_mapping_A ---
        # Condition 1: Unique expert (count=1) at its original location
        cond1_A = mask_1_2d.unsqueeze(0) & (placement_pattern == 1)
        # Condition 2: Two experts (count=2), select the one on the *minimum* device index
        cond2_A = mask_2_2d.unsqueeze(0) & (device_indices_3d == min_dev_idx_2d.unsqueeze(0))
        # Combine conditions: set to 1 if either condition is met
        expert_mapping_A = torch.where(cond1_A | cond2_A, 1, 0).int()


        # --- 5. Construct expert_mapping_B ---
        # Condition 1: Unique expert (count=1) at its original location (same as A)
        cond1_B = cond1_A # Reuse
        # Condition 2: Two experts (count=2), select the one on the *maximum* device index
        cond2_B = mask_2_2d.unsqueeze(0) & (device_indices_3d == max_dev_idx_2d.unsqueeze(0))
        # Combine conditions
        expert_mapping_B = torch.where(cond1_B | cond2_B, 1, 0).int()

        # --- 6. Call Position Function ---
        self.expert_position_A = self._construct_expert_mapping_from_placement_pattern(expert_mapping_A)
        self.expert_position_B = self._construct_expert_mapping_from_placement_pattern(expert_mapping_B)

        return 0 # Maintain original return signature


    def _construct_expert_mapping_from_placement_pattern(self, expert_mappingX):
            """
            计算 `expert_position`，从 `expert_mapping` 生成唯一的 global expert 索引 (向量化版本)。

            Args:
                expert_mappingX (torch.Tensor): 形状为 (num_devices, num_layers, num_epids) 的 0/1 张量。

            Returns:
                - torch.Tensor: 形状为 (num_layers, num_epids) 的张量，存储每个 expert 的唯一索引，或 -1。
            """
            num_devices, num_layers, num_epids = expert_mappingX.shape

            # 假设 self.placement_pattern 存在且形状与 expert_mappingX 相同
            # 如果 placement_pattern 的获取方式不同，需要调整
            placement_pattern = self.placement_pattern # 假设它在这里可用

            # 1. 全局计算累积和 (沿最后一个维度，即 num_epids)
            # cumsum_all shape: (num_devices, num_layers, num_epids)
            cumsum_all = torch.cumsum(placement_pattern, dim=2)

            # 2. 计算本地 0-based 索引
            # local_indices shape: (num_devices, num_layers, num_epids)
            local_indices = cumsum_all - 1

            # 3. 创建设备索引张量并计算设备偏移  device_indices shape: (num_devices,) -> (num_devices, 1, 1)
            device_indices = torch.arange(num_devices, device=self.device, dtype=torch.long).view(-1, 1, 1)
            # device_offset shape: (num_devices, 1, 1) -> broadcast to (num_devices, num_layers, num_epids)
            device_offset = device_indices * self.expert_per_device

            # 4. 计算所有位置的潜在全局索引  potential_global_indices shape: (num_devices, num_layers, num_epids)
            potential_global_indices = device_offset + local_indices

            # 5. 创建全局掩码
            # mask_all shape: (num_devices, num_layers, num_epids)
            mask_all = expert_mappingX == 1

            # 6. 初始化结果张量（使用一个中间态，形状与 potential_global_indices 相同）
            # 用 -1 填充，这样在后续取 max 时，没有专家的位置会保持 -1
            global_indices_masked = torch.full_like(potential_global_indices, -1, dtype=torch.long)

            # 7. 使用掩码，只在专家存在的位置填充计算出的全局索引
            global_indices_masked[mask_all] = potential_global_indices[mask_all]

            # 8. 沿着设备维度 (dim=0) 取最大值进行降维
            # 假设每个 (layer, epid) 最多只有一个设备拥有该专家。
            # 如果有多个设备（不应该发生），max 会取到索引值最大的那个设备计算出的全局索引。
            # 如果没有任何设备拥有该专家，max(-1, -1, ...) 结果是 -1。
            # expert_position shape: (num_layers, num_epids)
            # torch.max 返回 (values, indices)，我们只需要 values
            expert_position = torch.max(global_indices_masked, dim=0)[0]

            return expert_position.to(torch.int32)


