# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import torch

from .optimizers import Optimizer


class ResDis_ExpertsBalancer(Optimizer):
    """
    TokenBalance optimizer class inherits from Optimizer.

    This optimizer is designed to balance tokens across experts based on a specified capacity factor.
    It initializes necessary mappings based on cluster status and determines the expert-per-device count.
    """

    def __init__(self,
                 cluster_status,
                 rank=None,
                 is_rand_op=False,
                 max_count=16):
        """
        Initializes the TokenBalance optimizer instance.

        Parameters:
            cluster_status: Object containing the current status and mapping information of the cluster.

        """
        # Extract expert mapping information from the provided cluster status
        self.placement_pattern = cluster_status.placement_pattern
        self.world_size, self.num_layers, self.num_eps = self.placement_pattern.shape
        self.device = self.placement_pattern.device
        self.max_count = max_count

        # self.ep2pos_all = torch.zeros(self.num_layers,
        #                               self.num_eps,
        #                               self.max_count,
        #                               dtype = torch.int32,
        #                               device = self.device)

        # self.count_all = torch.zeros(self.num_layers, self.num_eps,
        #                               dtype = torch.int32,
        #                               device = self.device)

        # self._initial_ep2pos_all()

        self.ep2pos_all = cluster_status.expert_mapping.local_expert_mapping
        self.count_all = cluster_status.expert_mapping.redundant_count_per_expert

        if is_rand_op:
            self.optimize = self._Rand_optimize
        else:
            self.optimize = self._MoD_optimize

    """Optimizes based solely on scores, ignoring loads."""

    def _MoD_optimize(
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

        # ep2pos, ep_counts = self.dispatch_lis[layer_idx_moe]
        ep2pos, ep_counts = self.ep2pos_all[layer_idx_moe], self.count_all[
            layer_idx_moe]  #[256,20], [256]

        # available_counts = ep_counts[token_expert_id]
        # # 生成与 token_expert_id 同形的一个全局索引,从 0 到 token_expert_id.numel()-1
        # flat_indices = torch.arange(token_expert_id.numel(), device=token_expert_id.device).reshape(token_expert_id.shape)
        # # 用全局索引对 available_counts 取模,确保得到的索引总是处于 [0, available_counts) 范围内
        # # 这个操作是确定性的,且对每个 ep 内部的多次出现,会按 token_expert_id 中的位置不同而产生不同的值
        # rand_indices = flat_indices % available_counts
        # # 从映射表 ep2pos 中采样对应的 posid
        # pos_ids = ep2pos[token_expert_id, rand_indices]

        # 可以合并操作减少中间变量
        pos_ids = ep2pos[
            token_expert_id,
            torch.arange(token_expert_id.numel(), device=ep_counts.device).
            reshape(token_expert_id.shape) % ep_counts[token_expert_id]]
        # Return the updated token distribution
        return tokens, pos_ids, token_scores

    """Optimizes based solely on scores, ignoring loads."""

    def _Rand_optimize(
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
        torch.manual_seed(37)

        # ep2pos, ep_counts = self.dispatch_lis[layer_idx_moe]
        ep2pos, ep_counts = self.ep2pos_all[layer_idx_moe], self.count_all[
            layer_idx_moe]

        available_counts = ep_counts[token_expert_id]
        # 生成与 A 同形的随机数,乘以部署数,下取整得到随机索引
        rand_vals = torch.rand(token_expert_id.shape,
                               device=available_counts.device)
        rand_indices = (rand_vals * available_counts).long()

        # 从映射表 ep2pos 中采样对应的 posid
        pos_ids = ep2pos[token_expert_id, rand_indices]

        # Return the updated token distribution
        return tokens, pos_ids, token_scores

    def _initial_ep2pos_all(self):

        # num_moe_layers = self.placement_pattern.shape[1]
        # self.dispatch_lis = []
        # for moe_layer_id in range(num_moe_layers):
        #     self.dispatch_lis.append(   self._build_ep2pos_mapping_( self.placement_pattern[:,moe_layer_id,:])  )
        for layer_idx_moe in range(self.num_layers):
            ep2pos, count = self._build_ep2pos_mapping_(
                self.placement_pattern[:, layer_idx_moe, :])
            self.ep2pos_all[layer_idx_moe] += ep2pos
            self.count_all[layer_idx_moe] += count

        return 0

    def _build_ep2pos_mapping_(self, placement_pattern):
        """
        根据 placement_pattern 构造一个 ep -> [global posid] 的映射。

        说明：
        - placement_pattern 的形状为 (num_device, num_ep),表示哪些设备上部署了对应的专家。
        - 对于每个设备（行),从左向右遇到的第一个 1 编号为 0,第二个 1 编号为 1,依此类推。
        - 每个设备的全局偏移量为前面设备部署数的累计和,例如：
                如果第 0 个设备部署了 n0 个专家,那么第 0 设备的编号为 0~(n0-1),
                第 1 个设备的编号为 n0~(n0+n1-1)（其中 n1 为第 1 个设备的部署个数);
        - 对于 placement_pattern 中不部署的位置（值为 0),映射结果保持为 -1。

        参数：
        placement_pattern: Tensor, shape (num_device, num_ep),二值矩阵,指示哪些设备上有该专家部署。

        返回:
        ep2pos: Tensor, shape (num_ep, max_deployments),每一行存储该 ep 在各设备上的 global posid,
                对于不存在部署的位置,用 -1 填充。
        ep_counts: Tensor, shape (num_ep, ),每个 ep 的部署数（即可用的位置数)。
        """
        num_device, num_ep = placement_pattern.shape

        # 1. 对于每个设备（行),计算局部编号：
        # 对每行做累加,遇到的第一个1编号为 0,第二个1为 1,依此类推
        local_cumsum = torch.cumsum(placement_pattern, dim=1) - 1
        # 对于非部署位置,将 local_cumsum 设为 -1（无效编号)
        local_cumsum = local_cumsum * placement_pattern + (-1) * (
            1 - placement_pattern)

        # 2. 计算每个设备的全局偏移量：
        # offset[d] = sum_{i=0}^{d-1} (number of ones in placement_pattern[i, :])
        row_counts = torch.sum(placement_pattern,
                               dim=1)  # 每个设备的部署数,形状 (num_device,)
        offsets = torch.zeros(num_device,
                              dtype=torch.int32,
                              device=placement_pattern.device)
        if num_device > 0:
            offsets[1:] = torch.cumsum(row_counts, dim=0)[:-1]

        # 3. 计算 global_map = offset[device] + local编号
        global_map = local_cumsum + offsets.unsqueeze(1)
        # 保证对于无部署位置仍为 -1
        global_map = global_map * placement_pattern + (-1) * (
            1 - placement_pattern)

        # 4. 构造 ep2pos 映射：对于每个 ep（每一列),收集各设备上 global_map 中不为 -1 的值
        # 首先统计每个 ep 的部署数,用于确定映射表的第二个维度大小
        ep_counts = torch.sum(placement_pattern, dim=0).to(torch.int32)
        # max_count = int(ep_counts.max().item()) if ep_counts.numel() > 0 else 0
        # ep2pos = -torch.ones((num_ep, max_count), dtype=torch.int32, device=placement_pattern.device)

        ep2pos = -torch.ones((num_ep, self.max_count),
                             dtype=torch.int32,
                             device=placement_pattern.device)

        for ep in range(num_ep):
            pos_values = global_map[:, ep]  # shape (num_device,)
            valid = (pos_values != -1)
            valid_pos = pos_values[valid]
            if valid_pos.numel() > 0:
                # 按设备顺序有效的 global posid 已经过排序
                ep2pos[ep, :valid_pos.numel()] = valid_pos
        return ep2pos, ep_counts
