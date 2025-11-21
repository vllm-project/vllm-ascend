# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import time

import torch

from .optimizers import Optimizer


class TokenBalance(Optimizer):
    """
    TokenBalance optimizer class inherits from Optimizer.

    This optimizer is designed to balance tokens across experts based on a specified capacity factor.
    It initializes necessary mappings based on cluster status and determines the expert-per-device count.
    """

    def __init__(self, cluster_status, capacity_factor=1.0):
        """
        Initializes the TokenBalance optimizer instance.

        Parameters:
            cluster_status: Object containing the current status and mapping information of the cluster.
            capacity_factor (float, optional): Factor determining the allowed expert capacity relative
                                               to the average load. Default is 1.0.
        """

        # Extract expert mapping information from the provided cluster status
        self.expert_mapping = cluster_status.placement_pattern
        self.device = self.expert_mapping.device

        # Calculate the number of experts assigned per device.
        # It sums the expert mapping tensor on the first row of the first dimension.
        # Adding 0.5 before converting to integer ensures proper rounding.
        self.expert_per_device = int(
            torch.sum(self.expert_mapping[0][0]) + 0.5)

        # self.expert_per_device = int((torch.sum(self.expert_mapping[0][0].to(device)) + 0.5).item())
        self.expert_per_device = torch.tensor(int(
            (torch.sum(self.expert_mapping[0][0]) + 0.5).item()),
                                              dtype=torch.int32,
                                              device=self.device)

        # Construct a mapping from expert IDs back to their original positions.
        # This is required for later optimization steps.
        self._construct_expert_mapping_to_origin_pos()

    """Optimizes based solely on scores, ignoring loads."""

    def optimize(
        self, layer_idx_moe: int, tokens: torch.Tensor,
        token_expert_id: torch.Tensor, token_scores: torch.Tensor,
        cluster_status: list
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于 `expert_position` 进行 token 分配优化，直接映射到全局 expert ID。

        Args:
            layer_idx_moe: int
                当前 Mixture-of-Experts (MoE) 层的索引。
            tokens: torch.Tensor
                Token 数据，形状: [batch_size, seq_len, hidden_dim]。
            token_expert_id: torch.Tensor
                每个 token 初始分配的 expert ID，形状: [batch_size, seq_len, top_k]。
            token_scores: torch.Tensor
                每个 token 对应的 expert 评分，形状: [batch_size, seq_len, top_k]。
            cluster_status: list
                集群状态信息（当前不使用）。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 优化后的 token 分配信息：
                - tokens: 原始 token 张量，形状 [batch_size, seq_len, hidden_dim]。
                - token_expert_placement_ids: 更新后的 expert ID，形状 [batch_size, seq_len, top_k]。
                - token_scores: 原始 expert 评分张量，形状 [batch_size, seq_len, top_k]。
        """
        # token_expert_id_dtype = token_expert_id.dtype

        # 查询 `expert_position` 以获取全局 expert ID
        token_expert_placement_ids = self.expert_position[layer_idx_moe,
                                                          token_expert_id]

        return tokens, token_expert_placement_ids, token_scores

    def _map_topk_to_device_epid(
        self, layer_idx_moe: int, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        **Accepts `-1` as token_expert_id inputs, used for adaptive router selection.**

        Parameters:
            layer_idx_moe: int
                The current MoE layer index to query.
            topk_ids: torch.Tensor
                Tensor of shape (num_tokens, topk), containing expert IDs selected per token.
                A value of `-1` indicates an unselected expert.

        Dependencies:
            - self.expert_mapping: torch.Tensor
                Binary (0-1) tensor of shape (num_devices, num_layers, num_epids),
                indicating expert assignment per device and layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - topk_ids_clone: torch.Tensor
                Tensor of shape (num_tokens, topk), same as `topk_ids`, but with `-1` replaced by `0`
                to safely handle indexing. Original `-1` positions are tracked by `valid_mask`.
                - valid_mask: torch.Tensor
                Boolean tensor of shape (num_tokens, topk), indicating positions of valid expert IDs.
                (dtype=torch.bool)
                - topk_device_ids: torch.Tensor
                Tensor of shape (num_tokens, topk), containing the device ID for each expert selection.
                For invalid positions (`topk_ids == -1`), the device IDs are computed but should be ignored
                based on the `valid_mask`.
        """
        # Extract expert mapping for the specified MoE layer
        # Shape: (num_devices, num_epids)
        layer_expert_mapping = self.expert_mapping[:, layer_idx_moe, :]

        # Create a boolean mask indicating valid expert selections (not equal to -1)
        # Shape: (num_tokens, topk)
        valid_mask = topk_ids != -1
        # print('valid_mask ; XXXXXXXXXXXXXXXXXXXX', valid_mask.device, topk_ids.device )

        # Clone topk_ids to avoid modifying the original input tensor
        # topk_ids_clone = topk_ids.clone()

        # Replace invalid expert selections (-1) with 0 to prevent indexing errors
        topk_ids[~valid_mask] = 0

        # Calculate device IDs corresponding to each selected expert ID
        # For each token and top-k selection, find which device hosts the expert.
        # Note: For originally invalid positions, device IDs are computed but should be ignored.
        # Shape of indexing result: (num_devices, num_tokens, topk)
        # Shape after argmax: (num_tokens, topk)
        topk_device_ids = torch.argmax(layer_expert_mapping[:, topk_ids],
                                       dim=0)

        return topk_ids, valid_mask, topk_device_ids

    def _construct_expert_mapping_to_origin_pos(self):
        """
            计算 `expert_position`，从 `expert_mapping` 生成唯一的 global expert 索引。

            Returns:
                - self.expert_position: 形状为 (num_layers, num_epids) 的张量，存储每个 expert 的唯一索引。
            """
        num_devices, num_layers, num_epids = self.expert_mapping.shape

        # 初始化 expert_position，默认值为 -1（表示没有 expert）
        expert_position = torch.full((num_layers, num_epids),
                                     -1,
                                     dtype=torch.long,
                                     device=self.device)

        # 遍历所有设备
        for i in range(num_devices):
            for j in range(num_layers):
                # 计算该设备在该层的 cumulative sum（用于计算前面有多少个 expert）
                cumsum_expert = torch.cumsum(self.expert_mapping[i, j], dim=0)

                # 选取存在 expert 的位置
                mask = self.expert_mapping[i, j] == 1

                # 计算全局 expert 索引，并填充到 expert_position
                expert_position[
                    j,
                    mask] = i * self.expert_per_device + cumsum_expert[mask] - 1
        self.expert_position = expert_position.to(torch.int32)
        return 0
