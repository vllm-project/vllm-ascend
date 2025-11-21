# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

import torch
import torch_npu

from .optimizers import Optimizer


class ExpertsBalanceOptimizer(Optimizer):
    def __init__(self, cluster_status, batch_size=48, top_k_count=8) -> None:
        """
        Initializes the Expert Balance optimizer instance.

        Parameters:
            cluster_status: Object containing the current status and mapping information of the cluster.

        """

        # Extract expert mapping information from the provided cluster status
        self.redundant_expert_mapping = cluster_status.expert_mapping.redundant_expert_mapping
        self.device = self.redundant_expert_mapping.device
        # print(f"redundant_expert_mapping: {self.redundant_expert_mapping.size()}")
        # print(f"redundant_expert_mapping: {self.redundant_expert_mapping}")

        self.max_redundant_num = cluster_status.expert_mapping.get_max_redundant_expert_num()
        self.batch_size = int(os.environ.get("PTA_TORCHAIR_DECODE_GEAR_LIST", batch_size))
        self.top_k_count = top_k_count
        self.selector = torch.arange(self.batch_size, device=self.device) % self.max_redundant_num  # Shape: (batch_size,)
        self.selector = self.selector.view(self.batch_size, 1).expand(self.batch_size, self.top_k_count)  # Broadcast to (batch_size, expert_count)

        # print(f"selector: {self.selector.size()}")
        # print(f"selector: {self.selector}")

    def optimize(self,
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
        expert_mapping = self.redundant_expert_mapping[layer_idx_moe]
        return tokens, expert_mapping[self.selector, token_expert_id], token_scores

