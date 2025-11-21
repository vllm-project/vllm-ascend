# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from ..cluster_status import ClusterStatus
from .optimizers import Optimizer


class AdaRouter(Optimizer):

    def __init__(self,
                 cluster_status: ClusterStatus,
                 threshold: float = 0.9,
                 entropy_bound: float = 0.5,
                 method: str = 'threshold'):
        """
        Token-based adaptive selection of experts.

        Args:
            threshold (float): Hyper-parameter used in :meth:`select_experts_t`, default 0.9.
            entropy_bound (float): To be explored?
        """
        super().__init__()
        self._cluster_status = cluster_status
        if threshold > 1:
            raise ValueError("AdaRouter threshold must be ≤ 1")
        self._threshold = torch.tensor(1 - threshold, device='npu')
        self._entropy_bound = entropy_bound
        self._method = method

    def optimize(self, layer_id_moe, token, token_expert_ids, token_scores,
                 cluster_status):
        """
        Select a subset of top-k experts for each token based on threshold.

        inputs:
            layer_id_moe (int): moe layer id
            token (torch.Tensor): input token, shape [batch_size * seq_len]
            token_expert_ids (torch.Tensor): top-k indices per token, shape [batch_size * seq_len, top_k]
            token_scores (torch.Tensor): top-k scores per token, shape [batch_size * seq_len, top_k]
            cluster_status (ClusterStatus): cluster status

        returns:
            token (torch.Tensor): the same as input
            topk_ids (torch.Tensor): shape [bs * sl, top_k], indices after adaptive selection, padded with -1 for non-selected ones
            topk_weights (torch.Tensor): the same as topk_ids, but with weights, shape [bs * sl, top_k]
        """
        if self._method == 'threshold':
            topk_weights, topk_ids = self._select_experts_by_threshold(
                token_expert_ids, token_scores)
            return token, topk_ids, topk_weights
        else:
            return token, token_expert_ids, token_scores

    def _select_experts_by_threshold(self, token_expert_ids, token_scores):
        """
        selecting experts based on a universal threshold

        inputs:
            token_expert_id (torch.Tensor): top_k indices per token, shape [batch_size * seq_len, top_k]
            token_scores (torch.Tensor): top_k scores per token, shape [batch_size * seq_len, top_k]

        outputs:
            topk_ids (torch.Tensor): shape [bs * sl, top_k], indices after adaptive selection, padded with -1 for non-selected ones
            topk_weights (torch.Tensor): shape [bs * sl, top_k], rescaled to equal sum, padded with 0 for non-selected ones
        """
        mask = token_scores > self._threshold  # shape [batch_size * seq_len, top_k], bool
        token_scores = token_scores * mask.to(
            token_scores.dtype)  # masking weights
        token_expert_ids0 = token_expert_ids * mask + (
            ~mask) * -1  # pad with -1
        token_expert_ids = token_expert_ids0.to(token_expert_ids.dtype)
        token_scores = token_scores / (
            torch.sum(token_scores, dim=-1, keepdim=True) + 1e-12)
        return token_scores, token_expert_ids
