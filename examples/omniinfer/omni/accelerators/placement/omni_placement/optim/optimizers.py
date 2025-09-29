# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from abc import ABC

class Optimizer(ABC):
    """Base class for optimization strategies."""

    def optimize(self, layer_id, token, token_expert_id, token_scores, cluster_status):
        """
        Return the optimized expert selections in a proper order for all tokens.

        Args:
            layer_id (int): ID of the Mixture-of-Experts (MoE) layer. Dense layers are skipped, and IDs are reassigned starting from 0.
            token: The input token(s) being processed.
            token_expert_id (torch.Tensor): Tensor indicating the selected expert IDs for each token.
                                            Shape: (num_tokens, topk).
                                            A value of '-1' indicates the expert was not selected.
            token_scores (torch.Tensor): Importance scores corresponding to each selected expert per token.
                                         Shape: (num_tokens, topk).
            cluster_status (Any): Information about the current status or configuration of expert clusters.
                                  This can include expert load, availability, or other relevant metrics.

        Returns:
            list: Optimized list of expert IDs in the desired order for all tokens.
        """
        # Abstract method: subclasses must implement their own optimization logic.
        raise NotImplementedError("Subclasses must implement optimize")