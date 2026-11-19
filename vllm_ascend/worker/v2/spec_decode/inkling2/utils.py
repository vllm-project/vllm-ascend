# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def _select_context_anchor(
    target_features: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_rejected: torch.Tensor,
) -> torch.Tensor:
    """Select the last committed target features for each request's Markov head."""
    num_reqs = num_rejected.shape[0]
    ends = query_start_loc[1 : num_reqs + 1] - num_rejected
    return target_features[(ends - 1).clamp_min(0).long()]
