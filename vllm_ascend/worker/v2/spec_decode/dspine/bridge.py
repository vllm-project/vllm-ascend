# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def _build_predecessor_features(draft_features: torch.Tensor, anchor_features: torch.Tensor) -> torch.Tensor:
    """Align each draft row with the features of its predecessor.

    Draft features have shape [requests, draft_steps, channels], with at least
    one draft step. The first draft row uses the anchor codebook features.
    """
    return torch.cat((anchor_features.unsqueeze(1), draft_features[:, :-1]), dim=1)
