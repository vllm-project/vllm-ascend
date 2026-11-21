# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def _get_dfly_anchor_positions(
    query_start_loc: torch.Tensor,
    target_positions: torch.Tensor,
    num_rejected: torch.Tensor,
) -> torch.Tensor:
    """Get the next positions after each request's last accepted target token."""
    source_rows = query_start_loc[1 : num_rejected.shape[0] + 1] - num_rejected - 1
    return target_positions[source_rows.long()] + 1
