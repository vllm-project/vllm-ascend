# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from dataclasses import dataclass

import numpy as np
import torch
from vllm.distributed.eplb.policy import EplbStats


@dataclass
class AscendEplbStats(EplbStats):
    """One planning snapshot, including STAIR's temporal and topology inputs."""

    sample_counts: np.ndarray | None = None
    rank_node_ids: np.ndarray | None = None
    last_committed_mean_ratios: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedLoadStats:
    """Policy-prepared load values and optional leading-axis weights.

    ``values`` is ``[layers, experts]`` or
    ``[samples, layers, experts]``. ``sample_counts`` is optional
    ``[samples]`` shared across ranks.
    """

    values: torch.Tensor
    sample_counts: np.ndarray | None = None
