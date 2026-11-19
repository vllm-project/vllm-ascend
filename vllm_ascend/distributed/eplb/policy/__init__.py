# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PreparedLoadStats:
    """Policy-prepared load values and optional leading-axis weights.

    ``values`` is ``[layers, experts]`` or
    ``[samples, layers, experts]``. ``sample_counts`` is optional
    ``[samples]`` shared across ranks.
    """

    values: torch.Tensor
    sample_counts: np.ndarray | None = None
