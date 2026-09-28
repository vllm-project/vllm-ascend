# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM signed Hadamard/codebook flow with the ops-nn TurboQuant ABI."""

import math
from collections.abc import Callable
from functools import lru_cache
from importlib import import_module

import numpy as np
import torch

from . import HEAD_DIM


@lru_cache
def _get_turbo_quant_op() -> Callable:
    try:
        import_module("cann_ops_nn")
        return torch.ops.cann_ops_nn.turbo_quant
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("DeepSeek V4 TurboQuant requires TurboQuant from a matching ops-nn package.") from exc


CENTROIDS = (
    -0.12091285,
    -0.09111122,
    -0.07112455,
    -0.05513602,
    -0.04132067,
    -0.02874970,
    -0.01700489,
    -0.00568677,
    0.00547294,
    0.01680406,
    0.02857605,
    0.04108622,
    0.05492980,
    0.07101817,
    0.09115373,
    0.12037795,
)


class TurboQuantLatent:
    """Own transform tensors per attention layer; initialize before graph capture."""

    def __init__(self):
        self.rotation = None
        self.centroids = None
        self.norm_lut = None

    def _initialize(self, device):
        if self.rotation is not None:
            if self.rotation.device != device:
                raise RuntimeError("TurboQuant transform cannot move devices after initialization")
            return
        h = np.ones((1, 1), dtype=np.float32)
        while h.shape[0] < HEAD_DIM:
            h = np.block([[h, h], [h, -h]])
        signs = np.random.default_rng(0).choice([-1.0, 1.0], HEAD_DIM).astype(np.float32)
        self.rotation = torch.tensor(signs[:, None] * h / math.sqrt(HEAD_DIM), device=device)
        self.centroids = torch.tensor(CENTROIDS, dtype=torch.float32, device=device)
        cent = np.asarray(CENTROIDS, dtype=np.float32)
        codes = np.arange(256)
        self.norm_lut = torch.tensor(cent[codes & 15] ** 2 + cent[codes >> 4] ** 2, device=device)

    def forward(self, x):
        self._initialize(x.device)
        return (x.float().reshape(-1, HEAD_DIM) @ self.rotation).to(x.dtype).reshape(x.shape)

    def inverse(self, x):
        self._initialize(x.device)
        return (x.float().reshape(-1, HEAD_DIM) @ self.rotation.T).to(x.dtype).reshape(x.shape)

    def compress(self, x):
        self._initialize(x.device)
        rotated = (x.float().reshape(-1, HEAD_DIM) @ self.rotation).contiguous()
        packed, norm = _get_turbo_quant_op()(rotated, self.centroids)
        # ops-nn returns the original norm. MixedQuantSparseFlashMla multiplies
        # centroids by scale directly; retain the old DS norm correction here.
        selected_norm = self.norm_lut[packed.long()].sum(-1).sqrt()
        scale = (norm.float() / selected_norm).to(torch.float16)
        return torch.cat((packed, scale.view(torch.uint8).reshape(-1, 2)), dim=-1).unsqueeze(1)
