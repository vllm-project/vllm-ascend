# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM signed Hadamard/codebook flow with the ops-nn TurboQuant ABI."""

import math
from collections.abc import Callable
from functools import cache, lru_cache
from importlib import import_module

import numpy as np
import torch

from . import HEAD_DIM

_DSV4_LLOYD_MAX_SAMPLE_COUNT = 400_000


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
    """Initialize before graph capture; DSV4 layers share read-only transforms."""

    def __init__(self, *, dsv4: bool = False):
        self._dsv4 = dsv4
        self.inverse_rotation: torch.Tensor | None = None
        self.rotation: torch.Tensor | None = None
        self.centroids: torch.Tensor | None = None
        self.norm_lut: torch.Tensor | None = None

    def _initialize(self, device):
        if self.rotation is not None:
            if self.rotation.device != device:
                raise RuntimeError("TurboQuant transform cannot move devices after initialization")
            return
        if self._dsv4:
            self.rotation, self.inverse_rotation, self.centroids, self.norm_lut = _get_dsv4_transforms(device)
            return
        self._initialize_transforms(device)

    def _initialize_transforms(self, device):
        h = np.ones((1, 1), dtype=np.float32)
        while h.shape[0] < HEAD_DIM:
            h = np.block([[h, h], [h, -h]])
        rng = np.random.default_rng(0)
        if self._dsv4:
            # Historical DSV4 code drew Lloyd-Max training samples before the
            # signs. Preserve that sequence even though the codebook is fixed.
            rng.standard_normal(_DSV4_LLOYD_MAX_SAMPLE_COUNT)
        signs = rng.choice([-1.0, 1.0], HEAD_DIM).astype(np.float32)
        self.rotation = torch.tensor(signs[:, None] * h / math.sqrt(HEAD_DIM), device=device)
        self.centroids = torch.tensor(CENTROIDS, dtype=torch.float32, device=device)
        cent = np.asarray(CENTROIDS, dtype=np.float32)
        codes = np.arange(256)
        self.norm_lut = torch.tensor(cent[codes & 15] ** 2 + cent[codes >> 4] ** 2, device=device)

    def forward(self, x):
        self._initialize(x.device)
        rotation = self.rotation
        assert rotation is not None
        return (x.float().reshape(-1, HEAD_DIM) @ rotation).to(x.dtype).reshape(x.shape)

    def inverse(self, x):
        self._initialize(x.device)
        rotation = self.rotation
        assert rotation is not None
        inverse_rotation = self.inverse_rotation if self._dsv4 else rotation.T
        assert inverse_rotation is not None
        return (x.float().reshape(-1, HEAD_DIM) @ inverse_rotation).to(x.dtype).reshape(x.shape)

    def compress(self, x):
        self._initialize(x.device)
        rotation = self.rotation
        centroids = self.centroids
        norm_lut = self.norm_lut
        assert rotation is not None
        assert centroids is not None
        assert norm_lut is not None
        rotated = (x.float().reshape(-1, HEAD_DIM) @ rotation).contiguous()
        packed, norm = _get_turbo_quant_op()(rotated, centroids)
        # ops-nn returns the original norm. MixedQuantSparseFlashMla multiplies
        # centroids by scale directly; retain the old DS norm correction here.
        if self._dsv4 and packed.device.type == "npu":
            from vllm_ascend.ops.triton.turboquant_finalize import turboquant_finalize

            return turboquant_finalize(packed, norm, norm_lut)
        selected_norm = norm_lut[packed.long()].sum(-1).sqrt()
        scale = (norm.float() / selected_norm).to(torch.float16)
        return torch.cat((packed, scale.view(torch.uint8).reshape(-1, 2)), dim=-1).unsqueeze(1)


@cache
def _get_dsv4_transforms(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Share read-only tensors across DSV4 layers, initialized before graph
    # capture. SFA keeps its original per-layer initialization and math path.
    store = TurboQuantLatent(dsv4=True)
    store._initialize_transforms(device)
    assert store.rotation is not None
    assert store.centroids is not None
    assert store.norm_lut is not None
    return store.rotation, store.rotation.T.contiguous(), store.centroids, store.norm_lut
