# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Config:
    batch: int = 1
    tokens: int = 6
    hidden: int = 4096
    heads: int = 64
    dim: int = 128
    lower_bound: float = -5.0
    eps: float = 1e-5


def make_inputs(c, device="cpu", seed=7):
    torch.manual_seed(seed)
    b, tokens, h, d, k = c.batch, c.tokens, c.heads, c.dim, c.hidden

    def rand(shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, device=device) * scale).to(dtype)

    return {
        "x": rand((b, tokens, k), 0.3),
        "w_in": rand((3 * h * d + h + 2 * d, k), k**-0.5),
        "w_f": rand((h * d, d), d**-0.5),
        "w_g": rand((h * d, d), d**-0.5),
        "w_o": rand((k, h * d), (h * d) ** -0.5),
        "conv_w": rand((3, h, d, 4), 0.3),
        "history": rand((b, 3, h, d, 3), 0.2),
        "state": rand((b, h, d, d), 0.02, torch.float32),
        "a_log": rand((h,), 0.2, torch.float32),
        "dt_bias": rand((h, d), 0.2, torch.float32),
        "norm_w": rand((d,), 0.2) + 1,
    }
