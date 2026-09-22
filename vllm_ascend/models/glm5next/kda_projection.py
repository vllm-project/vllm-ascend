# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The f/g checkpoint and execution contract, independent of local TP width."""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

KDA_BATCHED_GATE_MAX_TOKENS = 128
KDA_BATCHED_GATE_MAX_WIDTH = 4096


class KDAFGProjection(nn.Module):
    """Own the canonical weights and dispatch without a derived weight cache.

    Checkpoints always supply f_b_proj/g_b_proj. Kernel-format weights are
    implementation-specific: merged ND for narrow projections, separate NZ
    capable weights for wide projections. Kernel-format reloads must use this
    instance's named_parameters(), just like other packed vLLM layers.
    """

    checkpoint_weight_names = ("f_b_proj.weight", "g_b_proj.weight")

    def __init__(
        self,
        head_dim: int,
        projection_size: int,
        local_projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self._merged = None
        if local_projection_size <= KDA_BATCHED_GATE_MAX_WIDTH:
            self._merged = MergedColumnParallelLinear(
                head_dim,
                [projection_size, projection_size],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}._merged",
            )
            self._merged.skip_weight_nz_conversion = True
        else:
            self._f = ColumnParallelLinear(
                head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}._f",
            )
            self._g = ColumnParallelLinear(
                head_dim,
                projection_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}._g",
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, weight in weights:
            shard = self.checkpoint_weight_names.index(name)
            if self._merged is not None:
                param = self._merged.weight
                param.weight_loader(param, weight, shard)
                loaded.add("_merged.weight")
            else:
                projection = self._f if shard == 0 else self._g
                param = projection.weight
                param.weight_loader(param, weight)
                loaded.add("_f.weight" if shard == 0 else "_g.weight")
        return loaded

    def forward(self, fg_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._merged is None:
            f_a, g_a = fg_a.split(self.head_dim, dim=-1)
            return self._f(f_a)[0], self._g(g_a)[0]
        weight = self._merged.weight.view(2, -1, self.head_dim)
        if fg_a.shape[0] > KDA_BATCHED_GATE_MAX_TOKENS:
            f_a, g_a = fg_a.split(self.head_dim, dim=-1)
            return torch.nn.functional.linear(f_a, weight[0]), torch.nn.functional.linear(g_a, weight[1])
        fg_a = fg_a.reshape(-1, 2, self.head_dim).transpose(0, 1)
        f, g = torch.bmm(fg_a, weight.transpose(1, 2)).unbind(0)
        return f, g
