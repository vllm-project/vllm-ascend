# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse row-parallel GEMM with sequence-parallel reduce-scatter on Ascend."""

from __future__ import annotations

from typing import Any

import torch
import torch.fx as fx
from torch._inductor.pattern_matcher import Match, PatternMatcherPass
from vllm.compilation.passes.inductor_pass import enable_fake_mode
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern

_MIN_REDUCE_DIM = 256
_MAX_REDUCE_DIM_EXCLUSIVE = 65535
_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

_GEMM_OP = torch.ops.vllm.unquantized_gemm.default
_REDUCE_SCATTER_OP = torch.ops.vllm.reduce_scatter.default
_PAD_OP = torch.ops.aten.constant_pad_nd.default


def _static_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _arg(node: fx.Node, index: int, name: str, default=None):
    return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)


class _MatmulReduceScatterPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, tp_size: int, group_name: str):
        super().__init__(vllm_config)
        self.tp_size = tp_size
        self.group_name = group_name

    def _empty(self, *shape: int) -> torch.Tensor:
        dtype = self.dtype if self.dtype in _SUPPORTED_DTYPES else torch.bfloat16
        device = self.vllm_config.device_config.device if self.vllm_config.device_config else "cpu"
        return torch.empty(shape, device=device, dtype=dtype)

    def get_inputs(self) -> list[torch.Tensor]:
        return [self._empty(8, 512), self._empty(256, 512)]

    @enable_fake_mode
    def register(self, pm_pass: PatternMatcherPass) -> None:
        super().register(pm_pass)

    def get_extra_check(self):
        def supported(match: Match) -> bool:
            gemm = next((node for node in match.nodes if node.target is _GEMM_OP), None)
            reduce_scatter = next((node for node in match.nodes if node.target is _REDUCE_SCATTER_OP), None)
            if gemm is None or reduce_scatter is None:
                return False
            if _arg(gemm, 2, "bias") is not None:
                return False
            if _arg(reduce_scatter, 1, "dim") != 0:
                return False
            if _arg(reduce_scatter, 2, "world_size") != self.tp_size:
                return False
            if _arg(reduce_scatter, 3, "group_name") != self.group_name:
                return False
            x, weight = gemm.args[0], gemm.args[1]
            if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
                return False
            x_val, weight_val = x.meta.get("val"), weight.meta.get("val")
            if x_val is None or weight_val is None or x_val.dim() != 2 or weight_val.dim() != 2:
                return False
            if x_val.dtype not in _SUPPORTED_DTYPES or x_val.dtype != weight_val.dtype:
                return False
            reduce_dim = _static_int(x_val.shape[1])
            return reduce_dim is not None and _MIN_REDUCE_DIM <= reduce_dim < _MAX_REDUCE_DIM_EXCLUSIVE

        return supported

    def pattern_key(self) -> str:
        return f"{self.__class__.__name__}_{self.tp_size}_{self.group_name}"


class MatmulReduceScatterPattern(_MatmulReduceScatterPattern):
    """Match ``gemm -> reduce_scatter``."""

    def get_pattern(self):
        def pattern(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            gemm = torch.ops.vllm.unquantized_gemm.default(x, weight, None)
            return torch.ops.vllm.reduce_scatter.default(
                gemm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.group_name,
            )

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.ops.vllm.npu_matmul_reduce_scatter.default(x, weight, self.tp_size, self.group_name)

        return replacement


class PaddedMatmulReduceScatterPattern(_MatmulReduceScatterPattern):
    """Match ``gemm -> padding -> reduce_scatter`` and move padding to input."""

    def get_scalar_workaround(self) -> dict[str, float | int] | None:
        return {"pad_rows": 13}

    def get_extra_check(self):
        base_check = super().get_extra_check()

        def supported(match: Match) -> bool:
            if not base_check(match):
                return False
            pad = next((node for node in match.nodes if node.target is _PAD_OP), None)
            if pad is None or _arg(pad, 2, "value", 0) not in (0, 0.0):
                return False
            widths = _arg(pad, 1, "pad")
            pad_val = pad.meta.get("val")
            return (
                isinstance(widths, (list, tuple))
                and len(widths) == 4
                and all(width == 0 for width in widths[:3])
                and pad_val is not None
                and pad_val.dim() == 2
            )

        return supported

    def get_pattern(self):
        def pattern(x: torch.Tensor, weight: torch.Tensor, pad_rows: int) -> torch.Tensor:
            gemm = torch.ops.vllm.unquantized_gemm.default(x, weight, None)
            padded = torch.ops.aten.constant_pad_nd.default(gemm, [0, 0, 0, pad_rows], 0.0)
            return torch.ops.vllm.reduce_scatter.default(
                padded,
                dim=0,
                world_size=self.tp_size,
                group_name=self.group_name,
            )

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, weight: torch.Tensor, pad_rows: int) -> torch.Tensor:
            padded = torch.ops.aten.constant_pad_nd.default(x, [0, 0, 0, pad_rows], 0.0)
            return torch.ops.vllm.npu_matmul_reduce_scatter.default(padded, weight, self.tp_size, self.group_name)

        return replacement


class MatmulReduceScatterFusionPass(VllmInductorPass):
    """Apply the padded and unpadded MMRS pattern replacements."""

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.pattern_match_passes = PatternMatcherPass(pass_name="matmul_reduce_scatter_fusion_pass")

        if self.tp_size not in _SUPPORTED_WORLD_SIZES:
            return

        self.group_name = get_tp_group().unique_name
        MatmulReduceScatterPattern(config, self.tp_size, self.group_name).register(self.pattern_match_passes)
        PaddedMatmulReduceScatterPattern(config, self.tp_size, self.group_name).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph) -> None:
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s matmul-reduce-scatter patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.tp_size in _SUPPORTED_WORLD_SIZES
