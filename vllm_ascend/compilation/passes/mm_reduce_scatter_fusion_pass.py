# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse row-parallel GEMM with sequence-parallel reduce-scatter on Ascend.

Sequence parallelism produces ``unquantized_gemm -> [constant_pad_nd] ->
reduce_scatter``. CANN can pipeline these operations through
``npu_mm_reduce_scatter_base``. Padding is moved to the GEMM input: appending
zero rows before or after a bias-free matrix multiplication is equivalent.
"""

from __future__ import annotations

import contextlib
from collections import Counter

import torch
import torch.fx as fx
from vllm.compilation.passes.fx_utils import is_func
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)

_MIN_REDUCE_DIM = 256
_MAX_REDUCE_DIM_EXCLUSIVE = 65535
_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _arg(node: fx.Node, index: int, name: str, default=None):
    """Return a positional or keyword FX argument."""
    return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)


def _static_int(value) -> int | None:
    return value if isinstance(value, int) else None


class MatmulReduceScatterFusionPass(VllmInductorPass):
    """Replace supported GEMM + reduce-scatter pairs with the CANN fused op."""

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.gemm_op = torch.ops.vllm.unquantized_gemm.default
        self.reduce_scatter_op = torch.ops.vllm.reduce_scatter.default
        self.pad_op = torch.ops.aten.constant_pad_nd.default
        self.fused_op = torch.ops.vllm.npu_matmul_reduce_scatter.default

    @property
    def tp_group(self):
        return get_tp_group()

    @property
    def tp_size(self) -> int:
        return get_tensor_model_parallel_world_size()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.tp_size in _SUPPORTED_WORLD_SIZES

    def __call__(self, graph: fx.Graph | fx.GraphModule) -> None:
        self.begin()
        fx_graph = graph.graph if isinstance(graph, fx.GraphModule) else graph
        matches: list[tuple[fx.Node, fx.Node, fx.Node | None]] = []
        skipped: Counter[str] = Counter()
        for node in fx_graph.nodes:
            if not is_func(node, self.reduce_scatter_op):
                continue
            match, reason = self._match(node)
            if match is None:
                skipped[reason] += 1
            else:
                matches.append((node, *match))

        for reduce_scatter, gemm, pad in matches:
            self._replace(fx_graph, reduce_scatter, gemm, pad)

        logger.info("MMRS fusion: matched=%d, skipped=%s", len(matches), dict(skipped))
        self.end_and_log()

    def _match(self, reduce_scatter: fx.Node) -> tuple[tuple[fx.Node, fx.Node | None] | None, str]:
        if _arg(reduce_scatter, 1, "dim") != 0:
            return None, "scatter_dim"
        if _arg(reduce_scatter, 2, "world_size") != self.tp_size:
            return None, "world_size"
        if _arg(reduce_scatter, 3, "group_name") != self.tp_group.unique_name:
            return None, "group_name"

        producer = _arg(reduce_scatter, 0, "tensor")
        if not isinstance(producer, fx.Node):
            return None, "non_node_input"

        pad = None
        if is_func(producer, self.pad_op):
            if not self._is_trailing_row_pad(producer):
                return None, "padding"
            if len(producer.users) != 1:
                return None, "pad_extra_consumer"
            pad = producer
            producer = _arg(pad, 0, "self")
            if not isinstance(producer, fx.Node):
                return None, "pad_non_node_input"

        if not is_func(producer, self.gemm_op):
            return None, "producer"
        if len(producer.users) != 1:
            return None, "gemm_extra_consumer"
        if _arg(producer, 2, "bias") is not None:
            return None, "bias"
        if not self._shapes_supported(producer):
            return None, "shape_or_dtype"
        return (producer, pad), ""

    def _is_trailing_row_pad(self, pad: fx.Node) -> bool:
        if _arg(pad, 2, "value", 0) not in (0, 0.0):
            return False
        widths = _arg(pad, 1, "pad")
        return (
            isinstance(widths, (list, tuple))
            and len(widths) == 4
            and all(width == 0 for width in widths[:3])
            and self._rank_of(pad) == 2
        )

    def _shapes_supported(self, gemm: fx.Node) -> bool:
        x, weight = _arg(gemm, 0, "x"), _arg(gemm, 1, "weight")
        if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
            return False
        x_val, weight_val = x.meta.get("val"), weight.meta.get("val")
        if x_val is None or weight_val is None or x_val.dim() != 2 or weight_val.dim() != 2:
            return False
        if x_val.dtype not in _SUPPORTED_DTYPES or x_val.dtype != weight_val.dtype:
            return False
        reduce_dim = _static_int(x_val.shape[1])
        return reduce_dim is not None and _MIN_REDUCE_DIM <= reduce_dim < _MAX_REDUCE_DIM_EXCLUSIVE

    @staticmethod
    def _rank_of(node: fx.Node) -> int | None:
        value = node.meta.get("val")
        return None if value is None else value.dim()

    def _replace(self, graph: fx.Graph, reduce_scatter: fx.Node, gemm: fx.Node, pad: fx.Node | None) -> None:
        x, weight = _arg(gemm, 0, "x"), _arg(gemm, 1, "weight")
        with graph.inserting_before(reduce_scatter):
            if pad is not None:
                x = self._insert_pad(graph, x, pad)
            fused = graph.call_function(
                self.fused_op,
                args=(x, weight, self.tp_size, self.tp_group.unique_name),
            )
        fused.meta.update(reduce_scatter.meta)
        reduce_scatter.replace_all_uses_with(fused)
        graph.erase_node(reduce_scatter)
        if pad is not None and not pad.users:
            graph.erase_node(pad)
        if not gemm.users:
            graph.erase_node(gemm)

    def _insert_pad(self, graph: fx.Graph, x: fx.Node, original_pad: fx.Node) -> fx.Node:
        rows = _arg(original_pad, 1, "pad")[3]
        padded = graph.call_function(self.pad_op, args=(x, [0, 0, 0, rows], 0.0))
        x_val = x.meta["val"]
        rows_val = rows.meta["val"] if isinstance(rows, fx.Node) else rows
        fake_mode = getattr(x_val, "fake_mode", None)
        with contextlib.ExitStack() as stack:
            if fake_mode is not None:
                stack.enter_context(fake_mode)
            padded.meta["val"] = self.pad_op(x_val, [0, 0, 0, rows_val], 0.0)
        return padded
