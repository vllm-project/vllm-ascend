#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Fuse a row-parallel matmul with the sequence-parallel reduce-scatter.

Sequence parallelism leaves the row-parallel projection unreduced and scatters
its output along the token dimension instead of all-reducing it, which produces

    unquantized_gemm -> [constant_pad_nd] -> vllm.reduce_scatter

in the FX graph. CANN exposes ``npu_mm_reduce_scatter_base``, which runs the
matmul and the reduce-scatter as one pipelined kernel, so this pass rewrites
the sequence into a single ``vllm.npu_matmul_reduce_scatter`` call.

The padding that sequence parallelism inserts to make the token count divisible
by the TP size is moved in front of the matmul. Zero rows in the left-hand side
produce zero rows in the product, so ``pad(x @ w)`` and ``pad(x) @ w`` are
equivalent, and the operator then receives the aligned ``m`` its kernel requires.

The counterpart of this pass upstream is ``AsyncTPPass``, which is selected by
the same ``pass_config.fuse_gemm_comms`` switch but emits ``symm_mem`` fused
collectives that Ascend does not provide.
"""

from __future__ import annotations

import contextlib

import torch
import torch.fx as fx
from vllm.compilation.passes.fx_utils import is_func
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.logger import logger

# The CANN kernel keeps the reduction on the cube core, which restricts the
# contracted dimension to [256, 65535). Other sizes keep the unfused path.
_MIN_REDUCE_DIM = 256
_MAX_REDUCE_DIM_EXCLUSIVE = 65535
# torch_npu.npu_mm_reduce_scatter_base only supports all-mesh HCCS topologies.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _arg(node: fx.Node, index: int, name: str, default=None):
    """Read an argument that tracing may have passed positionally or by name."""
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _static_int(value) -> int | None:
    """Return ``value`` as an int, or None when it is only known symbolically."""
    if isinstance(value, int):
        return value
    return None


class MatmulReduceScatterFusionPass(VllmInductorPass):
    """Rewrite matmul + reduce-scatter pairs into the fused CANN operator."""

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.tp_group = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.gemm_op = torch.ops.vllm.unquantized_gemm.default
        self.reduce_scatter_op = torch.ops.vllm.reduce_scatter.default
        self.pad_op = torch.ops.aten.constant_pad_nd.default
        self.fused_op = torch.ops.vllm.npu_matmul_reduce_scatter.default

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.tp_size in _SUPPORTED_WORLD_SIZES

    def __call__(self, graph) -> None:
        self.begin()
        fx_graph = graph.graph if isinstance(graph, fx.GraphModule) else graph
        # Collect first: rewriting erases nodes that the traversal still holds.
        matches = []
        for node in fx_graph.nodes:
            if not is_func(node, self.reduce_scatter_op):
                continue
            match = self._match(node)
            if match is not None:
                matches.append((node, *match))
        for reduce_scatter, gemm, pad in matches:
            self._replace(fx_graph, reduce_scatter, gemm, pad)
        logger.debug("Fused %s matmul reduce-scatter patterns", len(matches))
        self.end_and_log()

    def _match(self, reduce_scatter: fx.Node) -> tuple[fx.Node, fx.Node | None] | None:
        """Return the ``(gemm, pad)`` nodes to fuse, or None when unsupported."""
        if _arg(reduce_scatter, 1, "dim") != 0:
            return None
        if _arg(reduce_scatter, 2, "world_size") != self.tp_size:
            return None
        if _arg(reduce_scatter, 3, "group_name") != self.tp_group.unique_name:
            return None

        producer = _arg(reduce_scatter, 0, "tensor")
        if not isinstance(producer, fx.Node):
            return None

        pad = None
        if is_func(producer, self.pad_op):
            if not self._is_trailing_row_pad(producer):
                return None
            if len(producer.users) != 1:
                return None
            pad = producer
            producer = _arg(pad, 0, "self")
            if not isinstance(producer, fx.Node):
                return None

        if not is_func(producer, self.gemm_op):
            return None
        # A second consumer still needs the unreduced local product.
        if len(producer.users) != 1:
            return None
        # Bias is added before the reduction, so folding the padding in front of
        # the matmul would move the bias into the padded rows as well.
        if _arg(producer, 2, "bias") is not None:
            return None
        if not self._shapes_supported(producer):
            return None
        return producer, pad

    def _is_trailing_row_pad(self, pad: fx.Node) -> bool:
        """Whether ``pad`` only appends zero rows to a 2D tensor."""
        if _arg(pad, 2, "value", 0) not in (0, 0.0):
            return False
        pad_widths = _arg(pad, 1, "pad")
        if not isinstance(pad_widths, (list, tuple)) or len(pad_widths) != 4:
            return False
        # [last_dim_before, last_dim_after, row_before, row_after]
        if any(width != 0 for width in pad_widths[:3]):
            return False
        return self._rank_of(pad) == 2

    def _shapes_supported(self, gemm: fx.Node) -> bool:
        x = _arg(gemm, 0, "x")
        weight = _arg(gemm, 1, "weight")
        if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
            return False

        x_val = x.meta.get("val")
        weight_val = weight.meta.get("val")
        if x_val is None or weight_val is None:
            return False
        if x_val.dim() != 2 or weight_val.dim() != 2:
            return False
        if x_val.dtype not in _SUPPORTED_DTYPES or x_val.dtype != weight_val.dtype:
            return False

        reduce_dim = _static_int(x_val.shape[1])
        if reduce_dim is None or not _MIN_REDUCE_DIM <= reduce_dim < _MAX_REDUCE_DIM_EXCLUSIVE:
            return False
        return True

    def _rank_of(self, node: fx.Node) -> int | None:
        val = node.meta.get("val")
        return None if val is None else val.dim()

    def _replace(
        self,
        fx_graph: fx.Graph,
        reduce_scatter: fx.Node,
        gemm: fx.Node,
        pad: fx.Node | None,
    ) -> None:
        x = _arg(gemm, 0, "x")
        weight = _arg(gemm, 1, "weight")

        with fx_graph.inserting_before(reduce_scatter):
            if pad is not None:
                x = self._insert_pad(fx_graph, x, pad)
            fused = fx_graph.call_function(
                self.fused_op,
                args=(x, weight, self.tp_size, self.tp_group.unique_name),
            )
        fused.meta.update(reduce_scatter.meta)

        reduce_scatter.replace_all_uses_with(fused)
        # Erase in reverse topological order so each node is already unused.
        fx_graph.erase_node(reduce_scatter)
        if pad is not None and not pad.users:
            fx_graph.erase_node(pad)
        if not gemm.users:
            fx_graph.erase_node(gemm)

    def _insert_pad(self, fx_graph: fx.Graph, x: fx.Node, pad: fx.Node) -> fx.Node:
        """Re-apply ``pad``'s trailing row padding to the matmul input."""
        rows = _arg(pad, 1, "pad")[3]
        padded = fx_graph.call_function(self.pad_op, args=(x, [0, 0, 0, rows], 0.0))
        x_val = x.meta["val"]
        rows_val = rows.meta["val"] if isinstance(rows, fx.Node) else rows
        # Downstream backends read the shape of every node from its metadata.
        fake_mode = getattr(x_val, "fake_mode", None)
        with contextlib.ExitStack() as stack:
            if fake_mode is not None:
                stack.enter_context(fake_mode)
            padded.meta["val"] = self.pad_op(x_val, [0, 0, 0, rows_val], 0.0)
        return padded
