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
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock, patch

import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass import (
    MatmulReduceScatterFusionPass,
)

TP_SIZE = 2
TP_GROUP_NAME = "tp:0"
# The contracted dimension the CANN kernel supports starts at 256.
REDUCE_DIM = 512
OUTPUT_DIM = 256

GEMM_OP = torch.ops.vllm.unquantized_gemm.default
REDUCE_SCATTER_OP = torch.ops.vllm.reduce_scatter.default
PAD_OP = torch.ops.aten.constant_pad_nd.default
FUSED_OP = torch.ops.vllm.npu_matmul_reduce_scatter.default


def _build_graph(
    *,
    num_tokens: int = 8,
    pad_rows: int | None = 2,
    bias: bool = False,
    dim: int = 0,
    reduce_dim: int = REDUCE_DIM,
    extra_gemm_user: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> fx.GraphModule:
    """Build the FX graph sequence parallelism leaves for a row-parallel linear."""
    graph = fx.Graph()
    fake_mode = FakeTensorMode()

    def placeholder(name: str, *shape: int) -> fx.Node:
        node = graph.placeholder(name)
        with fake_mode:
            node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    def call(target, args, *shape: int) -> fx.Node:
        node = graph.call_function(target, args=args)
        with fake_mode:
            node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    x = placeholder("x", num_tokens, reduce_dim)
    weight = placeholder("weight", OUTPUT_DIM, reduce_dim)
    bias_node = placeholder("bias", OUTPUT_DIM) if bias else None

    gemm = call(GEMM_OP, (x, weight, bias_node), num_tokens, OUTPUT_DIM)
    reduced_rows = num_tokens
    scattered = gemm
    if pad_rows is not None:
        reduced_rows = num_tokens + pad_rows
        scattered = call(PAD_OP, (gemm, [0, 0, 0, pad_rows], 0.0), reduced_rows, OUTPUT_DIM)

    reduce_scatter = call(
        REDUCE_SCATTER_OP,
        (scattered, dim, TP_SIZE, TP_GROUP_NAME),
        reduced_rows // TP_SIZE,
        OUTPUT_DIM,
    )
    outputs = [reduce_scatter]
    if extra_gemm_user:
        outputs.append(call(torch.ops.aten.clone.default, (gemm,), num_tokens, OUTPUT_DIM))
    graph.output(tuple(outputs))
    return fx.GraphModule(torch.nn.Module(), graph)


def _targets(graph_module: fx.GraphModule) -> list:
    return [node.target for node in graph_module.graph.nodes if node.op == "call_function"]


class TestMatmulReduceScatterFusionPass(TestBase):
    def _make_pass(self) -> MatmulReduceScatterFusionPass:
        tp_group = MagicMock()
        tp_group.unique_name = TP_GROUP_NAME
        module = "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass"
        with (
            patch(f"{module}.get_tp_group", return_value=tp_group),
            patch(f"{module}.get_tensor_model_parallel_world_size", return_value=TP_SIZE),
        ):
            return MatmulReduceScatterFusionPass(VllmConfig())

    def test_padded_matmul_reduce_scatter_is_fused(self):
        graph_module = _build_graph()

        self._make_pass()(graph_module)

        targets = _targets(graph_module)
        self.assertEqual(targets.count(FUSED_OP), 1)
        self.assertNotIn(GEMM_OP, targets)
        self.assertNotIn(REDUCE_SCATTER_OP, targets)

    def test_padding_moves_in_front_of_the_matmul(self):
        graph_module = _build_graph(pad_rows=2)

        self._make_pass()(graph_module)

        fused = next(node for node in graph_module.graph.nodes if node.target is FUSED_OP)
        padded_input, weight, world_size, group_name = fused.args
        self.assertIs(padded_input.target, PAD_OP)
        # The padding now consumes the matmul input rather than its output.
        self.assertEqual(padded_input.args[0].op, "placeholder")
        self.assertEqual(padded_input.args[1], [0, 0, 0, 2])
        self.assertEqual(padded_input.meta["val"].shape[0], 10)
        self.assertEqual(weight.op, "placeholder")
        self.assertEqual(world_size, TP_SIZE)
        self.assertEqual(group_name, TP_GROUP_NAME)

    def test_unpadded_matmul_reduce_scatter_is_fused(self):
        graph_module = _build_graph(pad_rows=None)

        self._make_pass()(graph_module)

        targets = _targets(graph_module)
        self.assertEqual(targets.count(FUSED_OP), 1)
        self.assertNotIn(PAD_OP, targets)

    def test_fused_output_shape_is_preserved(self):
        graph_module = _build_graph()
        expected = next(
            node.meta["val"].shape for node in graph_module.graph.nodes if node.target is REDUCE_SCATTER_OP
        )

        self._make_pass()(graph_module)

        fused = next(node for node in graph_module.graph.nodes if node.target is FUSED_OP)
        self.assertEqual(fused.meta["val"].shape, expected)

    def test_bias_is_not_fused(self):
        # Bias is added before the reduction, so it must not reach the padded rows.
        graph_module = _build_graph(bias=True)

        self._make_pass()(graph_module)

        self.assertNotIn(FUSED_OP, _targets(graph_module))

    def test_extra_matmul_consumer_is_not_fused(self):
        graph_module = _build_graph(extra_gemm_user=True)

        self._make_pass()(graph_module)

        self.assertNotIn(FUSED_OP, _targets(graph_module))

    def test_reduce_scatter_on_other_dim_is_not_fused(self):
        graph_module = _build_graph(dim=1)

        self._make_pass()(graph_module)

        self.assertNotIn(FUSED_OP, _targets(graph_module))

    def test_small_reduce_dim_is_not_fused(self):
        graph_module = _build_graph(reduce_dim=128)

        self._make_pass()(graph_module)

        self.assertNotIn(FUSED_OP, _targets(graph_module))

    def test_float32_is_not_fused(self):
        graph_module = _build_graph(dtype=torch.float32)

        self._make_pass()(graph_module)

        self.assertNotIn(FUSED_OP, _targets(graph_module))

    def test_unsupported_world_size_skips_the_pass(self):
        tp_group = MagicMock()
        tp_group.unique_name = TP_GROUP_NAME
        module = "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass"
        with (
            patch(f"{module}.get_tp_group", return_value=tp_group),
            patch(f"{module}.get_tensor_model_parallel_world_size", return_value=16),
        ):
            fusion_pass = MatmulReduceScatterFusionPass(VllmConfig())

        self.assertFalse(fusion_pass.is_applicable_for_range(MagicMock()))
