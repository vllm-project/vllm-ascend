# SPDX-License-Identifier: Apache-2.0

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
    scattered = gemm
    reduced_rows = num_tokens
    if pad_rows is not None:
        reduced_rows += pad_rows
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

    def _apply(self, graph_module: fx.GraphModule) -> None:
        tp_group = MagicMock()
        tp_group.unique_name = TP_GROUP_NAME
        module = "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass"
        with (
            patch(f"{module}.get_tp_group", return_value=tp_group),
            patch(f"{module}.get_tensor_model_parallel_world_size", return_value=TP_SIZE),
        ):
            self._make_pass()(graph_module)

    def test_padded_pattern_is_fused_and_padding_moves_to_input(self):
        graph_module = _build_graph()
        self._apply(graph_module)

        self.assertEqual(_targets(graph_module).count(FUSED_OP), 1)
        self.assertNotIn(GEMM_OP, _targets(graph_module))
        fused = next(node for node in graph_module.graph.nodes if node.target is FUSED_OP)
        padded_input = fused.args[0]
        self.assertIs(padded_input.target, PAD_OP)
        self.assertEqual(padded_input.args[0].op, "placeholder")
        self.assertEqual(padded_input.meta["val"].shape, (10, REDUCE_DIM))

    def test_unpadded_pattern_is_fused(self):
        graph_module = _build_graph(pad_rows=None)
        self._apply(graph_module)
        self.assertEqual(_targets(graph_module).count(FUSED_OP), 1)
        self.assertNotIn(PAD_OP, _targets(graph_module))

    def test_fused_output_shape_is_preserved(self):
        graph_module = _build_graph()
        expected = next(node.meta["val"].shape for node in graph_module.graph.nodes if node.target is REDUCE_SCATTER_OP)
        self._apply(graph_module)
        fused = next(node for node in graph_module.graph.nodes if node.target is FUSED_OP)
        self.assertEqual(fused.meta["val"].shape, expected)

    def test_unsupported_patterns_are_not_fused(self):
        for kwargs in (
            {"bias": True},
            {"extra_gemm_user": True},
            {"dim": 1},
            {"reduce_dim": 128},
            {"dtype": torch.float32},
        ):
            with self.subTest(**kwargs):
                graph_module = _build_graph(**kwargs)
                self._apply(graph_module)
                self.assertNotIn(FUSED_OP, _targets(graph_module))

    def test_unsupported_world_size_skips_the_pass(self):
        module = "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass"
        with patch(f"{module}.get_tensor_model_parallel_world_size", return_value=16):
            fusion_pass = MatmulReduceScatterFusionPass(VllmConfig())
            self.assertFalse(fusion_pass.is_applicable_for_range(MagicMock()))
