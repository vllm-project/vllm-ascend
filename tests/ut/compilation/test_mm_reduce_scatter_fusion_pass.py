# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from unittest.mock import MagicMock, patch

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.config import VllmConfig

from tests.ut.base import TestBase


@torch.library.custom_op("mmrs_test::npu_mm_reduce_scatter_base", mutates_args=())
def _fake_mm_reduce_scatter(
    x1: torch.Tensor,
    x2: torch.Tensor,
    hcom: str,
    world_size: int,
    *,
    reduce_op: str = "sum",
) -> torch.Tensor:
    return x1.new_empty((x1.shape[0] // world_size, x2.shape[1]))


@_fake_mm_reduce_scatter.register_fake
def _fake_mm_reduce_scatter_fake(
    x1: torch.Tensor, x2: torch.Tensor, hcom: str, world_size: int, *, reduce_op: str = "sum"
) -> torch.Tensor:
    return x1.new_empty((x1.shape[0] // world_size, x2.shape[1]))


_stubbed_nge = importlib.util.find_spec("npugraph_ex") is None and importlib.util.find_spec("torchair") is None
if _stubbed_nge:
    sys.modules["npugraph_ex"] = MagicMock()
try:
    from vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass import (
        MatmulReduceScatterFusionPass,
    )
finally:
    if _stubbed_nge:
        del sys.modules["npugraph_ex"]

TP_SIZE = 2
TP_GROUP_NAME = "tp:0"
HCOM_NAME = "test_hcom"
REDUCE_DIM = 512
OUTPUT_DIM = 256

GEMM_OP = torch.ops.vllm.unquantized_gemm.default
REDUCE_SCATTER_OP = torch.ops.vllm.reduce_scatter.default
PAD_OP = torch.ops.aten.constant_pad_nd.default
CAT_OP = torch.ops.aten.cat.default
FULL_OP = torch.ops.aten.full.default
FUSED_OP = torch.ops.mmrs_test.npu_mm_reduce_scatter_base.default


def _build_graph(
    *,
    num_tokens: int = 8,
    pad_rows: int | None = 2,
    bias: bool = False,
    dim: int = 0,
    reduce_dim: int = REDUCE_DIM,
    extra_gemm_user: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    world_size: int = TP_SIZE,
    group_name: str = TP_GROUP_NAME,
    cat_dim: int = 0,
    reverse_cat_inputs: bool = False,
    valid_zeros_shape: bool = True,
    use_constant_pad: bool = False,
    pad_value: float = 0.0,
    valid_padding: bool = True,
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
    gemm_args = (x, weight, bias_node) if bias else (x, weight)
    gemm = call(GEMM_OP, gemm_args, num_tokens, OUTPUT_DIM)
    scattered = gemm
    reduced_rows = num_tokens
    if pad_rows is not None:
        reduced_rows += pad_rows
        if use_constant_pad:
            widths = [0, 0, 0, pad_rows] if valid_padding else [0, 1, 0, pad_rows]
            scattered = call(PAD_OP, (gemm, widths, pad_value), reduced_rows, OUTPUT_DIM)
        else:
            zeros_columns = OUTPUT_DIM if valid_zeros_shape else OUTPUT_DIM + 1
            zeros = graph.call_function(
                FULL_OP,
                args=([pad_rows, zeros_columns], 0),
                kwargs={"dtype": dtype, "layout": torch.strided, "device": torch.device("cpu"), "pin_memory": False},
            )
            with fake_mode:
                zeros.meta["val"] = torch.empty((pad_rows, zeros_columns), dtype=dtype)
            cat_inputs = [zeros, gemm] if reverse_cat_inputs else [gemm, zeros]
            cat_args = (cat_inputs,) if cat_dim == 0 else (cat_inputs, cat_dim)
            scattered = call(CAT_OP, cat_args, reduced_rows, OUTPUT_DIM)
    reduce_scatter = call(
        REDUCE_SCATTER_OP,
        (scattered, dim, world_size, group_name),
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


@patch(
    "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass.DeviceOperator.npu_mm_reduce_scatter_base",
    new=_fake_mm_reduce_scatter,
)
class TestMatmulReduceScatterFusionPass(TestBase):
    def _make_pass(self) -> MatmulReduceScatterFusionPass:
        tp_group = MagicMock()
        tp_group.unique_name = TP_GROUP_NAME
        tp_group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = HCOM_NAME
        module = "vllm_ascend.compilation.passes.mm_reduce_scatter_fusion_pass"
        with (
            patch(f"{module}.get_tp_group", return_value=tp_group),
            patch(f"{module}.get_tensor_model_parallel_world_size", return_value=TP_SIZE),
        ):
            config = VllmConfig()
            config.model_config = MagicMock(dtype=torch.bfloat16)
            return MatmulReduceScatterFusionPass(config)

    def _apply(self, graph_module: fx.GraphModule) -> None:
        tp_group = MagicMock()
        tp_group.unique_name = TP_GROUP_NAME
        tp_group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = HCOM_NAME
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
        self.assertNotIn(CAT_OP, _targets(graph_module))
        self.assertNotIn(FULL_OP, _targets(graph_module))
        fused = next(node for node in graph_module.graph.nodes if node.target is FUSED_OP)
        padded_input = fused.args[0]
        self.assertIs(padded_input.target, PAD_OP)
        self.assertIs(fused.args[1].target, torch.ops.aten.permute.default)
        self.assertEqual(fused.args[1].args[1], [1, 0])
        self.assertEqual(fused.args[2:4], (HCOM_NAME, TP_SIZE))
        self.assertEqual(padded_input.args[0].op, "placeholder")
        self.assertEqual(padded_input.meta["val"].shape, (10, REDUCE_DIM))

    def test_unpadded_pattern_is_fused(self):
        graph_module = _build_graph(pad_rows=None)
        self._apply(graph_module)
        self.assertEqual(_targets(graph_module).count(FUSED_OP), 1)
        self.assertNotIn(PAD_OP, _targets(graph_module))

    def test_legacy_constant_pad_pattern_is_fused(self):
        graph_module = _build_graph(use_constant_pad=True)
        self._apply(graph_module)

        self.assertEqual(_targets(graph_module).count(FUSED_OP), 1)
        fused = next(node for node in graph_module.graph.nodes if node.target is FUSED_OP)
        self.assertIs(fused.args[0].target, PAD_OP)
        self.assertEqual(fused.args[0].args[1], [0, 0, 0, 2])
        self.assertEqual(fused.args[0].args[0].op, "placeholder")

    def test_nge_registration_preserves_padding_scalar(self):
        # Exercise the arguments sent to NGE using its underlying matcher,
        # without importing an NPU backend in CPU-only unit tests.
        backend_pass = pm.PatternMatcherPass()

        def register_backend(**kwargs):
            pm.register_replacement(**kwargs, trace_fn=pm.fwd_only, pass_dicts=backend_pass)

        module = "vllm_ascend.compilation.passes.base_pattern"
        with (
            patch(f"{module}._registered_patterns", set()),
            patch(f"{module}.nge.register_replacement", side_effect=register_backend),
        ):
            self._make_pass()

        for pad_rows in (None, 1, 2, 7):
            with self.subTest(pad_rows=pad_rows):
                graph_module = _build_graph(num_tokens=7 if pad_rows in (1, 7) else 8, pad_rows=pad_rows)
                expected = next(
                    node.meta["val"].shape for node in graph_module.graph.nodes if node.target is REDUCE_SCATTER_OP
                )
                self.assertEqual(backend_pass.apply(graph_module.graph), 1)
                fused = next(node for node in graph_module.graph.nodes if node.target is FUSED_OP)
                self.assertEqual(fused.meta["val"].shape, expected)
                if pad_rows is None:
                    self.assertNotIn(PAD_OP, _targets(graph_module))
                else:
                    padded_input = fused.args[0]
                    self.assertIs(padded_input.target, PAD_OP)
                    self.assertEqual(padded_input.args[1], [0, 0, 0, pad_rows])
                    self.assertEqual(padded_input.args[0].op, "placeholder")

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
            {"world_size": 4},
            {"group_name": "tp:1"},
            {"reduce_dim": 128},
            {"dtype": torch.float32},
            {"cat_dim": 1},
            {"reverse_cat_inputs": True},
            {"valid_zeros_shape": False},
            {"use_constant_pad": True, "pad_value": 1.0},
            {"use_constant_pad": True, "valid_padding": False},
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
