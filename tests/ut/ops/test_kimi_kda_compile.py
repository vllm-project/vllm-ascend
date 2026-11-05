# SPDX-License-Identifier: Apache-2.0
"""CPU Dynamo/AOT regression for KDA metadata dispatch.

Run directly with PyTorch installed, without importing the NPU plugin:
    python tests/ut/ops/test_kimi_kda_compile.py -v

The production boundary and dispatcher are extracted from source; numerical KDA
kernels are mocked. This checks compilation semantics, not NPU kernel accuracy.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch._dynamo.backends.common import aot_autograd


class TestKDACompile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_kda.py"
        tree = ast.parse(source.read_text())
        functions = {"_ascend_kda_forward", "_ascend_kda_forward_fake"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
        attention = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendKimiK3DeltaAttention"
        )
        attention.bases = []
        attention.decorator_list = []
        attention.body = [
            node for node in attention.body if isinstance(node, ast.FunctionDef) and node.name == "_forward"
        ]
        # Keep the CPU registration separate from an installed NPU plugin.
        for node in ast.walk(attention):
            if isinstance(node, ast.Attribute) and node.attr == "vllm" and ast.unparse(node.value) == "torch.ops":
                node.attr = "kda_compile_regression"
        # Capture segmentation is NPU-specific; the CPU test keeps its body.
        cls.context = SimpleNamespace(attn_metadata=None, no_compile_layers={})
        cls.namespace = {
            "torch": torch,
            "get_forward_context": lambda: cls.context,
            "eager_break_during_capture": lambda fn: fn,
        }
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                *nodes,
                attention,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), cls.namespace)
        cls.layer_type = cls.namespace["AscendKimiK3DeltaAttention"]
        cls.library = torch.library.Library("kda_compile_regression", "FRAGMENT")

        def register(op_name, op_func, mutates_args, fake_impl):
            cls.library.define(op_name + torch.library.infer_schema(op_func, mutates_args=mutates_args))
            cls.library.impl(op_name, op_func, "CPU")
            cls.library._register_fake(op_name, fake_impl)

        # Execute the production registration too, including its mutation schema.
        registration = next(
            node
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "direct_register_custom_op"
        )
        cls.namespace["direct_register_custom_op"] = register
        exec(compile(ast.Module(body=[registration], type_ignores=[]), str(source), "exec"), cls.namespace)

    def setUp(self):
        torch._dynamo.reset()
        self.context.attn_metadata = None
        self.context.no_compile_layers.clear()

    def make_layer(self, multiplier):
        layer = self.layer_type()
        layer.prefix = "model.layers.0.self_attn"
        layer.use_fused_kda = True

        def fused(mixed_qkv, g1, g2, beta, out):
            metadata = self.context.attn_metadata
            if metadata is None:
                out.zero_()
            else:
                out.copy_(mixed_qkv * multiplier + metadata[layer.prefix])

        layer._kda_forward_impl = MagicMock(side_effect=lambda *args, **kwargs: fused(*args))
        self.context.no_compile_layers[layer.prefix] = layer
        return layer

    def test_none_metadata_trace_keeps_runtime_dispatch(self):
        for use_aot in (False, True):
            with self.subTest(use_aot=use_aot):
                self.setUp()
                layer = self.make_layer(2)
                graphs = []

                def backend(graph, inputs, graphs=graphs, use_aot=use_aot):
                    graphs.append(graph)
                    if use_aot:
                        return aot_autograd(fw_compiler=lambda gm, _: gm.forward)(graph, inputs)
                    return graph.forward

                def forward(x, layer=layer):
                    out = torch.empty_like(x)
                    layer._forward(x, x, x, x, out, beta_is_preprocessed=True)
                    return out + 1

                compiled = torch.compile(forward, backend=backend, fullgraph=True)
                x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
                torch.testing.assert_close(compiled(x), torch.ones_like(x))
                self.assertEqual(len(graphs), 1)
                self.assertIn("kda_compile_regression.ascend_kda_forward", graphs[0].code)
                self.assertNotIn("zero_", graphs[0].code)
                self.assertEqual(layer._kda_forward_impl.call_count, 1)

                # Reuse the graph traced with None for both prefill/decode-like
                # metadata changes and a fresh model with the same layer prefix.
                for metadata in (3, 7):
                    self.context.attn_metadata = {layer.prefix: metadata}
                    torch.testing.assert_close(compiled(x), x * 2 + metadata + 1)
                replacement = self.make_layer(5)
                self.context.attn_metadata = {layer.prefix: 11}
                torch.testing.assert_close(compiled(x), x * 5 + 12)
                replacement._kda_forward_impl.assert_called_once()
                self.assertEqual(len(graphs), 1)

    def test_ascendc_dispatch_preserves_beta_flag_and_dummy_zero(self):
        layer = self.make_layer(2)
        layer.use_fused_kda = False
        x = torch.ones(2, 3)
        out = torch.full_like(x, torch.nan)
        layer._forward(x, x, x, x, out)
        torch.testing.assert_close(out, torch.zeros_like(x))
        layer._kda_forward_impl.assert_called_once()
        layer._kda_forward_impl = MagicMock(side_effect=lambda *args, **kwargs: args[4].copy_(args[0]))
        for flag in (False, True):
            layer._forward(x, x, x, x, out, beta_is_preprocessed=flag)
            self.assertEqual(layer._kda_forward_impl.call_args.kwargs, {"beta_is_preprocessed": flag})
            self.assertIs(layer._kda_forward_impl.call_args.args[4], out)
            torch.testing.assert_close(out, x)


if __name__ == "__main__":
    unittest.main()
