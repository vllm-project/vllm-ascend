# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; NPU quantization and NZ format conversion are mocked."""

import ast
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from tests.ut.ops.test_kimi_v30_review import ROOT, load_class


class TestKimiLatentProjection(unittest.TestCase):
    def setUp(self):
        self.context = NS(cudagraph_runtime_mode="NONE")
        self.cast = Mock(side_effect=lambda weight, **kwargs: weight.clone())
        self.ops = NS(
            npu_format_cast=lambda weight, fmt, **kwargs: weight.clone(),
            npu_quant_matmul=Mock(side_effect=lambda x, w, *args, **kwargs: torch.zeros(x.shape[0], w.shape[-1])),
            npu_grouped_matmul=Mock(side_effect=AssertionError("latent projection selected GMM")),
            npu_dynamic_mx_quant=Mock(
                side_effect=lambda x, **kwargs: (x.to(torch.float8_e4m3fn), torch.zeros(x.shape[0], 2, 2))
            ),
            float8_e8m0fnu=object(),
        )
        ns = dict(
            torch=torch,
            torch_npu=self.ops,
            PreparedLinearInput=NS,
            RowParallelLinear=type("RowParallelLinear", (), {}),
            maybe_trans_nz=self.cast,
            _should_trans_nz=lambda weight: True,
        )
        scheme_cls = load_class(
            "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
            "AscendW8A8MXFP8DynamicLinearMethod",
            {"apply", "_quantize_input", "process_weights_after_loading", "restore_weights_for_rl_loading"},
            ns,
        )
        self.scheme = scheme_cls()
        self.scheme.group_size = 32
        self.scheme.dynamic_mx_quant_scale_alg = 0
        self.scheme.apply_single_group_gmm = Mock(side_effect=AssertionError("latent projection selected GMM"))
        scheme = self.scheme

        class ReplicatedLinear(torch.nn.Module):
            # Stand in for vLLM's standard wrapper; execute the real quant method.
            def __init__(self, input_size, output_size, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(output_size, input_size).to(torch.float8_e4m3fn), False)
                self.weight_scale = torch.nn.Parameter(
                    torch.ones(output_size, input_size // 32, dtype=torch.uint8), False
                )
                self.quant_method = NS(quant_method=scheme, apply=scheme.apply)

            def forward(self, x):
                return self.quant_method.apply(self, x, bias=None), None

        ns.update(
            ReplicatedLinear=ReplicatedLinear,
            is_forward_context_available=lambda: True,
            get_forward_context=lambda: self.context,
            CUDAGraphMode=NS(NONE="NONE"),
        )
        # Include the former constructor so this test catches a return to the
        # eager-GMM/graph-QMM split instead of just checking source text.
        load_class(
            "vllm_ascend/models/kimi_k3.py",
            "AscendKimiK3RoutedLinear",
            {"__init__", "forward", "apply_projection"},
            ns,
            base="ReplicatedLinear",
        )
        tree = ast.parse((ROOT / "vllm_ascend/models/kimi_k3.py").read_text())
        moe = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendKimiMoE")
        init = next(n for n in moe.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        latent = next(
            n
            for n in init.body
            if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute) and n.test.attr == "use_latent_moe"
        )
        model = NS(use_latent_moe=True, moe_hidden_size=64, latent_moe_use_norm=False)
        ns.update(
            self=model,
            hidden_size=128,
            latent_quant_config=object(),
            prefix="model.layers.0.block_sparse_moe",
            AscendKimiRoutedOutputTransform=lambda norm, projection: projection,
        )
        exec(compile(ast.Module(body=[latent], type_ignores=[]), "latent_constructor", "exec"), ns)
        self.layers = (model.routed_expert_down_proj, model.routed_expert_up_proj)

    def test_nz_quant_matmul_is_identical_across_runtime_modes(self):
        for layer in self.layers:
            output_size, input_size = layer.weight.shape
            self.scheme.process_weights_after_loading(layer)
            self.assertEqual(layer.weight.shape, (input_size, output_size))
            self.assertEqual(layer.weight_scale.shape, (input_size // 64, output_size, 2))
            self.assertFalse(layer._mxfp8_gmm_nz_layout)
            self.assertEqual(self.cast.call_args.args[0].shape, (input_size, output_size))
            x = torch.ones(3, input_size, dtype=torch.bfloat16)
            quantized = x.to(torch.float8_e4m3fn)
            scale = torch.zeros(3, input_size // 64, 2)
            inputs = (x, (quantized, scale), NS(quantized=quantized, scale=scale, output_dtype=x.dtype))
            for mode in ("NONE", "FULL", "PIECEWISE"):
                self.context.cudagraph_runtime_mode = mode
                for value in inputs:
                    with self.subTest(mode=mode, input_type=type(value), input_size=input_size):
                        self.ops.npu_quant_matmul.reset_mock()
                        with patch.object(torch, "full", side_effect=AssertionError("unexpected group-list Fill")):
                            output, bias = layer(value)
                        self.assertEqual(output.shape, (3, output_size))
                        self.assertIsNone(bias)
                        self.ops.npu_quant_matmul.assert_called_once()
                        args = self.ops.npu_quant_matmul.call_args.args
                        self.assertIs(args[1], layer.weight)
                        self.assertIs(args[2], layer.weight_scale)
        self.scheme.apply_single_group_gmm.assert_not_called()
        self.ops.npu_grouped_matmul.assert_not_called()
        self.assertEqual(self.cast.call_count, 2)

    def test_weight_reload_preserves_quant_matmul_layout_and_buffers(self):
        for layer in self.layers:
            self.scheme.process_weights_after_loading(layer)
            pointers = layer.weight.data_ptr(), layer.weight_scale.data_ptr()
            shapes = layer.weight.shape, layer.weight_scale.shape
            self.scheme.restore_weights_for_rl_loading(layer)
            layer.weight.data.copy_(torch.full_like(layer.weight, 2))
            self.scheme.process_weights_after_loading(layer)
            self.assertEqual((layer.weight.shape, layer.weight_scale.shape), shapes)
            self.assertEqual((layer.weight.data_ptr(), layer.weight_scale.data_ptr()), pointers)
            self.assertFalse(layer._mxfp8_gmm_nz_layout)
            torch.testing.assert_close(layer.weight.float(), torch.full(shapes[0], 2.0))


if __name__ == "__main__":
    unittest.main()
