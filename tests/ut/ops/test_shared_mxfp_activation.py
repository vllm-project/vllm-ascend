# SPDX-License-Identifier: Apache-2.0
"""Dependency-free contract tests; kernel accuracy requires Ascend hardware.

Load the shared-expert methods from their AST so these tests also run without
torch_npu/vLLM. Tensor metadata and device calls are mocked.
"""

import ast
import unittest
from collections import namedtuple
from contextlib import nullcontext
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

QuantType = Enum("QuantType", "W8A8MXFP W8A8 W4A8 W4A8MXFP W8A8FP W4A4MXFP NONE")
PreparedLinearInput = namedtuple("PreparedLinearInput", "quantized scale output_dtype")


class TestSharedMXFPActivation(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[3]
        source = self.root / "vllm_ascend/ops/fused_moe/shared_experts.py"
        tree = ast.parse(source.read_text())
        paths = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SharedExpertMLPPath")
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSharedExperts")
        names = {
            "_apply_output_gate",
            "_can_fuse_w8a8_mxfp_activation",
            "_prepare_w8a8_mxfp_down_input",
            "_prepare_gate_up_input",
            "enqueue_gate_up_after_router",
            "_finish_linear_wrapped_mlp",
            "_finish_shared_mlp",
            "_select_mlp_path",
            "part2",
            "validate_consistency",
        }
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        constant_names = {"MXFP8_ACTIVATION_GROUP_SIZE", "SWIGLU_MX_INPUT_ALIGNMENT", "SWIGLU_MX_QUANT_MODE"}
        constants = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constant_names for t in n.targets)
        ]
        self.quant = QuantType
        self.activation_cls = type("AscendSiluAndMul", (), {})
        self.scheme_cls = type("AscendW8A8MXFP8DynamicLinearMethod", (), {})
        scheme_module = ModuleType("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8")
        setattr(scheme_module, self.scheme_cls.__name__, self.scheme_cls)
        patcher = patch.dict("sys.modules", {scheme_module.__name__: scheme_module})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.prepared_cls = PreparedLinearInput
        self.quantized = SimpleNamespace(shape=(36, 2048))
        self.fused = Mock(return_value=(self.quantized, "scale", None))
        self.torch = SimpleNamespace(
            bfloat16="bf16",
            float8_e4m3fn="fp8",
            ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_swiglu_group_quant=self.fused)),
        )
        self.namespace: dict[str, Any] = {
            "torch": self.torch,
            "QuantType": self.quant,
            "Enum": Enum,
            "auto": auto,
            "SubmittedSharedExpertGateUp": SimpleNamespace,
            "npu_stream_switch": lambda *args, **kwargs: nullcontext(),
            "shared_experts_calculation_stream": lambda: None,
            "PreparedLinearInput": self.prepared_cls,
            "has_lora": bool,
            "AscendSiluAndMul": self.activation_cls,
            "F": SimpleNamespace(sigmoid=lambda x: x),
            "logger": Mock(),
        }
        self.load_ast(source, [paths, *constants, *methods])
        self.paths = self.namespace["SharedExpertMLPPath"]
        executor_cls = type("Executor", (), {name: self.namespace[name] for name in names})
        self.executor = executor_cls()
        scheme = self.scheme_cls()
        scheme.dynamic_mx_quant_scale_alg = 0
        scheme.group_size = 32
        down = Mock(return_value=(7, None))
        down.quant_method = SimpleNamespace(quant_method=scheme)
        down.mxfp8_tp_padding = (0, 0)
        down.input_is_parallel = True
        down.weight_scale = object()
        self.executor.layer = SimpleNamespace(
            down_proj=down,
            gate_up_proj=SimpleNamespace(weight_scale=object()),
            act_fn=self.activation_cls(),
            expert_gate=None,
        )
        self.executor.quant_type = self.quant.W8A8MXFP
        self.executor.lora_context = None
        self.executor.multistream_overlap = True
        self.executor.swiglu_limit = 0.0
        self.gate_up = SimpleNamespace(dtype="bf16", ndim=2, shape=(36, 4096))
        self.executor.part1 = Mock(return_value=self.gate_up)
        self.executor.apply_activation = Mock(return_value="original_activation")
        self.executor._prepare_linear_input_for_overlap = Mock(side_effect=lambda layer, x: x)
        self.executor._wait_for_routed_stage = Mock()
        self.executor._wait_for_milestone = Mock()
        self.events = SimpleNamespace(shared_activation_overlap_start="activation", routed_combine_start="combine")
        self.submitted = SimpleNamespace(hidden_states="input", gate_up=self.gate_up)

    def load_ast(self, source, nodes):
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        exec(compile(module, str(source), "exec"), self.namespace)

    def run_mlp(self):
        self.submitted.path = self.executor._select_mlp_path()
        return self.executor._finish_shared_mlp(self.submitted, self.events)

    def test_mxfp_path_preserves_gate_up_preparation_and_linear(self):
        prepared_input = self.prepared_cls("quantized_hidden", "input_scale", "bf16")
        self.executor._prepare_linear_input_for_overlap.side_effect = None
        self.executor._prepare_linear_input_for_overlap.return_value = prepared_input
        result = self.executor._prepare_gate_up_input("input")
        self.assertIs(result, prepared_input)
        self.executor._prepare_linear_input_for_overlap.assert_called_once_with(
            self.executor.layer.gate_up_proj, "input"
        )
        prepared_execution = SimpleNamespace(hidden_states="input", gate_up_input=result, local_dp_metadata=None)
        submitted = self.executor.enqueue_gate_up_after_router(prepared_execution, "router")
        self.executor._wait_for_milestone.assert_called_once_with("router", "router_output_ready")
        self.executor.part1.assert_called_once_with(prepared_input)
        self.assertIs(submitted.path, self.paths.W8A8_MXFP_LINEAR)
        self.assertIs(submitted.gate_up, self.gate_up)
        self.fused.assert_not_called()

    def test_fused_input_reaches_down_and_preserves_expert_gate(self):
        self.executor.layer.expert_gate = Mock(return_value=(2, None))
        self.assertEqual(self.run_mlp(), 14)
        self.fused.assert_called_once_with(
            self.gate_up, topk_weight=None, group_index=None, dst_type="fp8", quant_mode=2, clamp_value=0.0
        )
        self.executor.apply_activation.assert_not_called()
        self.executor._prepare_linear_input_for_overlap.assert_not_called()
        self.executor.layer.down_proj.assert_called_once_with(self.prepared_cls(self.quantized, "scale", "bf16"))
        self.executor.layer.expert_gate.assert_called_once_with("input")

    def test_unsupported_inputs_keep_original_preparation(self):
        down = self.executor.layer.down_proj
        scheme = down.quant_method.quant_method
        cases = [
            (self.executor, "lora_context", object()),
            (self.executor, "swiglu_limit", 7.0),
            (self.executor.layer, "act_fn", object()),
            (scheme, "dynamic_mx_quant_scale_alg", 1),
            (scheme, "group_size", 64),
            (down, "mxfp8_tp_padding", (0, 16)),
            (down, "input_is_parallel", False),
            (self.gate_up, "dtype", "fp16"),
            (self.gate_up, "ndim", 3),
            (self.gate_up, "shape", (5, 250)),
            (down.quant_method, "quant_method", object()),
        ]
        for target, name, value in cases:
            with self.subTest(name=name, value=value), patch.object(target, name, value):
                self.fused.reset_mock()
                self.executor.apply_activation.reset_mock()
                self.executor._prepare_linear_input_for_overlap.reset_mock()
                down.reset_mock()
                self.assertEqual(self.run_mlp(), 7)
                self.fused.assert_not_called()
                self.executor.apply_activation.assert_called_once_with(self.gate_up)
                self.executor._prepare_linear_input_for_overlap.assert_called_once_with(down, "original_activation")
                down.assert_called_once_with("original_activation")

    def test_mx_scale_pair_alignment_falls_back(self):
        for width in (0, 64, 192, 320):
            with self.subTest(width=width):
                self.gate_up.shape = (36, width)
                self.fused.reset_mock()
                self.executor.apply_activation.reset_mock()
                self.assertEqual(self.run_mlp(), 7)
                self.fused.assert_not_called()
                self.executor.apply_activation.assert_called_once_with(self.gate_up)

    def test_aligned_profile_shapes_fuse(self):
        for shape in ((1, 128), (30, 4096), (36, 4096)):
            with self.subTest(shape=shape):
                self.gate_up.shape = shape
                self.fused.reset_mock()
                self.executor.apply_activation.reset_mock()
                self.assertEqual(self.run_mlp(), 7)
                self.fused.assert_called_once()
                self.executor.apply_activation.assert_not_called()

    def test_existing_quantization_dispatch_is_preserved(self):
        expected = {
            "W8A8": "A8_INT_FUSED",
            "W4A8": "A8_INT_FUSED",
            "W4A8MXFP": "A8_MXFP_FUSED",
            "W8A8MXFP": "W8A8_MXFP_LINEAR",
        }
        for quant in self.quant:
            with self.subTest(quant=quant):
                self.executor.quant_type = quant
                self.assertEqual(self.executor._select_mlp_path().name, expected.get(quant.name, "LINEAR_WRAPPER"))
                self.assertEqual(
                    self.executor._can_fuse_w8a8_mxfp_activation(self.gate_up), quant is self.quant.W8A8MXFP
                )

    def test_lora_selects_generic_wrapper_for_every_quantization(self):
        self.executor.lora_context = object()
        for quant in self.quant:
            with self.subTest(quant=quant):
                self.executor.quant_type = quant
                self.assertIs(self.executor._select_mlp_path(), self.paths.LINEAR_WRAPPER)
                self.assertEqual(self.run_mlp(), 7)
        self.fused.assert_not_called()

    def test_other_fused_paths_keep_their_finishers(self):
        self.executor._run_a8_int_mlp = Mock(return_value="int_output")
        self.executor._run_a8_mxfp_mlp = Mock(return_value="mxfp_output")
        for quant, expected in (
            (self.quant.W8A8, "int_output"),
            (self.quant.W4A8, "int_output"),
            (self.quant.W4A8MXFP, "mxfp_output"),
        ):
            with self.subTest(quant=quant):
                self.executor.quant_type = quant
                self.assertEqual(self.run_mlp(), expected)
        self.executor.apply_activation.assert_not_called()
        self.fused.assert_not_called()

    def test_generic_wrapper_does_not_try_mxfp_fusion(self):
        for quant in (self.quant.W8A8FP, self.quant.W4A4MXFP, self.quant.NONE):
            with self.subTest(quant=quant):
                self.executor.quant_type = quant
                with patch.object(self.executor, "_prepare_w8a8_mxfp_down_input") as prepare_mxfp:
                    self.assertEqual(self.run_mlp(), 7)
                    prepare_mxfp.assert_not_called()

    def test_fusion_precedes_combine_wait_and_down_follows_it(self):
        order = Mock()
        order.attach_mock(self.executor._wait_for_routed_stage, "wait_stage")
        order.attach_mock(self.fused, "fused_activation")
        order.attach_mock(self.executor.layer.down_proj, "down")
        self.run_mlp()
        self.assertEqual(
            order.mock_calls,
            [
                call.wait_stage(self.events, "activation", "shared_activation_overlap_start"),
                call.fused_activation(
                    self.gate_up, topk_weight=None, group_index=None, dst_type="fp8", quant_mode=2, clamp_value=0.0
                ),
                call.wait_stage(self.events, "combine", "routed_combine_start"),
                call.down(self.prepared_cls(self.quantized, "scale", "bf16")),
            ],
        )

    def test_consistency_validation_uses_runtime_activation_path(self):
        self.executor.layer = Mock(return_value=7, **vars(self.executor.layer))
        self.executor.shared_expert_input_size = 6144
        self.executor.in_dtype = "bf16"
        self.torch.rand = Mock(return_value=0.75)
        self.torch.allclose = Mock(return_value=True)
        for width, should_fuse in ((4096, True), (192, False)):
            with self.subTest(width=width):
                self.gate_up.shape = (36, width)
                self.fused.reset_mock()
                self.executor.apply_activation.reset_mock()
                self.torch.allclose.reset_mock()
                self.executor.validate_consistency()
                self.torch.allclose.assert_called_once_with(7, 7)
                self.assertEqual(self.fused.call_count, int(should_fuse))
                self.assertEqual(self.executor.apply_activation.call_count, int(not should_fuse))

    def test_fallback_keeps_preparation_before_combine_and_down_after_it(self):
        self.gate_up.shape = (36, 192)
        order = Mock()
        order.attach_mock(self.executor._wait_for_routed_stage, "wait_stage")
        order.attach_mock(self.executor.apply_activation, "activation")
        order.attach_mock(self.executor._prepare_linear_input_for_overlap, "prepare")
        order.attach_mock(self.executor.layer.down_proj, "down")
        self.assertEqual(self.run_mlp(), 7)
        self.assertIs(self.submitted.path, self.paths.W8A8_MXFP_LINEAR)
        self.fused.assert_not_called()
        self.assertEqual(
            order.mock_calls,
            [
                call.wait_stage(self.events, "activation", "shared_activation_overlap_start"),
                call.activation(self.gate_up),
                call.prepare(self.executor.layer.down_proj, "original_activation"),
                call.wait_stage(self.events, "combine", "routed_combine_start"),
                call.down("original_activation"),
            ],
        )

    def test_missing_operator_falls_back(self):
        del self.torch.ops._C_ascend.npu_swiglu_group_quant
        self.assertEqual(self.run_mlp(), 7)
        self.executor.apply_activation.assert_called_once_with(self.gate_up)

    def test_linear_consumes_fused_input_without_requantizing(self):
        source = self.root / "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == self.scheme_cls.__name__)
        apply = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
        qmm = Mock(return_value="output")
        self.namespace["torch_npu"] = SimpleNamespace(float8_e8m0fnu="e8m0", npu_quant_matmul=qmm)
        self.load_ast(source, [apply])
        scheme = self.executor.layer.down_proj.quant_method.quant_method
        scheme._quantize_input = Mock()
        self.executor.layer.down_proj.weight = "weight"
        prepared = self.executor._prepare_w8a8_mxfp_down_input(self.gate_up)
        result = self.namespace["apply"](scheme, self.executor.layer.down_proj, prepared)
        self.assertEqual(result, "output")
        scheme._quantize_input.assert_not_called()
        self.assertIs(qmm.call_args.args[0], self.quantized)
        self.assertEqual(qmm.call_args.kwargs["pertoken_scale"], "scale")
        self.assertEqual(qmm.call_args.kwargs["output_dtype"], "bf16")
        self.assertEqual(qmm.call_args.kwargs["group_sizes"], [1, 1, 32])


if __name__ == "__main__":
    unittest.main()
