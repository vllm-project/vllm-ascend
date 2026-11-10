# SPDX-License-Identifier: Apache-2.0
"""Exercise production stream ordering without importing torch_npu or vLLM.

The methods are loaded from their AST; only tensor operations and device calls
are mocked. Kernel accuracy and actual Cube/Vector overlap require Ascend NPU.
"""

import ast
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call


class TestSharedActivationOverlap(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[3]
        MoEActivation = Enum("MoEActivation", "SITU SILU SWIGLUOAI SWIGLUOAI_UNINTERLEAVE SWIGLUSTEP GELU GELU_TANH")
        self.activation = MoEActivation
        self.order = Mock()
        self.events: list[object] = []
        self.stream = Mock()

        def record_event():
            event = object()
            self.events.append(event)
            return event

        self.stream.record_event.side_effect = record_event
        self.order.attach_mock(self.stream.record_event, "record_event")
        self.order.attach_mock(self.stream.wait_event, "wait_event")
        self.namespace: dict[str, Any] = {
            "torch": SimpleNamespace(
                float32="fp32",
                float16="fp16",
                bfloat16="bf16",
                int8="int8",
                float8_e4m3fn="fp8",
                uint8="uint8",
                npu=SimpleNamespace(current_stream=lambda: self.stream),
            ),
            "MoEActivation": self.activation,
            "Enum": Enum,
            "auto": auto,
            "dataclass": dataclass,
            "_EXTRA_CTX": SimpleNamespace(moe_comm_method=object()),
            "get_ascend_config": lambda: SimpleNamespace(multistream_overlap_shared_expert=True),
        }
        quant_source = self.root / "vllm_ascend/quantization/quant_type.py"
        self.load_ast(
            quant_source,
            [n for n in ast.parse(quant_source.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef)],
        )
        self.quant_type = self.namespace["QuantType"]
        comm_source = self.root / "vllm_ascend/ops/fused_moe/moe_comm_method.py"
        comm_tree = ast.parse(comm_source.read_text(encoding="utf-8"))
        comm_names = {"MoECommMethod", "MC2CommImpl", "AlltoAllCommImpl", "AllGatherCommImpl"}
        nodes = [n for n in comm_tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedExpertsResult"]
        for cls in comm_tree.body:
            if not isinstance(cls, ast.ClassDef) or cls.name not in comm_names:
                continue
            # Keep scheduling class attributes and the production method,
            # without importing or initializing communication machinery.
            cls.body = [
                n
                for n in cls.body
                if isinstance(n, ast.Assign) or isinstance(n, ast.FunctionDef) and n.name == "fused_experts"
            ] or [ast.Pass()]
            if cls.name == "MoECommMethod":
                cls.bases = []
            nodes.append(cls)
        self.load_ast(comm_source, nodes)
        mlp_source = self.root / "vllm_ascend/ops/fused_moe/moe_mlp.py"
        mlp_tree = ast.parse(mlp_source.read_text(encoding="utf-8"))
        self.load_ast(
            mlp_source,
            [
                n
                for n in mlp_tree.body
                if isinstance(n, ast.FunctionDef) and n.name in ("apply_moe_mlp", "maybe_record_event")
            ],
        )

    def load_ast(self, source, nodes):
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        exec(compile(module, str(source), "exec"), self.namespace)

    def run_routed(self, backend, activation=None, quant_type=None, fused=True, legacy=False):
        self.order.reset_mock()
        self.events.clear()
        activation = self.activation.SITU if activation is None else activation
        quant_type = self.quant_type.W4A8MXFP if quant_type is None else quant_type
        fused_input = SimpleNamespace(
            hidden_states=SimpleNamespace(dtype="bf16"),
            activation=activation,
            quant=SimpleNamespace(quant_type=quant_type, is_quant=quant_type != self.quant_type.NONE),
            lora_context=None,
        )
        dispatch_output = SimpleNamespace(combine_metadata=object(), group_list_type=1, group_list=object())
        dispatcher = SimpleNamespace(
            token_dispatch=Mock(return_value=dispatch_output), token_combine=Mock(return_value="routed_output")
        )
        self.order.attach_mock(dispatcher.token_dispatch, "dispatch")
        self.order.attach_mock(dispatcher.token_combine, "combine")
        self.namespace["build_token_dispatch_input"] = Mock(return_value="dispatch_input")
        self.namespace["build_mlp_compute_input"] = Mock(return_value=fused_input)
        quant = SimpleNamespace(
            supports_fused_activation=Mock(return_value=fused),
            apply_gmm1_act_quant=Mock(return_value=("quantized_activation", "scale")),
            apply_gmm1=Mock(return_value="gate_up"),
            apply_act_quant=Mock(return_value=("quantized_activation", "scale")),
            apply_gmm2=Mock(return_value="mlp_output"),
        )
        self.namespace["_unified_apply_activation"] = Mock(return_value="activation_output")
        for name in ("apply_gmm1_act_quant", "apply_gmm1", "apply_act_quant", "apply_gmm2"):
            self.order.attach_mock(getattr(quant, name), name)
        self.order.attach_mock(self.namespace["_unified_apply_activation"], "activation")
        executor = self.namespace[backend]()
        executor.token_dispatcher = dispatcher
        executor.moe_config = object()
        executor._apply_mlp = Mock(side_effect=lambda mlp: self.namespace["apply_moe_mlp"](mlp, quant))
        result = executor.fused_experts(fused_input, None if legacy else quant)
        self.assertEqual(result.routed_out, "routed_output")
        self.assertIs(result.expert_tokens, dispatch_output.group_list)
        self.assertEqual(result.group_list_type, dispatch_output.group_list_type)
        dispatcher.token_combine.assert_called_once_with(
            hidden_states="mlp_output", combine_metadata=dispatch_output.combine_metadata
        )
        quant.apply_gmm2.assert_called_once_with(fused_input, "quantized_activation", "scale")
        if legacy:
            executor._apply_mlp.assert_called_once_with(fused_input)
        else:
            executor._apply_mlp.assert_not_called()
        return result

    def assert_routed_schedule(self, result, fused=True):
        expected = ["record_event", "dispatch"]
        expected.extend(["apply_gmm1_act_quant"] if fused else ["apply_gmm1", "activation", "apply_act_quant"])
        expected.extend(["record_event", "apply_gmm2", "record_event", "combine"])
        self.assertEqual([c[0] for c in self.order.mock_calls], expected)
        self.assertEqual(len(self.events), 3)
        self.assertIs(result.before_dispatch_evt, self.events[0])
        self.assertIs(result.before_gmm2_evt, self.events[-2])
        self.assertIs(result.before_combine_evt, self.events[-1])
        self.assertIs(result.shared_activation_overlap_start_evt, result.before_gmm2_evt)

    def test_fused_activations_and_quantizations_wait_until_gmm2(self):
        for backend in ("MC2CommImpl", "AlltoAllCommImpl", "AllGatherCommImpl"):
            for activation in self.activation:
                for quant_type in self.quant_type:
                    with self.subTest(backend=backend, activation=activation, quant_type=quant_type):
                        result = self.run_routed(backend, activation=activation, quant_type=quant_type)
                        self.assert_routed_schedule(result)

    def test_nonfused_activations_and_quantizations_wait_until_gmm2(self):
        for backend in ("MC2CommImpl", "AlltoAllCommImpl", "AllGatherCommImpl"):
            for activation in self.activation:
                for quant_type in self.quant_type:
                    with self.subTest(backend=backend, activation=activation, quant_type=quant_type):
                        result = self.run_routed(backend, activation=activation, quant_type=quant_type, fused=False)
                        self.assert_routed_schedule(result, fused=False)

    def test_legacy_mlp_override_keeps_event_contract(self):
        for backend in ("MC2CommImpl", "AlltoAllCommImpl", "AllGatherCommImpl"):
            for fused in (False, True):
                for activation in self.activation:
                    for quant_type in self.quant_type:
                        with self.subTest(backend=backend, activation=activation, quant_type=quant_type, fused=fused):
                            result = self.run_routed(
                                backend, activation=activation, quant_type=quant_type, fused=fused, legacy=True
                            )
                            self.assert_routed_schedule(result, fused=fused)

    def make_shared_executor(self):
        source = self.root / "vllm_ascend/ops/fused_moe/shared_experts.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        self.load_ast(source, [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SharedExpertMLPPath"])
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSharedExperts")
        names = {
            "_run_a8_int_mlp",
            "_run_a8_mxfp_mlp",
            "_prepare_w8a8_mxfp_down_input",
            "_finish_linear_wrapped_mlp",
            "_finish_shared_mlp",
            "_wait_for_routed_stage",
            "_wait_for_milestone",
        }
        self.load_ast(source, [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names])
        shared_cls = type("SharedExecutor", (), {name: self.namespace[name] for name in names})
        shared = shared_cls()
        shared.multistream_overlap = True
        shared.situ_activation = None
        shared.swiglu_limit = 0.0
        shared.layer = SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight_scale_fp32="gate_up_scale"),
            down_proj=SimpleNamespace(weight="down_weight", weight_scale="down_scale"),
        )
        shared.part2 = Mock(return_value="shared_output")
        shared._apply_output_gate = Mock(return_value="shared_output")
        shared.apply_activation = Mock(return_value="shared_activation")
        shared._prepare_linear_input_for_overlap = Mock(return_value="prepared_down_input")
        shared._can_fuse_w8a8_mxfp_activation = Mock(return_value=False)
        quantized = ("shared_quantized", "shared_scale")
        shared_ops = SimpleNamespace(
            situ_mx_quant=Mock(return_value=quantized),
            dequant_situ_quant=Mock(return_value=quantized),
            npu_dequant_swiglu_quant=Mock(return_value=quantized),
            npu_swiglu_group_quant=Mock(return_value=(*quantized, None)),
        )
        self.namespace["torch"].ops = SimpleNamespace(_C_ascend=shared_ops)
        self.namespace["torch_npu"] = SimpleNamespace(npu_quant_matmul=Mock(return_value="shared_down_output"))
        self.namespace["SITU_MX_DST_TYPE_E4M3FN"] = 1
        self.namespace["SWIGLU_MX_QUANT_MODE"] = 2
        self.namespace["PreparedLinearInput"] = lambda data, scale, dtype: (data, scale, dtype)
        self.namespace["get_current_hardware_profile"] = lambda: SimpleNamespace(supports=lambda _: False)
        self.namespace["HardwareCapability"] = SimpleNamespace(FUSED_SWIGLU_TUNING_ARGS=object())
        for name, operation in vars(shared_ops).items():
            self.order.attach_mock(operation, name)
        self.order.attach_mock(self.namespace["torch_npu"].npu_quant_matmul, "quant_down")
        self.order.attach_mock(shared.part2, "down")
        self.order.attach_mock(shared._apply_output_gate, "output_gate")
        self.order.attach_mock(shared.apply_activation, "shared_activation")
        self.order.attach_mock(shared._prepare_linear_input_for_overlap, "prepare_down")
        return shared

    def assert_shared_schedule(self, shared, routed, path, operations):
        submitted = SimpleNamespace(
            hidden_states=SimpleNamespace(dtype="bf16"),
            gate_up=SimpleNamespace(dtype="bf16"),
            gate_up_input_scale="gate_up_input_scale",
            path=path,
        )
        milestones = SimpleNamespace(
            has_fine_grained_stage_events=True,
            shared_activation_overlap_start=routed.shared_activation_overlap_start_evt,
            routed_combine_start=routed.before_combine_evt,
        )
        self.order.reset_mock()
        self.assertEqual(shared._finish_shared_mlp(submitted, milestones), "shared_output")
        self.assertEqual([c[0] for c in self.order.mock_calls], operations)
        self.assertEqual(
            self.stream.wait_event.call_args_list,
            [call(routed.before_gmm2_evt), call(routed.before_combine_evt)],
        )
        return submitted

    def test_shared_int_activation_quant_waits_for_gmm2_and_down_waits_for_combine(self):
        shared = self.make_shared_executor()
        path = self.namespace["SharedExpertMLPPath"].A8_INT_FUSED
        for backend in ("MC2CommImpl", "AlltoAllCommImpl", "AllGatherCommImpl"):
            for situ in (False, True):
                with self.subTest(backend=backend, situ=situ):
                    shared.situ_activation = SimpleNamespace(beta=1.0, linear_beta=None) if situ else None
                    routed = self.run_routed(backend, quant_type=self.quant_type.W8A8)
                    operation = "dequant_situ_quant" if situ else "npu_dequant_swiglu_quant"
                    submitted = self.assert_shared_schedule(
                        shared, routed, path, ["wait_event", operation, "wait_event", "quant_down", "output_gate"]
                    )
                    self.namespace["torch_npu"].npu_quant_matmul.assert_called_once_with(
                        "shared_quantized",
                        "down_weight",
                        "down_scale",
                        pertoken_scale="shared_scale",
                        bias=None,
                        output_dtype=submitted.hidden_states.dtype,
                    )

    def test_shared_mxfp_activation_quant_waits_for_gmm2_and_down_waits_for_combine(self):
        shared = self.make_shared_executor()
        path = self.namespace["SharedExpertMLPPath"].A8_MXFP_FUSED
        for backend in ("MC2CommImpl", "AlltoAllCommImpl", "AllGatherCommImpl"):
            for situ in (False, True):
                with self.subTest(backend=backend, situ=situ):
                    shared.situ_activation = SimpleNamespace(beta=1.0, linear_beta=None) if situ else None
                    routed = self.run_routed(backend)
                    operation = "situ_mx_quant" if situ else "npu_swiglu_group_quant"
                    submitted = self.assert_shared_schedule(
                        shared, routed, path, ["wait_event", operation, "wait_event", "down"]
                    )
                    shared.part2.assert_called_once_with(submitted.hidden_states, ("shared_quantized", "shared_scale"))
                    if situ:
                        self.namespace["torch"].ops._C_ascend.situ_mx_quant.assert_called_once_with(
                            x=submitted.gate_up, beta=1.0, linear_beta=0.0, activate_left=True, dst_type=1
                        )

    def test_shared_linear_activation_and_preparation_wait_for_gmm2(self):
        shared = self.make_shared_executor()
        paths = self.namespace["SharedExpertMLPPath"]
        for backend in ("MC2CommImpl", "AlltoAllCommImpl", "AllGatherCommImpl"):
            for path, fused_activation in (
                (paths.LINEAR_WRAPPER, False),
                (paths.W8A8_MXFP_LINEAR, False),
                (paths.W8A8_MXFP_LINEAR, True),
            ):
                with self.subTest(backend=backend, path=path, fused_activation=fused_activation):
                    shared._can_fuse_w8a8_mxfp_activation.return_value = fused_activation
                    routed = self.run_routed(backend, activation=self.activation.SILU)
                    operations = (
                        ["npu_swiglu_group_quant"] if fused_activation else ["shared_activation", "prepare_down"]
                    )
                    submitted = self.assert_shared_schedule(
                        shared, routed, path, ["wait_event", *operations, "wait_event", "down"]
                    )
                    down_input = (
                        ("shared_quantized", "shared_scale", "bf16") if fused_activation else "prepared_down_input"
                    )
                    shared.part2.assert_called_once_with(submitted.hidden_states, down_input)


if __name__ == "__main__":
    unittest.main()
