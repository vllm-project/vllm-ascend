# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for model-selected MoonEP dispatch and paired MXFP8 buffers."""

import ast
import importlib.util
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_moonep_no_host_sync as sync_cases
import torch
from test_moonep_real_tokens import SOURCE, dispatch_test_modules, dispatch_test_quant

ROOT = Path(__file__).resolve().parents[3]
OLD = "VLLM_ASCEND_MOONEP_DISPATCH_DTYPE"


class AutoDispatchTests(unittest.TestCase):
    def setUp(self):
        self.modules = dispatch_test_modules()
        self.quant_type = self.modules["vllm_ascend.quantization.quant_type"].QuantType
        tree = ast.parse(SOURCE.read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_moonep_dispatch_transport")
        self.ns = {"torch": torch, "torch_npu": self.modules["torch_npu"]}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), "exec"), self.ns)

    def choose(self, quant=None, tensor=None):
        if quant is None:
            quant = dispatch_test_quant(self.modules, "mxfp8")
        if tensor is None:
            tensor = torch.ones(2, 1024, dtype=torch.bfloat16)
        with patch.dict(sys.modules, self.modules):
            return self.ns["_moonep_dispatch_transport"](quant, tensor)

    def test_w4a8_mxfp_is_selected_from_quant_metadata(self):
        self.assertEqual(self.choose(), "mxfp8")

    def test_both_e8m0_metadata_representations_are_accepted(self):
        for dtype in (torch.float8_e8m0fnu, self.modules["torch_npu"].float8_e8m0fnu):
            with self.subTest(dtype=dtype):
                quant = dispatch_test_quant(self.modules, "mxfp8")
                quant.mxfp.per_token_scale_dtype = dtype
                self.assertEqual(self.choose(quant), "mxfp8")

    def test_actual_quant_builder_preserves_backend_scale_identifier(self):
        path = ROOT / "vllm_ascend/ops/fused_moe/dataclass/moe_quant.py"
        selected = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        ns = {
            "__name__": __name__,
            "dataclass": dataclass,
            "torch": torch,
            "torch_npu": self.modules["torch_npu"],
            "QuantType": self.quant_type,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])), str(path), "exec"
            ),
            ns,
        )
        backend_dtype = self.modules["torch_npu"].float8_e8m0fnu
        quant = ns["build_quant_params"](
            quant_type=self.quant_type.W4A8MXFP,
            comm_quant_mode=None,
            is_per_channel_weight=False,
            mxfp_act_quant_type=torch.float8_e4m3fn,
            mxfp_weight_quant_type=None,
            mxfp_scale_dtype=backend_dtype,
            mxfp_per_token_scale_dtype=backend_dtype,
            mxfp_use_bf16=True,
        )
        self.assertEqual(quant.mxfp.per_token_scale_dtype, backend_dtype)
        self.assertEqual(self.choose(quant), "mxfp8")

    def test_byte_carrier_dtype_does_not_declare_e8m0_metadata(self):
        quant = dispatch_test_quant(self.modules, "mxfp8")
        quant.mxfp.per_token_scale_dtype = torch.uint8
        with self.assertRaisesRegex(NotImplementedError, "E8M0"):
            self.choose(quant)

    def test_other_quant_types_retain_bf16_without_wire_dependency(self):
        for kind in self.quant_type:
            if kind == self.quant_type.W4A8MXFP:
                continue
            with self.subTest(kind=kind):
                quant = SimpleNamespace(quant_type=kind, mxfp=None)
                self.assertEqual(self.choose(quant), "bf16")

    def test_standard_implicit_activation_and_scale_metadata_are_accepted(self):
        quant = dispatch_test_quant(self.modules, "mxfp8")
        quant.mxfp.act_quant_type = None
        quant.mxfp.per_token_scale_dtype = None
        self.assertEqual(self.choose(quant), "mxfp8")

    def test_missing_mxfp_metadata_and_incompatible_formats_are_errors(self):
        quant = dispatch_test_quant(self.modules, "mxfp8")
        quant.mxfp = None
        with self.assertRaisesRegex(ValueError, "quantization parameters"):
            self.choose(quant)
        quant = dispatch_test_quant(self.modules, "mxfp8")
        quant.mxfp.act_quant_type = torch.int8
        with self.assertRaisesRegex(NotImplementedError, "E4M3"):
            self.choose(quant)
        quant = dispatch_test_quant(self.modules, "mxfp8")
        quant.mxfp.per_token_scale_dtype = torch.float32
        with self.assertRaisesRegex(NotImplementedError, "E8M0"):
            self.choose(quant)

    def test_missing_e8m0_runtime_support_is_not_a_bf16_fallback(self):
        with patch.object(torch, "float8_e8m0fnu", None), self.assertRaisesRegex(NotImplementedError, "E8M0"):
            self.choose()

    def test_original_bf16_input_and_contiguity_are_required(self):
        for dtype in (torch.float16, torch.float32, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "original BF16"):
                self.choose(tensor=torch.ones(2, 1024).to(dtype))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.choose(tensor=torch.ones(1024, 2, dtype=torch.bfloat16).t())

    def test_wire_geometry_boundaries_and_alignment_are_preserved(self):
        for hidden in (1024, 2048, 4096, 16384):
            with self.subTest(hidden=hidden):
                self.assertEqual(self.choose(tensor=torch.empty(1, hidden, dtype=torch.bfloat16)), "mxfp8")
        for hidden in (16, 1023, 1025, 1536, 17408):
            with self.subTest(hidden=hidden), self.assertRaisesRegex(ValueError, "wire requires H"):
                self.choose(tensor=torch.empty(1, hidden, dtype=torch.bfloat16))

    def test_old_values_cannot_override_or_break_model_selection(self):
        for raw in ("bf16", "mxfp8", "", "invalid"):
            with self.subTest(raw=raw), patch.dict(os.environ, {OLD: raw}, clear=True):
                self.assertEqual(self.choose(), "mxfp8")
                self.assertEqual(self.choose(dispatch_test_quant(self.modules, "bf16")), "bf16")

    def test_old_environment_registration_and_fixed_attribute_are_removed(self):
        spec = importlib.util.spec_from_file_location("auto_dispatch_envs", ROOT / "vllm_ascend/envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        self.assertNotIn(OLD, envs.env_variables)
        self.assertFalse(hasattr(envs, OLD))
        tree = ast.parse(SOURCE.read_text())
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "activation_transport" for n in ast.walk(tree)))
        self.assertNotIn(
            OLD, {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        )

    def test_failed_mxfp8_dependencies_or_execution_never_retry_bf16(self):
        for message in ("wire library missing", "wire ABI mismatch", "bounded interface missing", "dispatch failed"):
            with self.subTest(message=message):
                case = sync_cases.NoHostCountSyncTests(methodName="runTest").make_dispatch("mxfp8")
                fn, dispatcher, inputs, _, _, _, _, modules = case
                buffer, _ = fn.__globals__["_select_buffer_for_step"]()
                buffer.dispatch = Mock(side_effect=AssertionError("No BF16 fallback"))
                buffer.dispatch_mxfp8 = Mock(side_effect=RuntimeError(message))
                with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, message):
                    fn(dispatcher, inputs)
                buffer.dispatch.assert_not_called()

    def test_real_wire_loader_rejects_missing_library_abi_and_bounded_interface(self):
        wire = self.modules["ascend_moonep.mxfp8_transport"]
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            wire.Mxfp8Wire("/__moonep_cpu_test_missing__/wire.so")
        fake = SimpleNamespace(moonep_mxfp8_wire_abi=Mock(return_value=999))
        with patch.object(wire.ctypes, "CDLL", return_value=fake), self.assertRaisesRegex(RuntimeError, "ABI mismatch"):
            wire.Mxfp8Wire(__file__)
        fake.moonep_mxfp8_wire_abi.return_value = wire.WIRE_ABI
        fake.moonep_mxfp8_wire = Mock()
        with patch.object(wire.ctypes, "CDLL", return_value=fake), self.assertRaisesRegex(RuntimeError, "bounded"):
            wire.Mxfp8Wire(__file__)

    def test_auto_choice_is_logged_once_per_quant_and_shape(self):
        case = sync_cases.NoHostCountSyncTests(methodName="runTest").make_dispatch("bf16")
        fn, dispatcher, inputs, _, _, _, _, modules = case
        stream = io.StringIO()
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"MOONEP_SLOT_STATS": "0"}),
            redirect_stdout(stream),
        ):
            fn(dispatcher, inputs)
            fn(dispatcher, inputs)
        self.assertEqual(stream.getvalue().count("[moonep-dispatch-auto]"), 1)
        self.assertIn("transport=bf16", stream.getvalue())

    def mixed_buffer(self, udma=False):
        """Use the real SDK wire adapter with CPU codec/communication doubles."""
        case = sync_cases.NoHostCountSyncTests(methodName="runTest").make_dispatch("bf16")
        fn, dispatcher, inputs, _, _, _, _, modules = case
        wire_module = modules["ascend_moonep.mxfp8_transport"]
        wire_module.__package__ = "ascend_moonep"
        hidden, capacity, rows = 1024, 4, 8
        events = []
        weights = torch.arange(rows, dtype=torch.float32)
        sym = torch.empty(rows, hidden, dtype=torch.bfloat16)
        cu = torch.tensor([2, 3, 3, 3, 4, 4], dtype=torch.int32)
        plan = SimpleNamespace(experts_to_copy=torch.tensor([[2, -1]]))
        ctx = {
            "S": capacity,
            "H": hidden,
            "NvS": rows,
            "K": 2,
            "E": 4,
            "B": 2,
            "rank": 0,
            "_dispatch_use_udma": udma,
            "hidden_buf": sym,
            "hidden_buf_local": sym,
            "weights_buf_local": weights,
            "after_planning_hook": lambda p: events.append("prefetch"),
        }
        ctx["kernels"] = {"planning": SimpleNamespace(compute=lambda *a, **kw: events.append("planning"))}

        class CpuWire:
            library_path = Path("/host-only-wire")

            def pack(self, q, scale, packet):
                events.append("pack")
                data = torch.cat((q.view(torch.uint8), scale.reshape(q.shape[0], -1)), dim=1)
                packet.view(torch.uint8).copy_(data)

            def unpack_bounded(self, received, x, scale, counts):
                events.append("unpack_bounded")
                # This CPU test double reads a CPU count. The real device codec
                # consumes the original device pointer without a host read.
                count = int(counts[0])
                data = received.view(torch.uint8)[:count]
                x[:count].view(torch.uint8).copy_(data[:, :hidden])
                scale[:count].reshape(count, -1).copy_(data[:, hidden:])

        def quantize(x, **kwargs):
            events.append("quant")
            return x.to(torch.float8_e4m3fn), torch.full((x.shape[0], hidden // 64, 2), 7, dtype=torch.uint8)

        def scatter(wire_ctx, packet, route, p):
            events.append("dispatch_mxfp8")
            indices = torch.tensor([0, 1, 0, 1])
            wire_ctx["hidden_buf_local"][:4].copy_(packet[indices])

        class Buffer:
            transport = None

            def _require_ctx(self):
                return ctx

            def _ensure_shmem_views(self):
                pass

            def _reused_plan_outputs(self, context):
                return plan, cu

            def dispatch(self, hidden_states, route, topk, tpe, **kwargs):
                events.extend(("planning", "prefetch", "dispatch_bf16"))
                sym[:4].copy_(hidden_states[torch.tensor([0, 1, 0, 1])])
                return sym[:], weights[:], cu, plan

            def dispatch_mxfp8(self, hidden_states, route, topk, tpe, **kwargs):
                if self.transport is None:
                    self.transport = wire_module.Mxfp8Transport(self, wire_storage=kwargs["wire_storage"])
                return self.transport.dispatch(hidden_states, route, topk, tpe, num_tokens=kwargs["num_tokens"])

        buffer = Buffer()
        fn.__globals__["_select_buffer_for_step"] = lambda *args: (buffer, capacity)
        fn.__globals__["_MOONEP_RUNTIME"] = SimpleNamespace(_buffers={(4, 2, hidden): {capacity: buffer}})
        modules.update(
            {
                "torch_npu": SimpleNamespace(
                    npu_dynamic_mx_quant=quantize, float8_e8m0fnu=modules["torch_npu"].float8_e8m0fnu
                ),
                "ascend_moonep.dispatch": SimpleNamespace(launch_dispatch=scatter),
                "ascend_moonep.launchers": SimpleNamespace(_env_use_udma=lambda stage: udma),
            }
        )

        def invoke(transport):
            inputs.quant = dispatch_test_quant(modules, transport)
            with (
                patch.dict(sys.modules, modules),
                patch.object(wire_module, "Mxfp8Wire", return_value=CpuWire()),
                patch.dict(os.environ, {"MOONEP_SLOT_STATS": "0"}),
                redirect_stdout(io.StringIO()),
            ):
                return fn(dispatcher, inputs)

        return SimpleNamespace(invoke=invoke, inputs=inputs, buffer=buffer, events=events, ctx=ctx)

    def test_same_buffer_alternates_formats_without_stale_scale_or_activation(self):
        h = self.mixed_buffer()
        h.inputs.hidden_states.fill_(2)
        first = h.invoke("mxfp8")
        self.assertTrue(torch.all(first.hidden_states[:4].to(torch.float32) == 2))
        self.assertTrue(torch.all(first.dynamic_scale[:4] == 7))
        scale_pointer = first.dynamic_scale.data_ptr()
        activation_pointer = first.hidden_states.data_ptr()
        h.inputs.hidden_states.fill_(3)
        middle = h.invoke("bf16")
        self.assertEqual(middle.hidden_states.dtype, torch.bfloat16)
        self.assertIsNone(middle.dynamic_scale)
        self.assertTrue(torch.all(middle.hidden_states[:4] == 3))
        h.inputs.hidden_states.fill_(4)
        last = h.invoke("mxfp8")
        self.assertEqual(last.hidden_states.data_ptr(), activation_pointer)
        self.assertEqual(last.dynamic_scale.data_ptr(), scale_pointer)
        self.assertTrue(torch.all(last.hidden_states[:4].to(torch.float32) == 4))
        self.assertTrue(torch.all(last.dynamic_scale[:4] == 7))
        self.assertEqual(h.events.count("quant"), 2)

    def test_mxfp8_preserves_planning_prefetch_quant_dispatch_bounded_unpack_order(self):
        for udma in (False, True):
            with self.subTest(udma=udma):
                h = self.mixed_buffer(udma=udma)
                result = h.invoke("mxfp8")
                self.assertEqual(
                    h.events, ["planning", "prefetch", "quant", "pack", "dispatch_mxfp8", "unpack_bounded"]
                )
                self.assertEqual(result.group_list.tolist(), [2, 1, 1, 0])
                self.assertEqual(result.dynamic_scale.data_ptr(), h.buffer.transport.scales.data_ptr())


if __name__ == "__main__":
    unittest.main()
