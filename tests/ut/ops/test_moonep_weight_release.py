# SPDX-License-Identifier: Apache-2.0
"""CPU checks for conditional weight handoff and Slot publication cleanup."""
import ast
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
BALANCE = ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py"
COMM = ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py"


def flat_bytes(tensor, nbytes):
    return tensor.detach().view(torch.uint8).reshape(-1)[:nbytes]


class WeightHandoffTests(unittest.TestCase):
    def make_state(self, quantized=False):
        tree = ast.parse(BALANCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_LayerWeightState")
        names = {"publish", "_release_sources", "destroy"}
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = {"torch": torch, "_flat_bytes": flat_bytes, "_RELEASE_CARRIER_NOTICE": [False]}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(BALANCE), "exec"), namespace)
        state_type = type("WeightStateHarness", (), {name: namespace[name] for name in names})
        state = state_type()
        layer = torch.nn.Module()
        dtype = torch.int8 if quantized else torch.bfloat16
        layer.w13_weight = torch.nn.Parameter(torch.arange(48).reshape(2, 4, 6).to(dtype), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.arange(24).reshape(2, 3, 4).to(dtype), requires_grad=False)
        refs = {"w13": layer.w13_weight, "w2": layer.w2_weight}
        if quantized:
            layer.w13_weight_scale = torch.nn.Parameter(torch.arange(8, dtype=torch.uint8), requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(torch.arange(4, dtype=torch.uint8), requires_grad=False)
            refs.update(w13_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale)
        state.homes = {}
        state._specs = []
        for name, ref in refs.items():
            nbytes = ref.numel() * ref.element_size()
            state.homes[name] = (torch.zeros(nbytes // 2, dtype=torch.bfloat16) if quantized
                                 else torch.zeros_like(ref))
            state._specs.append(SimpleNamespace(name=name, ref=ref, nbytes=nbytes))
        state._layer = layer
        state._w13_src, state._w2_src = layer.w13_weight, layer.w2_weight
        state._published = False
        state._vmm_tensors = []
        originals = {name: ref.detach().clone() for name, ref in refs.items()}
        return state, refs, originals

    def test_bf16_handoff_keeps_parameter_identity_and_weight_values(self):
        for old_switch in ("0", "1"):
            with self.subTest(old_switch=old_switch):
                state, refs, originals = self.make_state()
                lookup = {id(state._w13_src): state}
                with patch.dict(os.environ, {"MOONEP_RELEASE_WEIGHTS": old_switch}):
                    state.publish()
                self.assertTrue(state._published)
                self.assertTrue(state._released)
                self.assertIs(lookup[id(state._layer.w13_weight)], state)
                for name, attr in (("w13", "w13_weight"), ("w2", "w2_weight")):
                    parameter = getattr(state._layer, attr)
                    self.assertIs(parameter, refs[name])
                    self.assertEqual(parameter.data_ptr(), state.homes[name].data_ptr())
                    self.assertTrue(torch.equal(parameter, originals[name]))

    def test_quantized_weights_and_scales_keep_original_storage(self):
        for old_switch in ("0", "1"):
            with self.subTest(old_switch=old_switch):
                state, refs, originals = self.make_state(quantized=True)
                pointers = {name: ref.data_ptr() for name, ref in refs.items()}
                with patch.dict(os.environ, {"MOONEP_RELEASE_WEIGHTS": old_switch}):
                    state.publish()
                self.assertFalse(getattr(state, "_released", False))
                for name, ref in refs.items():
                    self.assertEqual(ref.data_ptr(), pointers[name])
                    self.assertTrue(torch.equal(ref, originals[name]))
                    nbytes = ref.numel() * ref.element_size()
                    self.assertTrue(torch.equal(flat_bytes(state.homes[name], nbytes),
                                                flat_bytes(originals[name], nbytes)))

    def test_publication_is_idempotent(self):
        state, _, _ = self.make_state()
        with patch.dict(os.environ, {"MOONEP_RELEASE_WEIGHTS": "0"}):
            state.publish()
            with patch.object(state, "_release_sources", side_effect=AssertionError("Already published")):
                state.publish()

    def test_shape_mismatch_refuses_storage_swap(self):
        state, refs, _ = self.make_state()
        pointer = refs["w13"].data_ptr()
        state.homes["w13"] = state.homes["w13"].reshape(-1)
        with self.assertRaisesRegex(RuntimeError, "shape/dtype changed"):
            state._release_sources()
        self.assertEqual(refs["w13"].data_ptr(), pointer)

    def test_dtype_mismatch_refuses_storage_swap(self):
        state, refs, _ = self.make_state()
        pointer = refs["w13"].data_ptr()
        state.homes["w13"] = state.homes["w13"].float()
        with self.assertRaisesRegex(RuntimeError, "shape/dtype changed"):
            state._release_sources()
        self.assertEqual(refs["w13"].data_ptr(), pointer)

    def test_reparameterized_layer_is_not_overwritten(self):
        state, _, _ = self.make_state()
        state._layer.w13_weight = torch.nn.Parameter(torch.ones_like(state._w13_src), requires_grad=False)
        state._layer.w2_weight = torch.nn.Parameter(torch.ones_like(state._w2_src), requires_grad=False)
        parameters = (state._layer.w13_weight, state._layer.w2_weight)
        state._release_sources()
        self.assertIs(state._layer.w13_weight, parameters[0])
        self.assertIs(state._layer.w2_weight, parameters[1])
        self.assertFalse(getattr(state, "_released", False))

    def test_destroy_refuses_to_unmap_handed_off_home_storage(self):
        state, _, _ = self.make_state()
        allocation = Mock()
        state._vmm_tensors = [SimpleNamespace(_vmm_allocation=allocation)]
        state.publish()
        with self.assertRaisesRegex(RuntimeError, "home chunks back"):
            state.destroy()
        allocation.destroy.assert_not_called()

    def test_quantized_skip_keeps_existing_destroy_path(self):
        state, _, _ = self.make_state(quantized=True)
        allocation = Mock()
        state._vmm_tensors = [SimpleNamespace(_vmm_allocation=allocation)]
        state.publish()
        state.destroy()
        allocation.destroy.assert_called_once()


class SlotPublicationCleanupTests(unittest.TestCase):
    def invoke(self, old_switch, fail_cleanup=False):
        tree = ast.parse(COMM.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MoonEPRuntime")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_initialize_locked")
        tail = method.body[-2:]
        self.assertIsInstance(tail[0], ast.For)
        self.assertIsInstance(tail[1], ast.Try)
        events = []

        def empty_cache():
            events.append("empty_cache")
            if fail_cleanup:
                raise RuntimeError("Cleanup probe")

        def memory_info(device):
            events.append("memory_info")
            return 60 * 2**30, 96 * 2**30

        npu = SimpleNamespace(current_device=lambda: 0, empty_cache=empty_cache, mem_get_info=memory_info)
        states = [SimpleNamespace(publish=lambda i=i: events.append(f"publish{i}")) for i in range(2)]
        namespace = {"torch": SimpleNamespace(npu=npu),
                     "get_ep_group": lambda: SimpleNamespace(rank_in_group=0),
                     "self": SimpleNamespace(_states=states)}
        output = io.StringIO()
        with patch.dict(os.environ, {"MOONEP_RELEASE_WEIGHTS": old_switch,
                                     "MOONEP_SLOT_MODE": old_switch}), redirect_stdout(output):
            exec(compile(ast.Module(body=tail, type_ignores=[]), str(COMM), "exec"), namespace)
        return events, output.getvalue()

    def test_slot_cleans_cache_after_all_publications_even_with_old_switch_zero(self):
        for old_switch in ("0", "1"):
            events, output = self.invoke(old_switch)
            self.assertEqual(events, ["publish0", "publish1", "empty_cache", "memory_info"])
            self.assertIn("weight publication complete", output)
            self.assertNotIn("home chunks now back", output)

    def test_off_does_not_publish_or_clean_slot_weights(self):
        tree = ast.parse(COMM.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "ensure_runtime_initialized")
        initialize = Mock(side_effect=AssertionError("Off must not initialize Slot weights"))
        namespace = {"_MOONEP_RUNTIME": SimpleNamespace(_ensure_initialized=initialize)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(COMM), "exec"), namespace)
        config = SimpleNamespace(get_ascend_config=lambda: SimpleNamespace(enable_moonep=False))
        with patch.dict(sys.modules, {"vllm_ascend.ascend_config": config}):
            namespace["ensure_runtime_initialized"]()
        initialize.assert_not_called()

    def test_cleanup_failure_keeps_existing_nonfatal_report(self):
        events, output = self.invoke("0", fail_cleanup=True)
        self.assertEqual(events, ["publish0", "publish1", "empty_cache"])
        self.assertIn("empty_cache/report failed", output)


if __name__ == "__main__":
    unittest.main()
