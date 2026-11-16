# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the Slot-only MoonEP service and independent native paths."""
import ast
import importlib.util
import os
import sys
import threading
import unittest
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from test_moonep_real_tokens import Metadata, load_method

ROOT = Path(__file__).resolve().parents[3]
COMM = ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py"


def function_node(path, name, class_name=None):
    body = ast.parse(path.read_text()).body
    if class_name is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def load_function(node, namespace):
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(COMM), "exec"), namespace)
    return namespace[node.name]


class SlotOnlyForwardTests(unittest.TestCase):
    def make_forward(self, fail_mlp=False, missing_state=False, fused_weights=False):
        events = []
        weight = [] if fused_weights else object()
        state = object()
        tls = SimpleNamespace(state=None)
        metadata = SimpleNamespace(state=state, buffer=object(), plan=object())
        dispatch_output = SimpleNamespace(combine_metadata=metadata,
                                         group_list=torch.tensor([2, 1, 1, 0]), group_list_type=1)
        output = object()

        def dispatch(value):
            self.assertIs(tls.state, state)
            events.append("dispatch")
            return dispatch_output

        def mlp(**kwargs):
            self.assertIs(kwargs["state"], state)
            events.append("slot_mlp")
            if fail_mlp:
                raise RuntimeError("Slot MLP probe")
            return object(), object()

        def combine(value, meta):
            self.assertIs(meta, metadata)
            events.append("weighted_combine")
            return output

        namespace = {
            "_SLOT_TLS": tls, "FusedExpertsResult": SimpleNamespace,
            "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(record_event=object))),
            "_MOONEP_RUNTIME": SimpleNamespace(_state_by_w13={} if missing_state else {id(weight): state}),
        }
        node = function_node(COMM, "fused_experts", "MoonEPCommImpl")
        forward = load_function(node, namespace)
        obj = SimpleNamespace(token_dispatcher=SimpleNamespace(token_dispatch=dispatch, token_combine=combine),
                              use_fusion_ops=False, moe_config=object())
        inputs = SimpleNamespace(layer=SimpleNamespace(w13_weight=weight), lora_context=None)
        modules = {
            "vllm_ascend.ops.fused_moe.moonep_balance": SimpleNamespace(run_slot_forward=mlp, select_prefetch_layer=lambda layer: None),
            "vllm_ascend.ops.fused_moe.dataclass.token_dispatcher": SimpleNamespace(
                build_token_dispatch_input=lambda **kwargs: object()),
        }
        return forward, obj, inputs, modules, events, tls, output

    def test_legacy_zero_one_and_unset_all_use_slot_forward(self):
        for legacy in (None, "0", "1"):
            with self.subTest(legacy=legacy):
                forward, obj, inputs, modules, events, tls, output = self.make_forward()
                with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, modules):
                    if legacy is not None:
                        os.environ["MOONEP_SLOT_MODE"] = legacy
                    result = forward(obj, inputs, quant_method=object())
                self.assertIs(result.routed_out, output)
                self.assertEqual(events, ["dispatch", "slot_mlp", "weighted_combine"])
                self.assertEqual(result.expert_tokens.tolist(), [2, 1, 1, 0])
                self.assertIsNone(tls.state)

    def test_missing_layer_state_fails_without_owner_fallback(self):
        forward, obj, inputs, modules, events, _, _ = self.make_forward(missing_state=True)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "balance state missing"):
            forward(obj, inputs, quant_method=object())
        self.assertEqual(events, [])

    def test_fused_weight_list_remains_unsupported(self):
        forward, obj, inputs, modules, events, _, _ = self.make_forward(fused_weights=True)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(NotImplementedError, "single"):
            forward(obj, inputs, quant_method=object())
        self.assertEqual(events, [])

    def test_slot_tls_is_cleared_after_mlp_failure(self):
        forward, obj, inputs, modules, events, tls, _ = self.make_forward(fail_mlp=True)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "Slot MLP probe"):
            forward(obj, inputs, quant_method=object())
        self.assertEqual(events, ["dispatch", "slot_mlp"])
        self.assertIsNone(tls.state)

    def test_combine_rejects_missing_source_weights(self):
        owner_combine = Mock(side_effect=AssertionError("No owner combine fallback"))
        namespace = {"torch": torch, "time": SimpleNamespace(perf_counter=lambda: 0),
                     "MoonEPCombineMetadata": Metadata}
        combine = load_method("token_combine", namespace)
        meta = Metadata(num_tokens=1, source_topk_weights_sk=None,
                        buffer=SimpleNamespace(combine=owner_combine))
        with self.assertRaisesRegex(ValueError, "source-rank route weights"):
            combine(SimpleNamespace(), torch.ones(1, 4), meta)
        owner_combine.assert_not_called()

    def test_runtime_creates_slot_states_without_legacy_flag(self):
        method = function_node(COMM, "_initialize_locked", "_MoonEPRuntime")
        start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.ImportFrom)
                     and any(a.name == "_build_slot_states" for a in n.names))
        end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr == "_state_by_w13" for t in n.targets))
        layer = SimpleNamespace(w13_weight=object())
        entries = [dict(layer=layer, E=256, H=4096), dict(layer=layer, E=256, H=4096), dict(layer=None)]

        state = SimpleNamespace(_w13_src=layer.w13_weight, layer_index=0)
        builder = Mock(return_value=[state])

        for legacy in ("0", "1"):
            with self.subTest(legacy=legacy):
                runtime = SimpleNamespace()
                namespace = {
                    "torch": SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 0)),
                    "get_ep_group": lambda: SimpleNamespace(rank_in_group=0),
                    "_registry_lock": threading.Lock(), "_moonep_layer_registry": entries, "self": runtime,
                }
                module = SimpleNamespace(_build_slot_states=builder)
                with patch.dict(os.environ, {"MOONEP_SLOT_MODE": legacy}), patch.dict(
                    sys.modules, {"vllm_ascend.ops.fused_moe.moonep_balance": module}
                ):
                    exec(compile(ast.Module(body=method.body[start:end + 1], type_ignores=[]), str(COMM), "exec"),
                         namespace)
                builder.assert_called_with(entries, 0, 0)
                self.assertEqual(len(runtime._states), 1)
                self.assertEqual(runtime._states[0].layer_index, 0)
                self.assertIs(runtime._state_by_w13[id(layer.w13_weight)], runtime._states[0])


class NativeSelectionTests(unittest.TestCase):
    def test_slot_environment_registration_is_removed_and_other_defaults_remain(self):
        spec = importlib.util.spec_from_file_location("slot_only_envs", ROOT / "vllm_ascend/envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        self.assertNotIn("MOONEP_SLOT_MODE", envs.env_variables)
        for legacy in ("0", "1"):
            with patch.dict(os.environ, {"MOONEP_SLOT_MODE": legacy}, clear=True):
                self.assertFalse(hasattr(envs, "MOONEP_SLOT_MODE"))
                self.assertFalse(envs.MOONEP_KEEP_MC2)
                self.assertFalse(hasattr(envs, "VLLM_ASCEND_MOONEP_DISPATCH_DTYPE"))
                self.assertFalse(hasattr(envs, "VLLM_ASCEND_MOONEP_FUSED_PUBLISH"))
                self.assertFalse(hasattr(envs, "MOONEP_SLOT_QUANT"))

    def test_compatibility_mode_query_does_not_read_legacy_environment(self):
        node = function_node(ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py", "slot_mode_enabled")
        query = load_function(node, {})
        with patch.dict(os.environ, {"MOONEP_SLOT_MODE": "0"}):
            self.assertTrue(query())

    def test_native_off_does_not_register_shmem_service(self):
        class Comm(Enum):
            ALLTOALL = auto()
            ALLGATHER = auto()
            MC2 = auto()
            FUSED_MC2 = auto()
            SHMEM = auto()
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                methods = {}
                namespace = {"_MoECommMethods": methods, "MoECommType": Comm,
                             "_moe_config_key": lambda kind, config: (kind, ()),
                             "get_ascend_config": lambda: SimpleNamespace(enable_moonep=enabled)}
                for name in ("AlltoAllCommImpl", "AllGatherCommImpl", "MC2CommImpl", "FusedMC2CommImpl"):
                    namespace[name] = lambda config: object()
                node = function_node(ROOT / "vllm_ascend/ops/fused_moe/moe_comm_method.py", "setup_moe_comm_method")
                setup = load_function(node, namespace)
                shmem = Mock(return_value=object())
                with patch.dict(sys.modules, {"vllm_ascend.ops.fused_moe.moonep_comm":
                                               SimpleNamespace(MoonEPCommImpl=shmem)}):
                    setup(SimpleNamespace(ep_size=8))
                self.assertEqual((Comm.SHMEM, ()) in methods, enabled)
                self.assertEqual(shmem.call_count, int(enabled))
                self.assertIn((Comm.MC2, ()), methods)

    def test_keep_mc2_policy_remains_independent(self):
        class Comm(Enum):
            MC2 = auto()
            FUSED_MC2 = auto()
            ALLTOALL = auto()
            SHMEM = auto()
        path = ROOT / "vllm_ascend/ascend_forward_context.py"
        blocks = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Attribute) and n.test.attr == "enable_moonep"
                  and any(isinstance(x, ast.Name) and x.id == "keep_mc2" for x in ast.walk(n))]
        self.assertEqual(len(blocks), 1)
        cases = [(False, False, Comm.MC2, Comm.MC2), (True, True, Comm.MC2, Comm.MC2),
                 (True, True, Comm.FUSED_MC2, Comm.FUSED_MC2),
                 (True, False, Comm.MC2, Comm.SHMEM), (True, True, Comm.ALLTOALL, Comm.SHMEM)]
        for enabled, keep, initial, expected in cases:
            with self.subTest(enabled=enabled, keep=keep, initial=initial):
                namespace = {"get_ascend_config": lambda: SimpleNamespace(enable_moonep=enabled),
                             "get_moonep_tokens_capacity": lambda: 4096,
                             "envs": SimpleNamespace(MOONEP_KEEP_MC2=keep), "MoECommType": Comm,
                             "moe_comm_type": initial, "num_tokens": 16}
                exec(compile(ast.Module(body=blocks, type_ignores=[]), str(path), "exec"), namespace)
                self.assertIs(namespace["moe_comm_type"], expected)


if __name__ == "__main__":
    unittest.main()
