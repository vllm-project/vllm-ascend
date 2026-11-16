# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fixed startup warmup and SHMEM initialization ordering."""
import ast
import importlib.util
import io
import os
import sys
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/moonep_comm.py"
OP_NAME = "npu_vllm_quant_lightning_indexer_metadata"


def load_function(name, namespace, class_name=None):
    tree = ast.parse(SOURCE.read_text())
    body = tree.body
    if class_name is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class WarmupTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.calls = []
        self.query_lengths, self.key_lengths = object(), object()
        self.lens = SimpleNamespace(clone=Mock(side_effect=[self.query_lengths, self.key_lengths]))
        self.ones = Mock(return_value=self.lens)
        self.fail_operator = False
        self.fail_sync = False

        def operator(**kwargs):
            self.events.append("operator")
            self.calls.append(kwargs)
            if self.fail_operator:
                raise RuntimeError("Operator probe")

        def synchronize():
            self.events.append("synchronize")
            if self.fail_sync:
                raise RuntimeError("Synchronize probe")

        self.operators = SimpleNamespace(**{OP_NAME: operator})
        self.namespace = {
            "os": os, "_AICPU_WARMUP_OP": OP_NAME,
            "_EXTRA_CTX": SimpleNamespace(capturing=False),
            "get_ep_group": lambda: SimpleNamespace(rank_in_group=0, world_size=8),
            "torch": SimpleNamespace(
                ops=SimpleNamespace(_C_ascend=self.operators), int32="int32",
                ones=self.ones, npu=SimpleNamespace(current_device=lambda: 0, synchronize=synchronize)),
        }
        self.warmup = load_function("_warmup_aicpu_custom_scheduler", self.namespace)

    def invoke(self, old_switch="0"):
        output = io.StringIO()
        with patch.dict(os.environ, {"MOONEP_AICPU_WARMUP": old_switch}), redirect_stdout(output):
            result = self.warmup()
        return result, output.getvalue()

    def test_old_switch_zero_and_one_both_attempt_minimal_warmup(self):
        for old_switch in ("0", "1"):
            with self.subTest(old_switch=old_switch):
                self.setUp()
                result, output = self.invoke(old_switch)
                self.assertTrue(result)
                self.assertEqual(self.events, ["operator", "synchronize"])
                self.ones.assert_called_once_with(1, dtype="int32", device="npu:0")
                self.assertEqual(self.lens.clone.call_count, 2)
                call = self.calls[0]
                self.assertIs(call["actual_seq_lengths_query"], self.query_lengths)
                self.assertIs(call["actual_seq_lengths_key"], self.key_lengths)
                self.assertEqual((call["batch_size"], call["max_seqlen_q"], call["max_seqlen_k"]), (1, 1, 1))
                self.assertEqual(call["device"], "npu:0")
                self.assertIn("aicpu-warmup done", output)

    def test_capture_skips_operator_and_synchronize(self):
        self.namespace["_EXTRA_CTX"].capturing = True
        result, output = self.invoke()
        self.assertFalse(result)
        self.assertEqual(self.events, [])
        self.ones.assert_not_called()
        self.assertIn("skipped (capturing)", output)

    def test_unavailable_capture_context_keeps_existing_warmup_attempt(self):
        class MissingContext:
            @property
            def capturing(self):
                raise RuntimeError("No forward context")
        self.namespace["_EXTRA_CTX"] = MissingContext()
        result, _ = self.invoke()
        self.assertTrue(result)
        self.assertEqual(self.events, ["operator", "synchronize"])

    def test_unregistered_operator_is_skipped_without_allocating(self):
        delattr(self.operators, OP_NAME)
        result, output = self.invoke()
        self.assertFalse(result)
        self.ones.assert_not_called()
        self.assertEqual(self.events, [])
        self.assertIn("not registered", output)

    def test_operator_failure_keeps_nonfatal_report(self):
        self.fail_operator = True
        result, output = self.invoke()
        self.assertFalse(result)
        self.assertEqual(self.events, ["operator"])
        self.assertIn("aicpu-warmup FAILED", output)

    def test_synchronize_failure_keeps_nonfatal_report(self):
        self.fail_sync = True
        result, output = self.invoke()
        self.assertFalse(result)
        self.assertEqual(self.events, ["operator", "synchronize"])
        self.assertIn("aicpu-warmup FAILED", output)


class StartupOrderingTests(unittest.TestCase):
    def make_initialization(self, bootstrap):
        events = []
        spec = importlib.util.spec_from_file_location(
            "bootstrap_warmup_envs", SOURCE.parents[2] / "envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        namespace = {"envs": envs}
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MoonEPRuntime")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_initialize_locked")
        block = next(n for n in method.body if isinstance(n, ast.If) and
                     isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not) and
                     isinstance(n.test.operand, ast.Call) and
                     isinstance(n.test.operand.func, ast.Attribute) and
                     n.test.operand.func.attr == "is_initialized")
        namespace.update({
            "os": os, "group": object(), "use_udma": False, "prefetch_overlap": False,
            "get_ep_group": lambda: SimpleNamespace(rank_in_group=0),
            "_warmup_aicpu_custom_scheduler": lambda: events.append("warmup"),
            "ShmemRuntime": SimpleNamespace(
                is_initialized=lambda: False,
                init_with_ip_port=lambda **kwargs: events.append("shmem_ipport"),
                init_with_buffer=lambda **kwargs: events.append("shmem_uid")),
        })
        modules = {
            "ascend_moonep.buffer_c": SimpleNamespace(symmetric_descriptors=lambda: []),
            "vllm.config": SimpleNamespace(get_current_vllm_config=lambda: SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_master_ip="127.0.0.1",
                                                data_parallel_size_local=8, data_parallel_size=8))),
        }

        def initialize():
            with patch.dict(os.environ, {"MOONEP_SHMEM_BOOTSTRAP_TYPE": bootstrap,
                                         "MOONEP_SHMEM_BOOTSTRAP_PORT": "28765"}), patch.dict(
                sys.modules, modules
            ), redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), "exec"), namespace)
        return events, initialize

    def test_warmup_precedes_both_shmem_bootstrap_paths(self):
        for bootstrap, event in (("ipport", "shmem_ipport"), ("uid", "shmem_uid")):
            with self.subTest(bootstrap=bootstrap):
                events, initialize = self.make_initialization(bootstrap)
                initialize()
                self.assertEqual(events, ["warmup", event])

    def test_runtime_initialization_does_not_repeat_startup_warmup(self):
        events, initialize = self.make_initialization("ipport")
        runtime = SimpleNamespace(_lock=threading.Lock(), _ready=False, _fail=None,
                                  _initialize_locked=initialize)
        ensure = load_function("_ensure_initialized", {}, "_MoonEPRuntime")
        ensure(runtime)
        ensure(runtime)
        self.assertEqual(events, ["warmup", "shmem_ipport"])

    def test_off_and_ep_one_keep_early_return(self):
        for enabled, world in ((False, 8), (True, 1)):
            with self.subTest(enabled=enabled, world=world):
                initialize = Mock()
                namespace = {"get_ep_group": lambda: SimpleNamespace(world_size=world),
                             "_MOONEP_RUNTIME": SimpleNamespace(_ensure_initialized=initialize)}
                ensure = load_function("ensure_runtime_initialized", namespace)
                module = SimpleNamespace(get_ascend_config=lambda: SimpleNamespace(enable_moonep=enabled))
                with patch.dict(sys.modules, {"vllm_ascend.ascend_config": module}):
                    ensure()
                initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
