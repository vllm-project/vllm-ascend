"""CPU checks for the strict SHMEM bootstrap type rename."""
import ast
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parents[3]
COMM = ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py"
NEW = "MOONEP_SHMEM_BOOTSTRAP_TYPE"
OLD = "MOONEP_SHMEM_BOOTSTRAP"


class BootstrapTypeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("bootstrap_type_envs", ROOT / "vllm_ascend/envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        cls = next(n for n in ast.parse(COMM.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "_MoonEPRuntime")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_initialize_locked")
        self.block = next(n for n in method.body if isinstance(n, ast.If)
                          and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not)
                          and isinstance(n.test.operand, ast.Call)
                          and isinstance(n.test.operand.func, ast.Attribute)
                          and n.test.operand.func.attr == "is_initialized")

    def harness(self, address="127.0.0.1", dp=8, local_dp=8, udma=False, overlap=False, initialized=False):
        events, calls = [], []
        group = object()
        descriptors = [object()]
        descriptors_fn = Mock(return_value=descriptors)
        config_fn = Mock(return_value=SimpleNamespace(parallel_config=SimpleNamespace(
            data_parallel_master_ip=address, data_parallel_size=dp, data_parallel_size_local=local_dp)))
        def ipport(**kwargs):
            events.append("ipport")
            calls.append(kwargs)
        def uid(**kwargs):
            events.append("uid")
            calls.append(kwargs)
        ns = {
            "os": os, "envs": self.envs, "group": group, "use_udma": udma, "prefetch_overlap": overlap,
            "get_ep_group": lambda: SimpleNamespace(rank_in_group=0),
            "_warmup_aicpu_custom_scheduler": lambda: events.append("warmup"),
            "ShmemRuntime": SimpleNamespace(
                is_initialized=lambda: initialized, init_with_ip_port=ipport, init_with_buffer=uid),
        }
        modules = {
            "ascend_moonep.buffer_c": SimpleNamespace(symmetric_descriptors=descriptors_fn),
            "vllm.config": SimpleNamespace(get_current_vllm_config=config_fn),
        }
        def run(environment):
            output = io.StringIO()
            with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, modules), redirect_stdout(output):
                exec(compile(ast.Module(body=[self.block], type_ignores=[]), str(COMM), "exec"), ns)
            return output.getvalue()
        return SimpleNamespace(events=events, calls=calls, run=run, group=group, descriptors=descriptors,
                               descriptors_fn=descriptors_fn, config_fn=config_fn)

    def test_registration_default_and_removed_old_name(self):
        self.assertIn(NEW, self.envs.env_variables)
        self.assertNotIn(OLD, self.envs.env_variables)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.envs.MOONEP_SHMEM_BOOTSTRAP_TYPE, "ipport")
            with self.assertRaises(AttributeError):
                getattr(self.envs, OLD)

    def test_only_two_exact_values_are_accepted(self):
        for value in ("ipport", "uid"):
            with self.subTest(value=value), patch.dict(os.environ, {NEW: value}, clear=True):
                self.assertEqual(self.envs.MOONEP_SHMEM_BOOTSTRAP_TYPE, value)

    def test_invalid_case_spaces_empty_and_typo_are_rejected(self):
        for value in ("", "IPPORT", "UID", " ipport", "uid ", "ip_port", "tcp", "0", "1"):
            with self.subTest(value=value), patch.dict(os.environ, {NEW: value}, clear=True):
                with self.assertRaisesRegex(ValueError, NEW + " must be ipport or uid"):
                    getattr(self.envs, NEW)

    def test_old_values_are_ignored_and_use_default_ipport(self):
        for value in ("uid", "ipport", "", "invalid"):
            with self.subTest(value=value):
                h = self.harness()
                h.run({OLD: value})
                self.assertEqual(h.events, ["warmup", "ipport"])

    def test_new_value_wins_over_old(self):
        for value in ("ipport", "uid"):
            with self.subTest(value=value):
                h = self.harness()
                h.run({NEW: value, OLD: "invalid"})
                self.assertEqual(h.events, ["warmup", value])

    def test_invalid_new_value_does_not_fall_back_or_initialize_shmem(self):
        h = self.harness()
        with self.assertRaisesRegex(ValueError, NEW):
            h.run({NEW: "invalid", OLD: "uid"})
        # Preserve the approved original read position, after startup warmup.
        self.assertEqual(h.events, ["warmup"])
        self.assertEqual(h.calls, [])

    def test_ipport_keeps_default_address_port_and_descriptors(self):
        h = self.harness()
        output = h.run({})
        self.assertEqual(h.events, ["warmup", "ipport"])
        self.assertEqual(h.calls[0], dict(
            group=h.group, ip_port="tcp://127.0.0.1:28765", descriptors=h.descriptors,
            enable_udma=False, qp_count=0, prefetch_overlap=False))
        self.assertIn("bootstrap_type=ipport", output)

    def test_ipport_keeps_explicit_multinode_address_and_port(self):
        h = self.harness(address="141.61.45.118", dp=16, local_dp=8)
        h.run({NEW: "ipport", "MOONEP_SHMEM_BOOTSTRAP_PORT": "28766"})
        self.assertEqual(h.calls[0]["ip_port"], "tcp://141.61.45.118:28766")

    def test_multinode_loopback_and_missing_master_remain_errors(self):
        for address in ("127.0.0.1", "localhost", "::1", None):
            with self.subTest(address=address):
                h = self.harness(address=address, dp=16, local_dp=8)
                with self.assertRaises(RuntimeError):
                    h.run({NEW: "ipport"})
                self.assertEqual(h.events, ["warmup"])
                self.assertEqual(h.calls, [])

    def test_uid_skips_ipport_only_config_and_port_parsing(self):
        h = self.harness(address=None)
        output = h.run({NEW: "uid", "MOONEP_SHMEM_BOOTSTRAP_PORT": "invalid"})
        self.assertEqual(h.events, ["warmup", "uid"])
        self.assertEqual(h.calls[0], dict(group=h.group, enable_udma=False, prefetch_overlap=False))
        h.config_fn.assert_not_called()
        h.descriptors_fn.assert_not_called()
        self.assertIn("bootstrap_type=uid", output)

    def test_udma_and_prefetch_overlap_arguments_are_preserved_for_both_types(self):
        for value in ("ipport", "uid"):
            with self.subTest(value=value):
                h = self.harness(udma=True, overlap=True)
                h.run({NEW: value})
                self.assertTrue(h.calls[0]["enable_udma"])
                self.assertTrue(h.calls[0]["prefetch_overlap"])

    def test_already_initialized_runtime_does_not_parse_or_initialize_again(self):
        h = self.harness(initialized=True)
        getter = Mock(side_effect=AssertionError("Existing runtime must not read bootstrap again"))
        with patch.dict(self.envs.env_variables, {NEW: getter}):
            h.run({NEW: "invalid"})
        getter.assert_not_called()
        self.assertEqual(h.events, [])
        self.assertEqual(h.calls, [])

    def test_runtime_read_uses_central_accessor_and_has_no_old_key(self):
        tree = ast.parse(COMM.read_text())
        reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                 and isinstance(n.value, ast.Name) and n.value.id == "envs" and n.attr == NEW]
        self.assertEqual(len(reads), 1)
        for path in (COMM, ROOT / "vllm_ascend/envs.py"):
            strings = {n.value for n in ast.walk(ast.parse(path.read_text()))
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertNotIn(OLD, strings)


    def test_port_registration_default_and_old_name_removed(self):
        self.assertIn("MOONEP_SHMEM_BOOTSTRAP_PORT", self.envs.env_variables)
        self.assertNotIn("MOONEP_SHMEM_IP_PORT", self.envs.env_variables)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.envs.MOONEP_SHMEM_BOOTSTRAP_PORT, 28765)
            with self.assertRaises(AttributeError):
                getattr(self.envs, "MOONEP_SHMEM_IP_PORT")

    def test_port_preserves_integer_parsing_without_new_range_checks(self):
        for raw, expected in (("28765", 28765), (" +028766 ", 28766), ("0", 0), ("-1", -1), ("70000", 70000)):
            with self.subTest(raw=raw), patch.dict(os.environ, {"MOONEP_SHMEM_BOOTSTRAP_PORT": raw}, clear=True):
                self.assertEqual(self.envs.MOONEP_SHMEM_BOOTSTRAP_PORT, expected)

    def test_old_port_values_are_ignored(self):
        for raw in ("28766", "0", "-1", "invalid", ""):
            with self.subTest(raw=raw):
                h = self.harness()
                h.run({"MOONEP_SHMEM_IP_PORT": raw})
                self.assertEqual(h.calls[0]["ip_port"], "tcp://127.0.0.1:28765")

    def test_new_port_takes_precedence_over_old(self):
        h = self.harness()
        h.run({"MOONEP_SHMEM_BOOTSTRAP_PORT": "28766", "MOONEP_SHMEM_IP_PORT": "invalid"})
        self.assertEqual(h.calls[0]["ip_port"], "tcp://127.0.0.1:28766")

    def test_invalid_new_port_does_not_fall_back_or_initialize(self):
        for raw in ("", "invalid", "1.5"):
            with self.subTest(raw=raw):
                h = self.harness()
                with self.assertRaises(ValueError):
                    h.run({"MOONEP_SHMEM_BOOTSTRAP_PORT": raw, "MOONEP_SHMEM_IP_PORT": "28765"})
                self.assertEqual(h.calls, [])
                self.assertEqual(h.events, ["warmup"])

    def test_uid_does_not_read_new_port_even_with_invalid_value(self):
        h = self.harness()
        getter = Mock(side_effect=AssertionError("UID must not read the port"))
        with patch.dict(self.envs.env_variables, {"MOONEP_SHMEM_BOOTSTRAP_PORT": getter}):
            h.run({NEW: "uid", "MOONEP_SHMEM_BOOTSTRAP_PORT": "invalid"})
        getter.assert_not_called()
        self.assertEqual(h.events, ["warmup", "uid"])

    def test_ipport_reads_port_once_at_initialization(self):
        h = self.harness()
        getter = Mock(return_value=28767)
        with patch.dict(self.envs.env_variables, {"MOONEP_SHMEM_BOOTSTRAP_PORT": getter}):
            h.run({NEW: "ipport"})
        getter.assert_called_once_with()
        self.assertEqual(h.calls[0]["ip_port"], "tcp://127.0.0.1:28767")

    def test_port_uses_existing_dp_master_ip_without_changing_it(self):
        h = self.harness(address="141.61.45.118", dp=16, local_dp=8)
        h.run({"MOONEP_SHMEM_BOOTSTRAP_PORT": "28768"})
        self.assertEqual(h.calls[0]["ip_port"], "tcp://141.61.45.118:28768")

    def test_initialized_runtime_does_not_read_port_again(self):
        h = self.harness(initialized=True)
        getter = Mock(side_effect=AssertionError("Existing runtime must not reread port"))
        with patch.dict(self.envs.env_variables, {"MOONEP_SHMEM_BOOTSTRAP_PORT": getter}):
            h.run({"MOONEP_SHMEM_BOOTSTRAP_PORT": "invalid"})
        getter.assert_not_called()
        self.assertEqual(h.calls, [])

    def test_port_runtime_uses_new_central_key_only(self):
        tree = ast.parse(COMM.read_text())
        reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                 and isinstance(n.value, ast.Name) and n.value.id == "envs"
                 and n.attr == "MOONEP_SHMEM_BOOTSTRAP_PORT"]
        self.assertEqual(len(reads), 1)
        for path in (COMM, ROOT / "vllm_ascend/envs.py"):
            strings = {n.value for n in ast.walk(ast.parse(path.read_text()))
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertNotIn("MOONEP_SHMEM_IP_PORT", strings)


if __name__ == "__main__":
    unittest.main()
