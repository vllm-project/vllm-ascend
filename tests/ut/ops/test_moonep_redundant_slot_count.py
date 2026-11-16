"""CPU compatibility checks for the per-rank redundant Slot count rename."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
OLD = "MOONEP_B"
NEW = "MOONEP_REDUNDANT_EXPERT_SLOT_COUNT"


class RedundantSlotCountTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("slot_count_envs", ROOT / "vllm_ascend/envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        comm = ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py"
        tree = ast.parse(comm.read_text())
        function = next(n for n in tree.body
                        if isinstance(n, ast.FunctionDef) and n.name == "_moonep_slot_count")
        namespace = {"envs": self.envs}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(comm), "exec"), namespace)
        self.buffer_count = lambda epn: namespace["_moonep_slot_count"](256, epn)

        balance = ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py"
        tree = ast.parse(balance.read_text())
        layout = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "_layer_weight_layout")
        end = next(i for i, n in enumerate(layout.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "F" for t in n.targets))
        layout.body = layout.body[:end] + [ast.Return(value=ast.Name(id="B", ctx=ast.Load()))]
        layer_namespace = {"envs": self.envs, "_LayerWeightLayout": object}
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[layout], type_ignores=[])), str(balance), "exec"), layer_namespace)
        self.layer_count = lambda epn: layer_namespace["_layer_weight_layout"](
            SimpleNamespace(w13_weight=SimpleNamespace(shape=(epn, 4, 8))), 256, 4)
        self.readers = {"buffer": self.buffer_count, "layer": self.layer_count}

    def values(self, environment, epn=8):
        with patch.dict(os.environ, environment, clear=True):
            values = [reader(epn) for reader in self.readers.values()]
        self.assertEqual(values[0], values[1])
        return values[0]

    def test_central_registration_and_raw_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn(NEW, self.envs.env_variables)
            self.assertNotIn(OLD, self.envs.env_variables)
            self.assertIsNone(getattr(self.envs, NEW))
            self.assertFalse(hasattr(self.envs, OLD))

    def test_default_is_resolved_per_layer(self):
        for epn in (1, 8, 32):
            with self.subTest(epn=epn):
                self.assertEqual(self.values({}, epn), epn)

    def test_old_name_alone_is_ignored_even_when_invalid(self):
        for value in ("4", "0", "invalid"):
            with self.subTest(value=value):
                self.assertEqual(self.values({OLD: value}), 8)
                with patch.dict(os.environ, {OLD: value}, clear=True):
                    self.assertFalse(hasattr(self.envs, OLD))

    def test_new_name_preserves_python_integer_parsing(self):
        # Values below epn only test the existing Python parser; the native
        # planner still independently requires B == E/EP.
        for value, expected in (("8", 8), ("1", 1), ("04", 4), (" +08 ", 8)):
            with self.subTest(value=value):
                self.assertEqual(self.values({NEW: value}), expected)

    def test_new_name_is_the_only_value_when_both_are_present(self):
        self.assertEqual(self.values({NEW: "4", OLD: "8"}), 4)
        self.assertEqual(self.values({NEW: "8", OLD: "invalid"}), 8)

    def test_invalid_new_value_cannot_fall_back_to_old_name(self):
        for name, reader in self.readers.items():
            with self.subTest(reader=name), patch.dict(os.environ, {NEW: "", OLD: "8"}, clear=True):
                with self.assertRaises(ValueError):
                    reader(8)

    def test_range_errors_identify_the_new_name(self):
        for value in ("0", "-1", "9"):
            for name, reader in self.readers.items():
                with self.subTest(value=value, reader=name), patch.dict(
                    os.environ, {NEW: value}, clear=True
                ):
                    with self.assertRaisesRegex(ValueError, NEW + " must be"):
                        reader(8)

    def test_malformed_integers_remain_errors_at_both_read_sites(self):
        for value in ("", "invalid", "1.5"):
            for name, reader in self.readers.items():
                with self.subTest(value=value, reader=name), patch.dict(
                    os.environ, {NEW: value}, clear=True
                ):
                    with self.assertRaises(ValueError):
                        reader(8)

    def test_both_runtime_sites_use_the_central_accessor(self):
        getter = Mock(return_value="4")
        with patch.dict(self.envs.env_variables, {NEW: getter}), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.buffer_count(8), 4)
            self.assertEqual(self.layer_count(8), 4)
        self.assertEqual(getter.call_count, 2)

    def test_environment_is_read_at_use_time_without_a_new_cache(self):
        with patch.dict(os.environ, {NEW: "4"}, clear=True):
            self.assertEqual(self.buffer_count(8), 4)
            os.environ[NEW] = "8"
            self.assertEqual(self.layer_count(8), 8)
            self.assertEqual(self.buffer_count(8), 8)

    def test_old_name_has_no_runtime_registration_read_or_alias(self):
        for relative in ("envs.py", "ops/fused_moe/moonep_comm.py", "ops/fused_moe/moonep_balance.py"):
            tree = ast.parse((ROOT / "vllm_ascend" / relative).read_text())
            self.assertFalse(any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                                 and OLD == n.value for n in ast.walk(tree)), relative)
            self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == OLD
                                 for n in ast.walk(tree)), relative)


if __name__ == "__main__":
    unittest.main()
