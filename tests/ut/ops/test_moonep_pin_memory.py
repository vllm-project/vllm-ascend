"""CPU compatibility tests for the platform's pinned-memory capability."""
import ast
from functools import cache
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]


def load_platform():
    source = ROOT / "vllm_ascend/platform.py"
    tree = ast.parse(source.read_text())
    platform = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUPlatform")
    method = next(n for n in platform.body
                  if isinstance(n, ast.FunctionDef) and n.name == "is_pin_memory_available")
    minimal_class = ast.ClassDef(
        name="NPUPlatform", bases=[], keywords=[], body=[method], decorator_list=[])
    namespace = {"os": os}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[minimal_class], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["NPUPlatform"]


def load_vllm_query():
    # Exercise the installed vLLM bridge and its real cache decorator without
    # importing a worker or initializing any device.
    spec = importlib.util.find_spec("vllm")
    source = Path(next(iter(spec.submodule_search_locations))) / "utils/platform_utils.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "is_pin_memory_available")
    namespace = {"cache": cache}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["is_pin_memory_available"]


class PinMemoryCapabilityTests(unittest.TestCase):
    def check_value(self, value):
        with patch.dict(os.environ):
            os.environ.pop("MOONEP_NO_PIN", None)
            if value is not None:
                os.environ["MOONEP_NO_PIN"] = value
            platform = load_platform()
            self.assertIs(platform.is_pin_memory_available(), True)

    def test_unset_keeps_pinned_memory_available(self):
        self.check_value(None)

    def test_old_zero_keeps_pinned_memory_available(self):
        self.check_value("0")

    def test_old_one_no_longer_disables_pinned_memory(self):
        self.check_value("1")

    def test_invalid_old_value_is_ignored(self):
        self.check_value("invalid")

    def test_class_and_instance_report_same_capability(self):
        platform = load_platform()
        self.assertIs(platform.is_pin_memory_available(), True)
        self.assertIs(platform().is_pin_memory_available(), True)

    def test_query_does_not_access_environment(self):
        platform = load_platform()
        with patch.object(os.environ, "get", side_effect=AssertionError("Unexpected environment read")):
            self.assertIs(platform.is_pin_memory_available(), True)

    def test_real_vllm_bridge_keeps_pinned_capability_with_old_switch(self):
        query = load_vllm_query()
        platform = load_platform()
        with patch.dict(sys.modules, {"vllm.platforms": types.SimpleNamespace(current_platform=platform)}):
            with patch.dict(os.environ, {"MOONEP_NO_PIN": "1"}):
                self.assertIs(query(), True)
            with patch.dict(os.environ, {"MOONEP_NO_PIN": "0"}):
                self.assertIs(query(), True)
            self.assertEqual(query.cache_info().misses, 1)
            self.assertEqual(query.cache_info().hits, 1)


if __name__ == "__main__":
    unittest.main()
