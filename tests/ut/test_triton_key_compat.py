# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock, patch

INIT_PATH = Path(__file__).resolve().parents[2] / "vllm_ascend" / "__init__.py"


class TestTritonKeyCompat(unittest.TestCase):
    def setUp(self):
        # Run the real package initialization without requiring Triton/NPU or
        # changing modules used by the rest of the CPU test suite.
        names = (
            "triton",
            "triton.compiler",
            "triton.compiler.compiler",
            "triton.runtime",
            "triton.runtime.cache",
            "triton.language",
            "triton.language.core",
            "triton.experimental",
            "triton.experimental.gluon",
            "triton.experimental.gluon.language",
            "vllm_ascend",
            "vllm_ascend.logger",
        )
        self.modules = {}
        for name in names:
            module = ModuleType(name)
            module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
            module.__path__ = []
            self.modules[name] = module
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(self.modules[parent], child, module)
        self.compiler = self.modules["triton.compiler.compiler"]
        self.runtime_key = Mock(return_value="compiler-source-fingerprint")
        vars(self.modules["triton.runtime.cache"])["triton_key"] = self.runtime_key
        modules_patch = patch.dict(sys.modules, self.modules)
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def test_new_layout_restores_legacy_import(self):
        namespace: dict[str, Any] = {}
        # Reproduce torch_npu's import before applying the compatibility alias.
        with self.assertRaises(ImportError):
            exec("from triton.compiler.compiler import triton_key", namespace)

        runpy.run_path(str(INIT_PATH))
        exec("from triton.compiler.compiler import triton_key", namespace)
        self.assertIs(namespace["triton_key"], self.runtime_key)
        self.runtime_key.assert_not_called()
        self.assertEqual(namespace["triton_key"](), "compiler-source-fingerprint")

        # Reinitialization must retain the same function rather than wrap it.
        runpy.run_path(str(INIT_PATH))
        self.assertIs(vars(self.compiler)["triton_key"], self.runtime_key)

    def test_existing_compiler_key_is_preserved(self):
        original_key = Mock(return_value="old-compiler-fingerprint")
        vars(self.compiler)["triton_key"] = original_key
        del vars(self.modules["triton.runtime.cache"])["triton_key"]

        runpy.run_path(str(INIT_PATH))

        self.assertIs(vars(self.compiler)["triton_key"], original_key)
        original_key.assert_not_called()

    def test_no_triton_remains_optional(self):
        missing_modules = {name: None for name in self.modules if name == "triton" or name.startswith("triton.")}
        with patch.dict(sys.modules, missing_modules):
            runpy.run_path(str(INIT_PATH))
            self.assertIsNone(sys.modules["triton"])
        self.runtime_key.assert_not_called()

    def test_unavailable_compiler_remains_optional(self):
        # CPU UT environments may expose Triton but stub its runtime modules.
        with patch.dict(sys.modules, {"triton.compiler.compiler": None}):
            runpy.run_path(str(INIT_PATH))

        self.runtime_key.assert_not_called()


if __name__ == "__main__":
    unittest.main()
