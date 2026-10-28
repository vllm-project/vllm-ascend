# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
            "triton.knobs",
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
        self.compiled_kernel: Any = type("CompiledKernel", (), {})
        vars(self.compiler)["CompiledKernel"] = self.compiled_kernel
        self.runtime_hooks = SimpleNamespace(launch_enter_hook=None, launch_exit_hook=None)
        vars(self.modules["triton.knobs"])["runtime"] = self.runtime_hooks
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

    def test_launch_hooks_forward_callbacks_and_metadata(self):
        calls = []

        def enter(metadata):
            calls.append(("enter", metadata))

        def exit_hook(metadata):
            calls.append(("exit", metadata))

        self.runtime_hooks.launch_enter_hook = enter
        self.runtime_hooks.launch_exit_hook = exit_hook
        runpy.run_path(str(INIT_PATH))

        binary = self.compiled_kernel()
        metadata = {"name": "test_kernel", "stream": 7}
        # torch_npu reads the class attributes for both the launcher scope and
        # its launch_metadata guard. Instance access must not bind callbacks.
        self.assertIs(binary.__class__.launch_enter_hook, enter)
        self.assertIs(binary.__class__.launch_exit_hook, exit_hook)
        binary.launch_enter_hook(metadata)
        binary.launch_exit_hook(metadata)
        self.assertEqual(calls, [("enter", metadata), ("exit", metadata)])

    def test_launch_hooks_follow_runtime_replacement(self):
        runpy.run_path(str(INIT_PATH))
        for name in ("launch_enter_hook", "launch_exit_hook"):
            with self.subTest(hook=name):
                descriptor = vars(self.compiled_kernel)[name]
                self.assertIsNone(getattr(self.compiled_kernel, name))
                hook = Mock()
                setattr(self.runtime_hooks, name, hook)
                self.assertIs(getattr(self.compiled_kernel, name), hook)
                setattr(self.runtime_hooks, name, None)
                self.assertIsNone(getattr(self.compiled_kernel, name))
                runpy.run_path(str(INIT_PATH))
                self.assertIs(vars(self.compiled_kernel)[name], descriptor)

    def test_existing_launch_hooks_are_preserved(self):
        original_exit = Mock()
        self.compiled_kernel.launch_enter_hook = None
        self.compiled_kernel.launch_exit_hook = original_exit
        # Old Triton has no knobs module. Also preserve exports if it does.
        for has_knobs in (True, False):
            with self.subTest(has_knobs=has_knobs):
                if not has_knobs:
                    del vars(self.modules["triton"])["knobs"]
                with patch.dict(sys.modules, {} if has_knobs else {"triton.knobs": None}):
                    runpy.run_path(str(INIT_PATH))
                self.assertIsNone(self.compiled_kernel.launch_enter_hook)
                self.assertIs(self.compiled_kernel.launch_exit_hook, original_exit)

    def test_only_missing_launch_hook_is_patched(self):
        self.compiled_kernel.launch_enter_hook = None
        self.runtime_hooks.launch_exit_hook = Mock()

        runpy.run_path(str(INIT_PATH))

        self.assertIsNone(self.compiled_kernel.launch_enter_hook)
        self.assertIs(self.compiled_kernel.launch_exit_hook, self.runtime_hooks.launch_exit_hook)


if __name__ == "__main__":
    unittest.main()
