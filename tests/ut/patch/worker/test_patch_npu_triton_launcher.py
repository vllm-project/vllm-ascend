# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_npu_triton_launcher.py"


def triton_version_uses_attrs_dict():
    return True


def _make_launcher(self):
    # Reduced torch_npu 2.10.0.post4 argument-generation contract. Keep both
    # branches so tests exercise the real ABI decision, not just a callback.
    fn = self.kernel.src.fn
    if triton_version_uses_attrs_dict():
        def_args = call_args = fn.arg_names
    else:
        def_args = [name for name in fn.arg_names if name not in self.config.kwargs]
        call_args = [name for i, name in enumerate(fn.arg_names) if i not in fn.constexprs]
    scope: dict[str, Any] = {"runner": self.kernel.run, "control": self.control}
    exec(
        f"def launcher({', '.join(def_args)}, stream):\n    return runner(*control, {', '.join(call_args)})",
        scope,
    )
    launcher = scope["launcher"]
    launcher.config = self.config
    launcher.store_cubin = True
    launcher.def_args = [name for name in def_args if name not in self.config.kwargs]
    launcher.call_args = [name for name in call_args if self.kernel.src.signature.get(name) != "constexpr"]
    return launcher


class TestNpuTritonLauncher(unittest.TestCase):
    def setUp(self):
        module = ModuleType("torch_npu._inductor.npu_triton_heuristics")
        self.result_type: Any = type("TritonCompileResultNpu", (), {"make_launcher": lambda self: self.launcher})
        self.uses_attrs = Mock(return_value=True)
        vars(module).update(TritonCompileResultNpu=self.result_type, triton_version_uses_attrs_dict=self.uses_attrs)
        with patch.dict(sys.modules, {module.__name__: module}):
            self.bind_constants = runpy.run_path(str(PATCH_PATH))["_bind_launcher_constants"]

    def result(self, launcher, names, constexprs, constants):
        result = self.result_type()
        result.launcher = launcher
        result.kernel = SimpleNamespace(
            src=SimpleNamespace(
                fn=SimpleNamespace(arg_names=names, constexprs=constexprs), signature=dict.fromkeys(names)
            )
        )
        result.compile_meta = {"constants": constants}
        return result

    def test_gdn_block_constants_and_launcher_metadata(self):
        calls = []

        def launcher(src, dst, x_numel, X0BLOCK, X0BLOCK_SUB, stream):
            calls.append((src, dst, x_numel, X0BLOCK, X0BLOCK_SUB, stream))

        # The generated wrapper in the failing CI only supplies these runtime args.
        with self.assertRaisesRegex(TypeError, "X0BLOCK"):
            launcher("src", "dst", 2, stream=7)  # type: ignore[call-arg]
        attrs = {
            "config": object(),
            "n_regs": 4,
            "n_spills": 0,
            "shared": 0,
            "store_cubin": True,
            "fn": object(),
            "bin": object(),
            "def_args": ["src", "dst", "x_numel"],
            "call_args": ["src", "dst", "x_numel"],
        }
        launcher.__dict__.update(attrs)
        result = self.result(
            launcher, ["src", "dst", "x_numel", "X0BLOCK", "X0BLOCK_SUB"], [3, 4], {"X0BLOCK": 128, "X0BLOCK_SUB": 32}
        )
        bound = result.make_launcher()
        bound("src", "dst", 2, stream=7)
        self.assertEqual(calls, [("src", "dst", 2, 128, 32, 7)])
        self.assertEqual(list(inspect.signature(bound).parameters), ["src", "dst", "x_numel", "stream"])
        for name, value in attrs.items():
            self.assertIs(getattr(bound, name), value)

    def test_interleaved_constants_extra_grid_and_hooks(self):
        enter, leave = Mock(), Mock()
        metadata = object()
        value = "quote' and backslash\\"

        def launcher(src, mode, dst, optional, grid_0, stream):
            enter(metadata)
            leave(metadata)
            return src, mode, dst, optional, grid_0, stream

        result = self.result(launcher, ["src", "mode", "dst", "optional"], [1, 3], {"mode": value, "optional": None})
        bound = result.make_launcher()
        self.assertEqual(bound("src", "dst", 5, stream=9), ("src", value, "dst", None, 5, 9))
        enter.assert_called_once_with(metadata)
        leave.assert_called_once_with(metadata)

    def test_preserves_explicit_constexpr_and_runtime_specialization(self):
        def launcher(src, block, size, stream):
            return src, block, size, stream

        # A constexpr absent from compile constants remains caller-supplied;
        # a specialized runtime argument must not be removed from the signature.
        result = self.result(launcher, ["src", "block", "size"], [1], {"size": 1})
        self.assertIs(result.make_launcher(), launcher)

    def test_legacy_and_already_fixed_launchers_are_unchanged(self):
        def launcher(src, stream):
            return src, stream

        result = self.result(launcher, ["src", "block"], [1], {"block": 128})
        self.assertIs(result.make_launcher(), launcher)
        self.uses_attrs.return_value = False
        self.assertIs(result.make_launcher(), launcher)

        def legacy_launcher(src, block, stream):
            return src, stream

        result.launcher = legacy_launcher
        self.assertIs(result.make_launcher(), legacy_launcher)

    def test_config_values_are_per_launcher(self):
        def launcher(src, block, stream):
            return src, block, stream

        result = self.result(launcher, ["src", "block"], [1], {"block": 32})
        first = result.make_launcher()
        result.compile_meta["constants"]["block"] = 64
        second = result.make_launcher()
        self.assertEqual(first("src", stream=1), ("src", 32, 1))
        self.assertEqual(second("src", stream=1), ("src", 64, 1))

    def test_compact_signature_matches_strict_npu_launch_arity(self):
        calls = []

        # NPULauncher uses nine control args plus len(src.signature) args.
        # A fixed Python signature rejects extra constants just like its C stub.
        def runner(g0, g1, g2, stream, function, packed, metadata, enter, leave, src, dst, x_numel):
            enter(metadata)
            calls.append((g0, g1, g2, stream, function, packed, metadata, src, dst, x_numel))
            leave(metadata)

        for names in (
            ["src", "dst", "x_numel", "X0BLOCK", "X0BLOCK_SUB"],
            ["src", "X0BLOCK", "dst", "X0BLOCK_SUB", "x_numel"],
        ):
            with self.subTest(names=names):
                constants = {"X0BLOCK": 128, "X0BLOCK_SUB": 32}
                result = self.result(None, names, [names.index(n) for n in constants], constants)
                result.config = SimpleNamespace(kwargs=constants)
                result.kernel.src.signature = {"src": "*fp32", "dst": "*fp32", "x_numel": "i32"}
                result.kernel.run = runner
                enter, leave = Mock(), Mock()
                result.control = (2, 1, 1, 7, "kernel", "packed", "metadata", enter, leave)
                launcher = self.bind_constants(_make_launcher)(result)
                launcher("src", "dst", 2, stream=7)
                self.assertEqual(calls[-1], (2, 1, 1, 7, "kernel", "packed", "metadata", "src", "dst", 2))
                self.assertEqual(launcher.def_args, ["src", "dst", "x_numel"])
                self.assertEqual(launcher.call_args, ["src", "dst", "x_numel"])
                self.assertIs(launcher.config, result.config)
                self.assertTrue(launcher.store_cubin)
                enter.assert_called_once_with("metadata")
                leave.assert_called_once_with("metadata")
                # No per-launch wrapper and no mutation of shared backend globals.
                self.assertIs(launcher.__globals__["runner"], runner)
                self.assertTrue(triton_version_uses_attrs_dict())

    def test_full_signature_retains_constants_in_strict_runner(self):
        def runner(g0, g1, g2, stream, function, packed, metadata, enter, leave, src, dst, x_numel, block, sub):
            return src, dst, x_numel, block, sub

        names = ["src", "dst", "x_numel", "X0BLOCK", "X0BLOCK_SUB"]
        constants = {"X0BLOCK": 128, "X0BLOCK_SUB": 32}
        result = self.result(None, names, [3, 4], constants)
        result.kernel.src.signature = dict(zip(names, ["*fp32", "*fp32", "i32", "constexpr", "constexpr"]))
        result.config = SimpleNamespace(kwargs=constants)
        result.kernel.run = runner
        result.control = (2, 1, 1, 7, "kernel", "packed", None, None, None)
        launcher = self.bind_constants(_make_launcher)(result)
        self.assertEqual(launcher("src", "dst", 2, stream=7), ("src", "dst", 2, 128, 32))


if __name__ == "__main__":
    unittest.main()
