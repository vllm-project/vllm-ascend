# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TestVitMropeSetupPatch(unittest.TestCase):
    def load_patch(self, missing=None):
        target = "vllm.model_executor.layers.rotary_embedding.mrope_vit_setup"
        modules = {}
        parts = target.split(".")
        for end in range(1, len(parts) + 1):
            name = ".".join(parts[:end])
            modules[name] = types.ModuleType(name)
            modules[name].__path__ = []
            if end > 1:
                setattr(modules[".".join(parts[: end - 1])], parts[end - 1], modules[name])
        upstream = modules[target]
        original = object()
        upstream._vit_mrope_setup_kernel = original
        # Model modules may import the wrapper before the worker patch runs.
        exec("def vit_mrope_setup(): return _vit_mrope_setup_kernel", upstream.__dict__)
        alias = upstream.vit_mrope_setup
        jit_options = {}

        def jit(**kwargs):
            jit_options.update(kwargs)
            return lambda fn: fn

        triton_utils = types.ModuleType("vllm.triton_utils")
        triton_utils.triton = types.SimpleNamespace(jit=jit)
        triton_utils.tl = types.SimpleNamespace(constexpr=int)
        modules["vllm.triton_utils"] = triton_utils
        if missing == target:
            modules[target] = None
            delattr(modules[".".join(parts[:-1])], parts[-1])
        elif missing:
            del modules[target]
            delattr(modules[".".join(parts[:-1])], parts[-1])

        path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_mrope_vit_setup.py"
        spec = importlib.util.spec_from_file_location("_test_vit_mrope_patch", path)
        module = importlib.util.module_from_spec(spec)
        real_import = builtins.__import__

        def import_with_missing_dependency(name, *args, **kwargs):
            if name == target:
                raise ModuleNotFoundError(name=missing)
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, modules):
            if missing and missing != target:
                with patch("builtins.__import__", side_effect=import_with_missing_dependency):
                    spec.loader.exec_module(module)
            else:
                spec.loader.exec_module(module)
        return module, upstream, alias, original, jit_options

    def test_existing_wrapper_alias_uses_replacement(self):
        module, upstream, alias, original, options = self.load_patch()
        self.assertIs(alias(), module._vit_mrope_setup_kernel)
        self.assertIs(upstream._vit_mrope_setup_kernel, module._vit_mrope_setup_kernel)
        self.assertIsNot(alias(), original)
        self.assertEqual(options["do_not_specialize"], ["num_tokens", "num_grids"])
        self.assertEqual(options["do_not_specialize_on_alignment"], ["grids_ptr"])

    def test_older_vllm_without_setup_module(self):
        module, _, alias, original, _ = self.load_patch(
            missing="vllm.model_executor.layers.rotary_embedding.mrope_vit_setup"
        )
        self.assertIsNone(module.mrope_vit_setup)
        self.assertIs(alias(), original)

    def test_missing_dependency_is_not_silenced(self):
        with self.assertRaises(ModuleNotFoundError):
            self.load_patch(missing="unexpected_dependency")


if __name__ == "__main__":
    unittest.main()
