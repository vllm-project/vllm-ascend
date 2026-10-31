# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def _load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TestFLALazyImports(unittest.TestCase):
    def _module(self, name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        self.modules[name] = module
        return module

    def setUp(self):
        self.registry: dict[str, type] = {}
        self.registrations: list[type] = []
        registry, registrations = self.registry, self.registrations

        class CustomOp:
            @classmethod
            def register_oot(cls, *, name):
                def register(op):
                    if name in registry:
                        raise AssertionError(f"duplicate registration: {name}")
                    registry[name] = op
                    registrations.append(op)
                    return op

                return register

        class FusedRMSNormGated:
            def __new__(cls, *args, **kwargs):
                return object.__new__(registry.get("FusedRMSNormGated", cls))

            def __init__(self, hidden_size, eps=1e-5, activation="swish"):
                self.weight, self.bias = object(), None
                self.eps, self.activation = eps, activation

        self.modules: dict[str, ModuleType] = {}
        for name in ["vllm", "vllm.model_executor", "vllm.third_party", "vllm.third_party.flash_linear_attention"]:
            self._module(name, __path__=[])
        self._module("vllm.third_party.flash_linear_attention.ops", __path__=[])
        self._module("vllm.third_party.flash_linear_attention.ops.kda", FusedRMSNormGated=FusedRMSNormGated)
        self._module("vllm.model_executor.custom_op", CustomOp=CustomOp)
        self._module("vllm_ascend", __path__=[str(ROOT / "vllm_ascend")])
        self._module("vllm_ascend.ops", __path__=[str(ROOT / "vllm_ascend/ops")])
        self._module("vllm_ascend.ops.triton", __path__=[])
        self._module("vllm_ascend.ops.triton.kda", __path__=[])
        self.kernel = Mock(return_value=object())
        self._module("vllm_ascend.ops.triton.kda.kda", rms_norm_gated=self.kernel)
        utils = self._module("vllm_ascend.utils", REGISTERED_ASCEND_OPS={})
        self.register = _load_function("vllm_ascend/utils.py", "register_ascend_fla_customop", utils.__dict__)
        self.layernorm = self._module("vllm_ascend.ops.layernorm")
        _load_function("vllm_ascend/ops/layernorm.py", "__getattr__", self.layernorm.__dict__)
        context = patch.dict(sys.modules, self.modules)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        sys.modules.pop("vllm_ascend.ops.fused_rms_norm_gated", None)
        self.upstream_class = FusedRMSNormGated

    def test_registration_is_deferred_until_requested(self):
        self.assertEqual(self.registry, {})
        self.assertNotIn("vllm_ascend.ops.fused_rms_norm_gated", sys.modules)
        cls = self.register()
        self.assertTrue(issubclass(cls, self.upstream_class))
        self.assertIs(self.registry["FusedRMSNormGated"], cls)
        self.assertIs(self.modules["vllm_ascend.utils"].REGISTERED_ASCEND_OPS["FusedRMSNormGated"], cls)

    def test_existing_class_export_and_repeated_registration_keep_identity(self):
        cls = self.layernorm.AscendFusedRMSNormGated
        self.assertIs(self.register(), cls)
        self.assertIs(self.layernorm.AscendFusedRMSNormGated, cls)
        self.assertEqual(self.registrations, [cls])

    def test_unknown_attribute_does_not_load_fla(self):
        with self.assertRaises(AttributeError):
            _ = self.layernorm.nonexistent_attribute
        self.assertEqual(self.registrations, [])

    def test_concurrent_registration_uses_one_class(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            classes = list(pool.map(lambda _: self.register(), range(32)))
        self.assertTrue(all(cls is classes[0] for cls in classes))
        self.assertEqual(self.registrations, [classes[0]])

    def test_upstream_dispatch_preserves_kernel_arguments(self):
        cls = self.register()
        layer = cast(Any, self.upstream_class(8, eps=1e-6, activation="sigmoid"))
        self.assertIsInstance(layer, cls)
        x, gate, residual, out = object(), object(), object(), object()
        result = layer.forward_oot(x, gate, residual=residual, prenorm=True, residual_in_fp32=True, out=out)
        self.assertIs(result, self.kernel.return_value)
        self.kernel.assert_called_once_with(
            x,
            gate,
            layer.weight,
            layer.bias,
            "sigmoid",
            residual=residual,
            eps=1e-6,
            prenorm=True,
            residual_in_fp32=True,
            out=out,
        )

    def test_worker_registers_after_device_selection_before_model_runner(self):
        events: list[str] = []

        def select_device():
            events.append("device")
            return object()

        namespace = {
            "register_ascend_fla_customop": lambda: events.append("register"),
            "init_workspace_manager": Mock(),
            "NPUModelRunner": lambda *args: events.append("model_runner"),
        }
        init_device = _load_function("vllm_ascend/worker/worker.py", "init_device", namespace)
        worker = SimpleNamespace(
            _init_device=select_device,
            use_v2_model_runner=False,
            vllm_config=object(),
            rank=1,
        )
        init_device(worker)
        self.assertEqual(events, ["device", "register", "model_runner"])


class TestFLAImportBoundaries(unittest.TestCase):
    def test_discovery_paths_have_no_eager_fla_imports(self):
        paths = ["device/device_op.py", "ops/kda.py", "ops/layernorm.py", "models/glm5next/kda.py"]

        class CheckImports(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Function bodies execute at use time, not module discovery.
                return

            def visit_ImportFrom(self, node):
                if node.module and node.module.startswith("vllm.third_party.flash_linear_attention"):
                    raise AssertionError(f"eager FLA import at line {node.lineno}: {node.module}")

        for path in paths:
            with self.subTest(path=path):
                CheckImports().visit(ast.parse((ROOT / "vllm_ascend" / path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
