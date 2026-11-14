# SPDX-License-Identifier: Apache-2.0
"""Portable checks for direct GPUWorker inheritance in RFC #18920.

Run with unittest without importing the accelerator stack. Runtime Worker
startup remains an integration check for the supported vLLM dependency.
"""

import ast
import copy
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "vllm_ascend/worker/worker.py"


def worker_class():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")


def worker_method(name):
    method = copy.deepcopy(
        next(node for node in worker_class().body if isinstance(node, ast.FunctionDef) and node.name == name)
    )
    module = ast.Module(body=[method], type_ignores=[])
    scope = {}
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), scope)
    return scope[name]


class WorkerInheritanceContract(unittest.TestCase):
    def test_real_parent_constructor_is_used(self):
        cls = worker_class()
        self.assertEqual([ast.unparse(base) for base in cls.bases], ["GPUWorker"])
        constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        calls = [ast.unparse(node.func) for node in ast.walk(constructor) if isinstance(node, ast.Call)]
        self.assertIn("super().__init__", calls)
        self.assertNotIn("WorkerBase.__init__", calls)

    def test_npu_profiler_validation(self):
        validate = worker_method("_validate_profiler_config")
        for config in (None, SimpleNamespace(profiler=None), SimpleNamespace(profiler="torch")):
            with self.subTest(config=config):
                validate(SimpleNamespace(profiler_config=config))
        with self.assertRaisesRegex(ValueError, "torch NPU profiler"):
            validate(SimpleNamespace(profiler_config=SimpleNamespace(profiler="cuda")))

    def test_executor_rejects_elastic_work_and_shutdown_is_idempotent(self):
        name = "vllm_ascend.worker.worker_interfaces"
        module = types.ModuleType(name)
        path = ROOT / "vllm_ascend/worker/worker_interfaces.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        factory = worker_method("_make_elastic_ep_executor")
        with patch.dict("sys.modules", {name: module}):
            executor = factory(SimpleNamespace())
        self.assertEqual(vars(executor), {})
        for method in ("load_model", "reconfigure"):
            with self.subTest(method=method), self.assertRaisesRegex(NotImplementedError, "elastic"):
                executor.execute(method)
        executor.shutdown()
        executor.shutdown()


if __name__ == "__main__":
    unittest.main()
