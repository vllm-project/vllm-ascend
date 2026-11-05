# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class TestMiniMaxVisionBinding(unittest.TestCase):
    def setUp(self):
        # Execute the production loader in isolation: importing the full model
        # requires the NPU stack and would also initialize worker dependencies.
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/minimax_m3/minimax_m3_vl.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        loader = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_load_vllm_minimax_m3_common_module"
        )
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.common = root / "models/minimax_m3/common"
        self.common.mkdir(parents=True)
        self.original = Mock(return_value="old")
        self.replacement = Mock(return_value="ascend")
        self.source = types.ModuleType("_test_mrope_source")
        self.source.vit_mrope_setup = self.original
        self.namespace = {
            "Path": Path,
            "vllm": types.SimpleNamespace(__file__=str(root / "__init__.py")),
            "import_from_path": self.import_from_path,
            "vit_mrope_setup": self.replacement,
        }
        exec(compile(ast.Module(body=[loader], type_ignores=[]), str(path), "exec"), self.namespace)
        self.load = self.namespace[loader.name]

    @staticmethod
    def import_from_path(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_model_calls_replacement_without_worker_patch(self):
        (self.common / "vision_tower.py").write_text(
            "from _test_mrope_source import vit_mrope_setup\n"
            "class Vision:\n"
            "    def run(self):\n"
            "        return vit_mrope_setup()\n",
            encoding="utf-8",
        )
        with patch.dict(sys.modules, {"_test_mrope_source": self.source}):
            first = self.load("vision_tower")
            instance = first.Vision()
            self.assertEqual(instance.run(), "ascend")
            # Loading the file again creates a different module. Both existing
            # instances and the new class must still call the Ascend function.
            second = self.load("vision_tower")
            self.assertIsNot(first, second)
            self.assertEqual(second.Vision().run(), "ascend")
            self.assertEqual(instance.run(), "ascend")
            self.assertIs(self.source.vit_mrope_setup, self.original)
            self.original.assert_not_called()

    def test_preprocessing_is_not_patched(self):
        (self.common / "mm_preprocess.py").write_text("sentinel = 42\n", encoding="utf-8")
        with patch.dict(sys.modules):
            module = self.load("mm_preprocess")
            self.assertEqual(module.sentinel, 42)
            self.assertFalse(hasattr(module, "vit_mrope_setup"))

    def test_missing_source(self):
        with self.assertRaisesRegex(ImportError, "common source was not found"):
            self.load("vision_tower")

    def test_missing_package_location(self):
        self.namespace["vllm"].__file__ = None
        with self.assertRaisesRegex(ImportError, "Unable to locate"):
            self.load("vision_tower")


if __name__ == "__main__":
    unittest.main()
