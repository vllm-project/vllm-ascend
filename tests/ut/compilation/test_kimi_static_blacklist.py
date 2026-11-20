# SPDX-License-Identifier: Apache-2.0
"""Exercise static runtime initialization without importing NPU packages."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TestKimiStaticBlacklist(unittest.TestCase):
    def test_quant_matmul_is_allowed_while_other_blacklist_entries_are_preserved(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/compiler_interface.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if getattr(n, "name", None) == "_initialize_static_kernel_runtime")
        setter = Mock()
        env = {"LOCAL_WORLD_SIZE": "8"}
        namespace = dict(os=SimpleNamespace(environ=env), logger=Mock(), _set_static_kernel_blacklist=setter)
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        # Initialization must still set the blacklist when topology is already configured.
        namespace["_initialize_static_kernel_runtime"](None)
        entries = setter.call_args.args[0]
        self.assertIn("FusedInferAttentionScore", entries)
        self.assertIn("QuantLightningIndexer", entries)
        self.assertIn("TopKV2", entries)
        self.assertIn("GenericBlockSparseAttn", entries)
        self.assertIn("TransposeQuantBatchMatMul", entries)
        self.assertNotIn("QuantBatchMatmulV3", entries)
        self.assertEqual(env, {"LOCAL_WORLD_SIZE": "8"})


if __name__ == "__main__":
    unittest.main()
