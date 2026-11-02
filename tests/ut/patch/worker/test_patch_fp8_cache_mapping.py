# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise the mapping fallback without importing NPU worker side effects."""

import ast
import types
import unittest
from pathlib import Path


class TestFP8CacheMapping(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_kv_cache_dtype.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_fp8_kv_cache_dtype_flip"
        )
        self.dtype = types.SimpleNamespace(uint8=object(), float8_e4m3fn=object())
        self.mapping = {"fp8": self.dtype.uint8, "int8": object()}
        namespace = {"torch": self.dtype, "_torch_utils": types.SimpleNamespace(STR_DTYPE_TO_TORCH_DTYPE=self.mapping)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        self.apply = namespace[function.name]

    def test_raw_byte_fallback_and_idempotency(self):
        int8 = self.mapping["int8"]
        self.apply()
        self.apply()
        self.assertIs(self.mapping["fp8"], self.dtype.float8_e4m3fn)
        self.assertIs(self.mapping["int8"], int8)

    def test_registered_native_and_custom_mappings_are_preserved(self):
        for native in (self.dtype.float8_e4m3fn, object()):
            self.mapping["fp8"] = native
            self.apply()
            self.assertIs(self.mapping["fp8"], native)

    def test_missing_mapping_is_not_invented(self):
        del self.mapping["fp8"]
        self.apply()
        self.assertNotIn("fp8", self.mapping)


if __name__ == "__main__":
    unittest.main()
