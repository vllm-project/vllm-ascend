# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Exercise the actual fallback without importing unrelated worker patches."""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def load_fp8_flip():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_kv_cache_dtype.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_fp8_kv_cache_dtype_flip"
    )
    return compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec")


class TestWorkerFp8DtypeFallback(unittest.TestCase):
    def exercise(self, *, registration_api, native_mapping):
        native, raw, signed = object(), object(), object()
        mapping = {"fp8": native if native_mapping else raw, "int8": signed, "mxfp8": native}
        alias = mapping
        cache = ModuleType("vllm.config.cache")
        if registration_api:
            cache.register_kv_cache_dtype = object()
        modules = {name: ModuleType(name) for name in ("vllm", "vllm.config")}
        modules["vllm.config.cache"] = cache
        namespace = {
            "torch": SimpleNamespace(float8_e4m3fn=native),
            "_torch_utils": SimpleNamespace(STR_DTYPE_TO_TORCH_DTYPE=mapping),
        }
        exec(load_fp8_flip(), namespace)
        with patch.dict(sys.modules, modules):
            namespace["_apply_fp8_kv_cache_dtype_flip"]()
            namespace["_apply_fp8_kv_cache_dtype_flip"]()
        self.assertIs(alias, mapping)
        self.assertIs(mapping["fp8"], native)
        self.assertIs(mapping["int8"], signed)
        self.assertIs(mapping["mxfp8"], native)

    def test_registration_api_without_native_mapping(self):
        # The API can exist while no handler has run in this worker.
        self.exercise(registration_api=True, native_mapping=False)

    def test_legacy_build_without_registration_api(self):
        self.exercise(registration_api=False, native_mapping=False)

    def test_native_mapping_is_idempotent(self):
        self.exercise(registration_api=True, native_mapping=True)
