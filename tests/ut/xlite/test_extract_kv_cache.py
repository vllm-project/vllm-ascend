# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise cache routing without requiring the optional xlite extension."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


class TestExtractKVCache(unittest.TestCase):
    def test_index_mask_compatibility(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/xlite/xlite.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        model = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV32XliteModel"
        )
        method = next(
            node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "extract_kv_cache"
        )
        # Compile the actual method with postponed annotations; only its optional
        # extension dependency is replaced, leaving cache routing unchanged.
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        namespace: dict[str, Any] = {}
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        extract = namespace["extract_kv_cache"]
        dummy = object()
        cases: list[tuple[dict[str, list[bool]], list[tuple[object, ...]], list[tuple[object, ...]]]] = [
            ({}, [("i0",), ("k0", "v0"), ("i1",), ("k1", "v1")], [("k0", "v0", "i0"), ("k1", "v1", "i1")]),
            (
                {"index_full_mask": []},
                [("i0",), ("k0", "v0"), ("i1",), ("k1", "v1")],
                [("k0", "v0", "i0"), ("k1", "v1", "i1")],
            ),
            (
                {"index_full_mask": [False, True]},
                [("k0", "v0"), ("i1",), ("k1", "v1")],
                [("k0", "v0", dummy), ("k1", "v1", "i1")],
            ),
        ]
        with patch.dict("sys.modules", {"vllm_ascend.xlite.utils": SimpleNamespace(_DUMMY_TENSOR=dummy)}):
            for attributes, caches, expected in cases:
                with self.subTest(attributes=attributes):
                    adapter = SimpleNamespace(xlite_config=SimpleNamespace(n_layers=2, **attributes))
                    self.assertEqual(extract(adapter, caches), expected)
