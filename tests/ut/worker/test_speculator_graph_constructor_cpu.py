# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Preserve omitted and explicit Core graph settings across API revisions."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("kwargs", [{}, {"varlen_decode": False}, {"varlen_decode": True}])
def test_optional_varlen_argument_reaches_parent_unchanged(kwargs):
    source = Path(__file__).resolve().parents[3] / ("vllm_ascend/worker/v2/spec_decode/target_dependent_ar/aclgraph.py")
    owner = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    constructor = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    for argument in constructor.args.args:
        argument.annotation = None

    class Parent:
        def __init__(self, *args, **received):
            self.received = received
            self._capture_descs = {}

        def needs_capture(self):
            return False

    owner.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    owner.body = [constructor]
    namespace = {"Parent": Parent, "Any": object, "collect_sorted_captured_token_sizes": lambda descs: []}
    module = ast.fix_missing_locations(ast.Module(body=[owner], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    manager = namespace[owner.name](None, None, None, 4, **kwargs)
    assert manager.received == {"lora_capture_cases": None, **kwargs}
