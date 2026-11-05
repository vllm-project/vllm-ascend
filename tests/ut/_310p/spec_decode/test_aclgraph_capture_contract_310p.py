# SPDX-License-Identifier: Apache-2.0
"""Exercise the 310P capture override without loading an NPU runtime."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/_310p/worker/v2/spec_decode/aclgraph.py"


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("pcp_manager", [None, object()])
def test_capture_forwards_pcp_manager(monkeypatch, full_graph, pcp_manager):
    tree = ast.parse(SOURCE.read_text())
    original = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AutoRegressiveAclGraphManager310"
    )
    capture = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "capture")
    # Execute the production override, replacing only its runtime dependencies.
    cls = ast.ClassDef(
        name=original.name,
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[capture],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    calls = []

    class Parent:
        def capture(self, *args, **kwargs):
            calls.append(kwargs)

    class Batch:
        @classmethod
        def make_dummy(cls, *args, **kwargs):
            return SimpleNamespace()

    original_dummy = Batch.__dict__["make_dummy"]
    attention = types.ModuleType("vllm_ascend.attention.attention_v1")
    attention.AscendAttentionState = SimpleNamespace(SpecDecoding=object())
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    namespace = dict(Parent=Parent, AscendInputBatch=Batch, logger=SimpleNamespace(info=lambda *a: None))
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    manager = namespace[original.name]()
    manager.is_draft_model_prefill = True
    manager.cudagraph_mode = SimpleNamespace(has_full_cudagraphs=lambda: full_graph)
    manager.capture(*([None] * 6), pcp_manager=pcp_manager)
    assert len(calls) == 1
    assert calls[0]["pcp_manager"] is pcp_manager
    if full_graph:
        assert Batch.make_dummy.__func__ is original_dummy.__func__
