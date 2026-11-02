# SPDX-License-Identifier: Apache-2.0
"""Capture warmups must not publish a real connector step's Host mirrors."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "vllm_ascend/attention/sfa_hisparse.py"


def load_helper():
    tree = ast.parse(BASE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "finish_hisparse_kv_update")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BASE), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize("filename", ["sfa_hisparse.py", "sfa_hisparse_optimized.py"])
def test_capture_builder_marks_only_its_fresh_metadata_and_forwards_arguments(filename):
    path = BASE.with_name(filename)
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendHiSparseMetadataBuilder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build_for_cudagraph_capture")
    calls = []

    class Parent:
        def build_for_cudagraph_capture(self, common, **kwargs):
            calls.append((common, kwargs))
            return self.build()

        def build(self):
            return SimpleNamespace(hisparse_for_capture=False)

    cls = ast.ClassDef(
        name=cls.name, bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    namespace = {"Parent": Parent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    builder = namespace[cls.name]()
    common, draft = object(), object()
    capture = builder.build_for_cudagraph_capture(common, draft_index=draft)
    assert calls == [(common, {"draft_index": draft})]
    assert capture.hisparse_for_capture is True
    assert builder.build().hisparse_for_capture is False


def test_repeated_capture_forwards_do_not_submit_host_mirrors():
    finish = load_helper()
    calls = []
    cache = SimpleNamespace(finish_kv_update=lambda: calls.append(True))
    for _ in range(5):
        finish(cache, SimpleNamespace(hisparse_for_capture=True))
    assert calls == []
    finish(cache, SimpleNamespace(hisparse_for_capture=False))
    finish(cache, SimpleNamespace())
    assert calls == [True, True]


def test_real_forward_duplicate_errors_remain_visible():
    finish = load_helper()

    def duplicate():
        raise RuntimeError("HiSparse layer 0 mirrored twice.")

    with pytest.raises(RuntimeError, match="mirrored twice"):
        finish(SimpleNamespace(finish_kv_update=duplicate), SimpleNamespace(hisparse_for_capture=False))
