# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Architecture selection must not import an unrelated optional model."""

import ast
import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def classifier():
    # Exercise the production selection without importing NPU-only dependencies.
    source = Path(__file__).resolve().parents[3] / (
        "vllm_ascend/worker/v2/spec_decode/target_dependent_ar/speculator.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendTargetDependentARSpeculator")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "set_attn")
    assert isinstance(method.body[-1], ast.If)
    function = ast.FunctionDef(
        name="classify",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[method.body[-1]],
        decorator_list=[],
    )
    names = (
        "AscendDSABackend",
        "AscendMLABackend",
        "AscendSFABackend",
        "AscendSFAIndexerBackend",
        "AscendAttentionBackend",
    )
    namespace = {name: type(name, (), {}) for name in names}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("AscendDSABackend", "DSA"),
        ("AscendMLABackend", "MLA"),
        ("AscendSFABackend", "SFA"),
        ("AscendSFAIndexerBackend", "SFA"),
    ],
)
def test_known_backend_with_qsa_unavailable(classifier, monkeypatch, backend, expected):
    original = builtins.__import__

    def import_without_qsa(name, *args, **kwargs):
        if name == "vllm_ascend.models.qwen4_exp.qsa":
            raise AssertionError("Known backend imported an unavailable QSA model")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_qsa)
    instance = SimpleNamespace(attn_backend=classifier[backend])
    classifier["classify"](instance)
    assert instance.attn_architecture == expected


@pytest.mark.parametrize("expected", ["QSA", "GQA", "unsupported"])
def test_optional_backend_keeps_classification_and_rejection(classifier, monkeypatch, expected):
    qsa = type("AscendQSABackend", (classifier["AscendAttentionBackend"],), {})
    module = ModuleType("vllm_ascend.models.qwen4_exp.qsa")
    module.AscendQSABackend = qsa
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = {"QSA": qsa, "GQA": classifier["AscendAttentionBackend"], "unsupported": type("Unsupported", (), {})}[
        expected
    ]
    instance = SimpleNamespace(attn_backend=backend)
    if expected == "unsupported":
        with pytest.raises(ValueError, match="Unsupported attention backend"):
            classifier["classify"](instance)
    else:
        classifier["classify"](instance)
        assert instance.attn_architecture == expected
