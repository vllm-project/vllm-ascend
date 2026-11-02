# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Architecture selection must not import an unrelated optional model."""

import ast
import builtins
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

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
    qsa = type("AscendQSABackend", (classifier["AscendAttentionBackend"],), {"draft_attention_architecture": "QSA"})
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


@pytest.mark.parametrize("requires_live_rows,padded", [(False, True), (True, True), (True, False)])
def test_backend_owns_live_query_boundary_contract(requires_live_rows, padded):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/attn_utils.py"
    node = next(
        n
        for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.If) and "requires_unpadded_cpu_query_lens" in ast.unparse(n.test)
    )

    @dataclass
    class Metadata:
        query_start_loc_cpu: object
        seq_lens: object

    queries = list(range(5))
    lengths = object()
    original = Metadata(queries, lengths)
    namespace = dict(
        attn_group=SimpleNamespace(backend=SimpleNamespace(requires_unpadded_cpu_query_lens=requires_live_rows)),
        num_reqs=4,
        num_actual_reqs=2 if padded else 4,
        common_attn_metadata=original,
        replace=replace,
    )
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec")
    exec(code, namespace)
    result = namespace["common_attn_metadata"]
    assert result.query_start_loc_cpu == (queries[:3] if requires_live_rows and padded else queries)
    assert result.seq_lens is lengths
    assert original.query_start_loc_cpu is queries


def test_qsa_backend_declares_its_worker_contracts():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/qwen4_exp/qsa.py"
    owner = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "AscendQSABackend"
    )
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in owner.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)
    }
    assert values["draft_attention_architecture"] == "QSA"
    assert values["requires_unpadded_cpu_query_lens"] is True
