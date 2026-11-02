# SPDX-License-Identifier: Apache-2.0
"""Actual cold profile selection, without importing Torch or initializing NPU."""

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "hisparse_profile_test", ROOT / "vllm_ascend/distributed/kv_transfer/hisparse/optimization_profile.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_baseline_needs_no_optional_core_package(tmp_path):
    assert MODULE.select_profile("baseline", tmp_path) == "baseline"


def test_unknown_profile_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        MODULE.select_profile("automatic", tmp_path)


@pytest.mark.parametrize("changed", [None, "missing", "runtime", "worker"])
def test_exact_source_pairing_before_any_import(tmp_path, monkeypatch, changed):
    expected = {}
    for relative in ["runtime.py", "worker.py"]:
        path = tmp_path / relative
        path.write_bytes(relative.encode())
        expected[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "CORE_SOURCES", expected)
    if changed == "missing":
        (tmp_path / "worker.py").unlink()
    elif changed:
        (tmp_path / (changed + ".py")).write_bytes(b"a different source contract")
    if changed:
        with pytest.raises(RuntimeError):
            MODULE.select_profile("a5_c8_62592", tmp_path)
    else:
        assert MODULE.select_profile("a5_c8_62592", tmp_path) == "a5_c8_62592"


def test_source_pairing_accepts_windows_checkout_line_endings(tmp_path, monkeypatch):
    original = b"# same public source\nvalue = 1\n"
    monkeypatch.setattr(MODULE, "CORE_SOURCES", {"worker.py": hashlib.sha256(original).hexdigest()})
    (tmp_path / "worker.py").write_bytes(original.replace(b"\n", b"\r\n"))
    assert MODULE.select_profile("a5_c8_62592", tmp_path) == "a5_c8_62592"


@pytest.mark.parametrize("changed", [None, "runtime", "worker", "missing"])
def test_main_pairing_rejects_partial_or_different_source_sets(tmp_path, monkeypatch, changed):
    legacy = {}
    current = {}
    for name in ["runtime", "worker"]:
        relative = f"v1/hisparse/{name}.py"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# reviewed main {name}\n".encode())
        legacy[relative] = hashlib.sha256(f"# historical {name}\n".encode()).hexdigest()
        current[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "CORE_SOURCES", legacy)
    monkeypatch.setattr(MODULE, "MAIN_CORE_SOURCES", current)
    if changed == "missing":
        (tmp_path / "v1/hisparse/worker.py").unlink()
    elif changed:
        (tmp_path / f"v1/hisparse/{changed}.py").write_bytes(f"# historical {changed}\n".encode())
    if changed:
        with pytest.raises(RuntimeError):
            MODULE.select_profile("a5_c8_main", tmp_path)
        # Probe a main-version file: accepting it requires the entire main set.
        probe = "worker" if changed == "runtime" else "runtime"
        with pytest.raises(RuntimeError):
            MODULE.verify_core_source(tmp_path / f"v1/hisparse/{probe}.py", legacy[f"v1/hisparse/{probe}.py"])
    else:
        assert MODULE.select_profile("a5_c8_main", tmp_path) == "a5_c8_main"
        MODULE.verify_core_source(tmp_path / "v1/hisparse/runtime.py", legacy["v1/hisparse/runtime.py"])
        with pytest.raises(RuntimeError):
            MODULE.select_profile("a5_c8_62592", tmp_path)


@pytest.mark.parametrize("type_parameters", [None, [], [ast.Name(id="T", ctx=ast.Load())]])
def test_certificate_guard_normalizes_only_empty_python312_fields(type_parameters):
    policy = ROOT / "vllm_ascend/patch/worker/patch_hisparse_lazy_mtp_mirror.py"
    definition = next(
        node
        for node in ast.parse(policy.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_certificate_source_digest"
    )

    def parse(source):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and type_parameters is not None:
                if "type_params" not in node._fields:
                    node._fields = (*node._fields, "type_params")
                node.type_params = type_parameters
        return tree

    namespace = {
        "ast": SimpleNamespace(parse=parse, walk=ast.walk, dump=ast.dump),
        "sha256": hashlib.sha256,
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(policy), "exec"), namespace)
    certificate = "def certified(value):\n    return value + 1\n"
    ordinary = ast.parse(certificate)
    for node in ast.walk(ordinary):
        if "type_params" in node._fields and getattr(node, "type_params", None) == []:
            node._fields = tuple(name for name in node._fields if name != "type_params")
    expected = hashlib.sha256(ast.dump(ordinary).encode()).hexdigest()
    actual = namespace["_certificate_source_digest"](certificate)
    assert (actual == expected) is not bool(type_parameters)
