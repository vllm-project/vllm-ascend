# SPDX-License-Identifier: Apache-2.0
"""Actual cold profile selection, without importing Torch or initializing NPU."""

import hashlib
import importlib.util
from pathlib import Path

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
