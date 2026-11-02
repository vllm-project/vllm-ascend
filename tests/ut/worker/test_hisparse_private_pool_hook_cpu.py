# SPDX-License-Identifier: Apache-2.0
"""The reviewed Core allocation boundary must reach its backend before views."""

import ast
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/hisparse/runtime_adapter.py"


def load_installer(monkeypatch, backend, verify):
    calls = []

    def original(size):
        calls.append(size)
        return "original", size

    common = SimpleNamespace(allocate_pinned_host_pool=original)
    for name in ["vllm", "vllm.v1", "vllm.v1.hisparse", "vllm.v1.hisparse.backend"]:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.v1.hisparse"].runtime = common
    sys.modules["vllm.v1.hisparse.backend"].get_hisparse_backend = lambda: backend
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "install_private_host_allocator"
    )
    namespace = {"wraps": wraps, "_verify_source": verify, "COMMON_RUNTIME_SHA256": "reviewed"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[node.name], common, original, calls


def test_private_pool_reaches_selected_allocator_once_and_preserves_owners(monkeypatch):
    owners, calls, checks = (object(), object()), [], []

    def allocate(size):
        calls.append(size)
        return owners

    install, common, original, fallback = load_installer(
        monkeypatch, SimpleNamespace(allocate_private_host_pool=allocate), lambda fn, sha: checks.append((fn, sha))
    )
    install()
    wrapped = common.allocate_pinned_host_pool
    install()
    assert common.allocate_pinned_host_pool is wrapped
    assert wrapped(8193) is owners and calls == [8193] and fallback == []
    assert checks == [(original, "reviewed")]


def test_non_owning_backend_uses_original_allocator(monkeypatch):
    install, common, _, calls = load_installer(monkeypatch, SimpleNamespace(), lambda *_: None)
    install()
    assert common.allocate_pinned_host_pool(4096) == ("original", 4096)
    assert calls == [4096]


def test_unknown_core_is_rejected_before_installation(monkeypatch):
    def reject(*_):
        raise RuntimeError("Unknown Core source")

    install, common, original, calls = load_installer(monkeypatch, SimpleNamespace(), reject)
    with pytest.raises(RuntimeError, match="Unknown Core"):
        install()
    assert common.allocate_pinned_host_pool is original and calls == []
