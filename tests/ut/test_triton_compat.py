from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

MODULE_PATH = Path(__file__).parents[2] / "vllm_ascend" / "_triton_compat.py"
SPEC = importlib.util.spec_from_file_location("vllm_ascend_triton_compat_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)


@pytest.fixture(autouse=True)
def isolate_gluon_modules(monkeypatch: pytest.MonkeyPatch):
    original = {
        name: module
        for name, module in sys.modules.items()
        if name == "triton.experimental" or name.startswith("triton.experimental.gluon")
    }
    for name in list(original):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in list(sys.modules):
        if name == "triton.experimental" or name.startswith("triton.experimental.gluon"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    sys.modules.update(original)


def test_modern_triton_loads_real_gluon_without_stubs(monkeypatch: pytest.MonkeyPatch):
    loaded: list[str] = []

    def import_module(name: str):
        loaded.append(name)
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        return module

    monkeypatch.setattr(compat, "_triton_version", lambda: compat.Version("3.6.0"))
    monkeypatch.setattr(compat.importlib, "import_module", import_module)

    compat.ensure_gluon_compatibility()

    assert loaded == ["triton.experimental.gluon", "triton.experimental.gluon.language"]


def test_legacy_triton_gets_complete_parent_child_hierarchy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(compat, "_triton_version", lambda: compat.Version("3.5.0"))

    compat.ensure_gluon_compatibility()

    experimental = sys.modules["triton.experimental"]
    gluon = sys.modules["triton.experimental.gluon"]
    language = sys.modules["triton.experimental.gluon.language"]
    assert experimental.gluon is gluon
    assert gluon.language is language
    assert experimental.__path__ == []
    assert gluon.__path__ == []


def test_modern_triton_import_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(compat, "_triton_version", lambda: compat.Version("3.6.0"))

    def fail_import(name: str):
        raise ImportError(f"broken modern Gluon: {name}")

    monkeypatch.setattr(compat.importlib, "import_module", fail_import)

    with pytest.raises(ImportError, match="broken modern Gluon"):
        compat.ensure_gluon_compatibility()
