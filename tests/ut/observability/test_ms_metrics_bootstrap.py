# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

_REPOSITORY_ROOT = Path(__file__).parents[3]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "vllm_ascend" / "observability" / "ms_metrics"


def _load_bootstrap(monkeypatch, module_name: str = "test_ms_metrics_bootstrap_mod"):
    spec = importlib.util.spec_from_file_location(module_name, _PACKAGE_ROOT / "bootstrap.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Fresh module each time so the idempotent guard resets.
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_ensure_prometheus_multiproc_dir_keeps_user_path(monkeypatch, tmp_path):
    user_dir = tmp_path / "user_prom"
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(user_dir))
    bootstrap = _load_bootstrap(monkeypatch)

    path = bootstrap.ensure_prometheus_multiproc_dir()

    assert path == str(user_dir)
    assert user_dir.is_dir()


def test_ensure_prometheus_multiproc_dir_calls_vllm_helper(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    bootstrap = _load_bootstrap(monkeypatch, "test_ms_metrics_bootstrap_auto")

    called = {"n": 0}

    def _fake_setup():
        called["n"] += 1
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/auto_prom_dir")

    fake_prom = ModuleType("vllm.v1.metrics.prometheus")
    fake_prom.setup_multiprocess_prometheus = _fake_setup
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics.prometheus", fake_prom)
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics", ModuleType("vllm.v1.metrics"))

    path = bootstrap.ensure_prometheus_multiproc_dir()

    assert called["n"] == 1
    assert path == "/tmp/auto_prom_dir"


def test_enable_ms_metric_collection_skips_without_package(monkeypatch):
    bootstrap = _load_bootstrap(monkeypatch, "test_ms_metrics_bootstrap_skip")
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name.startswith("ms_service_metric"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    assert bootstrap.enable_ms_metric_collection() is False


def test_enable_ms_metric_collection_writes_on(monkeypatch):
    bootstrap = _load_bootstrap(monkeypatch, "test_ms_metrics_bootstrap_on")

    manager = MagicMock()
    manager.connect.return_value = True
    manager.send_control_command.return_value = (True, 2, 0, True)

    shm_mod = ModuleType("ms_service_metric.utils.shm_manager")
    shm_mod.POSIX_IPC_AVAILABLE = True
    shm_mod.SharedMemoryManager = MagicMock(return_value=manager)

    utils_mod = ModuleType("ms_service_metric.utils")
    pkg = ModuleType("ms_service_metric")
    monkeypatch.setitem(sys.modules, "ms_service_metric", pkg)
    monkeypatch.setitem(sys.modules, "ms_service_metric.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "ms_service_metric.utils.shm_manager", shm_mod)

    assert bootstrap.enable_ms_metric_collection() is True
    manager.connect.assert_called_once_with(create=True)
    manager.send_control_command.assert_called_once_with(is_start=True, force=False, send_signal=True)
    manager.disconnect.assert_called_once()


def test_bootstrap_ms_metrics_defaults_on_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path / "prom"))
    bootstrap = _load_bootstrap(monkeypatch, "test_ms_metrics_bootstrap_default")

    calls = {"ensure": 0, "enable": 0}

    def _fake_ensure():
        calls["ensure"] += 1
        return str(tmp_path / "prom")

    def _fake_enable():
        calls["enable"] += 1
        return True

    monkeypatch.setattr(bootstrap, "ensure_prometheus_multiproc_dir", _fake_ensure)
    monkeypatch.setattr(bootstrap, "enable_ms_metric_collection", _fake_enable)

    bootstrap._bootstrapped = False
    bootstrap.bootstrap_ms_metrics()
    assert calls == {"ensure": 1, "enable": 1}

    bootstrap.bootstrap_ms_metrics()
    assert calls == {"ensure": 1, "enable": 1}
