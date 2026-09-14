# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import logging
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from .rfork_test_support import RFORK_ROOT, _load_module, _stub


@pytest.fixture
def runtime(monkeypatch):
    """Load the real lease/session code with external runtime dependencies stubbed."""
    prefix = "vllm_ascend.model_loader.rfork"

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name):
        full_name = f"{prefix}.{name}"
        spec = importlib.util.spec_from_file_location(full_name, RFORK_ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        return module

    stub("vllm.logger", logger=logging.getLogger("rfork-release-test"))
    stub("vllm.utils.network_utils", get_ip=lambda: "127.0.0.1", join_host_port=lambda host, port: f"{host}:{port}")
    stub(f"{prefix}.identity", build_seed_key=lambda **kwargs: "model-key")
    stub(f"{prefix}.transfer_backend", RForkTransferBackend=Mock)

    class StartupError(RuntimeError):
        def __init__(self, message, *, handle=None):
            super().__init__(message)
            self.handle = handle

    stub(
        f"{prefix}.seed_server",
        RForkSeedServerHandle=Mock,
        RForkSeedServerStartupError=StartupError,
        start_rfork_server=Mock(),
    )
    types = load("types")
    config = load("config")
    load("manifest")
    load("seed_client")
    client = load("planner_client")
    session = load("session")
    monkeypatch.setattr(session.atexit, "register", lambda callback: None)
    cfg = config.RForkConfig(
        "model", "strategy", "http://planner", request_timeout_sec=0.1, lease_release_retry_interval_sec=0.001
    )
    identity = types.RForkIdentity(0, 0, compatibility_fingerprint="fingerprint")
    lease = types.SeedLease("127.0.0.1", 1234, "private-user-id", 0, "model-key")
    return SimpleNamespace(types=types, client=client, session=session, config=cfg, identity=identity, lease=lease)


@pytest.fixture
def tensor_runtime(monkeypatch):
    _stub(monkeypatch, "vllm", __path__=[])
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-tensor-safety-test"))
    _stub(monkeypatch, "vllm.utils", __path__=[])
    _stub(
        monkeypatch,
        "vllm.utils.network_utils",
        get_ip=lambda: "127.0.0.1",
        get_open_port=lambda: 12345,
        join_host_port=lambda host, port: f"{host}:{port}",
    )
    for module_name in (
        "vllm_ascend",
        "vllm_ascend.model_loader",
        "vllm_ascend.model_loader.rfork",
    ):
        _stub(monkeypatch, module_name, __path__=[])

    @dataclass(frozen=True)
    class _SeedTransferInfo:
        session_id: str
        weights: dict
        formats: dict | None = None

    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.types", SeedTransferInfo=_SeedTransferInfo)
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.manifest", "manifest.py")
    tensor_layout = _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.tensor_layout", "tensor_layout.py")
    transfer_backend = _load_module(
        monkeypatch, "vllm_ascend.model_loader.rfork.transfer_backend", "transfer_backend.py"
    )
    return SimpleNamespace(
        tensor_layout=tensor_layout,
        transfer_backend=transfer_backend,
        RForkTransferBackend=transfer_backend.RForkTransferBackend,
        SeedTransferInfo=_SeedTransferInfo,
    )
