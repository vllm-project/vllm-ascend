# SPDX-License-Identifier: Apache-2.0
"""Real TCP lifecycle tests with a fake allocation backend; no NPU needed."""

import importlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def pool_module(monkeypatch):
    package = types.ModuleType("_shared_pool_ut")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module = importlib.import_module("_shared_pool_ut.shared_pool")

    class Buffer:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.fail_release = False

        def view(self, start, end):
            return types.SimpleNamespace(close=lambda: None)

        def close(self):
            if self.fail_release:
                raise RuntimeError("device release failed")
            self.closed = True

    monkeypatch.setattr(module, "MappedHostBuffer", Buffer)
    monkeypatch.setattr(module, "retry_pending_rollbacks", lambda path: None)
    yield module
    for name in list(sys.modules):
        if name.startswith("_shared_pool_ut."):
            sys.modules.pop(name)


@pytest.fixture
def owner(pool_module, tmp_path):
    owner = pool_module.SharedPoolOwner(tmp_path / "pool", "model-a", device="npu:0")
    owner.add("codes", 32, 256, torch.int8)
    owner.ready()
    yield owner
    owner.close(timeout=2)


def test_identity_rejection_does_not_hold_lease(pool_module, owner):
    with pytest.raises(ValueError, match="model-compatible"):
        pool_module.SharedPoolClient(owner.root, "model-b", device="npu:0")
    owner.close(timeout=1)
    assert owner.closed


def test_drain_waits_for_consumer_and_rejects_new_attach(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    try:
        with pytest.raises(TimeoutError):
            owner.close()
        assert not owner.buffers["codes"].closed
        with pytest.raises(ValueError, match="not ready"):
            pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    finally:
        client.close()
    owner.close(timeout=1)
    assert owner.buffers["codes"].closed


def test_failed_device_release_retains_lease_until_retry(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
    buffer = client.buffers["codes"]
    try:
        buffer.fail_release = True
        with pytest.raises(RuntimeError, match="device release failed"):
            client.close()
        with pytest.raises(TimeoutError):
            owner.close()
        assert not owner.buffers["codes"].closed
    finally:
        buffer.fail_release = False
        client.close()
    owner.close(timeout=1)


def test_failed_owner_release_does_not_skip_other_buffers(pool_module, owner):
    first = owner.buffers["codes"]
    second = pool_module.MappedHostBuffer()
    owner.buffers["second"] = second
    second.fail_release = True
    with pytest.raises(RuntimeError, match="device release failed"):
        owner.close()
    assert first.closed and not second.closed and not owner.closed
    second.fail_release = False
    owner.close()
    assert owner.closed and second.closed


def test_failed_rollback_keeps_consumer_lease_until_retry(pool_module, owner, monkeypatch):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)

    def fail_rollback(path):
        raise RuntimeError("rollback failed")

    monkeypatch.setattr(pool_module, "retry_pending_rollbacks", fail_rollback)
    with pytest.raises(RuntimeError, match="rollback failed"):
        client.close()
    assert client.buffers["codes"].closed and not client.closed
    with pytest.raises(TimeoutError):
        owner.close()
    monkeypatch.setattr(pool_module, "retry_pending_rollbacks", lambda path: None)
    client.close()
    owner.close(timeout=1)


def test_layout_rejected_before_mapping(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    try:
        with pytest.raises(ValueError, match="layout mismatch"):
            client.view("codes", 0, 16, rows=33, width=256, dtype=torch.int8)
        assert not client.buffers
    finally:
        client.close()


def test_consumer_reattaches_without_reloading_owner(pool_module, owner):
    original = owner.buffers["codes"]
    for _ in range(2):
        client = pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
        client.close()
    assert owner.buffers["codes"] is original
    assert not original.closed


def test_private_directory_required(pool_module, owner):
    owner.root.chmod(0o755)
    with pytest.raises(ValueError, match="mode0700"):
        pool_module.SharedPoolClient(owner.root, "model-a", device="npu:0")
    owner.root.chmod(0o700)


def test_launcher_waits_for_compatible_live_owner(pool_module, owner):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_launch.py"
    spec = importlib.util.spec_from_file_location("_engram_launcher_ut", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.wait_ready(owner.root, "model-a", 5)
    with pytest.raises(ValueError, match="identity mismatch"):
        launcher.wait_ready(owner.root, "model-b", 5)
    owner.close(timeout=1)
    with pytest.raises(TimeoutError):
        launcher.wait_ready(owner.root, "model-a", 1)


def test_launcher_forwards_termination_to_consumer(pool_module, owner, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        (model / name).write_text(json.dumps({}))
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_launch.py"
    spec = importlib.util.spec_from_file_location("_engram_launcher_signal_ut", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    owner.manifest["identity"] = launcher.model_identity(model)
    owner._publish_manifest()
    process = subprocess.Popen(
        [
            sys.executable,
            str(path),
            "--role",
            "consumer",
            "--model",
            str(model),
            "--pool-dir",
            str(owner.root),
            "--",
            sys.executable,
            "-u",
            "-c",
            "import time; print('started', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "started"
        process.terminate()
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()


def test_network_bootstrap_uses_separate_local_directories(pool_module, owner, tmp_path):
    owner.network_bootstrap = True
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_launch.py"
    spec = importlib.util.spec_from_file_location("_engram_network_bootstrap_ut", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    descriptor = owner.manifest["tensors"]["codes"]["descriptor"]
    (owner.root / descriptor).write_bytes(b"fake-vmm-fabric-handle")
    local = tmp_path / "consumer-local" / "pool"
    local.parent.mkdir()
    host, port = owner.manifest["endpoint"]
    launcher.bootstrap_local(local, "model-a", host, port, 5)
    assert (local / descriptor).read_bytes() == b"fake-vmm-fabric-handle"
    assert (local / (descriptor + ".ready")).is_file()
    assert local.stat().st_mode & 0o777 == 0o700
    client = pool_module.SharedPoolClient(local, "model-a", device="npu:0")
    try:
        client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
        with pytest.raises(TimeoutError):
            owner.close()
    finally:
        client.close()


def test_network_bootstrap_requires_opt_in_and_correct_model(pool_module, owner, tmp_path):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_launch.py"
    spec = importlib.util.spec_from_file_location("_engram_network_denied_ut", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    host, port = owner.manifest["endpoint"]
    local = tmp_path / "consumer-local"
    with pytest.raises(ValueError, match="rejected bootstrap"):
        launcher.bootstrap_local(local, "model-a", host, port, 5)
    assert not local.exists()
    owner.network_bootstrap = True
    with pytest.raises(ValueError, match="rejected bootstrap"):
        launcher.bootstrap_local(local, "other-model", host, port, 5)
    assert not local.exists()


def test_stale_network_descriptors_cannot_attach_to_new_generation(pool_module, owner, tmp_path):
    owner.network_bootstrap = True
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_launch.py"
    spec = importlib.util.spec_from_file_location("_engram_stale_generation_ut", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    descriptor = owner.manifest["tensors"]["codes"]["descriptor"]
    (owner.root / descriptor).write_bytes(b"fake-vmm-fabric-handle")
    host, port = owner.manifest["endpoint"]
    local = tmp_path / "consumer-local"
    launcher.bootstrap_local(local, "model-a", host, port, 5)
    owner.manifest["generation"] = "replacement-generation"
    with pytest.raises(ValueError, match="not ready or model-compatible"):
        pool_module.SharedPoolClient(local, "model-a", device="npu:0")


def test_consumer_launcher_bootstraps_in_separate_process(pool_module, owner, tmp_path):
    owner.network_bootstrap = True
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        (model / name).write_text("{}")
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_launch.py"
    spec = importlib.util.spec_from_file_location("_engram_network_process_ut", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    owner.manifest["identity"] = launcher.model_identity(model)
    owner._publish_manifest()
    descriptor = owner.manifest["tensors"]["codes"]["descriptor"]
    (owner.root / descriptor).write_bytes(b"fake-vmm-fabric-handle")
    local = tmp_path / "other-pod-local"
    host, port = owner.manifest["endpoint"]
    result = subprocess.run(
        [
            sys.executable,
            str(path),
            "--role",
            "consumer",
            "--model",
            str(model),
            "--pool-dir",
            str(local),
            "--owner-host",
            host,
            "--owner-port",
            str(port),
            "--",
            sys.executable,
            "-c",
            "print('service-started')",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == "service-started"
    assert (local / descriptor).read_bytes() == b"fake-vmm-fabric-handle"
    assert owner.root != local
