# SPDX-License-Identifier: Apache-2.0
"""Real TCP leases plus CPU-backed discontiguous address-table integration."""

import ctypes
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture
def stack(monkeypatch):
    package = types.ModuleType("_sharded_pool_ut")
    package.__path__ = [str(Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/engram_shard_layout.py"
    spec = importlib.util.spec_from_file_location("_owner_layout_ut", path)
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    monkeypatch.setitem(sys.modules, "vllm_ascend.engram_shard_layout", layout)
    pool = importlib.import_module(package.__name__ + ".shared_pool")
    shards = importlib.import_module(package.__name__ + ".shared_shards")
    registry = {}

    class Buffer:
        def __init__(self, path, rows, width, dtype, *, owner, **kwargs):
            self.path = Path(path)
            self.closed = False
            self.views = []
            if owner:
                self.data = torch.empty((rows, width), dtype=dtype)
                registry[str(self.path)] = self.data
                self.path.write_text(str(self.path))
            else:
                self.data = registry[self.path.read_text()]

        def publish(self, start, source):
            self.data[start : start + len(source)].copy_(source)

        def view(self, start, end, *, chunk_rows=1 << 22):
            result = types.SimpleNamespace(
                ptrs=torch.tensor(
                    [
                        self.data.data_ptr() + row * self.data.shape[1] * self.data.element_size()
                        for row in range(start, end, chunk_rows)
                    ],
                    dtype=torch.int64,
                ),
                closed=False,
            )

            def close():
                result.closed = True

            result.close = close
            self.views.append(result)
            return result

        def close(self):
            assert all(view.closed for view in self.views)
            self.closed = True

    monkeypatch.setattr(pool, "MappedHostBuffer", Buffer)
    monkeypatch.setattr(pool, "retry_pending_rollbacks", lambda path: None)
    launcher_path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_launch.py"
    spec = importlib.util.spec_from_file_location("_sharded_launcher_ut", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    yield types.SimpleNamespace(pool=pool, shards=shards, layout=layout, launcher=launcher)
    for name in list(sys.modules):
        if name.startswith(package.__name__ + "."):
            sys.modules.pop(name)


@pytest.fixture(params=[2, 4])
def deployment(stack, tmp_path, request):
    owners = [
        stack.pool.SharedPoolOwner(tmp_path / f"backing-{i}", "model-a", device="npu:0", network_bootstrap=True)
        for i in range(request.param)
    ]
    membership = dict(
        schema=1,
        deployment_id="test",
        chunk_rows=8,
        owners=[dict(host=o.manifest["endpoint"][0], port=o.manifest["endpoint"][1]) for o in owners],
    )
    rows = 37
    for rank, owner in enumerate(owners):
        owner.manifest["sharding"] = dict(
            layout_id=stack.layout.layout_identity(membership, "model-a"), owner_rank=rank, membership=membership
        )
        start, end = stack.layout.row_ranges(rows, len(owners), 8)[rank]
        for layer in (1, 14):
            for kind, width, dtype in [("weight", 256, torch.int8), ("scale", 8, torch.float32)]:
                buffer = owner.add(
                    f"layers.{layer}.{kind}", end - start, width, dtype, global_rows=rows, row_start=start
                )
                values = torch.arange(start, end, dtype=torch.int64).remainder(127).to(dtype)
                buffer.publish(0, values[:, None].expand(-1, width))
        owner.ready()
    root = tmp_path / "consumer"
    stack.launcher.bootstrap_shards(root, "model-a", membership, 5)
    yield types.SimpleNamespace(owners=owners, root=root, membership=membership, rows=rows)
    for owner in owners:
        owner.close(timeout=2)


def test_two_layers_share_sessions_and_global_addresses(stack, deployment):
    pool = stack.shards.ShardedPool(deployment.root, "model-a", device="npu:0")
    first, second = pool.acquire(), pool.acquire()
    try:
        for lease, layer in [(first, 1), (second, 14)]:
            for suffix, width, dtype in [("weight", 256, torch.int8), ("scale", 8, torch.float32)]:
                # Non-chunk-aligned head start reproduces the old concatenation hazard.
                view = lease.view(f"layers.{layer}.{suffix}", 5, 35, rows=deployment.rows, width=width, dtype=dtype)
                assert view.global_rows and tuple(view.tensor.shape) == (30, width)
                for row in range(5, 35):
                    pointer = (
                        view.ptrs[row // 8].item() + (row % 8) * width * torch.empty((), dtype=dtype).element_size()
                    )
                    value = (
                        ctypes.c_int8.from_address(pointer).value
                        if dtype == torch.int8
                        else ctypes.c_float.from_address(pointer).value
                    )
                    assert value == row
        assert len(pool.clients) == len(deployment.owners)
        assert all(len(owner._leases) == 1 for owner in deployment.owners)
        first.close()
        assert not pool.closed
        with pytest.raises(TimeoutError):
            deployment.owners[0].close()
    finally:
        first.close()
        second.close()
    assert pool.closed
    assert all(not owner._leases for owner in deployment.owners)


def test_layout_mismatch_rejected_before_any_mapping(stack, deployment):
    owner = deployment.owners[-1]
    key = "layers.14.scale"
    owner.manifest["tensors"][key]["row_start"] += 1
    with pytest.raises(ValueError, match="consistent table"):
        stack.shards.ShardedPool(deployment.root, "model-a", device="npu:0")
    assert all(not o._leases for o in deployment.owners)
    assert all(not buffer.views for o in deployment.owners for buffer in o.buffers.values())


def test_generation_change_rejected_before_mapping(stack, deployment):
    owner = deployment.owners[-1]
    owner.manifest["generation"] = "new-generation"
    owner._publish_manifest()
    with pytest.raises(ValueError):
        stack.shards.ShardedPool(deployment.root, "model-a", device="npu:0")
    assert all(not o._leases for o in deployment.owners)


def test_membership_rejects_duplicates(stack):
    peer = dict(host="localhost", port=1)
    with pytest.raises(ValueError, match="Duplicate"):
        stack.layout.validate_owner_set(dict(schema=1, deployment_id="test", owners=[peer, peer]))


def test_checkpoint_loader_reads_only_owner_rows(stack, tmp_path):
    import ast
    import threading

    from safetensors import safe_open
    from safetensors.torch import save_file

    rows = 37
    model = tmp_path / "checkpoint"
    model.mkdir()
    tensors = {}
    for layer in (1, 14):
        tensors[f"layers.{layer}.engram.embed.weight"] = (
            torch.arange(rows * 256).remainder(127).to(torch.int8).reshape(rows, 256)
        )
        tensors[f"layers.{layer}.engram.embed.scale"] = torch.arange(rows * 8).float().reshape(rows, 8)
    save_file(tensors, model / "weights.safetensors")
    (model / "config.json").write_text(
        json.dumps(
            dict(text_config=dict(engram_head_dim=256, engram_layer_ids=[1, 14], engram_num_embeddings=[rows, rows]))
        )
    )
    (model / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(dict(weight_map={key: "weights.safetensors" for key in tensors}))
    )
    # Execute the actual loader, excluding module CLI/NPU initialization.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_tables")
    scope = dict(Path=Path, json=json, torch=torch, safe_open=safe_open, row_ranges=stack.layout.row_ranges)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    membership = dict(owners=[{}, {}, {}, {}], chunk_rows=8)
    all_data = {key: [] for key in tensors}
    for rank, (start, end) in enumerate(stack.layout.row_ranges(rows, 4, 8)):
        owner = stack.pool.SharedPoolOwner(tmp_path / f"loader-{rank}", "model-a", device="npu:0")
        try:
            scope["load_tables"](owner, model, threading.Event(), membership=membership, owner_rank=rank)
            for key, buffer in owner.buffers.items():
                assert len(buffer.data) == end - start
                assert torch.equal(buffer.data, tensors[key][start:end])
                assert owner.manifest["tensors"][key]["row_start"] == start
                all_data[key].append(buffer.data.clone())
        finally:
            owner.close()
    for key, pieces in all_data.items():
        assert torch.equal(torch.cat(pieces), tensors[key])


def load_registry():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_registry.py"
    spec = importlib.util.spec_from_file_location("_owner_registry_ut", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_discovers_all_owners_without_model_readiness(stack):
    import concurrent.futures
    import threading

    module = load_registry()
    registry = module.OwnerRegistry(4)
    server = module.make_server("127.0.0.1", 0, registry)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    stack.launcher.discover_owners,
                    url,
                    "a" * 64,
                    5,
                    host="127.0.0.1",
                    port=8100 + i,
                    node_id=f"pod-{3 - i}",
                )
                for i in range(4)
            ]
            results = [future.result() for future in futures]
        assert sorted(rank for _, rank in results) == list(range(4))
        assert all(spec == results[0][0] for spec, _ in results)
        spec, rank = stack.launcher.discover_owners(url, "a" * 64, 5)
        assert rank is None and spec == results[0][0]
        assert len(spec["owners"]) == 4
        with pytest.raises(ValueError, match="HTTP 409"):
            stack.launcher.discover_owners(url, "a" * 64, 5, host="127.0.0.2", port=8100, node_id="replacement")
        with pytest.raises(ValueError, match="checkpoint mismatch"):
            stack.launcher.discover_owners(url, "b" * 64, 5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_registry_rejects_conflicting_registration():
    registry = load_registry().OwnerRegistry(2)
    entry = dict(node_id="pod-0", host="127.0.0.1", port=29699, identity="a" * 64)
    assert registry.register(entry)["state"] == "collecting"
    with pytest.raises(ValueError, match="endpoint"):
        registry.register(dict(entry, port=29700))
    with pytest.raises(ValueError, match="duplicated"):
        registry.register(dict(entry, node_id="pod-1"))
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        registry.register(dict(entry, node_id="pod-1", host="127.0.0.2", identity="b" * 64))
    assert registry.status()["registered"] == 1


def test_registry_cli_works_without_device_imports():
    import subprocess

    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram_registry.py"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--owners",
            "2",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 7, result.stderr


def test_pending_views_do_not_attach_or_read_table(stack, deployment):
    import time

    manifest_path = deployment.root / "manifest.json"
    ready = json.loads(manifest_path.read_text())
    loading = dict(ready, state="loading", expires_at=time.time() + 10)
    manifest_path.write_text(json.dumps(loading))
    pool = stack.shards.ShardedPool(deployment.root, "model-a", device="cpu")
    lease = pool.acquire()
    view = lease.view("layers.1.weight", 5, 35, rows=37, width=256, dtype=torch.int8)
    parameter = torch.nn.Parameter(view.tensor, requires_grad=False)
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    model._shared_pool_client = lease
    try:
        assert not pool.clients and not pool.attached
        assert not any(owner._leases for owner in deployment.owners)
        assert view.ptrs is None and view.tensor.device.type == "meta"
        assert parameter.shape == (30, 256)
        manifest_path.write_text(json.dumps(ready))
        stack.shards.finalize_shared_engram(model, model)
        assert pool.attached
        assert all(len(owner._leases) == 1 for owner in deployment.owners)
        assert model.weight is parameter and view.tensor.device.type == "meta"
        for row in (0, 7, 8, 23, 24, 36):
            address = view.ptrs[row // 8].item() + (row % 8) * 256
            assert ctypes.c_int8.from_address(address).value == row % 127
        pointers = view.ptrs
        stack.shards.finalize_shared_engram(model)
        assert view.ptrs is pointers
    finally:
        lease.close()
    assert not any(owner._leases for owner in deployment.owners)


@pytest.mark.parametrize("failure", ["failed", "timeout", "membership", "generation"])
def test_pending_failure_never_leaves_a_mapping(stack, deployment, failure):
    import time

    path = deployment.root / "manifest.json"
    ready = json.loads(path.read_text())
    loading = dict(ready, state="loading", expires_at=time.time() + 10)
    path.write_text(json.dumps(loading))
    pool = stack.shards.ShardedPool(deployment.root, "model-a", device="cpu")
    lease = pool.acquire()
    view = lease.view("layers.1.weight", 0, 37, rows=37, width=256, dtype=torch.int8)
    if failure == "failed":
        loading["state"] = "failed"
    elif failure == "timeout":
        pool.deadline = time.time() - 1
    elif failure == "membership":
        loading["membership"]["deployment_id"] = "replacement"
    else:
        loading = ready
        loading["generations"][0] = "replacement"
    path.write_text(json.dumps(loading))
    try:
        with pytest.raises((RuntimeError, TimeoutError, ValueError)):
            pool.finalize()
        assert view.ptrs is None
        assert not any(owner._leases for owner in deployment.owners)
    finally:
        lease.close()


def test_pending_bootstrap_publishes_ready_atomically(stack, deployment, tmp_path):
    root = tmp_path / "pending-consumer"
    plan = stack.launcher.begin_shard_loading(root, "model-a", deployment.membership, 5)
    assert plan["state"] == "loading"
    with pytest.raises(FileExistsError):
        stack.launcher.begin_shard_loading(root, "model-a", deployment.membership, 5)
    stack.launcher.bootstrap_shards(root, "model-a", deployment.membership, 5, pending=True)
    ready = json.loads((root / "manifest.json").read_text())
    assert ready["state"] == "ready" and ready["membership"] == plan["membership"]
    pool = stack.shards.ShardedPool(root, "model-a", device="cpu")
    pool.close()


def test_closed_pending_view_is_not_reopened(stack, deployment):
    import time

    path = deployment.root / "manifest.json"
    ready = json.loads(path.read_text())
    path.write_text(json.dumps(dict(ready, state="loading", expires_at=time.time() + 10)))
    pool = stack.shards.ShardedPool(deployment.root, "model-a", device="cpu")
    first, second = pool.acquire(), pool.acquire()
    view = first.view("layers.1.weight", 0, 37, rows=37, width=256, dtype=torch.int8)
    first.close()
    path.write_text(json.dumps(ready))
    try:
        pool.finalize()
        assert view.ptrs is None
    finally:
        second.close()


def test_overlap_launcher_starts_service_before_owners_ready(stack, deployment, tmp_path):
    import concurrent.futures
    import subprocess

    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        (model / name).write_text("{}")
    identity = stack.launcher.model_identity(model)
    for rank, owner in enumerate(deployment.owners):
        owner.manifest.update(identity=identity, state="loading")
        owner.manifest["sharding"]["layout_id"] = stack.layout.layout_identity(deployment.membership, identity)
        owner._publish_manifest()
    spec = tmp_path / "owners.json"
    spec.write_text(json.dumps(deployment.membership))
    root = tmp_path / "overlap"
    source = Path(__file__).resolve().parents[3]
    launcher = source / "vllm_ascend/shared_engram_launch.py"
    # Isolate the real stdlib launcher from the CPU environment's NPU package imports.
    entry = (
        "import sys,types,runpy; p=types.ModuleType('vllm_ascend'); "
        f"p.__path__=[{str(source / 'vllm_ascend')!r}]; sys.modules['vllm_ascend']=p; "
        f"sys.argv[0]={str(launcher)!r}; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    service = (
        "import json,time,pathlib; "
        f"p=pathlib.Path({str(root / 'manifest.json')!r}); "
        "assert json.loads(p.read_text())['state']=='loading'; print('started-before-ready',flush=True); "
        "exec(\"for _ in range(800):\\n if json.loads(p.read_text())['state']=='ready': break\\n time.sleep(.01)\\n"
        "else: raise RuntimeError('ready gate timed out')\"); print('ready',flush=True)"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            entry,
            "--role",
            "consumer",
            "--model",
            str(model),
            "--pool-dir",
            str(root),
            "--owner-set",
            str(spec),
            "--overlap-load",
            "--timeout",
            "8",
            "--",
            sys.executable,
            "-c",
            service,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            started = executor.submit(process.stdout.readline)
            try:
                assert started.result(timeout=5).strip() == "started-before-ready"
            finally:
                # A serial-start regression must fail promptly instead of deadlocking the test.
                for owner in deployment.owners:
                    owner.ready()
        output, error = process.communicate(timeout=12)
        assert process.returncode == 0, error
        assert output.strip() == "ready"
        assert json.loads((root / "manifest.json").read_text())["state"] == "ready"
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=12)


@pytest.fixture
def lease_table():
    import ast

    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/embedding.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendParallelEngramEmbedding")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "close_host_offload")
    scope = dict(torch=torch, nn=torch.nn)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)

    class Table(torch.nn.Module):
        close_host_offload = scope["close_host_offload"]

        def __init__(self, lease, layer):
            super().__init__()
            self._shared_pool_client = lease
            self.storage_dtype, self.scale_dtype = torch.int8, torch.float32
            self._codes_uva = lease.view(f"layers.{layer}.weight", 0, 37, rows=37, width=256, dtype=torch.int8)
            self._scales_uva = lease.view(f"layers.{layer}.scale", 0, 37, rows=37, width=8, dtype=torch.float32)
            self.weight = torch.nn.Parameter(self._codes_uva.tensor, requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(self._scales_uva.tensor, requires_grad=False)

    return Table


def test_close_synchronizes_then_releases_real_leases(stack, deployment, lease_table, monkeypatch):
    pool = stack.shards.ShardedPool(deployment.root, "model-a", device="cpu")
    model = torch.nn.ModuleList([lease_table(pool.acquire(), layer) for layer in (1, 14)])
    calls = []

    def synchronize():
        assert all(owner._leases for owner in deployment.owners)
        assert all(table._codes_uva.ptrs is not None for table in model)
        calls.append("sync")

    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(synchronize=synchronize), raising=False)
    stack.shards.close_shared_engram(model, model)
    assert calls == ["sync"]
    assert pool.closed and not any(owner._leases for owner in deployment.owners)
    assert all(table.weight.numel() == 0 and table._shared_pool_client is None for table in model)
    stack.shards.close_shared_engram(model)
    assert calls == ["sync"]


def test_failed_synchronize_keeps_mapping_and_lease(stack, deployment, lease_table, monkeypatch):
    pool = stack.shards.ShardedPool(deployment.root, "model-a", device="cpu")
    table = lease_table(pool.acquire(), 1)

    def failed():
        raise RuntimeError("device work not finished")

    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(synchronize=failed), raising=False)
    with pytest.raises(RuntimeError, match="device work not finished"):
        stack.shards.close_shared_engram(table)
    assert table._codes_uva.ptrs is not None and not pool.closed
    assert all(owner._leases for owner in deployment.owners)
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    stack.shards.close_shared_engram(table)
    assert not any(owner._leases for owner in deployment.owners)


def test_cleanup_error_attempts_other_tables_and_can_retry(stack, deployment, lease_table, monkeypatch):
    pool = stack.shards.ShardedPool(deployment.root, "model-a", device="cpu")
    first, second = (lease_table(pool.acquire(), layer) for layer in (1, 14))
    model = torch.nn.ModuleList([first, second])
    original = first.close_host_offload

    def failed():
        raise RuntimeError("unmap failed")

    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(synchronize=lambda: None), raising=False)
    monkeypatch.setattr(first, "close_host_offload", failed)
    with pytest.raises(RuntimeError, match="Failed to close 1"):
        stack.shards.close_shared_engram(model)
    assert second._shared_pool_client is None
    assert first._codes_uva.ptrs is not None and pool.references == 1
    assert all(owner._leases for owner in deployment.owners)
    monkeypatch.setattr(first, "close_host_offload", original)
    stack.shards.close_shared_engram(model)
    assert pool.closed and not any(owner._leases for owner in deployment.owners)


@pytest.mark.parametrize("failure", ["draining", "generation", "identity", "token"])
def test_owner_health_rejects_changed_active_generation(stack, deployment, failure):
    checks = stack.launcher.shard_owner_checks(deployment.root, "model-a")
    stack.launcher.check_shard_owners(checks, "model-a")
    owner = deployment.owners[-1]
    if failure == "token":
        checks[-1]["token"] = "wrong-capability"
    elif failure == "generation":
        owner.manifest["generation"] = "replacement"
    elif failure == "identity":
        owner.manifest["identity"] = "replacement"
    else:
        owner.manifest["state"] = "draining"
    with pytest.raises(RuntimeError, match="unavailable, draining or replaced"):
        stack.launcher.check_shard_owners(checks, "model-a")


def test_launcher_stops_service_when_remote_owner_drains(stack, deployment, tmp_path):
    import concurrent.futures
    import os
    import subprocess

    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        (model / name).write_text("{}")
    identity = stack.launcher.model_identity(model)
    for owner in deployment.owners:
        owner.manifest["identity"] = identity
        owner.manifest["sharding"]["layout_id"] = stack.layout.layout_identity(deployment.membership, identity)
        owner._publish_manifest()
    spec = tmp_path / "owners.json"
    spec.write_text(json.dumps(deployment.membership))
    source = Path(__file__).resolve().parents[3]
    launcher = source / "vllm_ascend/shared_engram_launch.py"
    entry = (
        "import sys,types,runpy; p=types.ModuleType('vllm_ascend'); "
        f"p.__path__=[{str(source / 'vllm_ascend')!r}]; sys.modules['vllm_ascend']=p; "
        f"sys.argv[0]={str(launcher)!r}; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            entry,
            "--role",
            "consumer",
            "--model",
            str(model),
            "--pool-dir",
            str(tmp_path / "supervised"),
            "--owner-set",
            str(spec),
            "--",
            sys.executable,
            "-u",
            "-c",
            "import os,time; print(os.getpid(),flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            child = int(executor.submit(process.stdout.readline).result(timeout=5))
        deployment.owners[-1].manifest["state"] = "draining"
        _, error = process.communicate(timeout=10)
        assert process.returncode != 0
        assert "Shared Engram became unavailable; service stopped" in error
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        assert all(owner.manifest["state"] == "ready" for owner in deployment.owners[:-1])
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=12)
