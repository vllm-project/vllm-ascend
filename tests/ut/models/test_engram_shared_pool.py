# SPDX-License-Identifier: Apache-2.0
"""Real TCP lifecycle tests with a fake allocation backend; no NPU needed."""

import ast
import importlib
import importlib.util
import socket
import sys
import threading
import types
from contextlib import ExitStack, contextmanager
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

    class Runtime:
        @staticmethod
        def acquire(*args, **kwargs):
            return types.SimpleNamespace(rank=0, close=lambda: None)

    monkeypatch.setattr(module, "MemfabricRuntime", Runtime)
    yield module
    for name in list(sys.modules):
        if name.startswith("_shared_pool_ut."):
            sys.modules.pop(name)


def free_endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


@pytest.fixture
def owner(pool_module, tmp_path):
    owner = pool_module.SharedPoolOwner(free_endpoint(), "model-a", device="npu:0")
    owner.add("codes", 32, 256, torch.int8)
    owner.ready()
    yield owner
    owner.close(timeout=2)


def test_identity_rejection_does_not_hold_lease(pool_module, owner):
    with pytest.raises(ValueError, match="model-compatible"):
        pool_module.SharedPoolClient(owner.endpoint, "model-b", device="npu:0")
    owner.close(timeout=1)
    assert owner.closed


def test_endpoint_discovery_requires_ready_and_identity(pool_module, tmp_path):
    owner = pool_module.SharedPoolOwner(free_endpoint(), "model-a", device="npu:0")
    try:
        owner.add("codes", 32, 256, torch.int8)
        with pytest.raises(ValueError, match="not ready"):
            pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
        owner.ready()
        client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
        assert client.manifest["generation"] == owner.generation
        assert "token" not in client.manifest
        client.close()
    finally:
        owner.close(timeout=1)


def test_new_owner_has_new_generation(pool_module):
    endpoint = free_endpoint()
    owner = pool_module.SharedPoolOwner(endpoint, "model-a", device="npu:0")
    first = owner.generation
    owner.close(timeout=1)
    other = pool_module.SharedPoolOwner(free_endpoint(), "model-a", device="npu:0")
    try:
        assert other.generation != first
    finally:
        other.close(timeout=1)


def test_default_store_never_uses_control_port(pool_module, monkeypatch):
    endpoint = free_endpoint()
    control_port = int(endpoint.rsplit(":", 1)[1])
    real_socket = socket.socket

    class Socket:
        def __init__(self, *args, **kwargs):
            self.socket = real_socket(*args, **kwargs)

        def bind(self, address):
            if address[1] == 0:
                try:
                    return self.socket.bind((address[0], control_port))
                except OSError:
                    pass
            return self.socket.bind(address)

        def __getattr__(self, name):
            return getattr(self.socket, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.socket.close()

    monkeypatch.setattr(pool_module.socket, "socket", Socket)
    owner = pool_module.SharedPoolOwner(endpoint, "model-a", device="npu:0")
    try:
        assert owner.manifest["memfabric"]["store_url"] != f"tcp://{endpoint}"
    finally:
        owner.close(timeout=1)


def test_same_store_endpoint_rejected_without_leaking_listener(pool_module):
    endpoint = free_endpoint()
    with pytest.raises(ValueError, match="endpoints must differ"):
        pool_module.SharedPoolOwner(endpoint, "model-a", device="npu:0", store_url=f"tcp://{endpoint}")
    with socket.socket() as probe:
        probe.bind(pool_module.parse_endpoint(endpoint))


def test_failed_release_ack_wakes_draining_owner(pool_module, owner, monkeypatch):
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    original_send = pool_module._send
    ack_attempted = threading.Event()
    completed = threading.Event()
    notified = threading.Event()
    waiting = threading.Event()
    original_wait = owner._condition.wait

    def wait(timeout):
        waiting.set()
        result = original_wait(timeout)
        if result:
            notified.set()
        return result

    def send(wire, message):
        if message == {"released": True}:
            ack_attempted.set()
            raise OSError("connection lost while acknowledging release")
        return original_send(wire, message)

    monkeypatch.setattr(pool_module, "_send", send)
    monkeypatch.setattr(owner._condition, "wait", wait)

    def close_owner():
        owner.close(timeout=2)
        completed.set()

    drain = threading.Thread(target=close_owner)
    drain.start()
    assert waiting.wait(1)
    pool_module._send(client._wire, {"op": "release", "generation": client.manifest["generation"]})
    assert ack_attempted.wait(1)
    assert notified.wait(1), "owner was not notified when release acknowledgement failed"
    assert completed.wait(2)
    drain.join()
    client._wire.close()
    client._conn.close()


def test_blocked_release_ack_does_not_hold_owner_lock(pool_module, owner, monkeypatch):
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    original_send = pool_module._send
    ack_started = threading.Event()
    unblock_ack = threading.Event()

    def send(wire, message):
        if message == {"released": True}:
            ack_started.set()
            assert unblock_ack.wait(3)
        return original_send(wire, message)

    monkeypatch.setattr(pool_module, "_send", send)
    pool_module._send(client._wire, {"op": "release", "generation": client.manifest["generation"]})
    assert ack_started.wait(1)
    completed = threading.Event()

    def close_owner():
        owner.close(timeout=1)
        completed.set()

    drain = threading.Thread(target=close_owner)
    drain.start()
    try:
        assert completed.wait(1.5), "blocked acknowledgement held the owner lock"
    finally:
        unblock_ack.set()
        drain.join(timeout=2)
        client._wire.close()
        client._conn.close()


def test_stale_generation_release_does_not_retire_lease(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    pool_module._send(client._wire, {"op": "release", "generation": "stale"})
    with pytest.raises(ConnectionError):
        pool_module._receive(client._wire)
    with pytest.raises(TimeoutError):
        owner.close()
    client._wire.close()
    client._conn.close()
    # Fake-storage test only: the simulated reader never created a device mapping.
    with owner._condition:
        owner._leases.clear()
    owner.close(timeout=1)


def test_owner_cli_uses_shared_backend_and_endpoint(pool_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.models.deepseek_v41.engram.shared_pool", pool_module)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    package = types.ModuleType("vllm_ascend")
    package.__path__ = []
    ops = types.ModuleType("vllm_ascend.ops")
    monkeypatch.setattr(package, "ops", ops, raising=False)
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops)
    monkeypatch.setattr(torch, "npu", types.SimpleNamespace(set_device=lambda _: None), raising=False)
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/shared_engram.py"
    spec = importlib.util.spec_from_file_location("_shared_engram_cli_ut", path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    monkeypatch.setattr(cli, "threading", types.SimpleNamespace(Event=lambda: types.SimpleNamespace(wait=lambda: None)))
    monkeypatch.setattr(cli, "checkpoint_identity", lambda _: "model-a")
    monkeypatch.setattr(cli, "load_tables", lambda *args: None)
    captured = []

    class Owner:
        closed = False
        generation = "generation"

        def __init__(self, endpoint, identity, **kwargs):
            captured.append((endpoint, identity, kwargs))

        def ready(self):
            pass

        def close(self, timeout):
            self.closed = True

    monkeypatch.setattr(cli, "SharedPoolOwner", Owner)
    monkeypatch.setattr(
        sys, "argv", ["shared_engram", "--backend", "a3", "--model", "model", "--endpoint", "owner:29500"]
    )
    cli.main()
    assert captured == [("owner:29500", "model-a", {"device": "npu:0", "store_url": None, "world_size": 65})]
    monkeypatch.setattr(
        sys, "argv", ["shared_engram", "--backend", "a2", "--model", "model", "--endpoint", "owner:29500"]
    )
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["shared_engram", "--backend", "a3", "--model", "model"])
    with pytest.raises(SystemExit):
        cli.main()


def test_drain_waits_for_consumer_and_rejects_new_attach(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    try:
        with pytest.raises(TimeoutError):
            owner.close()
        assert not owner.buffers["codes"].closed
        with pytest.raises(ValueError, match="not ready"):
            pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    finally:
        client.close()
    owner.close(timeout=1)
    assert owner.buffers["codes"].closed


def test_failed_device_release_retains_lease_until_retry(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
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


def test_client_failure_releases_other_buffers_and_retries(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
    failing = client.buffers["codes"]
    other = pool_module.MappedHostBuffer()
    client.buffers["other"] = other
    failing.fail_release = True
    with pytest.raises(RuntimeError, match="device release failed"):
        client.close()
    assert other.closed and not failing.closed and not client.closed
    with pytest.raises(TimeoutError):
        owner.close()
    failing.fail_release = False
    client.close()
    assert client.closed and failing.closed
    owner.close(timeout=1)


def test_failed_mapping_construction_keeps_client_lease_until_cleanup(pool_module, owner, monkeypatch):
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    pending = pool_module.MappedHostBuffer()
    pending.fail_release = True

    def fail_mapping(*args, **kwargs):
        raise pool_module.MappingCleanupError(pending)

    monkeypatch.setattr(pool_module, "MappedHostBuffer", fail_mapping)
    with pytest.raises(pool_module.MappingCleanupError):
        client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
    with pytest.raises(RuntimeError, match="device release failed"):
        client.close()
    with pytest.raises(TimeoutError):
        owner.close()
    pending.fail_release = False
    client.close()
    owner.close(timeout=1)
    assert client.closed and owner.closed


def test_failed_owner_mapping_construction_is_retained_for_retry(pool_module, tmp_path, monkeypatch):
    owner = pool_module.SharedPoolOwner(free_endpoint(), "model-a", device="npu:0")
    pending = pool_module.MappedHostBuffer()
    pending.fail_release = True

    def fail_mapping(*args, **kwargs):
        raise pool_module.MappingCleanupError(pending)

    monkeypatch.setattr(pool_module, "MappedHostBuffer", fail_mapping)
    with pytest.raises(pool_module.MappingCleanupError):
        owner.add("codes", 32, 256, torch.int8)
    with pytest.raises(RuntimeError, match="device release failed"):
        owner.close()
    assert not owner.closed
    pending.fail_release = False
    owner.close()
    assert owner.closed and pending.closed


def test_owner_failure_releases_other_buffers_and_retries(pool_module, owner):
    failing = owner.buffers["codes"]
    other = pool_module.MappedHostBuffer()
    owner.buffers["other"] = other
    failing.fail_release = True
    with pytest.raises(RuntimeError, match="device release failed"):
        owner.close()
    assert other.closed and not failing.closed and not owner.closed
    failing.fail_release = False
    owner.close(timeout=1)
    assert owner.closed and failing.closed


def test_model_construction_rolls_back_first_layer_when_second_attach_fails(pool_module, owner):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    model = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41Model"
    )
    init = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    block = next(
        n
        for n in init.body
        if isinstance(n, ast.With)
        and isinstance(n.items[0].context_expr, ast.Call)
        and isinstance(n.items[0].context_expr.func, ast.Name)
        and n.items[0].context_expr.func.id == "ExitStack"
    )
    code = ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[]))
    calls = []

    class Embedding:
        def __init__(self, rows, dim, heads, slot, **kwargs):
            self.client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
            if slot == 1:
                self.client.close()
                raise RuntimeError("second attach failed")

        def bind_checkpoint(self, *args):
            pass

        def close_host_offload(self):
            calls.append("close first")
            self.client.close()

    layer = types.SimpleNamespace(engram=types.SimpleNamespace(embed_tokens=None))
    model_obj = types.SimpleNamespace(
        engram_layout=types.SimpleNamespace(primes=[[[32]], [[32]]]),
        engram_weight_root="unused",
        engram_dp_shared_memory=False,
        layers=[layer, layer],
    )
    config = types.SimpleNamespace(engram_layer_ids=[0, 1], engram_num_embeddings=[32, 32], engram_head_dim=256)
    vc = types.SimpleNamespace(load_config=types.SimpleNamespace(load_format="real"), quant_config=object())
    ns = dict(
        ExitStack=ExitStack,
        self=model_obj,
        config=config,
        vllm_config=vc,
        shared_pool_endpoint="pool",
        shared_pool_identity="model-a",
        cpu_offload=True,
        torch=torch,
        AscendParallelEngramEmbedding=Embedding,
        preflight_engram_checkpoint=lambda *args: None,
        engram_storage_dtype=lambda *args: torch.int8,
        get_current_hardware_profile=lambda: types.SimpleNamespace(supports=lambda _: False),
        HardwareCapability=types.SimpleNamespace(ENGRAM_MXFP8=object()),
        rollback_shared_engram=pool_module.rollback_shared_engram,
    )
    with pytest.raises(RuntimeError, match="second attach failed"):
        exec(compile(code, str(path), "exec"), ns)
    assert calls == ["close first"]
    owner.close(timeout=1)
    assert owner.closed


def test_merged_model_initializes_rotation_and_registers_replay():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    model = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41Model"
    )
    init = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    block = next(
        n
        for n in init.body
        if isinstance(n, ast.With)
        and isinstance(n.items[0].context_expr, ast.Call)
        and isinstance(n.items[0].context_expr.func, ast.Name)
        and n.items[0].context_expr.func.id == "ExitStack"
    )
    tree = ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[]))
    swa = object()
    attn = types.SimpleNamespace(dsa_attn=types.SimpleNamespace(swa_cache_layer=swa))
    layer = types.SimpleNamespace(engram=types.SimpleNamespace(embed_tokens=None), self_attn=attn)
    model_obj = types.SimpleNamespace(
        engram_layout=types.SimpleNamespace(primes=[[[32]]]),
        engram_weight_root="unused",
        engram_root="unused",
        engram_dp_shared_memory=False,
        layers=[layer],
        has_engram=True,
        device="cpu",
        end_layer=1,
    )
    model_obj.register_buffer = lambda name, value, **kwargs: setattr(model_obj, name, value)
    config = types.SimpleNamespace(
        engram_layer_ids=[0], engram_num_embeddings=[32], engram_head_dim=256, hidden_size=64
    )
    context: dict[str, object] = {}
    vc = types.SimpleNamespace(
        load_config=types.SimpleNamespace(load_format="real"),
        quant_config=object(),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=types.SimpleNamespace(max_cudagraph_capture_size=32, static_forward_context=context),
    )
    rotation = torch.stack([torch.eye(32), -torch.eye(32)])
    replay = object()

    class Embedding:
        def __init__(self, *args, **kwargs):
            pass

        def bind_checkpoint(self, *args):
            pass

    ns = dict(
        ExitStack=ExitStack,
        self=model_obj,
        config=config,
        vllm_config=vc,
        shared_pool_endpoint=None,
        shared_pool_identity=None,
        cpu_offload=True,
        torch=torch,
        AscendParallelEngramEmbedding=Embedding,
        preflight_engram_checkpoint=lambda *args: None,
        engram_storage_dtype=lambda *args: torch.int8,
        get_current_hardware_profile=lambda: types.SimpleNamespace(supports=lambda _: False),
        HardwareCapability=types.SimpleNamespace(ENGRAM_MXFP8=object()),
        get_ascend_config=lambda: types.SimpleNamespace(multistream_engram_overlap=False),
        get_rotation_path=lambda _: "rotation",
        load_engram_rotation_block=lambda *args: rotation,
        create_engram_hash_state=lambda *args: "hash",
        make_decoder_replay=lambda *args: replay,
        prefix="model",
        rollback_shared_engram=lambda fn: fn(),
    )
    exec(compile(tree, str(path), "exec"), ns)
    torch.testing.assert_close(model_obj.engram_rotation, rotation)
    assert model_obj.engram_hash == "hash" and model_obj.decoder_replay_layers is replay
    assert context == {"model.decoder_replay": replay}


def test_shutdown_retry_after_model_deletion_releases_tcp_lease(pool_module, owner):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    shutdown = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    embedding = types.SimpleNamespace(_shared_pool_client=client, close_host_offload=client.close)

    class Parent:
        def shutdown(self):
            if hasattr(self, "model"):
                del self.model
                raise RuntimeError("empty cache failed")

    runner_class = ast.parse("class Runner(Parent):\n    pass").body[0]
    assert isinstance(runner_class, ast.ClassDef)
    runner_class.body = [shutdown]
    tree = ast.fix_missing_locations(ast.Module(body=[runner_class], type_ignores=[]))
    ns = dict(Parent=Parent, C8KVBlockZeroer=type("C8KVBlockZeroer", (), {}))
    exec(compile(tree, str(path), "exec"), ns)
    runner = ns["Runner"]()
    runner.model = types.SimpleNamespace(modules=lambda: [embedding])
    with pytest.raises(RuntimeError, match="empty cache failed"):
        runner.shutdown()
    assert not client.closed
    runner.shutdown()
    assert client.closed
    owner.close(timeout=1)
    assert owner.closed


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_outer_model_init_failure_releases_attached_engram(pool_module, owner, cleanup_fails):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AscendDeepseekV41LLMForCausalLM")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    init.args.kwonlyargs[0].annotation = None
    runner_class = ast.parse("class Model(Base):\n    pass").body[0]
    assert isinstance(runner_class, ast.ClassDef)
    runner_class.body = [init]
    tree = ast.fix_missing_locations(ast.Module(body=[runner_class], type_ignores=[]))
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")

    def close():
        if cleanup_fails:
            raise RuntimeError("cleanup failed")
        client.close()

    embedding = types.SimpleNamespace(_shared_pool_client=client, close_host_offload=close)
    inner = types.SimpleNamespace(modules=lambda: [embedding])

    def fail_head(*args, **kwargs):
        raise RuntimeError("head failed")

    ns = dict(
        Base=object,
        normalize_deepseek_v41_config=lambda config: config,
        DeviceOperator=types.SimpleNamespace(get_dsv41_packed_cache_ops=lambda: None),
        maybe_prefix=lambda *args: "model",
        get_pp_group=lambda: types.SimpleNamespace(is_last_rank=True),
        ParallelLMHead=fail_head,
        ExitStack=ExitStack,
        rollback_shared_engram=pool_module.rollback_shared_engram,
    )
    exec(compile(tree, str(path), "exec"), ns)
    ns["Model"].model_cls = staticmethod(lambda **kwargs: inner)
    config = types.SimpleNamespace(vocab_size=1, hidden_size=1)
    vc = types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=config), quant_config=None)
    try:
        with pytest.raises(RuntimeError, match="head failed"):
            ns["Model"](vllm_config=vc)
        if cleanup_fails:
            assert pool_module._failed_startup_cleanups[-1] is close
            with pytest.raises(TimeoutError):
                owner.close()
            pool_module._failed_startup_cleanups.clear()
        else:
            assert client.closed
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["context_exit", "attribute"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_vl_constructor_failure_releases_language_model_lease(pool_module, owner, failure, cleanup_fails):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/vl_model.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AscendDeepseekV41ForCausalLM")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    runner_class = ast.parse("class VL(Base):\n    pass").body[0]
    assert isinstance(runner_class, ast.ClassDef)
    runner_class.body = [init]
    tree = ast.fix_missing_locations(ast.Module(body=[runner_class], type_ignores=[]))
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._shared_pool_client = client

        def close_host_offload(self):
            if cleanup_fails:
                raise RuntimeError("cleanup failed")
            client.close()
            self._shared_pool_client = None

    class Language(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.embedding = Embedding()

        @property
        def requires_uncompiled_fallback(self):
            raise RuntimeError("attribute failed")

    class Base(torch.nn.Module):
        @contextmanager
        def _mark_tower_model(self, *args):
            yield

        @contextmanager
        def _mark_language_model(self, *args):
            yield
            if failure == "context_exit":
                raise RuntimeError("context_exit failed")

    ns = dict(
        Base=Base,
        nn=torch.nn,
        torch=torch,
        maybe_prefix=lambda *args: "language_model",
        rollback_shared_engram=pool_module.rollback_shared_engram,
    )
    exec(compile(tree, str(path), "exec"), ns)
    ns["VL"].language_model_cls = Language
    config = types.SimpleNamespace(vision_n_layers=0)
    vc = types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=config, multimodal_config=None))
    try:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            ns["VL"](vllm_config=vc)
        if cleanup_fails:
            assert isinstance(pool_module._failed_startup_cleanups[-1].__self__, Embedding)
            with pytest.raises(TimeoutError):
                owner.close()
            pool_module._failed_startup_cleanups.clear()
        else:
            assert client.closed
    finally:
        client.close()


def test_v2_shutdown_finds_shared_embedding_inside_vl_wrapper(pool_module, owner):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    shutdown = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._shared_pool_client = client

        def close_host_offload(self):
            client.close()
            self._shared_pool_client = None

    wrapper = torch.nn.Module()
    language = torch.nn.Module()
    embedding = Embedding()
    language.add_module("embedding", embedding)
    wrapper.add_module("language_model", language)

    class Parent:
        def shutdown(self):
            pass

    runner_class = ast.parse("class Runner(Parent):\n    pass").body[0]
    assert isinstance(runner_class, ast.ClassDef)
    runner_class.body = [shutdown]
    tree = ast.fix_missing_locations(ast.Module(body=[runner_class], type_ignores=[]))
    ns = dict(Parent=Parent, C8KVBlockZeroer=type("C8KVBlockZeroer", (), {}))
    exec(compile(tree, str(path), "exec"), ns)
    runner = ns["Runner"]()
    runner.model = wrapper
    runner.shutdown()
    assert client.closed and embedding._shared_pool_client is None
    owner.close(timeout=1)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_embedding_constructor_failure_after_mapping_releases_lease(pool_module, owner, cleanup_fails):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/embedding.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AscendParallelEngramEmbedding")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    cleanup_block = next(n for n in init.body if isinstance(n, ast.Try))
    tree = ast.fix_missing_locations(ast.Module(body=[cleanup_block], type_ignores=[]))
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")

    def close():
        if cleanup_fails:
            raise RuntimeError("cleanup failed")
        client.close()

    embedding = types.SimpleNamespace(
        _shared_pool_client=types.SimpleNamespace(close=close), _weight_loader=None, vocab_start_idx=0
    )

    def fail_attributes(*args):
        raise RuntimeError("parameter setup failed")

    ns = dict(
        self=embedding,
        weight=torch.empty(1),
        scales=torch.empty(1),
        nn=torch.nn,
        set_weight_attrs=fail_attributes,
        cpu_offload=True,
        rollback_shared_engram=pool_module.rollback_shared_engram,
    )
    with pytest.raises(RuntimeError, match="parameter setup failed"):
        exec(compile(tree, str(path), "exec"), ns)
    if cleanup_fails:
        assert pool_module._failed_startup_cleanups[-1] is close
        with pytest.raises(TimeoutError):
            owner.close()
        pool_module._failed_startup_cleanups.clear()
        client.close()
    else:
        assert client.closed and embedding._shared_pool_client is None
    owner.close(timeout=1)


def test_layout_rejected_before_mapping(pool_module, owner):
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    try:
        with pytest.raises(ValueError, match="layout mismatch"):
            client.view("codes", 0, 16, rows=33, width=256, dtype=torch.int8)
        assert not client.buffers
    finally:
        client.close()


def test_consumer_reattaches_without_reloading_owner(pool_module, owner):
    original = owner.buffers["codes"]
    for _ in range(2):
        client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
        client.close()
    assert owner.buffers["codes"] is original
    assert not original.closed


def test_old_schema_rejected_and_lease_released(pool_module, owner):
    owner.manifest["schema"] = 1
    with pytest.raises(ValueError, match="manifest"):
        pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    owner.close(timeout=1)
    assert owner.closed


def test_invalid_bm_object_rejected_before_mapping(pool_module, owner):
    owner.manifest["tensors"]["codes"]["object_id"] = -1
    client = pool_module.SharedPoolClient(owner.endpoint, "model-a", device="npu:0")
    try:
        with pytest.raises(ValueError, match="object id"):
            client.view("codes", 0, 16, rows=32, width=256, dtype=torch.int8)
        assert not client.buffers
        assert client.runtime is None
    finally:
        client.close()
