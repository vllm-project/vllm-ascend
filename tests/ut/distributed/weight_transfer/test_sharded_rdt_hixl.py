"""CPU contract tests; these do not establish CANN/NPU correctness."""

import ast
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[4]
MODULE_DIR = ROOT / "vllm_ascend/distributed/weight_transfer"
ALIGNMENT = 2 << 20


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    device = SimpleNamespace(type="npu")

    def __init__(self, address=ALIGNMENT, size=ALIGNMENT):
        self.address, self.size = address, size

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self.address

    def untyped_storage(self):
        return self

    def numel(self):
        return self.size

    def element_size(self):
        return 1


@pytest.fixture
def session(monkeypatch):
    calls: list[tuple[Any, ...]] = []

    class Engine:
        state = 1
        status = 0

        def initialize(self, endpoint, options):
            return 0

        def register_mem(self, desc, kind):
            calls.append(("register", desc.address, desc.size))
            return 0, desc.address

        def connect(self, endpoint):
            calls.append(("connect", endpoint))
            return 0

        def transfer_async(self, endpoint, op, operations):
            calls.append(("read", endpoint, op, operations))
            return 0, 10

        def get_transfer_status(self, request):
            calls.append(("poll", request))
            return self.status, self.state

        def disconnect(self, endpoint):
            calls.append(("disconnect", endpoint))
            return 0

        def deregister_mem(self, handle):
            calls.append(("deregister", handle))
            return 0

        def finalize(self):
            calls.append(("finalize",))

    binding = SimpleNamespace(
        SUCCESS=0,
        Hixl=Engine,
        MemDesc=lambda address, size: SimpleNamespace(address=address, size=size),
        MemType=SimpleNamespace(MEM_DEVICE=0),
        TransferOp=SimpleNamespace(READ=0),
        TransferOpDesc=lambda **kwargs: SimpleNamespace(**kwargs),
        TransferStatus=SimpleNamespace(WAITING=0, COMPLETED=1, FAILED=3),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(device=object, Tensor=Tensor, npu=SimpleNamespace(set_device=lambda device: None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "acl",
        SimpleNamespace(rt=SimpleNamespace(get_context=lambda: (object(), 0), set_context=lambda context: 0)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.network_utils",
        SimpleNamespace(
            get_ip=lambda: "127.0.0.1", get_open_port=lambda: 10000, join_host_port=lambda host, port: f"{host}:{port}"
        ),
    )
    module = load_file("hixl_session_test", MODULE_DIR / "hixl_session.py")
    result = module.HixlSession(SimpleNamespace(type="npu"), binding=binding)
    return module, result, calls


def test_read_uses_local_target_and_remote_source_and_consumes_once(session):
    _, transport, calls = session
    target = Tensor()
    transport.register(target)
    request = transport.read("source:2000", target, 4 * ALIGNMENT, 16)
    operation = next(c for c in calls if c[0] == "read")[3][0]
    assert (operation.local_addr, operation.remote_addr, operation.len) == (ALIGNMENT, 4 * ALIGNMENT, 16)
    transport.wait(request)
    with pytest.raises(ValueError, match="already consumed"):
        transport.wait(request)


def test_close_disconnects_before_deregistering_and_is_idempotent(session):
    _, transport, calls = session
    target = Tensor()
    transport.register(target)
    transport.wait(transport.read("source:2000", target, 4 * ALIGNMENT, 16))
    transport.close()
    transport.close()
    names = [c[0] for c in calls]
    assert names[-3:] == ["disconnect", "deregister", "finalize"]
    assert not transport._regions


def test_pending_request_prevents_storage_release(session):
    _, transport, calls = session
    target = Tensor()
    transport.register(target)
    transport.read("source:2000", target, 4 * ALIGNMENT, 16)
    with pytest.raises(RuntimeError, match="Drain"):
        transport.close()
    assert not any(c[0] == "deregister" for c in calls)
    assert transport._regions


@pytest.mark.parametrize("address,size", [(1, ALIGNMENT), (ALIGNMENT, 1)])
def test_registration_rejects_unaligned_regions(session, address, size):
    _, transport, _ = session
    with pytest.raises(ValueError, match="aligned"):
        transport.register(Tensor(address, size))


def test_transfer_failure_poisoning_retains_storage(session):
    _, transport, calls = session
    target = Tensor()
    transport.register(target)
    transport._engine.state = 3
    request = transport.read("source:2000", target, 4 * ALIGNMENT, 16)
    with pytest.raises(RuntimeError, match="READ failed"):
        transport.wait(request)
    with pytest.raises(RuntimeError, match="terminate"):
        transport.close()
    assert transport._regions
    assert not any(c[0] == "deregister" for c in calls)


def test_wait_timeout_preserves_request_and_storage(session, monkeypatch):
    module, transport, _ = session
    target = Tensor()
    transport.register(target)
    transport._engine.state = 0
    monkeypatch.setattr(module, "TRANSFER_TIMEOUT_S", -1)
    request = transport.read("source:2000", target, 4 * ALIGNMENT, 16)
    with pytest.raises(TimeoutError):
        transport.wait(request)
    assert request in transport._requests
    assert transport._regions


@pytest.mark.parametrize("nbytes", [0, -1, ALIGNMENT + 1])
def test_read_rejects_sizes_outside_target(session, nbytes):
    _, transport, calls = session
    target = Tensor()
    transport.register(target)
    with pytest.raises(ValueError, match="size"):
        transport.read("source:2000", target, 4 * ALIGNMENT, nbytes)
    assert not any(c[0] == "read" for c in calls)


def test_read_rejects_unregistered_target(session):
    _, transport, _ = session
    with pytest.raises(ValueError, match="not registered"):
        transport.read("source:2000", Tensor(), 4 * ALIGNMENT, 16)


def test_registration_limit_fails_before_calling_cann(session, monkeypatch):
    module, transport, calls = session
    monkeypatch.setattr(module, "MAX_REGIONS", 1)
    transport.register(Tensor())
    with pytest.raises(ValueError, match="limit"):
        transport.register(Tensor(4 * ALIGNMENT))
    assert len([c for c in calls if c[0] == "register"]) == 1


def test_wait_polls_pending_then_consumes_completed_handle(session, monkeypatch):
    _, transport, _ = session
    states = iter([(0, 0), (0, 1)])
    monkeypatch.setattr(transport._engine, "get_transfer_status", lambda request: next(states))
    target = Tensor()
    transport.register(target)
    request = transport.read("source:2000", target, 4 * ALIGNMENT, 16)
    transport.wait(request)
    assert not transport._requests


@pytest.mark.parametrize("close_fails", [False, True])
def test_producer_shutdown_releases_ipc_only_after_safe_close(close_fails):
    # Execute the actual producer lease/close methods without importing the
    # device-only module. A live reader must block teardown before HIXL close.
    tree = ast.parse((MODULE_DIR / "sharded_rdt_trainer.py").read_text(encoding="utf-8"))
    server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_HixlProducerServer")
    names = {"hixl_add_consumer", "hixl_remove_consumer", "shutdown"}
    body: list[ast.stmt] = [node for node in server.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "producer_lifecycle", "exec"), namespace)
    producer_type = type("Producer", (), {name: namespace[name] for name in names})
    producer = producer_type()
    calls = []
    producer._cache_cond = threading.Condition()
    producer._serve_lock = threading.Lock()
    producer._hixl_consumers = set()
    producer._hixl_closing = False
    cached_mapping = object()
    producer._base_cache = {"ipc_handle": cached_mapping}

    def close():
        assert producer._base_cache == {"ipc_handle": cached_mapping}
        calls.append("close")
        if close_fails:
            raise RuntimeError("HIXL disconnect failed")

    producer._hixl = SimpleNamespace(close=close)
    for name in ("_cache", "_gens", "_sharers", "_serve_rings", "_pack_dsts"):
        setattr(producer, name, {})
    producer.hixl_add_consumer("reader:1000")
    with pytest.raises(RuntimeError, match="inference"):
        producer.shutdown()
    assert not calls
    assert producer._base_cache == {"ipc_handle": cached_mapping}
    producer.hixl_remove_consumer("reader:1000")
    if close_fails:
        with pytest.raises(RuntimeError, match="disconnect failed"):
            producer.shutdown()
        assert calls == ["close"]
        assert producer._base_cache == {"ipc_handle": cached_mapping}
        return
    producer.shutdown()
    assert calls == ["close"]
    assert not producer._base_cache
    with pytest.raises(RuntimeError, match="shutting down"):
        producer.hixl_add_consumer("late-reader:2000")


@pytest.mark.parametrize("active", [True, False])
def test_ipc_export_resolves_current_npu_reducer(active):
    tree = ast.parse((MODULE_DIR / "sharded_rdt_trainer.py").read_text(encoding="utf-8"))
    reducer = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_reduce_npu_tensor"
    )
    rebuild = object()
    reductions = SimpleNamespace(reduce_tensor=lambda tensor: (object(), ()))
    namespace: dict[str, Any] = {
        "torch": SimpleNamespace(multiprocessing=SimpleNamespace(reductions=reductions)),
        "rebuild_npu_tensor": rebuild,
    }
    exec(compile(ast.Module(body=[reducer], type_ignores=[]), "npu_reducer", "exec"), namespace)
    # Simulate torch_npu replacing the reducer after this module was imported.
    if active:
        reductions.reduce_tensor = lambda tensor: (rebuild, ("npu handle",))
        assert namespace["_reduce_npu_tensor"](object()) == (rebuild, ("npu handle",))
    else:
        with pytest.raises(RuntimeError, match="not active"):
            namespace["_reduce_npu_tensor"](object())


@pytest.mark.parametrize(
    "active,configured,fails",
    [
        (True, True, False),
        (False, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_worker_shutdown_rpc_preserves_update_guard_and_propagates_failure(active, configured, fails):
    path = ROOT / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    method = next(
        node
        for node in worker_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "shutdown_weight_transfer_engine"
    )
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "worker_shutdown", "exec"), namespace)
    calls = []

    def shutdown():
        calls.append("shutdown")
        if fails:
            raise RuntimeError("HIXL close failed")

    engine = SimpleNamespace(shutdown=shutdown) if configured else None
    worker = SimpleNamespace(_weight_update_active=active, weight_transfer_engine=engine)
    if active:
        with pytest.raises(RuntimeError, match="during a weight update"):
            namespace["shutdown_weight_transfer_engine"](worker)
        assert not calls
    elif fails:
        with pytest.raises(RuntimeError, match="HIXL close failed"):
            namespace["shutdown_weight_transfer_engine"](worker)
        assert worker.weight_transfer_engine is engine
        assert calls == ["shutdown"]
    else:
        namespace["shutdown_weight_transfer_engine"](worker)
        assert calls == (["shutdown"] if configured else [])


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_public_backend_name_dispatches_both_factories(session, monkeypatch, platform):
    worker: Any = type("WorkerFactory", (), {})
    trainer: Any = type("TrainerFactory", (), {})
    original_worker, original_trainer = object(), object()
    worker._registry = {"sharded_rdt": lambda: original_worker}
    trainer._registry = {"sharded_rdt": lambda: original_trainer}
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.weight_transfer.factory",
        SimpleNamespace(WeightTransferEngineFactory=worker, WeightTransferTrainerFactory=trainer),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.platforms", SimpleNamespace(current_platform=SimpleNamespace(device_type=platform))
    )
    new_worker, new_trainer = object(), object()
    for suffix, symbol, result in (
        ("engine", "ShardedRDTHixlWeightTransferEngine", new_worker),
        ("trainer", "ShardedRDTHixlTrainerWeightTransferEngine", new_trainer),
    ):
        name = f"vllm_ascend.distributed.weight_transfer.sharded_rdt_{suffix}"
        module = ModuleType(name)
        setattr(module, symbol, result)
        monkeypatch.setitem(sys.modules, name, module)
    module = load_file("sharded_factory_test", MODULE_DIR / "__init__.py")
    module._register_sharded_rdt()
    module._register_sharded_rdt()
    assert list(worker._registry) == ["sharded_rdt"]
    assert list(trainer._registry) == ["sharded_rdt"]
    assert worker._registry["sharded_rdt"]() is (new_worker if platform == "npu" else original_worker)
    assert trainer._registry["sharded_rdt"]() is (new_trainer if platform == "npu" else original_trainer)


@pytest.mark.parametrize("supports_nixl,retry_fails", [(True, False), (False, False), (False, True)])
def test_cpu_trainer_import_scopes_ray_transport_patch(monkeypatch, supports_nixl, retry_fails):
    helper = load_file("hixl_cpu_import_test", Path(__file__).with_name("conftest.py"))
    calls = []
    imports = []

    def method(*args, **kwargs):
        if kwargs.get("tensor_transport") == "nixl" and not supports_nixl:
            raise ValueError("Invalid tensor transport NIXL, must be one of ['OBJECT_STORE']")
        calls.append(kwargs)
        return lambda function: function

    fake_ray = SimpleNamespace(method=method)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    original_import = helper.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name != "vllm.distributed.weight_transfer.sharded_rdt_trainer":
            return original_import(name, *args, **kwargs)
        imports.append(name)
        fake_ray.method(tensor_transport="nixl", num_returns=1)(lambda: None)
        fake_ray.method(tensor_transport="gloo", num_returns=2)(lambda: None)
        if retry_fails:
            raise RuntimeError("unrelated import failure")
        return ModuleType(name)

    monkeypatch.setattr(helper.importlib, "import_module", import_module)
    if retry_fails:
        with pytest.raises(RuntimeError, match="unrelated import failure"):
            helper._import_trainer_without_nixl_transport()
    else:
        helper._import_trainer_without_nixl_transport()
    assert fake_ray.method is method
    assert len(imports) == (1 if supports_nixl else 2)
    assert calls == [
        {"tensor_transport": "nixl", "num_returns": 1} if supports_nixl else {"num_returns": 1},
        {"tensor_transport": "gloo", "num_returns": 2},
    ]


def test_cpu_trainer_import_does_not_hide_other_value_errors(monkeypatch):
    helper = load_file("hixl_cpu_import_error_test", Path(__file__).with_name("conftest.py"))
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace())

    def import_module(*args, **kwargs):
        raise ValueError("invalid ownership metadata")

    monkeypatch.setattr(helper.importlib, "import_module", import_module)
    with pytest.raises(ValueError, match="invalid ownership metadata"):
        helper._import_trainer_without_nixl_transport()
