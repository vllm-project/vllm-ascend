# SPDX-License-Identifier: Apache-2.0
"""CPU faults test real P/D modules without importing a device runtime."""

import gc
import importlib
import socket
import sys
import threading
import weakref
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class Owner:
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


class Engine:
    def __init__(self):
        self._engine = self
        self.unique_id = "cpu-session"
        self.calls = []
        self.register_result = 0
        self.destroy_result = 0
        self.uninitialize_result = 0
        self.read_result = 0

    def configure(self, **kwargs):
        self.calls.append(("configure", kwargs))

    def get_transfer_engine(self, host):
        return self

    def register_memory(self, address, size):
        self.calls.append(("register", address, size))
        return self.register_result

    def batch_transfer_sync_read(self, *args):
        self.calls.append(("read",))
        if isinstance(self.read_result, BaseException):
            raise self.read_result
        return self.read_result

    def destroy(self):
        self.calls.append(("destroy",))
        return self.destroy_result

    def unInitialize(self):
        self.calls.append(("unInitialize",))
        return self.uninitialize_result


class Device:
    def __init__(self):
        self.device = 1
        self.calls = []
        self.error = None

    def current_device(self):
        return self.device

    def set_device(self, device):
        self.calls.append(("set_device", device))
        assert device == self.device

    def synchronize(self, device):
        self.calls.append(("sync", device))
        if self.error is not None:
            raise self.error


@pytest.fixture
def environment(monkeypatch):
    package = ModuleType("pd_cleanup_under_test")
    package.__path__ = [
        str(Path(__file__).resolve().parents[5] / "vllm_ascend/distributed/kv_transfer/kv_p2p/hisparse_pd")
    ]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    modules = SimpleNamespace(
        **{name: importlib.import_module(f"{package.__name__}.{name}") for name in ("protocol", "transport", "worker")}
    )
    for module in (modules.transport, modules.worker):
        for name in ("_retained_readers", "_retained_workers"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, {})
    engine, device = Engine(), Device()
    monkeypatch.setattr(modules.transport, "configure_urma_staging", lambda mib: {"swap_mib": mib})
    module = ModuleType("vllm_ascend.distributed.kv_transfer.utils.memfabric_transfer_engine")
    module.global_memfabric_te = engine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    parallel = ModuleType("vllm.distributed.parallel_state")
    parallel.get_tensor_model_parallel_rank = lambda: 1
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    monkeypatch.setattr(torch, "npu", device, raising=False)
    result = SimpleNamespace(**vars(modules), engine=engine, device=device, monkeypatch=monkeypatch)
    yield result
    assert not modules.transport._retained_readers
    assert not modules.worker._retained_workers


def owners():
    tensor = torch.empty(32, dtype=torch.int8)
    mapping = Owner(tensor=tensor)
    indexer = torch.empty(32)
    return [tensor, mapping, indexer]


def reader(environment, allocation_owners):
    return environment.transport.MemfabricReader(
        producer=True,
        device=1,
        host="127.0.0.1",
        protocol="device_urma",
        regions={(allocation_owners[0].data_ptr(), 32): "hisparse_source"},
        owners=allocation_owners,
    )


def worker(environment, allocation_owners, *, producer=False):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "host": "127.0.0.1",
                "control_port": 32100,
                "memfabric_transfer_protocol": "device_urma",
            }
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    result = environment.worker.HiSparsePDWorker(config, None, identity="model", producer=producer)
    result.owners = list(allocation_owners)
    result.reader = reader(environment, allocation_owners)
    result.reader.initialize()
    return result


def collect(*objects):
    return [weakref.ref(value) for value in objects]


def test_reader_construction_has_no_engine_side_effect(environment):
    allocation_owners = owners()
    r = reader(environment, allocation_owners)
    refs = collect(*allocation_owners)
    del allocation_owners
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert environment.engine.calls == []
    r.close()
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("missing", ["destroy", "unInitialize"])
def test_incomplete_teardown_api_rejected_before_registration(environment, missing):
    allocation_owners = owners()
    r = reader(environment, allocation_owners)
    refs = collect(r, *allocation_owners)
    method = getattr(environment.engine, missing)
    setattr(environment.engine, missing, None)
    with pytest.raises(RuntimeError, match="both destroy and unInitialize"):
        r.initialize()
    assert not any(call[0] == "register" for call in environment.engine.calls)
    del r, allocation_owners
    gc.collect()
    assert all(ref() is not None for ref in refs)
    setattr(environment.engine, missing, method)
    refs[0]().close()
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_registration_failure_retains_assigned_reader_and_real_owners(environment, cleanup_fails):
    environment.engine.register_result = -7
    environment.engine.destroy_result = -8 if cleanup_fails else 0

    def attempt():
        allocation_owners = owners()
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config={
                    "host": "127.0.0.1",
                    "control_port": 32100,
                    "memfabric_transfer_protocol": "device_urma",
                }
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        w = environment.worker.HiSparsePDWorker(config, None, identity="model", producer=True)
        weak_owners = collect(*allocation_owners)
        address = allocation_owners[0].data_ptr()
        environment.monkeypatch.setattr(
            environment.worker,
            "build_layouts",
            lambda *args, **kwargs: ([], {(address, 32): "hisparse_source"}, [ref() for ref in weak_owners]),
        )
        try:
            w.register({})
        except RuntimeError as error:
            assert "cannot register" in str(error)
            if cleanup_fails:
                assert "destroy returned -8" in str(error.__cause__)
                assert w.reader is not None
            result = collect(w, w.reader, *allocation_owners)
        environment.monkeypatch.setattr(environment.worker, "build_layouts", lambda *args, **kwargs: None)
        return result

    refs = attempt()  # Caller and exception traceback leave scope.
    gc.collect()
    if cleanup_fails:
        assert all(ref() is not None for ref in refs)
        environment.engine.destroy_result = 0
        refs[0]().close()
        gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("stage", ["destroy", "unInitialize"])
def test_teardown_retry_commits_successful_stages(environment, stage):
    allocation_owners = owners()
    r = reader(environment, allocation_owners)
    r.initialize()
    field = "destroy_result" if stage == "destroy" else "uninitialize_result"
    setattr(environment.engine, field, -9)
    refs = collect(r, *allocation_owners)
    with pytest.raises(RuntimeError, match=stage):
        r.close()
    del r, allocation_owners
    gc.collect()
    assert all(ref() is not None for ref in refs)
    setattr(environment.engine, field, 0)
    r = refs[0]()
    r.close()
    r.close()
    assert sum(call[0] == "destroy" for call in environment.engine.calls) == (2 if stage == "destroy" else 1)
    assert sum(call[0] == "unInitialize" for call in environment.engine.calls) == (1 if stage == "destroy" else 2)
    del r
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_read_failure_requires_device_drain_and_keeps_descriptors(environment):
    allocation_owners = owners()
    r = reader(environment, allocation_owners)
    r.initialize()
    environment.engine.read_result = -3
    with pytest.raises(RuntimeError, match="read failed"):
        r.read("source", [(101, 202, 32)])
    assert r._failed_reads == [("source", [101], [202], [32])]
    refs = collect(r, *allocation_owners)
    with pytest.raises(RuntimeError, match="Device drain"):
        r.close()
    assert not any(call[0] == "destroy" for call in environment.engine.calls)
    del r, allocation_owners
    gc.collect()
    assert all(ref() is not None for ref in refs)
    refs[0]().close(device_drained=True)
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("failure", ["device", "destroy", "unInitialize"])
def test_worker_close_failure_retains_owners_until_device_and_transport_drained(environment, failure):
    allocation_owners = owners()
    w = worker(environment, allocation_owners)
    refs = collect(w, w.reader, *allocation_owners)
    if failure == "device":
        environment.device.error = RuntimeError("device drain failed")
    else:
        setattr(environment.engine, "destroy_result" if failure == "destroy" else "uninitialize_result", -1)
    with pytest.raises(RuntimeError):
        w.close()
    if failure == "device":
        assert not any(call[0] == "destroy" for call in environment.engine.calls)
    del w, allocation_owners
    environment.device.error = None  # Drop independent exception traceback owner.
    gc.collect()
    assert all(ref() is not None for ref in refs)
    environment.engine.destroy_result = environment.engine.uninitialize_result = 0
    w = refs[0]()
    w.close()
    w.close()
    assert w.closed
    del w
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_active_source_lease_keeps_ack_server_usable(environment):
    allocation_owners = owners()
    w = worker(environment, allocation_owners, producer=True)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"tcp://127.0.0.1:{port}"
    w.server = environment.transport.LeaseServer(endpoint, w.registry)
    w.registry.publish("lease", "request", {"identity": "model", "rank": 1, "world_size": 2})
    claimed = environment.transport.rpc(
        endpoint,
        {"op": "acquire", "lease_id": "lease", "consumer": "consumer", "identity": "model", "rank": 1, "world_size": 2},
        2,
    )
    assert claimed["status"] == "ready"
    refs = collect(w, w.reader, w.server, *allocation_owners)
    with pytest.raises(RuntimeError, match="source leases"):
        w.close()
    with pytest.raises(RuntimeError, match="closing"):
        w.queue(SimpleNamespace(imports=[{}]))
    # Pending scheduler work cannot enter the already joined executor; the
    # independent control server must still accept the drained ACK below.
    w.pending.append(SimpleNamespace(imports=[{}]))
    with pytest.raises(RuntimeError, match="closing"):
        w.poll(set())
    del w, allocation_owners
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert not any(call[0] == "destroy" for call in environment.engine.calls)
    ack = environment.transport.rpc(
        endpoint, {"op": "release", "lease_id": "lease", "consumer": "consumer", "drained": True}, 2
    )
    assert ack["status"] == "released"
    refs[0]().close()
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_future_error_preserved_after_safe_teardown_or_as_cleanup_cause(environment, cleanup_fails):
    allocation_owners = owners()
    w = worker(environment, allocation_owners)
    future = Future()
    future.set_exception(ValueError("original read failure"))
    w.futures["request"] = future, threading.Event()
    refs = collect(w, w.reader, *allocation_owners)
    if cleanup_fails:
        environment.device.error = RuntimeError("device drain failed")
    with pytest.raises(ValueError, match="original read failure") as caught:
        w.close()
    if cleanup_fails:
        assert str(caught.value.__cause__) == "device drain failed"
    else:
        assert w.closed
    del caught, w, future, allocation_owners
    environment.device.error = None
    gc.collect()
    if cleanup_fails:
        assert all(ref() is not None for ref in refs)
        with pytest.raises(ValueError, match="original read failure"):
            refs[0]().close()
        gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("api_raises", [False, True])
def test_background_read_failure_keeps_worker_without_drained_ack(environment, api_raises):
    allocation_owners = owners()
    w = worker(environment, allocation_owners)
    w.kv_cache_config = SimpleNamespace(hisparse_shared_host_pool=False)
    job = {"request_id": "request", "ticket": {"lease_id": "lease", "num_tokens": 1}, "blocks": {}, "start_token": 0}
    manifest = {
        "version": environment.worker.VERSION,
        "ticket": job["ticket"],
        "layers": [],
        "blocks": {},
        "session": "peer",
    }
    environment.monkeypatch.setattr(environment.worker, "acquire", lambda *args, **kwargs: manifest)
    environment.monkeypatch.setattr(environment.worker, "plan_reads", lambda *args, **kwargs: [(101, 202, 32)])
    acknowledgements = []
    environment.monkeypatch.setattr(environment.worker, "rpc", lambda *args, **kwargs: acknowledgements.append(args))
    environment.engine.read_result = ValueError("backend read failed") if api_raises else -4
    future = w.executor.submit(w._load, job, threading.Event())
    w.futures["request"] = future, threading.Event()
    error_type = ValueError if api_raises else RuntimeError
    with pytest.raises(error_type, match="read failed"):
        future.result()
    refs = collect(w, w.reader, *allocation_owners)
    assert not acknowledgements
    assert ("set_device", 1) in environment.device.calls
    del w, future, allocation_owners
    environment.engine.read_result = 0  # The injector must not retain a traceback.
    gc.collect()
    assert all(ref() is not None for ref in refs)
    with pytest.raises(error_type, match="read failed"):
        refs[0]().close()
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert not acknowledgements


def test_server_close_failure_retains_worker_until_retry(environment):
    allocation_owners = owners()
    w = worker(environment, allocation_owners)

    class Server:
        fail = True

        def close(self):
            if self.fail:
                raise RuntimeError("server join failed")

    w.server = Server()
    refs = collect(w, w.reader, w.server, *allocation_owners)
    with pytest.raises(RuntimeError, match="server join failed"):
        w.close()
    del w, allocation_owners
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert not any(call[0] == "destroy" for call in environment.engine.calls)
    refs[2]().fail = False
    refs[0]().close()
    gc.collect()
    assert all(ref() is None for ref in refs)
