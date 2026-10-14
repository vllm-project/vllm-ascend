# SPDX-License-Identifier: Apache-2.0
"""CPU lane lifetime contracts using controlled engines, never hardware QoS."""

import ast
import copy
import ctypes
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "vllm_ascend/distributed/kv_transfer"
PREFIX = "vllm_ascend.distributed.kv_transfer."


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


# Execute complete production modules; isolate only heavyweight parent imports.
# Native objects are deliberate CPU fakes, with no legacy initialization method.
with patch.dict(sys.modules):
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(ROOT / "vllm_ascend")]
    sys.modules["vllm_ascend"] = package
    ENVS = load(ROOT / "vllm_ascend/envs.py", "vllm_ascend.envs")
    INIT = load(BASE / "qos_engine.py", PREFIX + "qos_engine")
    PD = load(BASE / "qos_pd_pool.py", PREFIX + "qos_pd_pool")
    STORE = load(BASE / "qos_store_pool.py", PREFIX + "qos_store_pool")
    LAYER = load(BASE / "kv_p2p/layerwise_qos.py", PREFIX + "kv_p2p.layerwise_qos")


class Engine:
    def __init__(self, port):
        self.port = port
        self.regions = {}
        self.reg_rc = self.unreg_rc = 0
        self.on_read = lambda *args: 0

    def initialize(self, host, metadata, protocol, device):
        self.args = (host, metadata, protocol, device)
        self.config = json.loads(os.environ[ENVS.MOONCAKE_ASCEND_RESOURCE_ENV])
        self.path = Path(os.environ[ENVS.MOONCAKE_TENT_CONFIG_ENV])
        self.tent = json.loads(self.path.read_text())
        return 0

    def get_rpc_port(self):
        return self.port

    def get_engine(self):
        return self

    def register_memory(self, ptr, size):
        if not self.reg_rc:
            self.regions[ptr] = size
        return self.reg_rc

    def unregister_memory(self, ptr):
        if not self.unreg_rc:
            self.regions.pop(ptr, None)
        return self.unreg_rc

    def batch_transfer_sync_write(self, *args):
        return self.on_read(*args)

    def batch_transfer_sync_read(self, *args):
        return self.on_read(*args)


class Store:
    def __init__(self, data):
        self.data = data
        self.closed = False

    def setup(self, **kwargs):
        self.kwargs = kwargs
        return 0

    def close(self):
        self.closed = True
        return 0

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, *replicate):
        if replicate == (None,):
            raise TypeError("upstream pybind does not accept None as config")
        self.replicate = replicate
        for key, addresses, lengths in zip(keys, addrs, sizes):
            self.data[key] = b"".join(ctypes.string_at(a, n) for a, n in zip(addresses, lengths))
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        for key, addresses, lengths in zip(keys, addrs, sizes):
            offset = 0
            for addr, size in zip(addresses, lengths):
                ctypes.memmove(addr, self.data[key][offset : offset + size], size)
                offset += size
        return [sum(parts) for parts in sizes]


class LaneContracts(unittest.TestCase):
    def setUp(self):
        self.engines, self.stores, self.data = [], [], {}
        env = patch.dict(os.environ, {"ASCEND_ENABLE_USE_FABRIC_MEM": "0"})
        env.start()
        self.addCleanup(env.stop)
        for name in (ENVS.MOONCAKE_ASCEND_RESOURCE_ENV, ENVS.MOONCAKE_TENT_CONFIG_ENV):
            os.environ.pop(name, None)
        modules = patch.dict(sys.modules, {PREFIX + "qos_pd_pool": PD, PREFIX + "qos_store_pool": STORE})
        modules.start()
        self.addCleanup(modules.stop)

    def engine(self):
        engine = Engine(23000 + len(self.engines))
        self.engines.append(engine)
        return engine

    def store(self):
        store = Store(self.data)
        self.stores.append(store)
        return store

    def pool(self, store=False, contribute=True):
        if store:
            result = STORE.QosStorePool(
                qos_values=[7, 0, 3, 3],
                default_qos=3,
                resource_config={},
                setup_kwargs=dict(
                    protocol="ascend",
                    local_hostname="localhost",
                    tenant_id="team-a",
                    global_segment_size=4096 if contribute else 0,
                    local_buffer_size=512 if contribute else 0,
                ),
                engine_factory=self.engine,
                store_factory=self.store,
            )
        else:
            result = PD.QosPDPool("localhost", "", [7, 0, 3, 3], 3, {}, self.engine)
        self.addCleanup(result.close)
        return result

    def test_tenant_capacity_and_no_environment_mutation(self):
        before = dict(os.environ)
        pool = self.pool(store=True)
        self.assertEqual(len(self.engines), 3)
        self.assertEqual([s.kwargs["tenant_id"] for s in self.stores], ["team-a"] * 3)
        self.assertEqual([s.kwargs["global_segment_size"] for s in self.stores], [0, 4096, 0])
        self.assertEqual([s.kwargs["local_buffer_size"] for s in self.stores], [0, 512, 0])
        self.assertIs(pool.default_store, self.stores[1])
        self.assertEqual(dict(os.environ), before)

    def test_noncontributing_client_capacity_remains_zero(self):
        self.pool(store=True, contribute=False)
        self.assertTrue(all(s.kwargs["global_segment_size"] == s.kwargs["local_buffer_size"] == 0 for s in self.stores))

    def test_all_nine_cross_lane_combinations_bytes_and_guards(self):
        pool = self.pool(store=True)
        source = ctypes.create_string_buffer(bytes(range(64)))
        target = ctypes.create_string_buffer(b"x" * 65)
        src, dst = ctypes.addressof(source), ctypes.addressof(target)
        pool.register_buffers([src, dst], [65, 66])
        for put in [0, 3, 7]:
            for get in [0, 3, 7]:
                with self.subTest(put=put, get=get):
                    ctypes.memset(dst, ord("x"), 66)
                    pool.transfer(put, "put", ["same-key"], [[src + 1, src + 17]], [[7, 19]])
                    pool.transfer(get, "get", ["same-key"], [[dst + 1, dst + 17]], [[7, 19]])
                    self.assertEqual(target.raw[1:8], source.raw[1:8])
                    self.assertEqual(target.raw[17:36], source.raw[17:36])
                    self.assertEqual(target.raw[:1] + target.raw[8:17] + target.raw[36:], b"x" * 40)
        self.assertEqual(list(self.data), ["same-key"])

    def test_registration_failure_each_lane_preserves_existing(self):
        for failed in range(3):
            pool = self.pool()
            engines = self.engines[-3:]
            pool.register_buffers([100], [10])
            engines[failed].reg_rc = -1
            with self.assertRaises(RuntimeError):
                pool.register_buffers([200], [10])
            self.assertTrue(all(e.regions == {100: 10} for e in engines))
            engines[failed].reg_rc = 0
            pool.close()

    def test_rollback_failure_poisoning_then_cleanup(self):
        pool = self.pool()
        self.engines[1].reg_rc = -1
        self.engines[0].unreg_rc = -1
        with self.assertRaisesRegex(RuntimeError, "broken"):
            pool.register_buffers([100], [10])
        with self.assertRaisesRegex(RuntimeError, "broken"):
            pool.read(0, "peer:1", [100], [200], [1])
        self.engines[0].unreg_rc = 0
        pool.close()
        self.assertTrue(all(not e.regions for e in self.engines))

    def test_overlap_overflow_and_size_change(self):
        pool = self.pool()
        pool.register_buffers([100], [10])
        pool.register_buffers([100], [10])
        for ptr, size in [(100, 11), (101, 2), ((1 << 64) - 1, 2), (True, 1)]:
            with self.assertRaises(ValueError):
                pool.register_buffers([ptr], [size])
        self.assertTrue(all(e.regions == {100: 10} for e in self.engines))

    def test_cross_lane_progress_and_close_waits_for_native_return(self):
        pool = self.pool()
        pool.register_buffers([100], [10])
        entered, release, other = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def blocked(*args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("fixture release missing")
            return 0

        self.engines[0].on_read = blocked

        def progress(*args):
            other.set()
            return 0

        self.engines[1].on_read = progress
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(pool.read, 0, "peer:1", [100], [200], [1])
            try:
                self.assertTrue(entered.wait(1))
                second = executor.submit(pool.read, 3, "peer:3", [100], [200], [1])
                self.assertTrue(other.wait(1))
                second.result(1)
                closing = executor.submit(pool.close)
                self.assertFalse(closing.done())
                self.assertTrue(all(e.regions for e in self.engines))
            finally:
                release.set()
            first.result(1)
            closing.result(1)
        self.assertTrue(all(not e.regions for e in self.engines))

    def test_same_lane_serializes_calls(self):
        pool = self.pool()
        pool.register_buffers([100], [10])
        entered, release, second_started = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        calls = []

        def blocked(*args):
            calls.append(1)
            entered.set()
            if not release.wait(2):
                raise TimeoutError("fixture release missing")
            return 0

        self.engines[0].on_read = blocked

        def second():
            second_started.set()
            return pool.read(0, "peer:1", [100], [200], [1])

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(pool.read, 0, "peer:1", [100], [200], [1])
            try:
                self.assertTrue(entered.wait(1))
                future = executor.submit(second)
                self.assertTrue(second_started.wait(1))
                self.assertEqual(calls, [1])
            finally:
                release.set()
            first.result(1)
            future.result(1)
        self.assertEqual(calls, [1, 1])

    def test_missing_upstream_method_never_falls_back(self):
        with self.assertRaises(AttributeError):
            PD.QosPDPool("host", "", [0, 3, 7], 0, {}, lambda: object())

    def test_close_idempotent_and_no_use_after_close(self):
        pool = self.pool()
        pool.register_buffers([100], [10])
        pool.close()
        pool.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pool.read(0, "peer:1", [100], [200], [1])

    def test_upstream_arguments_resource_and_distinct_snapshots(self):
        os.environ[ENVS.MOONCAKE_ASCEND_RESOURCE_ENV] = '{"comm_resource_config.service_level": 2}'
        pool = PD.QosPDPool("host", "device0", range(8), 3, engine_factory=self.engine)
        self.addCleanup(pool.close)
        self.assertEqual(len(set(pool.rpc_ports.values())), 8)
        self.assertEqual(len({e.path for e in self.engines}), 8)
        for qos, engine in enumerate(self.engines):
            self.assertEqual(engine.args, ("host", "P2PHANDSHAKE", "ascend", "device0"))
            self.assertEqual(engine.config, {"comm_resource_config.service_level": 2, "comm_resource_config.qos": qos})
            self.assertEqual(
                json.loads(engine.tent["transports"]["ascend_direct"]["global_resource_config"]), engine.config
            )
        pool.close()
        self.assertTrue(all(e.path.is_file() for e in self.engines))

    def test_existing_tent_preserved_and_only_resource_overridden(self):
        original: dict[str, Any] = {
            "metadata": {"type": "handshake"},
            "transports": {"tcp": {"enabled": True}, "ascend_direct": {"other": 9, "global_resource_config": "old"}},
            INIT.RESOURCE_CONFIG_KEY: "old-flat",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.json"
            before = json.dumps(original)
            path.write_text(before)
            os.environ[ENVS.MOONCAKE_TENT_CONFIG_ENV] = str(path)
            env = dict(os.environ)
            resource = {"comm_resource_config.service_level": 1, "comm_resource_config.qos": 7}
            saved = copy.deepcopy(resource)
            pool = PD.QosPDPool("host", "", [0, 3], 0, resource, self.engine)
            self.addCleanup(pool.close)
            self.assertEqual(dict(os.environ), env)
            self.assertEqual(path.read_text(), before)
            self.assertEqual(resource, saved)
            for engine in self.engines:
                expected = copy.deepcopy(original)
                expected.pop(INIT.RESOURCE_CONFIG_KEY)
                expected["transports"]["ascend_direct"]["global_resource_config"] = json.dumps(
                    engine.config, separators=(",", ":")
                )
                self.assertEqual(engine.tent, expected)

    def test_environment_restored_for_absent_empty_and_present_values(self):
        for previous in (None, "", "{}"):
            with self.subTest(previous=previous):
                if previous is None:
                    os.environ.pop(ENVS.MOONCAKE_ASCEND_RESOURCE_ENV, None)
                    os.environ.pop(ENVS.MOONCAKE_TENT_CONFIG_ENV, None)
                else:
                    os.environ[ENVS.MOONCAKE_ASCEND_RESOURCE_ENV] = previous
                    os.environ[ENVS.MOONCAKE_TENT_CONFIG_ENV] = ""
                before = dict(os.environ)
                self.pool().close()
                self.assertEqual(dict(os.environ), before)

    def test_factory_and_initialize_failures_restore_environment(self):
        before = dict(os.environ)
        for failure in ("factory", "exception", "rc"):

            def factory(failure=failure):
                if failure == "factory":
                    raise RuntimeError("factory failed")
                engine = self.engine()

                def initialize(*args):
                    if failure == "exception":
                        raise RuntimeError("native failed")
                    return -9

                engine.initialize = initialize
                return engine

            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                PD.QosPDPool("host", "", [0], 0, engine_factory=factory)
            self.assertEqual(dict(os.environ), before)

    def test_invalid_configuration_fails_before_native_construction(self):
        for resource in ("[]", "null", "broken", '{"store": {}}', '{"comm_resource_config": {}}'):
            with self.subTest(resource=resource):
                os.environ[ENVS.MOONCAKE_ASCEND_RESOURCE_ENV] = resource
                before = dict(os.environ)
                with self.assertRaises(ValueError):
                    PD.QosPDPool("host", "", [0], 0, engine_factory=self.engine)
                self.assertEqual(self.engines, [])
                self.assertEqual(dict(os.environ), before)
        os.environ.pop(ENVS.MOONCAKE_ASCEND_RESOURCE_ENV)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            os.environ[ENVS.MOONCAKE_TENT_CONFIG_ENV] = str(path)
            original: Any
            for original in ([], {"transports": []}, {"transports": {"ascend_direct": []}}):
                path.write_text(json.dumps(original))
                with self.assertRaises(ValueError):
                    PD.QosPDPool("host", "", [0], 0, engine_factory=self.engine)
                self.assertEqual(self.engines, [])

    def test_pd_and_store_construction_share_one_environment_lock(self):
        entered, release, store_started, store_factory_entered = [threading.Event() for _ in range(4)]
        before = dict(os.environ)

        def pd_factory():
            engine = self.engine()
            initialize = engine.initialize

            def blocked(*args):
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("release missing")
                return initialize(*args)

            engine.initialize = blocked
            return engine

        def store_factory():
            store_factory_entered.set()
            return self.engine()

        def create_store():
            store_started.set()
            return STORE.QosStorePool(
                qos_values=[7],
                default_qos=7,
                setup_kwargs={"protocol": "ascend", "local_hostname": "host"},
                engine_factory=store_factory,
                store_factory=self.store,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            pd = executor.submit(PD.QosPDPool, "host", "", [0], 0, None, pd_factory)
            try:
                self.assertTrue(entered.wait(1))
                store = executor.submit(create_store)
                self.assertTrue(store_started.wait(1))
                self.assertFalse(store_factory_entered.wait(0.05))
            finally:
                release.set()
            self.addCleanup(pd.result(2).close)
            self.addCleanup(store.result(2).close)
        self.assertEqual([e.config["comm_resource_config.qos"] for e in self.engines], [0, 7])
        self.assertEqual(dict(os.environ), before)

    def test_layerwise_write_uses_new_pool_and_preserves_route(self):
        pool = LAYER.QosPDWritePool("host", "dev", [0, 3, 7], 0, {}, self.engine)
        self.addCleanup(pool.close)
        pool.register_buffers([100], [20])
        calls = []
        for engine in self.engines:

            def record(*args, engine=engine):
                calls.append((engine.config["comm_resource_config.qos"], args))
                return 0

            engine.on_read = record
        before = dict(os.environ)
        self.assertEqual(pool.write(7, "peer:23007", [101], [200], [4]), 0)
        self.assertEqual(pool.read(3, "peer:23003", [102], [201], [5]), 0)
        self.assertEqual(calls, [(7, ("peer:23007", [101], [200], [4])), (3, ("peer:23003", [102], [201], [5]))])
        self.assertEqual(dict(os.environ), before)
        with self.assertRaises(ValueError):
            pool.write(7, "peer:23007", [120], [200], [1])

    def test_replication_config_is_forwarded_without_copy(self):
        pool = self.pool(store=True)
        source = ctypes.create_string_buffer(b"data")
        addr = ctypes.addressof(source)
        pool.register_buffers([addr], [5])
        config = object()
        pool.transfer(7, "put", ["key"], [[addr]], [[4]], config)
        self.assertIs(self.stores[2].replicate[0], config)
        pool.transfer(0, "put", ["key"], [[addr]], [[4]])
        self.assertEqual(self.stores[0].replicate, ())

    def test_store_partial_setup_closes_all_created_clients(self):
        def factory():
            store = self.store()
            if len(self.stores) == 2:
                store.setup = lambda **kwargs: -1
            return store

        before = dict(os.environ)
        with self.assertRaisesRegex(RuntimeError, "Store setup failed"):
            STORE.QosStorePool(
                qos_values=[0, 3, 7],
                default_qos=0,
                setup_kwargs={"protocol": "ascend", "local_hostname": "host"},
                engine_factory=self.engine,
                store_factory=factory,
            )
        self.assertEqual(len(self.stores), 2)
        self.assertTrue(all(s.closed for s in self.stores))
        self.assertEqual(dict(os.environ), before)

    def test_store_registration_rollback_and_close_order(self):
        pool = self.pool(store=True)
        self.engines[1].reg_rc = -1
        with self.assertRaises(RuntimeError):
            pool.register_buffers([100], [10])
        self.assertTrue(all(not e.regions for e in self.engines))
        self.engines[1].reg_rc = 0
        pool.register_buffers([100], [10])
        for engine, store in zip(self.engines, self.stores):
            unregister = engine.unregister_memory

            def checked(ptr, store=store, unregister=unregister):
                self.assertTrue(store.closed)
                return unregister(ptr)

            engine.unregister_memory = checked
        pool.close()
        self.assertTrue(all(not e.regions for e in self.engines))

    def test_duplicate_pd_ports_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "distinct RPC"):
            PD.QosPDPool("host", "", [0, 3], 0, {}, lambda: Engine(20000))

    def test_invalid_qos_values_never_construct_engines(self):
        for values, default in (([], 0), ([True], True), ([-1], -1), ([8], 8), ([0], 3), ([0], False)):
            with self.subTest(values=values, default=default), self.assertRaises(ValueError):
                PD.QosPDPool("host", "", values, default, {}, self.engine)
        self.assertFalse(self.engines)

    def test_backend_constructs_ascend_pool_through_upstream_imports(self):
        # Execute the unchanged setup method, including the real lazy import.
        path = BASE / "kv_pool/ascend_store/backend/mooncake_backend.py"
        cls = next(
            n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "MooncakeBackend"
        )
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_setup_store")
        namespace: dict[str, Any] = {"get_ip": lambda: "host"}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        native: Any = ModuleType("mooncake.engine")
        native.TransferEngine = self.engine
        store: Any = ModuleType("mooncake.store")
        store.MooncakeDistributedStore = self.store
        obj = SimpleNamespace(
            qos_policy=SimpleNamespace(
                priority_to_qos={0: 0, 3: 3, 7: 7}, default_priority=0, select=lambda q: q, resource_config={}
            ),
            config=SimpleNamespace(
                metadata_server="P2PHANDSHAKE",
                global_segment_size=64,
                local_buffer_size=32,
                protocol="ascend",
                device_name="dev",
                master_server_address="master",
                tenant_id="team",
            ),
            _contribute_memory=True,
            set_device=lambda: None,
        )
        with patch.dict(sys.modules, {"mooncake.engine": native, "mooncake.store": store}):
            result = namespace["_setup_store"](obj)
        self.addCleanup(obj.qos_pool.close)
        self.assertIsInstance(obj.qos_pool, STORE.QosStorePool)
        self.assertIs(result, self.stores[0])
        self.assertTrue(all(e.args[-1] == "dev" for e in self.engines))


if __name__ == "__main__":
    unittest.main()
