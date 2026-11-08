# SPDX-License-Identifier: Apache-2.0
"""CPU contracts; companion vLLM source must be discoverable on PYTHONPATH.

No SDK, NPU, or vLLM import is required.
"""

import ast
import time
import unittest
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib.machinery import PathFinder
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

vllm_spec = PathFinder.find_spec("vllm")
assert vllm_spec is not None and vllm_spec.submodule_search_locations
VLLM = Path(vllm_spec.submodule_search_locations[0])
STORE = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"


def load_classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)


class NativeWorkerReuseTests(unittest.TestCase):
    def setUp(self):
        namespace: dict[str, Any] = {"MooncakeBackend": object}
        load_classes(STORE / "backend/base.py", {"StoreCapacityError", "TransferCompletionUnknown"}, namespace)
        load_classes(STORE / "backend/mooncake_backend.py", {"NativeMooncakeStoreAdapter"}, namespace)
        self.capacity_error = namespace["StoreCapacityError"]
        self.unknown = namespace["TransferCompletionUnknown"]
        self.backend = Mock()
        self.store = namespace["NativeMooncakeStoreAdapter"](self.backend)
        self.namespace = namespace

    def checked_backend(self):
        self.namespace.update(ABC=ABC, abstractmethod=abstractmethod, Integral=Integral)
        load_classes(
            STORE / "backend/base.py",
            {"Backend", "BatchResultShapeError", "require_aligned_batch_results"},
            self.namespace,
        )
        load_classes(STORE / "backend/mooncake_backend.py", {"MooncakeBackend"}, self.namespace)
        backend = self.namespace["MooncakeBackend"].__new__(self.namespace["MooncakeBackend"])
        backend._closed = False
        backend._store_initialized = True
        backend._completion_unknown = False
        backend.store = Mock()
        backend._use_fabric_mem = False
        backend._use_store_independent_te = False
        backend._prefetch_conf = SimpleNamespace(use_dict_setup=False)
        backend._shared_te_registered_ptrs = set()
        backend.set_device = Mock()
        backend._build_replicate_config = Mock(return_value=object())
        return backend

    def test_legacy_close_inherits_noop_without_sdk_calls(self):
        backend = self.checked_backend()
        self.assertIs(type(backend).close, self.namespace["Backend"].close)
        backend._completion_unknown = True
        backend._shared_te_registered_ptrs = {100}
        backend.close()
        backend.store.close.assert_not_called()
        backend.store.unregister_buffer.assert_not_called()
        backend.set_device.assert_not_called()
        self.assertFalse(backend._closed)
        self.assertEqual(backend._shared_te_registered_ptrs, {100})

    def test_registration_rejection_keeps_successful_prefix_cleanup_possible(self):
        backend = self.checked_backend()
        engine = Mock()
        engine.register_memory.side_effect = [0, -1]
        engine.unregister_memory.return_value = 0
        self.namespace.update(
            global_te=SimpleNamespace(get_transfer_engine=lambda *args, **kwargs: engine, transfer_engine=engine),
            get_ip=lambda: "local",
        )
        with self.assertRaisesRegex(RuntimeError, "registration failed") as caught:
            backend.register_buffer_checked([100, 200], [16, 32])
        self.assertNotIsInstance(caught.exception, self.unknown)
        self.assertFalse(backend._completion_unknown)
        self.assertEqual(backend._shared_te_registered_ptrs, {100})
        backend.close_native_store()
        engine.unregister_memory.assert_called_once_with(100)
        backend.set_device.assert_called()
        self.assertEqual(backend._shared_te_registered_ptrs, set())
        self.assertIsNone(backend.store)
        self.assertTrue(backend._closed)

    def test_checked_registration_unregisters_with_the_actual_owner(self):
        backend = self.checked_backend()
        engine = Mock()
        engine.register_memory.return_value = engine.unregister_memory.return_value = 0
        self.namespace.update(
            global_te=SimpleNamespace(
                get_transfer_engine=lambda *args, **kwargs: engine,
                transfer_engine=engine,
            ),
            get_ip=lambda: "local",
        )
        backend.register_buffer_checked([100], [16])
        engine.register_memory.assert_called_once_with(100, 16)
        backend.store.register_buffer.assert_not_called()
        self.assertEqual(backend.unregister_buffer_checked(100), 0)
        engine.unregister_memory.assert_called_once_with(100)
        backend.store.unregister_buffer.assert_not_called()
        backend._use_store_independent_te = True
        backend.store.register_buffer.return_value = backend.store.unregister_buffer.return_value = 0
        backend.register_buffer_checked([200], [32])
        backend.store.register_buffer.assert_called_once_with(200, 32)
        self.assertEqual(backend.unregister_buffer_checked(200), 0)
        backend.store.unregister_buffer.assert_called_once_with(200)
        backend._completion_unknown = True
        with self.assertRaises(self.unknown):
            backend.unregister_buffer_checked(200)

    def native_worker(self):
        self.namespace.update(
            MooncakeStoreWorker=object,
            AscendNativeStoreSendingThread=object,
            AscendNativeStoreRecvingThread=object,
            contextmanager=contextmanager,
        )
        load_classes(STORE / "native_worker.py", {"AscendNativeStoreWorker"}, self.namespace)
        cls = self.namespace["AscendNativeStoreWorker"]
        worker = cls.__new__(cls)
        worker.store = self.store
        return worker

    @unittest.skipUnless(
        (VLLM / "distributed/kv_transfer/kv_connector/v1/mooncake/store/layout.py").is_file(),
        "requires vLLM split Mooncake Store worker",
    )
    def test_native_receiver_publishes_only_after_known_completion(self):
        steps = []
        worker = self.native_worker()

        class NativeThreadBase:
            def set_finished_request(self, req_id):
                steps.append("finished")

            def _record_operation(self, *args, **kwargs):
                pass

        self.namespace.update(
            KVTransferThread=NativeThreadBase,
            nullcontext=nullcontext,
            time=time,
            envs=SimpleNamespace(VLLM_MOONCAKE_STORE_TIER_LOG=False),
            logger=Mock(),
        )
        load_classes(
            VLLM / "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
            {"KVCacheStoreRecvingThread", "_rotate_list", "_sum_batch_bytes"},
            self.namespace,
        )
        load_classes(STORE / "native_worker.py", {"AscendNativeStoreRecvingThread"}, self.namespace)
        cls = self.namespace["AscendNativeStoreRecvingThread"]
        receiver = cls.__new__(cls)
        receiver.store = self.store
        self.backend._completion_unknown = False
        receiver._report_load_error = Mock()
        receiver.coord = SimpleNamespace(load_mask=lambda *args: ([True],))
        receiver.block_size = 4
        receiver.tp_rank = 0
        receiver.group_participates = [True]
        receiver.usable_disk_offload_buffer_budget_bytes = None
        receiver.request_queue = Mock()
        layout = SimpleNamespace(
            block_size=4,
            local_shard_ids=[0],
            key_for=lambda *args: "key",
            prepare_values=lambda *args: ([[100]], [[4]], [2]),
        )
        receiver.token_databases = [
            SimpleNamespace(
                store_layout=layout,
                hash_block_size=4,
                process_tokens=lambda *args: [(0, 4, b"hash")],
            )
        ]
        self.backend.get_checked.return_value = [4]
        request = SimpleNamespace(
            req_id="request",
            block_ids=([2],),
            block_hashes=[b"hash"],
            load_spec=SimpleNamespace(token_len=4, vllm_cached_tokens=0, tail_key_boundaries=()),
        )
        receiver._handle_request(request)
        self.assertEqual(steps, ["finished"])
        receiver.request_queue.task_done.assert_called_once_with()
        steps.clear()
        receiver.request_queue.reset_mock()
        error = self.unknown("pending native transfer")

        def pending_get(*args):
            self.backend._completion_unknown = True
            raise error

        self.backend.get_checked.side_effect = pending_get
        with self.assertRaises(self.unknown):
            receiver._handle_request(request)
        self.assertEqual(steps, [])
        receiver.request_queue.task_done.assert_not_called()
        with self.assertRaises(self.unknown):
            worker.get_finished(set(), None)
        with self.assertRaises(self.unknown):
            worker._close_store(self.store, False)
        self.backend.close_native_store.assert_not_called()

    def test_native_worker_rejects_custom_mem_pool(self):
        worker = self.native_worker()
        with self.assertRaisesRegex(ValueError, "custom_mem_pool"):
            worker._create_mem_pool({"custom_mem_pool": "NVLINK"}, object())

    def test_native_close_retains_pending_or_unknown_store_for_retry(self):
        self.namespace.update(
            KVCacheStoreSendingThread=object,
            KVCacheStoreRecvingThread=object,
            AscendNativeStoreSendingThread=object,
            AscendNativeStoreRecvingThread=object,
            contextmanager=contextmanager,
            logger=Mock(),
            time=SimpleNamespace(monotonic=Mock(side_effect=range(0, 100, 6)), sleep=Mock()),
        )
        load_classes(
            VLLM / "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
            {"MooncakeStoreWorker"},
            self.namespace,
        )
        if not hasattr(self.namespace["MooncakeStoreWorker"], "_close_store"):
            self.skipTest("requires vLLM native Store close hook (#66294)")
        load_classes(STORE / "native_worker.py", {"AscendNativeStoreWorker"}, self.namespace)
        cls = self.namespace["AscendNativeStoreWorker"]
        worker = cls.__new__(cls)
        worker.store = self.store
        worker.kv_send_thread = SimpleNamespace(name="send", request_queue=SimpleNamespace(unfinished_tasks=1))
        worker.kv_recv_threads = []
        self.backend._completion_unknown = False
        worker.close()
        self.assertIs(worker.store, self.store)
        self.backend.close_native_store.assert_not_called()
        worker.kv_send_thread.request_queue.unfinished_tasks = 0
        self.backend._completion_unknown = True
        worker.close()
        self.assertIs(worker.store, self.store)
        self.backend.close_native_store.assert_not_called()
        self.backend._completion_unknown = False
        worker.close()
        self.assertIsNone(worker.store)
        self.backend.close_native_store.assert_called_once_with()
        self.namespace["time"].sleep.assert_not_called()

    def test_device_context_failure_is_visible_to_model_worker(self):
        worker = self.native_worker()
        self.backend._completion_unknown = False
        error = RuntimeError("set_device failed")
        self.backend.set_device.side_effect = error
        with self.assertRaises(RuntimeError) as caught, worker._transfer_context():
            self.fail("device setup failure must not enter transfer")
        self.assertIs(caught.exception, error)
        with self.assertRaises(RuntimeError) as surfaced:
            self.store.raise_if_failed()
        self.assertIs(surfaced.exception, error)
        with self.assertRaises(RuntimeError):
            worker.get_finished(set(), None)
        self.assertFalse(self.backend._completion_unknown)

    def test_native_sdk_pressure_and_python_exceptions(self):
        backend = self.checked_backend()
        store = self.namespace["NativeMooncakeStoreAdapter"](backend)
        backend.store.batch_put_from_multi_buffers.return_value = [0, -200]
        self.assertEqual(store.batch_put_from_multi_buffers(["a", "b"], [[1], [2]], [[4], [4]]), [0, -200])
        self.assertFalse(backend._completion_unknown)
        backend.store.batch_put_from_multi_buffers.side_effect = TypeError("invalid native binding arguments")
        with self.assertRaises(TypeError):
            store.batch_put_from_multi_buffers(["a"], [[1]], [[4]])
        self.assertFalse(backend._completion_unknown)
        backend.store.batch_get_into_multi_buffers.side_effect = ValueError("invalid native binding arguments")
        with self.assertRaises(ValueError):
            store.batch_get_into_multi_buffers(["a"], [[1]], [[4]])
        self.assertFalse(backend._completion_unknown)

    def test_explicit_unknown_completion_remains_sticky(self):
        backend = self.checked_backend()
        backend.store.batch_get_into_multi_buffers.side_effect = self.unknown("pending native transfer")
        with self.assertRaises(self.unknown):
            backend.get_checked(["a"], [[1]], [[4]])
        self.assertTrue(backend._completion_unknown)

    def test_native_config_preserves_topology_offload_and_tenant(self):
        self.namespace.update(
            __name__=__name__,
            dataclass=dataclass,
            DEFAULT_GLOBAL_SEGMENT_SIZE=4096,
            DEFAULT_LOCAL_BUFFER_SIZE=4096,
            DEFAULT_TENANT_ID="default",
        )
        load_classes(VLLM / "distributed/mooncake_store.py", {"MooncakeStoreConfig"}, self.namespace)
        self.namespace.update(
            MooncakeStoreWorker=object,
            AscendNativeStoreSendingThread=object,
            AscendNativeStoreRecvingThread=object,
            contextmanager=contextmanager,
        )
        load_classes(STORE / "native_worker.py", {"AscendNativeStoreWorker"}, self.namespace)
        cls = self.namespace["AscendNativeStoreWorker"]
        worker = cls.__new__(cls)
        config = SimpleNamespace(
            metadata_server="metadata",
            master_server_address="master",
            protocol="ascend",
            device_name="device",
            mode="standalone-store",
            global_segment_size=0,
            local_buffer_size=8192,
            enable_offload=True,
            enable_ssd_offload=False,
            tenant_id="tenant",
        )
        self.namespace["AscendStoreConfig"] = SimpleNamespace(load_from_env=lambda: config)
        native = worker._load_store_config()
        self.assertEqual(native.mode, "standalone-store")
        self.assertTrue(native.enable_offload)
        self.assertEqual(native.global_segment_size, 0)
        self.assertEqual(native.local_buffer_size, 8192)
        self.assertEqual(native.tenant_id, "tenant")
        self.assertEqual(native.master_server_address, "master")
        self.assertIs(worker._ascend_store_config, config)

    def test_native_registration_precedes_transfer_thread_start(self):
        steps = []

        class UpstreamWorker:
            def register_kv_caches(self, caches):
                steps.append("upstream")

        self.namespace.update(
            MooncakeStoreWorker=UpstreamWorker,
            AscendNativeStoreSendingThread=object,
            AscendNativeStoreRecvingThread=object,
            contextmanager=contextmanager,
            register_native_cache_views=lambda worker, caches: steps.append("bind"),
        )
        load_classes(STORE / "native_worker.py", {"AscendNativeStoreWorker"}, self.namespace)
        cls = self.namespace["AscendNativeStoreWorker"]
        worker = cls.__new__(cls)
        worker._capacity_only = False
        worker.store = self.store
        self.backend._completion_unknown = False
        worker._init_lookup_key_prefixes = lambda: steps.append("prefixes")
        worker._start_transfer_threads = lambda: steps.append("threads")
        worker.register_kv_caches({"layer": object()})
        self.assertEqual(steps, ["bind", "prefixes", "threads"])

    def test_capacity_rejection_keeps_status_for_upstream_pressure(self):
        self.backend.put_checked.side_effect = self.capacity_error([0, -200])
        config = object()
        self.assertEqual(self.store.batch_put_from_multi_buffers(["a", "b"], [[1], [2]], [[4], [4]], config), [0, -200])
        self.backend.put_checked.assert_called_once_with(["a", "b"], [[1], [2]], [[4], [4]], replicate_config=config)

    def test_short_read_fails_without_marking_dma_unknown(self):
        self.backend.get_checked.return_value = [3]
        with self.assertRaisesRegex(RuntimeError, "Short Store GET") as caught:
            self.store.batch_get_into_multi_buffers(["a"], [[1]], [[4]])
        self.assertNotIsInstance(caught.exception, self.unknown)
        self.backend.get_checked.return_value = [4]
        self.assertEqual(self.store.batch_get_into_multi_buffers(["a"], [[1]], [[4]]), [4])

    def test_unknown_dma_survives_adapter_and_prevents_unregister(self):
        self.backend.put_checked.side_effect = self.unknown("pending")
        self.backend.get_checked.side_effect = self.unknown("pending")
        self.backend.unregister_buffer_checked.side_effect = self.unknown("pending")
        with self.assertRaises(self.unknown):
            self.store.batch_put_from_multi_buffers(["a"], [[1]], [[4]])
        with self.assertRaises(self.unknown):
            self.store.batch_get_into_multi_buffers(["a"], [[1]], [[4]])
        with self.assertRaises(self.unknown):
            self.store.unregister_buffer(1)
        self.backend.store.unregister_buffer.assert_not_called()

    def test_unknown_dma_prevents_save_pin_release(self):
        released = []

        class UpstreamSender:
            def finish_store_job(self, request):
                released.append(request)

        self.namespace["KVCacheStoreSendingThread"] = UpstreamSender
        load_classes(STORE / "native_worker.py", {"AscendNativeStoreSendingThread"}, self.namespace)
        sender = self.namespace["AscendNativeStoreSendingThread"]()
        sender.store = self.store
        self.backend._completion_unknown = True
        with self.assertRaises(self.unknown):
            sender.finish_store_job("job")
        self.assertEqual(released, [])
        self.backend._completion_unknown = False
        sender.finish_store_job("job")
        self.assertEqual(released, ["job"])


if __name__ == "__main__":
    unittest.main()
