# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

_WAIT_SECONDS = 5


class TestGlobalTE(unittest.TestCase):
    def setUp(self):
        # Mooncake is optional; these unit tests only exercise singleton state.
        mooncake = ModuleType("mooncake")
        engine_module = ModuleType("mooncake.engine")
        engine_module.__dict__["TransferEngine"] = MagicMock()
        mooncake.__dict__["engine"] = engine_module
        modules = patch.dict("sys.modules", {"mooncake": mooncake, "mooncake.engine": engine_module})
        modules.start()
        self.addCleanup(modules.stop)

        # ascend_store tests replace the canonical module during collection.
        # Load the real source privately without changing their mocked module.
        source = Path(__file__).parents[4] / "vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py"
        spec = importlib.util.spec_from_file_location("_test_global_te", source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.global_te_cls = module.GlobalTE

    def test_concurrent_getters_only_receive_initialized_engine(self):
        started = threading.Event()
        release = threading.Event()
        observed = threading.Event()
        engine = MagicMock()
        engine.ready = False

        def initialize(*args):
            started.set()
            if not release.wait(_WAIT_SECONDS):
                raise TimeoutError("initialization was not released")
            engine.ready = True
            return 0

        engine.initialize.side_effect = initialize
        global_te = self.global_te_cls()
        lock = threading.Lock()

        class ObservedLock:
            def __enter__(self):
                if started.is_set():
                    observed.set()
                lock.acquire()

            def __exit__(self, *args):
                lock.release()

        def second_getter():
            result = global_te.get_transfer_engine("localhost", None)
            was_ready = result.ready
            observed.set()
            return result, was_ready

        with (
            patch.object(global_te, "transfer_engine_lock", ObservedLock()),
            patch("mooncake.engine.TransferEngine", return_value=engine) as factory,
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(global_te.get_transfer_engine, "localhost", None)
                try:
                    self.assertTrue(started.wait(_WAIT_SECONDS))
                    second = pool.submit(second_getter)
                    self.assertTrue(observed.wait(_WAIT_SECONDS))
                finally:
                    release.set()
                self.assertIs(first.result(timeout=_WAIT_SECONDS), engine)
                returned_engine, was_ready = second.result(timeout=_WAIT_SECONDS)
            self.assertIs(returned_engine, engine)
            self.assertTrue(was_ready)
            factory.assert_called_once_with()
            engine.initialize.assert_called_once_with("localhost", "P2PHANDSHAKE", "ascend", "")

    def test_failed_initialization_is_not_retried(self):
        for failure in (-1, RuntimeError("native initialization failed")):
            with self.subTest(failure=failure):
                global_te = self.global_te_cls()
                failed = MagicMock()
                if isinstance(failure, Exception):
                    failed.initialize.side_effect = failure
                else:
                    failed.initialize.return_value = failure
                with patch("mooncake.engine.TransferEngine", return_value=failed) as factory:
                    with self.assertRaises(RuntimeError):
                        global_te.get_transfer_engine("localhost", "device0")
                    self.assertIsNone(global_te.transfer_engine)
                    for _ in range(3):
                        with self.assertRaisesRegex(RuntimeError, "Restart the process"):
                            global_te.get_transfer_engine("localhost", "device0")
                    factory.assert_called_once_with()
                    failed.initialize.assert_called_once_with("localhost", "P2PHANDSHAKE", "ascend", "device0")

    def test_failed_construction_is_not_retried(self):
        global_te = self.global_te_cls()
        with patch("mooncake.engine.TransferEngine", side_effect=RuntimeError("construction failed")) as factory:
            with self.assertRaisesRegex(RuntimeError, "construction failed"):
                global_te.get_transfer_engine("localhost", None)
            with self.assertRaisesRegex(RuntimeError, "Restart the process"):
                global_te.get_transfer_engine("localhost", None)
            self.assertIsNone(global_te.transfer_engine)
            factory.assert_called_once_with()

    def test_waiting_getter_does_not_retry_failed_initialization(self):
        started, release, waiting = (threading.Event() for _ in range(3))
        engine = MagicMock()

        def initialize(*args):
            started.set()
            if not release.wait(_WAIT_SECONDS):
                raise TimeoutError("initialization was not released")
            return -1

        engine.initialize.side_effect = initialize
        global_te = self.global_te_cls()
        lock = threading.Lock()

        class ObservedLock:
            def __enter__(self):
                if started.is_set():
                    waiting.set()
                lock.acquire()

            def __exit__(self, *args):
                lock.release()

        with (
            patch.object(global_te, "transfer_engine_lock", ObservedLock()),
            patch("mooncake.engine.TransferEngine", return_value=engine) as factory,
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(global_te.get_transfer_engine, "localhost", None)
                try:
                    self.assertTrue(started.wait(_WAIT_SECONDS))
                    second = pool.submit(global_te.get_transfer_engine, "localhost", None)
                    self.assertTrue(waiting.wait(_WAIT_SECONDS))
                finally:
                    release.set()
                with self.assertRaisesRegex(RuntimeError, "ret_value: -1"):
                    first.result(timeout=_WAIT_SECONDS)
                with self.assertRaisesRegex(RuntimeError, "Restart the process"):
                    second.result(timeout=_WAIT_SECONDS)
            self.assertIsNone(global_te.transfer_engine)
            factory.assert_called_once_with()
            engine.initialize.assert_called_once_with("localhost", "P2PHANDSHAKE", "ascend", "")

    def test_registration_uses_successfully_initialized_engine(self):
        global_te = self.global_te_cls()
        with self.assertRaisesRegex(AssertionError, "must be initialized"):
            global_te.register_buffer([100], [64])
        engine = MagicMock()
        engine.initialize.return_value = 0
        engine.register_memory.return_value = 0
        with patch("mooncake.engine.TransferEngine", return_value=engine):
            self.assertIs(global_te.get_transfer_engine("localhost", None), engine)
            global_te.register_buffer([100], [64])
            global_te.register_buffer([100], [64])
        engine.register_memory.assert_called_once_with(100, 64)

    def test_unregister_and_reregister_registered_buffers(self):
        global_te = self.global_te_cls()
        global_te.transfer_engine = MagicMock()
        global_te.transfer_engine.register_memory.return_value = 0
        global_te.transfer_engine.unregister_memory.return_value = 0
        global_te.transfer_engine.disconnect_all_peers.return_value = 0

        global_te.register_buffer([0x1000, 0x2000], [128, 256])
        global_te.unregister_buffer()

        self.assertFalse(global_te.is_register_buffer)
        global_te.transfer_engine.assert_has_calls(
            [
                call.register_memory(0x1000, 128),
                call.register_memory(0x2000, 256),
                call.disconnect_all_peers(),
                call.unregister_memory(0x1000),
                call.unregister_memory(0x2000),
            ]
        )
        self.assertEqual(global_te.transfer_engine.unregister_memory.call_count, 2)

        global_te.reregister_buffer()

        self.assertTrue(global_te.is_register_buffer)
        self.assertEqual(global_te.transfer_engine.register_memory.call_count, 4)

    def test_unregister_failure_leaves_completed_buffers_unregistered(self):
        global_te = self.global_te_cls()
        global_te.transfer_engine = MagicMock()
        global_te.transfer_engine.unregister_memory.side_effect = [0, -1]
        global_te.transfer_engine.disconnect_all_peers.return_value = 0
        global_te.registered_buffers = [(0x1000, 128), (0x2000, 256)]
        global_te.is_register_buffer = True

        with self.assertRaisesRegex(RuntimeError, "ptr=0x2000"):
            global_te.unregister_buffer()

        global_te.transfer_engine.register_memory.assert_not_called()
        self.assertTrue(global_te.is_register_buffer)

    def test_disconnect_failure_stops_memory_unregistration(self):
        global_te = self.global_te_cls()
        global_te.transfer_engine = MagicMock()
        global_te.transfer_engine.disconnect_all_peers.return_value = -1
        global_te.registered_buffers = [(0x1000, 128)]
        global_te.is_register_buffer = True

        with self.assertRaisesRegex(RuntimeError, "peer disconnection failed"):
            global_te.unregister_buffer()

        global_te.transfer_engine.unregister_memory.assert_not_called()
        self.assertTrue(global_te.is_register_buffer)

    def test_disconnect_all_peers_is_noop_before_engine_initialization(self):
        self.global_te_cls().disconnect_all_peers()
