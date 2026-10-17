# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import importlib.util
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

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

    def test_failed_initialization_can_be_retried(self):
        for failure in (-1, RuntimeError("native initialization failed")):
            with self.subTest(failure=failure):
                global_te = self.global_te_cls()
                failed, recovered = MagicMock(), MagicMock()
                if isinstance(failure, Exception):
                    failed.initialize.side_effect = failure
                else:
                    failed.initialize.return_value = failure
                recovered.initialize.return_value = 0
                with patch("mooncake.engine.TransferEngine", side_effect=[failed, recovered]) as factory:
                    with self.assertRaises(RuntimeError):
                        global_te.get_transfer_engine("localhost", "device0")
                    self.assertIsNone(global_te.transfer_engine)
                    self.assertIs(global_te.get_transfer_engine("localhost", "device0"), recovered)
                    self.assertIs(global_te.get_transfer_engine("localhost", "device0"), recovered)
                    self.assertEqual(factory.call_count, 2)
                    recovered.initialize.assert_called_once_with("localhost", "P2PHANDSHAKE", "ascend", "device0")

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
