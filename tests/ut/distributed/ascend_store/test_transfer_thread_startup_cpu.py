# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression for transfer-thread startup failures.

Run: python3 tests/ut/distributed/ascend_store/test_transfer_thread_startup_cpu.py
The production run method is invoked synchronously, so no timeouts, NPU, or
real worker threads are needed to prove an unset startup signal/dead waiter.
"""

import __future__

import ast
import contextlib
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "vllm_ascend").is_dir())
POOL = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[method_name]


class TransferThreadStartupTest(unittest.TestCase):
    def setUp(self):
        hardware = patch.dict(sys.modules, {"torch_npu": MagicMock(), "acl": MagicMock()})
        hardware.start()
        self.addCleanup(hardware.stop)
        self.failure = RuntimeError("device binding failed")

    def test_failed_start_signals_readiness_and_records_error(self):
        run = load_method(POOL / "kv_transfer.py", "KVTransferThread", "run", {"logger": MagicMock()})
        state = SimpleNamespace(
            _set_os_thread_name=lambda: None,
            m_store=SimpleNamespace(set_device=MagicMock(side_effect=self.failure)),
            ready_event=threading.Event(),
            _fatal_error=None,
        )
        # A real Thread sends this to threading.excepthook; its parent still
        # waits on ready_event, which is the observable defect.
        with contextlib.suppress(RuntimeError):
            run(state)
        self.assertTrue(state.ready_event.is_set(), "parent would wait forever on startup event")
        self.assertIs(state._fatal_error, self.failure)

    def test_successful_start_keeps_normal_handshake(self):
        run = load_method(POOL / "kv_transfer.py", "KVTransferThread", "run", {"logger": MagicMock()})
        requests = queue.Queue()
        requests.put(None)
        state = SimpleNamespace(
            _set_os_thread_name=lambda: None,
            m_store=SimpleNamespace(set_device=lambda: None),
            ready_event=threading.Event(),
            _fatal_error=None,
            request_queue=requests,
            _stopping=threading.Event(),
        )
        run(state)
        self.assertTrue(state.ready_event.is_set())
        self.assertIsNone(state._fatal_error)
        self.assertEqual(requests.unfinished_tasks, 0)

    def test_all_worker_readiness_waits_propagate_start_failure(self):
        cases = (
            ("bulk_send", "pool_worker.py", "KVPoolWorker", True, False, "send"),
            ("bulk_receive", "pool_worker.py", "KVPoolWorker", False, False, "receive"),
            ("layer_send", "layerwise/pool_worker.py", "LayerwiseKVPoolWorker", True, True, "layer_send"),
            ("key_send", "layerwise/pool_worker.py", "LayerwiseKVPoolWorker", True, False, "key_send"),
            ("layer_receive", "layerwise/pool_worker.py", "LayerwiseKVPoolWorker", False, True, "layer_receive"),
            ("key_receive", "layerwise/pool_worker.py", "LayerwiseKVPoolWorker", False, False, "key_receive"),
        )
        for name, filename, class_name, can_put, layerwise, failing_kind in cases:
            with self.subTest(path=name):
                failure = self.failure

                def make_thread(kind, failed_kind=failing_kind, error=failure):
                    class StartupThread:
                        def __init__(self, *args, **kwargs):
                            self.ready_event = next(arg for arg in args if isinstance(arg, threading.Event))

                        def start(self):
                            self.ready_event.set()

                        def raise_if_failed(self):
                            if kind == failed_kind:
                                raise error

                    return StartupThread

                namespace = {
                    "threading": threading,
                    "torch": SimpleNamespace(npu=SimpleNamespace(Event=MagicMock())),
                    "is_kv_save_role": lambda role, _: role == "kv_both",
                    "KVCacheStoreSendingThread": make_thread("send"),
                    "KVCacheStoreRecvingThread": make_thread("receive"),
                    "KVCacheStoreLayerSendingThread": make_thread("layer_send"),
                    "KVCacheStoreKeyLayerSendingThread": make_thread("key_send"),
                    "KVCacheStoreLayerRecvingThread": make_thread("layer_receive"),
                    "KVCacheStoreKeyLayerRecvingThread": make_thread("key_receive"),
                }
                start = load_method(POOL / filename, class_name, "_start_kv_transfer_threads", namespace)
                state = SimpleNamespace(
                    _transfer_threads_started=False,
                    can_put=can_put,
                    load_async=not can_put,
                    m_store=MagicMock(),
                    token_database=object(),
                    grouped_block_size=[128],
                    block_size=128,
                    tp_rank=0,
                    tp_size=1,
                    use_kvpp=False,
                    pcp_rank=0,
                    pcp_size=1,
                    dcp_size=1,
                    put_step=1,
                    kv_role="kv_both" if can_put else "kv_consumer",
                    group_uses_align_state=[False],
                    enable_kv_events=False,
                    tp_mismatch=False,
                    _invalid_block_ids=set(),
                    _invalid_block_ids_lock=threading.Lock(),
                    _record_kv_connector_operation=MagicMock(),
                    use_layerwise_transfer=layerwise,
                    use_block_key_layerwise=False,
                    layer_load_tasks=[[]],
                    num_layers=1,
                    consumer_is_to_put=False,
                    page_size_bytes=128,
                    layerwise_max_transfer_blocks=1,
                    layerwise_max_transfer_bytes=128,
                    _build_group_layer_builders=lambda: [],
                    _put_started_keys=set(),
                    _put_started_keys_lock=threading.Lock(),
                    _layerwise_session_tracker=None,
                    layerwise_offload=False,
                    h2d_stagger_us=0,
                    external_slot_release_waiter=None,
                    kv_send_thread=None,
                    _layer_load_aborted=threading.Event(),
                    num_kv_cache_groups=1,
                    _save_disabled=False,
                )
                with self.assertRaisesRegex(RuntimeError, "device binding failed") as raised:
                    start(state)
                self.assertIs(raised.exception, failure)
                self.assertFalse(state._transfer_threads_started)


if __name__ == "__main__":
    unittest.main()
