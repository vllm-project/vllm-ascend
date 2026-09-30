# SPDX-License-Identifier: Apache-2.0
"""Pinned source contracts with real Python threads and upstream runner ordering.

No NPU/native transfer/HTTP service is started. Production methods are executed
unchanged; backend, cache tensors, events and unrelated runner plumbing are fakes.
"""

import __future__

import ast
import copy
import ctypes
import importlib.util
import queue
import threading
import time
import unittest
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
VLLM = ROOT.parent / "vllm/vllm"
BASE = ROOT / "vllm_ascend/distributed/kv_transfer"
STORE = BASE / "kv_pool/ascend_store/kv_transfer.py"
LAYER = BASE / "kv_p2p/mooncake_layerwise_connector.py"


def source_class(path, name, names, bases, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    # Filter unrelated methods/imports, preserving each exercised function body.
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    node.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
    unit = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(unit, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


def function(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    unit = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(unit, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class ReceiveLifecycle(unittest.TestCase):
    def make(self, outcome=None, plain=False, callback=None):
        namespace = dict(
            threading=threading, queue=queue, defaultdict=defaultdict, ctypes=ctypes, time=time, logger=Mock()
        )
        namespace["Thread"] = threading.Thread
        source_class(
            STORE,
            "KVTransferThread",
            {
                "__init__",
                "run",
                "_set_os_thread_name",
                "_get_block_size",
                "_prepare_value",
                "_skip_null_blocks",
                "add_request",
                "set_finished_request",
                "get_and_clear_finished_requests",
                "raise_if_failed",
            },
            ["Thread"],
            namespace,
        )
        function(STORE, "record_failed_blocks", namespace)
        cls = source_class(
            STORE, "KVCacheStoreRecvingThread", {"__init__", "_handle_request"}, ["KVTransferThread"], namespace
        )
        calls = []

        def get(keys, addrs, sizes):
            calls.append(list(keys))
            if keys[0].startswith("bad"):
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome
            return [0] * len(keys)

        backend = NS(
            qos_pool=object(),
            set_device=Mock(),
            get=Mock(side_effect=get),
            get_request=Mock(side_effect=lambda rid, priority, *args: get(*args)),
        )
        db = NS(
            group_block_len=[[128]],
            load_mask=lambda *a: None,
            mask_allows_chunk=lambda *a: True,
            process_token_key_strings_with_block_ids=lambda length, hashes, blocks, *a, **kw: (
                (i * 128, (i + 1) * 128, f"{hashes[0]}-{i}", None, block) for i, block in enumerate(blocks)
            ),
            prepare_value=lambda start, end, blocks, **kw: ([1000 + start], [end - start], kw["block_id"]),
        )
        worker = cls(backend, db, 128, 0, record_operation=callback)

        def request(name, blocks):
            return NS(
                req_id=name,
                kv_priority=None if plain else 7,
                kv_cache_group_ids=[0],
                load_spec=NS(token_len=256, vllm_cached_tokens=0),
                block_ids_by_group=[blocks],
                block_hashes=[name],
                skip_null_blocks_by_group=[],
            )

        return worker, backend, request, calls

    def execute(self, worker, requests):
        for r in requests:
            worker.add_request(r)
        worker.add_request(None)
        worker.start()
        worker.join(3)
        self.assertFalse(worker.is_alive(), "receiving thread did not terminate")

    def test_request_get_exception_finishes_and_next_request_runs(self):
        t, backend, req, calls = self.make(RuntimeError("QOS_MATRIX_EXPECTED_GET_FAILURE"))
        self.execute(t, [req("bad", [5, 6]), req("good", [7, 8])])
        self.assertEqual(t.get_and_clear_finished_requests(), {"bad", "good"})
        self.assertEqual(t._invalid_block_ids, {5, 6})
        self.assertEqual(len(calls), 2)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)
        self.assertIsNone(t._fatal_error)
        self.assertEqual(backend.get_request.call_count, 2)
        backend.get.assert_not_called()

    def test_negative_codes_keep_existing_partial_failure_semantics(self):
        t, _, req, _ = self.make([0, -1])
        self.execute(t, [req("bad", [5, 6])])
        self.assertEqual(t._invalid_block_ids, {6})
        self.assertEqual(t.get_and_clear_finished_requests(), {"bad"})
        self.assertEqual(t.request_queue.unfinished_tasks, 0)

    def test_none_result_still_marks_failed_blocks_and_completes(self):
        t, _, req, _ = self.make(None)
        self.execute(t, [req("bad", [5, 6])])
        self.assertEqual(t._invalid_block_ids, {5, 6})
        self.assertEqual(t.get_and_clear_finished_requests(), {"bad"})
        self.assertEqual(t.request_queue.unfinished_tasks, 0)

    def test_plain_get_success_unchanged(self):
        t, backend, req, _ = self.make(plain=True)
        self.execute(t, [req("good", [7, 8])])
        self.assertEqual(t.get_and_clear_finished_requests(), {"good"})
        backend.get.assert_called_once()
        backend.get_request.assert_not_called()

    def test_plain_get_exception_is_not_silently_swallowed(self):
        error = RuntimeError("plain backend error")
        t, _, req, _ = self.make(error, plain=True)
        self.execute(t, [req("bad", [5, 6])])
        self.assertIs(t._fatal_error, error)
        self.assertEqual(t.get_and_clear_finished_requests(), set())

    def test_unrelated_callback_error_remains_fatal(self):
        error = RuntimeError("metrics bug")
        t, _, req, _ = self.make(callback=Mock(side_effect=error))
        self.execute(t, [req("good", [7, 8])])
        self.assertIs(t._fatal_error, error)
        self.assertEqual(t.get_and_clear_finished_requests(), set())

    def test_missing_load_spec_completes_once(self):
        t, backend, req, _ = self.make()
        r = req("empty", [5, 6])
        r.load_spec = None
        self.execute(t, [r])
        self.assertEqual(t.get_and_clear_finished_requests(), {"empty"})
        self.assertEqual(t.request_queue.unfinished_tasks, 0)
        backend.get_request.assert_not_called()


class LayerwiseLifecycle(unittest.TestCase):
    def setUp(self):
        # Execute the real runner ordering with a fake cache and transfer queue.
        self.events = []
        upstream = VLLM
        if not (upstream / "distributed/kv_transfer/kv_connector/v1/base.py").is_file():
            spec = importlib.util.find_spec("vllm")
            if spec is None or spec.origin is None:
                raise RuntimeError("vLLM source or installation required")
            upstream = Path(spec.origin).parent
        namespace = dict(
            contextmanager=contextmanager,
            copy=copy,
            logger=Mock(),
            MambaSpec=type("MambaSpec", (), {}),
            FullAttentionSpec=type("FullAttentionSpec", (), {}),
            SlidingWindowSpec=type("SlidingWindowSpec", (), {}),
        )

        class Metadata:
            def __init__(self):
                self.requests = {"r": NS(local_block_ids=[[1]], remote_block_ids=[[2]])}
                self.send_task = NS(group_rearrange_block_ids=None)

        class Event:
            def record(s):
                self.events.append("cache_event")

        namespace.update(
            torch=NS(npu=NS(Event=Event)),
            MooncakeLayerwiseConnectorMetadata=Metadata,
            SendTask=lambda **kw: NS(**kw, send_request={}, failed_requests={}),
        )
        namespace["Base"] = source_class(
            upstream / "distributed/kv_transfer/kv_connector/v1/base.py",
            "KVConnectorBase_V1",
            {"bind_connector_metadata", "clear_connector_metadata", "has_connector_metadata"},
            [],
            namespace,
        )
        self.connector_cls = source_class(
            LAYER,
            "MooncakeLayerwiseConnector",
            {"bind_connector_metadata", "start_load_kv", "on_kv_cache_written", "save_kv_layer"},
            ["Base"],
            namespace,
        )
        worker_cls = source_class(
            LAYER,
            "MooncakeLayerwiseConnectorWorker",
            {"start_load_kv", "on_kv_cache_written", "save_kv_layer"},
            [],
            namespace,
        )
        self.worker = worker_cls()
        self.worker.vllm_config = NS(kv_transfer_config=NS(is_kv_producer=True, is_kv_consumer=False))
        self.worker.total_layers = 1
        self.worker.index_to_name = {0: ["layer0"]}
        self.worker.layer_metadata = {"layer0": NS(tensor_group_idx=[0])}
        self.worker.num_kv_cache_groups = 1
        self.worker.kv_cache_specs = [object()]
        self.worker._align_remote_block_ids = Mock()
        self.worker._get_kv_split_metadata = lambda *a: {
            ("peer", 1): dict(local_block_ids=[1], remote_block_ids=[2], trans_count=1)
        }
        self.worker._get_kernel_block_ids = lambda blocks: blocks
        self.worker.pd_head_ratio = 1
        self.worker.enable_kv_quant = self.worker.enable_c8_quant = False
        self.worker.kv_send_layer_thread = NS(send_queue=queue.Queue())
        self.worker.qos_pool = object()
        self.worker.update_decoder_info = lambda _, meta: meta
        self.worker._mark_layer_reuse_pending = Mock()
        self.worker._complete_layer_reuse = Mock()
        original_start = self.worker.start_load_kv

        def start(meta):
            self.events.append("prepare")
            return original_start(meta)

        self.worker.start_load_kv = Mock(side_effect=start)
        self.connector = self.connector_cls()
        self.connector._is_kv_producer = True
        self.connector._connector_metadata = None
        self.connector.connector_worker = self.worker
        self.connector.wait_for_save = lambda: self.events.append("wait_save")
        self.connector.get_transfer_results = lambda _: NS(
            finished_sending=set(), finished_recving=set(), failed_recving=set()
        )
        self.connector.get_block_ids_with_load_errors = lambda: set()
        for name in (
            "get_kv_connector_stats",
            "get_kv_cache_events",
            "get_kv_connector_kv_cache_events",
            "build_connector_worker_meta",
        ):
            setattr(self.connector, name, lambda: None)
        namespace.update(
            KVConnectorBase=namespace["Base"],
            KVConnectorOutput=NS,
            get_kv_transfer_group=lambda: self.connector,
            get_forward_context=lambda: NS(),
        )
        self.mixin = source_class(
            upstream / "v1/worker/kv_connector_model_runner_mixin.py",
            "KVConnectorModelRunnerMixin",
            {"_get_kv_connector_output"},
            [],
            namespace,
        )
        self.Metadata = Metadata

    def run_forward(self, sync=False, meta=None, defer=False):
        meta = self.Metadata() if meta is None else meta
        output = NS(kv_connector_metadata=meta, has_sync_kv_loads=sync, finished_req_ids=set())
        with self.mixin._get_kv_connector_output(output, defer_finalize=defer):
            self.events.append("forward")
            self.connector.on_kv_cache_written("layer0")
            self.connector.save_kv_layer("layer0", [object(), object()], NS())
        return meta

    def test_deferred_load_runner_initializes_before_producer_forward(self):
        self.run_forward()
        self.assertEqual(self.events, ["prepare", "forward", "cache_event", "wait_save"])
        self.assertEqual(self.worker.current_layer, 1)
        self.worker.start_load_kv.assert_called_once()
        task = self.worker.kv_send_layer_thread.send_queue.get_nowait()
        self.assertEqual(task.layer_idx, 0)
        self.assertEqual(set(task.send_request), {"r"})
        self.assertIsNone(self.connector._connector_metadata)

    def test_sync_load_runner_does_not_prepare_twice(self):
        self.run_forward(sync=True)
        self.worker.start_load_kv.assert_called_once()
        self.assertEqual(self.worker.current_layer, 1)

    def test_rebinding_same_metadata_object_resets_next_step(self):
        meta = self.run_forward()
        self.run_forward(meta=meta)
        self.assertEqual(self.worker.start_load_kv.call_count, 2)
        tasks = [self.worker.kv_send_layer_thread.send_queue.get_nowait() for _ in range(2)]
        self.assertEqual([t.layer_idx for t in tasks], [0, 0])

    def test_empty_step_does_not_record_events_or_send(self):
        meta = self.Metadata()
        meta.requests.clear()
        self.run_forward(meta=meta)
        self.assertNotIn("cache_event", self.events)
        self.assertTrue(self.worker.kv_send_layer_thread.send_queue.empty())

    def test_consumer_keeps_upstream_deferred_timing(self):
        self.connector._is_kv_producer = False
        self.worker.start_load_kv = Mock(side_effect=lambda _: self.events.append("load"))
        out = NS(kv_connector_metadata=self.Metadata(), has_sync_kv_loads=False, finished_req_ids=set())
        with self.mixin._get_kv_connector_output(out):
            self.events.append("forward")
            self.worker.start_load_kv.assert_not_called()
        self.assertEqual(self.events, ["forward", "load", "wait_save"])

    def test_consumer_keeps_upstream_sync_timing(self):
        self.connector._is_kv_producer = False
        self.worker.start_load_kv = Mock(side_effect=lambda _: self.events.append("load"))
        out = NS(kv_connector_metadata=self.Metadata(), has_sync_kv_loads=True, finished_req_ids=set())
        with self.mixin._get_kv_connector_output(out):
            self.events.append("forward")
        self.assertEqual(self.events, ["load", "forward", "wait_save"])

    def test_producer_preparation_error_prevents_forward(self):
        self.worker.start_load_kv.side_effect = RuntimeError("metadata prepare failed")
        with self.assertRaisesRegex(RuntimeError, "metadata prepare failed"):
            self.run_forward()
        self.assertNotIn("forward", self.events)

    def test_closed_connector_rejects_bind_before_worker_use(self):
        self.connector._qos_closing = True
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            self.connector.bind_connector_metadata(self.Metadata())
        self.worker.start_load_kv.assert_not_called()

    def test_deferred_finalization_preserves_completed_layer_index(self):
        self.run_forward(defer=True)
        self.assertEqual(self.worker.current_layer, 1)
        self.assertNotIn("wait_save", self.events)
        self.assertIsNotNone(self.connector._connector_metadata)


if __name__ == "__main__":
    unittest.main()
