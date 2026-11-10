# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.distributed.ascend_store.test_cross_layer_kv import (
    CrossLayerKVPoolWorker,
    CrossLayerKVRecvingThread,
    FakeCrossLayerKVStore,
    KVPoolWorker,
    _plan_worker,
    _v41_layout,
    build_cross_layer_kv_trigger_map,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    ReqMeta,
)


def _capability_worker(
    method="dspark", architecture="DeepseekV41DSparkModel", use_v2=True, graph_mode="NONE", enforce_eager=True
):
    groups = _v41_layout(num_layers=12, sources=(2, 8), with_state=False, with_draft=True)
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name=graph_mode),
            static_forward_context={
                str(layer): SimpleNamespace(layer_idx=layer, supports_cross_layer_kv_transfer_hooks=True)
                for layer in range(12)
            },
        ),
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        parallel_config=SimpleNamespace(use_ubatching=False, use_sequence_parallel_moe=False),
        speculative_config=SimpleNamespace(
            method=method,
            enforce_eager=True,
            draft_model_config=SimpleNamespace(architectures=[architecture]),
        ),
    )
    store = FakeCrossLayerKVStore(bytearray(0x10000))
    if method is None:
        config.speculative_config = None
        store.supports_exclusive_allocation = False

    def initialize(worker, *_args):
        worker.vllm_config = config
        worker._extra_config = {"backend": "memcache", "use_layerwise": True}
        worker.pp_size = worker.pcp_size = worker.dcp_size = worker.tp_size = 1
        worker.tp_rank = 0
        worker.use_kvpp = worker.load_async = worker.enable_kv_events = False
        worker.can_put = True
        worker._transfer_threads_started = False
        worker.token_database = MagicMock()
        worker.grouped_block_size = [16] * len(groups)
        worker.group_layer_cache_entry_offsets = {
            group: list(range(len(names) + 1)) if group else [0, 2, 4] for group, names in groups.items()
        }
        worker.cacheable_group_ids = list(groups)
        worker.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=names) for names in groups.values()]
        )
        worker.hf_config = SimpleNamespace(num_hidden_layers=12, kv_source_layer_ids=[2, 8])
        worker.m_store = store
        worker._invalid_block_ids = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker.kv_recv_thread = None

    with (
        patch.object(KVPoolWorker, "__init__", autospec=True, side_effect=initialize),
        patch.object(KVPoolWorker, "_start_kv_transfer_threads", autospec=True),
    ):
        worker = CrossLayerKVPoolWorker(config, True)
        worker._start_kv_transfer_threads()
    return worker


class TestDSparkCrossLayerKVCapabilities(unittest.TestCase):
    def test_piecewise_and_eager_enable_the_same_target_load_pipeline(self):
        for method in [None, "dspark"]:
            for mode, enforce_eager in [("NONE", False), ("NONE", True), ("PIECEWISE", False)]:
                with self.subTest(method=method, mode=mode, enforce_eager=enforce_eager):
                    worker = _capability_worker(method=method, graph_mode=mode, enforce_eager=enforce_eager)
                    try:
                        self.assertTrue(worker._cross_layer_kv_load_capable)
                        self.assertEqual(worker.cross_layer_kv_trigger_layers, (2, 3, 7, 8, 11))
                        self.assertFalse(worker._cross_layer_kv_save_capable)
                    finally:
                        if worker.kv_recv_thread is not None:
                            worker.kv_recv_thread.stop()
                            worker.kv_recv_thread.join(timeout=2)

    def test_unnegotiated_full_graph_modes_keep_bulk(self):
        for mode in ["FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "UNKNOWN"]:
            with self.subTest(mode=mode):
                worker = _capability_worker(graph_mode=mode, enforce_eager=False)
                self.assertFalse(worker._cross_layer_kv_load_capable)
                self.assertFalse(worker._cross_layer_kv_save_capable)

    def test_dspark_enables_target_loads_and_keeps_speculative_saves_bulk(self):
        worker = _capability_worker()
        try:
            self.assertTrue(worker._cross_layer_kv_load_capable)
            self.assertEqual(worker.cross_layer_kv_trigger_layers, (2, 3, 7, 8, 11))
            # Even a future exclusive backend must not publish provisional
            # draft/verification pages via the incremental sender.
            self.assertFalse(worker._cross_layer_kv_save_capable)
        finally:
            if worker.kv_recv_thread is not None:
                worker.kv_recv_thread.stop()
                worker.kv_recv_thread.join(timeout=2)

    def test_other_speculative_methods_and_unknown_drafters_retain_bulk(self):
        for method, architecture, use_v2 in [
            ("eagle", "DeepseekV41DSparkModel", True),
            ("mtp", "DeepseekV41DSparkModel", True),
            ("dspark", "UnknownDraftModel", True),
            ("dspark", "DeepseekV41DSparkModel", False),
        ]:
            with self.subTest(method=method, architecture=architecture, use_v2=use_v2):
                worker = _capability_worker(method, architecture, use_v2)
                self.assertFalse(worker._cross_layer_kv_load_capable)
                self.assertFalse(worker._cross_layer_kv_save_capable)


class TestDSparkCrossLayerKVMixedLoads(unittest.TestCase):
    def _fixture(self):
        database = ChunkedTokenDatabase([KeyMetadata("m", 0, 0, 0), KeyMetadata("m", 0, 0, 1)], [16, 16], None)
        database.set_group_buffers(
            {0: [0x100, 0x200, 0x300, 0x400], 1: [0x500, 0x600, 0x700]},
            {0: [8, 4, 8, 4], 1: [8, 8, 8]},
            {0: [24, 24, 24, 24], 1: [24, 24, 24]},
            group_num_layers={0: 1, 1: 1},
        )
        database.load_mask = lambda *_args: None
        database.mask_allows_chunk = lambda *_args: True
        names = _v41_layout(num_layers=12, sources=(2, 8), with_state=False)[0]
        plan = build_cross_layer_kv_trigger_map({0: names}, (2, 8), {0: [0, 2, 4]})
        store = FakeCrossLayerKVStore(bytearray(0x10000))
        bulk_keys = []

        def get(keys, addrs, sizes):
            for key, destinations, lengths in zip(keys, addrs, sizes, strict=True):
                bulk_keys.append(key)
                offset = 0
                for addr, size in zip(destinations, lengths, strict=True):
                    store.device[addr : addr + size] = store.published[key][offset : offset + size]
                    offset += size
            return [0] * len(keys)

        store.get = get
        keys = []
        for group, block, value in [(0, 7, bytes(range(1, 25))), (1, 9, b"draft000draft111draft222")]:
            key = next(
                database.process_token_key_strings_with_block_ids(16, ["h0"], [block], 0, kv_cache_group_id=group)
            )[2]
            base = store.batch_alloc([key], [len(value)])[0]
            store.pool[base : base + len(value)] = value
            store.batch_write_finish([key], [0])
            keys.append(key)
        worker = _plan_worker(database, plan, store=store)
        worker.cacheable_group_ids = [0, 1]
        worker.grouped_block_size = [16, 16]
        worker.group_uses_align_state = [False, False]
        worker.tp_mismatch = False
        worker.tp_rank = 0
        worker.load_async = False
        worker.backend_name = "memcache"
        worker.num_kv_cache_groups = 2
        worker._failed_requests = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker._save_disabled = threading.Event()
        worker.ensure_store_initialized = lambda: None
        worker._record_kv_connector_operation = lambda *_args: None
        worker.kv_cache_config.kv_cache_groups.append(
            SimpleNamespace(layer_names=[f"mtp.{i}.self_attn.swa_cache" for i in range(3)])
        )
        receiver = CrossLayerKVRecvingThread(store, database, [16, 16], tp_rank=0)
        worker.kv_recv_thread = receiver
        request = ReqMeta(
            "r",
            token_len_chunk=16,
            block_ids=[[7], [9]],
            block_hashes=["h0"],
            kv_cache_group_ids=[0, 1],
            load_spec=SimpleNamespace(can_load=True, vllm_cached_tokens=0, kvpool_cached_tokens=16),
        )
        return worker, request, keys, bulk_keys

    def test_draft_is_ready_while_later_target_source_is_still_loading(self):
        worker, request, keys, bulk_keys = self._fixture()
        store, receiver = worker.m_store, worker.kv_recv_thread
        later_started, release_later = threading.Event(), threading.Event()
        original_copy = store.batch_copy

        def copy(gvas, addrs, sizes, direction):
            if direction == 1 and gvas[0] == store.alloc_base[keys[0]] + 12:
                later_started.set()
                if not release_later.wait(timeout=3):
                    raise TimeoutError("later target source was not released")
            return original_copy(gvas, addrs, sizes, direction)

        store.batch_copy = copy
        receiver.start()
        self.assertTrue(receiver.ready_event.wait(timeout=2))
        try:
            worker.start_load_kv(SimpleNamespace(requests=[request]))
            self.assertEqual(len(worker._cross_layer_kv_load_tasks), 1)
            worker.wait_for_cross_layer_kv_load(2)
            self.assertTrue(later_started.wait(timeout=2))
            self.assertFalse(worker._cross_layer_kv_load_tasks[0].done.is_set())
            self.assertEqual(bulk_keys, [keys[1]])
            self.assertEqual(
                b"".join(bytes(store.device[base + 216 : base + 224]) for base in [0x500, 0x600, 0x700]),
                b"draft000draft111draft222",
            )
            self.assertEqual(store.device[0x100 + 168 : 0x100 + 176], bytes(range(1, 9)))
            self.assertEqual(store.device[0x300 + 168 : 0x300 + 176], bytes(8))
            self.assertEqual(request.kv_cache_group_ids, [0, 1])
            self.assertFalse(hasattr(request.load_spec, "token_len"))
            release_later.set()
            worker.finish_cross_layer_kv_loads()
            self.assertEqual(store.device[0x300 + 168 : 0x300 + 176], bytes(range(13, 21)))
            worker.start_load_kv(SimpleNamespace(requests=[request]))
            self.assertEqual(bulk_keys, [keys[1]])
        finally:
            release_later.set()
            receiver.stop()
            receiver.join(timeout=2)

    def test_bulk_exception_waits_for_target_dma_before_propagating(self):
        worker, request, _keys, _bulk_keys = self._fixture()
        store, receiver = worker.m_store, worker.kv_recv_thread
        copy_started, release_copy, bulk_failed, returned = (threading.Event() for _ in range(4))
        original_copy = store.batch_copy
        errors = []

        def copy(*args):
            copy_started.set()
            if not release_copy.wait(timeout=3):
                raise TimeoutError("target copy was not released")
            return original_copy(*args)

        def get(*_args):
            bulk_failed.set()
            raise RuntimeError("draft restore failed")

        def load():
            try:
                worker.start_load_kv(SimpleNamespace(requests=[request]))
            except Exception as exc:
                errors.append(exc)
            finally:
                returned.set()

        store.batch_copy, store.get = copy, get
        receiver.start()
        self.assertTrue(receiver.ready_event.wait(timeout=2))
        caller = threading.Thread(target=load)
        caller.start()
        try:
            self.assertTrue(copy_started.wait(timeout=2))
            self.assertTrue(bulk_failed.wait(timeout=2))
            self.assertFalse(returned.wait(timeout=0.05))
            release_copy.set()
            caller.join(timeout=2)
            self.assertTrue(returned.is_set())
            self.assertEqual([str(error) for error in errors], ["draft restore failed"])
            self.assertTrue(worker._cross_layer_kv_load_tasks[0].done.is_set())
        finally:
            release_copy.set()
            caller.join(timeout=2)
            receiver.stop()
            receiver.join(timeout=2)

    def test_private_noncacheable_group_is_not_added_to_bulk_fallback(self):
        worker, request, keys, bulk_keys = self._fixture()
        request.kv_cache_group_ids.append(2)
        request.block_ids_by_group.append([11])
        receiver = worker.kv_recv_thread
        receiver.start()
        self.assertTrue(receiver.ready_event.wait(timeout=2))
        try:
            worker.start_load_kv(SimpleNamespace(requests=[request]))
            worker.finish_cross_layer_kv_loads()
            self.assertEqual(bulk_keys, [keys[1]])
            self.assertEqual(request.kv_cache_group_ids, [0, 1, 2])
        finally:
            receiver.stop()
            receiver.join(timeout=2)

    def test_bulk_failure_status_aborts_before_draft_and_survives_notification_clear(self):
        for result in [None, [-1], [], [0, 0]]:
            with self.subTest(result=result):
                worker, request, _keys, _bulk_keys = self._fixture()
                worker.m_store.get = lambda *_args, result=result: result
                receiver = worker.kv_recv_thread
                receiver.start()
                self.assertTrue(receiver.ready_event.wait(timeout=2))
                try:
                    with self.assertRaises(RuntimeError):
                        worker.start_load_kv(SimpleNamespace(requests=[request]))
                    self.assertEqual(worker._failed_requests, {"r"})
                    self.assertTrue(worker._save_disabled.is_set())
                    self.assertTrue(all(task.done.is_set() for task in worker._cross_layer_kv_load_tasks))
                    worker._failed_requests.clear()
                    with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
                        worker.prepare_cross_layer_kv_step(SimpleNamespace(requests=[]))
                finally:
                    receiver.stop()
                    receiver.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
