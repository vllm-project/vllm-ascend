#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerRangeReqMeta,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LayerTransferTask,
    LoadSpec,
    PoolKey,
    ReqMeta,
)

# isort: on
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
)


class FakeStore:
    def __init__(self, exists_result=None):
        self.exists_result = exists_result or []
        self.put_calls = []
        self.get_calls = []

    def set_device(self):
        pass

    def exists(self, keys):
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))


class FakeKey:
    def __init__(self, val):
        self._val = val

    def to_string(self):
        return self._val


class FakeTokenDatabase:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.group_block_len = [[block_size, block_size]]
        self.group_kv_caches_base_addr = [[0, block_size]]
        self.group_block_stride = {0: [block_size, block_size]}

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        for i, h in enumerate(block_hashes):
            start = i * self.block_size
            if start >= token_len:
                break
            end = min(start + self.block_size, token_len)
            if start < mask_num:
                continue
            yield start, end, PoolKey(meta, f"k{i}")

    def prepare_value(self, start, end, block_ids):
        block_id = block_ids[start // self.block_size]
        return [1000 + block_id], [end - start], block_id

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = block_ids[start // self.block_size]
        return [2000 + layer_id * 100 + block_id], [end - start], block_id

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes):
        return keys, addrs, sizes


class MaskedFakeTokenDatabase(FakeTokenDatabase):
    def __init__(self, block_size=16, masks=([True],)):
        super().__init__(block_size)
        self.masks = masks

    def store_mask(self, token_len, num_prompt_tokens=None):
        return self.masks

    def load_mask(self, block_hashes, token_len):
        return self.masks

    def mask_allows_chunk(self, masks, kv_cache_group_id, start):
        if masks is None:
            return True
        block_idx = start // self.block_size
        return block_idx < len(masks[kv_cache_group_id]) and masks[kv_cache_group_id][block_idx]


class RangeBatchFakeTokenDatabase(FakeTokenDatabase):
    def __init__(self):
        super().__init__()
        self.group_block_len = [[64, 32, 64, 32, 64, 32]]
        self.group_kv_caches_base_addr = [[100, 200, 500, 600, 1000, 2000]]
        self.group_block_stride = {0: [100, 50, 100, 50, 100, 50]}


class TestLayerBatchBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = LayerBatchBuilder(
            RangeBatchFakeTokenDatabase(),
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=96,
            num_layers=3,
        )

    def _build(self, block_ids, block_keys):
        request = ReqMeta(
            req_id="r1",
            block_ids=block_ids,
            save_block_keys=block_keys,
        )
        task = LayerTransferTask(
            layer_id=2,
            block_ranges=[LayerBlockRange(request, 0, len(block_ids))],
            use_key_major_ranges=True,
        )
        return self.builder.build(task)

    def test_builds_memcache_shared_data_without_block_keys(self):
        request = ReqMeta(
            req_id="r1",
            block_ids=[1],
            block_ids_np=np.asarray([1], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request, 0, 1)],
        )

        shared = self.builder.build_shared(task)

        self.assertIsNotNone(shared)
        assert shared is not None
        self.assertIsNone(shared.block_keys)
        np.testing.assert_array_equal(shared.block_ids_arr, [1])
        np.testing.assert_array_equal(shared.block_gvas_arr, [0x1000])

    def test_builds_key_major_range_batch_for_two_blocks_and_two_segments(self):
        batch = self._build([1, 2], ["key-1", "key-2"])

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertIsInstance(batch, LayerRangeReqMeta)
        self.assertEqual(batch.req_ids, ["r1"])
        self.assertEqual(batch.layer_id, 2)
        self.assertEqual(batch.block_ids, [1, 2])
        self.assertEqual(batch.keys, ["key-1", "key-2"])
        self.assertEqual(batch.all_buffers, [[1100, 2050], [1200, 2100]])
        self.assertEqual(batch.all_sizes, [[64, 32], [64, 32]])
        self.assertEqual(batch.all_offsets, [[192, 256], [192, 256]])

    def test_reuses_key_major_shared_data_across_layers(self):
        request = ReqMeta(
            req_id="r1",
            block_ids=[1, 2],
            save_block_keys=["key-1", "key-2"],
        )
        task = LayerTransferTask(
            layer_id=2,
            block_ranges=[LayerBlockRange(request, 0, 2)],
            use_key_major_ranges=True,
        )
        shared = self.builder.build_shared(task)

        self.assertIsNotNone(shared)
        assert shared is not None
        layer_0_batch = self.builder.build_addrs(shared, 0)
        layer_2_batch = self.builder.build_addrs(shared, 2)

        for batch in (layer_0_batch, layer_2_batch):
            self.assertIsInstance(batch, LayerRangeReqMeta)
            self.assertEqual(batch.block_ids, [1, 2])
            self.assertEqual(batch.keys, ["key-1", "key-2"])
            self.assertEqual(batch.all_sizes, [[64, 32], [64, 32]])

        self.assertEqual(layer_0_batch.all_buffers, [[200, 250], [300, 300]])
        self.assertEqual(layer_0_batch.all_offsets, [[0, 64], [0, 64]])
        self.assertEqual(layer_2_batch.all_buffers, [[1100, 2050], [1200, 2100]])
        self.assertEqual(layer_2_batch.all_offsets, [[192, 256], [192, 256]])

    def test_partial_only_failed_session_stays_an_empty_key_major_batch(self):
        request = ReqMeta(
            req_id="r1",
            block_ids=[1],
            save_last_block_key=None,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request, 0, 0, partial_block_index=0)],
            use_key_major_ranges=True,
        )

        shared = self.builder.build_shared(task)

        self.assertIsNotNone(shared)
        assert shared is not None
        self.assertEqual(shared.block_keys, [])
        np.testing.assert_array_equal(shared.block_ids_arr, [])
        batch = self.builder.build_addrs(shared, 0)
        self.assertIsInstance(batch, LayerRangeReqMeta)
        self.assertEqual(batch.keys, [])
        self.assertEqual(batch.all_buffers, [])

    def test_filters_none_key_with_its_aligned_block_and_range_values(self):
        batch = self._build([1, 2, 3], ["key-1", None, "key-3"])

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.block_ids, [1, 3])
        self.assertEqual(batch.keys, ["key-1", "key-3"])
        self.assertEqual(batch.all_buffers, [[1100, 2050], [1300, 2150]])
        self.assertEqual(batch.all_sizes, [[64, 32], [64, 32]])
        self.assertEqual(batch.all_offsets, [[192, 256], [192, 256]])


class TestKVTransferThread(unittest.TestCase):
    def _make_thread(self, exists_result=None):
        store = FakeStore(exists_result or [])
        db = FakeTokenDatabase()
        t = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        return t, store

    def test_add_request(self):
        t, _ = self._make_thread()
        req = MagicMock()
        t.add_request(req)
        self.assertFalse(t.request_queue.empty())

    def test_get_and_clear_finished_requests(self):
        t, _ = self._make_thread()
        t.set_finished_request("r1")
        t.set_finished_request("r2")
        finished = t.get_and_clear_finished_requests()
        self.assertEqual(finished, {"r1", "r2"})
        self.assertEqual(t.get_and_clear_finished_requests(), set())

    def test_lookup_all_exist(self):
        t, _ = self._make_thread([1, 1, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, True, True])

    def test_lookup_partial(self):
        t, _ = self._make_thread([1, 0, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, False, True])

    def test_lookup_exception(self):
        t, store = self._make_thread()
        store.exists = MagicMock(side_effect=Exception("conn fail"))
        result = t.lookup(["k1"])
        self.assertEqual(result, [False])

    def test_update_and_get_kv_events(self):
        t, _ = self._make_thread()
        event1 = BlockStored(
            block_hashes=["h1"],
            parent_block_hash=None,
            token_ids=[1, 2, 3],
            block_size=16,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        event2 = BlockStored(
            block_hashes=["h2"],
            parent_block_hash="h1",
            token_ids=[4, 5, 6],
            block_size=16,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        t.update_kv_event([event1, event2])
        events = t.get_kv_events()
        self.assertEqual(len(events), 2)
        # After get, events should be cleared
        self.assertEqual(len(t.get_kv_events()), 0)

    def test_handle_request_base_noop(self):
        t, _ = self._make_thread()
        # Base class _handle_request does nothing
        t._handle_request(MagicMock())


class TestKVCacheStoreSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, kv_role="kv_producer", enable_kv_event=False):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role=kv_role,
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=enable_kv_event,
        )
        return t, store

    def test_handle_request_puts_missing_keys(self):
        t, store = self._make_thread([1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 2)

    def test_handle_request_all_exist_no_put(self):
        t, store = self._make_thread([1, 1])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_not_in_stored(self):
        t, store = self._make_thread([0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_with_kv_event(self):
        t, store = self._make_thread([0], enable_kv_event=True)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
            token_ids=list(range(16)),
            original_block_size=16,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)

    def test_handle_request_consumer_role(self):
        t, store = self._make_thread([0], kv_role="kv_consumer")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)

    def test_add_dec_delete_stored_request(self):
        t, _ = self._make_thread()
        t.add_stored_request("r1")
        t.add_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 2)
        t.dec_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 1)
        t.delete_finished_stored_request("r1")
        self.assertNotIn("r1", t.stored_requests)

    def test_dec_nonexistent_request(self):
        t, _ = self._make_thread()
        t.dec_stored_request("nonexist")  # should not raise

    def test_delete_nonexistent_request(self):
        t, _ = self._make_thread()
        t.delete_finished_stored_request("nonexist")  # should not raise

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=event,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_dcp_size_gt_1(self):
        store = FakeStore([0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=2,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        # dcp_size > 1 means no slicing
        self.assertEqual(len(store.put_calls), 1)

    def test_handle_request_applies_store_mask(self):
        store = FakeStore([0, 0])
        db = MaskedFakeTokenDatabase(masks=([True, False],))
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)


class TestKVCacheStoreRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_applies_load_mask(self):
        store = FakeStore()
        db = MaskedFakeTokenDatabase(masks=([True, False],))
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        keys, _, _ = store.get_calls[0]
        self.assertEqual(len(keys), 1)


@unittest.skip("LayerMultiBlockReqMeta API is deprecated, tests need update for LayerTransferTask")
class _DeprecatedKVCacheStoreLayerSendingThreadTests(unittest.TestCase):
    __test__ = False

    def _make_thread(self, exists_result=None, num_layers=2):
        store = FakeStore(exists_result or [0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=32,
            ready_event=threading.Event(),
            num_layers=num_layers,
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            sync_save_events=[],
        )
        return t, store

    def _make_layer_req(self, layer_id=0, is_last_chunk=False, num_keys=2):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta, f"h{i}", layer_id) for i in range(num_keys)]
        return LayerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[i * 16 for i in range(num_keys)],
            ends=[(i + 1) * 16 for i in range(num_keys)],
            block_ids=list(range(num_keys)),
            layer_id=layer_id,
            is_last_chunk=is_last_chunk,
            current_event=None,
            token_ids=list(range(num_keys * 16)),
            original_block_size=16,
            block_hashes=[f"h{i}".encode() for i in range(num_keys)],
        )

    def test_handle_request_puts_missing(self):
        t, store = self._make_thread([1, 0])
        req = self._make_layer_req(layer_id=0)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)

    def test_handle_request_all_exist_not_last(self):
        t, store = self._make_thread([1, 1])
        req = self._make_layer_req(layer_id=0, is_last_chunk=False)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_all_exist_last_chunk_final_layer(self):
        t, store = self._make_thread([1, 1], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_empty_keys(self):
        t, store = self._make_thread()
        _meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[],
            starts=[],
            ends=[],
            block_ids=[],
            layer_id=0,
            is_last_chunk=True,
        )
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertNotIn("r1", finished)

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
            is_last_chunk=False,
            current_event=event,
        )
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_last_chunk_final_layer_with_missing(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_layerwise_kv_event_published_on_final_layer(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [maybe_convert_block_hash(b"h0")])
        self.assertEqual(events[0].token_ids, list(range(16)))
        self.assertEqual(events[0].block_size, 16)

    def test_layerwise_kv_event_not_published_before_final_layer(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=0, is_last_chunk=False, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(t.get_kv_events(), [])

    def test_layerwise_kv_event_uses_missing_blocks_from_previous_layers(self):
        t, store = self._make_thread([0], num_layers=2)
        first_layer_req = self._make_layer_req(layer_id=0, is_last_chunk=True, num_keys=1)
        t.add_stored_request(first_layer_req.req_id)
        t.request_queue.put(first_layer_req)
        t._handle_request(first_layer_req)
        t.m_store.exists_result = [1]
        final_layer_req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.request_queue.put(final_layer_req)
        t._handle_request(final_layer_req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [maybe_convert_block_hash(b"h0")])


@unittest.skip("LayerMultiBlockReqMeta API is deprecated, tests need update for LayerTransferTask")
class _DeprecatedKVCacheStoreLayerRecvingThreadTests(unittest.TestCase):
    __test__ = False

    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        get_event = threading.Event()
        t = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        self.assertTrue(get_event.is_set())


class TestKVCacheStoreLayerFinalization(unittest.TestCase):
    @staticmethod
    def _make_send_thread():
        store = FakeStore()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=FakeTokenDatabase(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=32,
            ready_event=threading.Event(),
            num_layers=1,
            layer_save_finished_events=[threading.Event()],
            sync_save_events=[MagicMock()],
        )
        request = ReqMeta(
            req_id="r1",
            block_ids=[1],
            block_ids_np=np.asarray([1], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
        )
        task = LayerTransferTask(0, [LayerBlockRange(request, 0, 1)])
        task.shared_block_data = thread.build_shared_data(task)
        return thread, [task]

    @staticmethod
    def _make_recv_thread():
        store = FakeStore()
        invalid_block_ids: set[int] = set()
        get_event = threading.Event()
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=FakeTokenDatabase(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=32,
            ready_event=threading.Event(),
            get_event=get_event,
            layer_load_finished_events=[threading.Event()],
            layer_save_finished_events=[threading.Event()],
            num_layers=1,
            invalid_block_ids=invalid_block_ids,
            invalid_block_ids_lock=threading.Lock(),
        )
        request = ReqMeta(
            req_id="r1",
            block_ids=[3],
            block_ids_np=np.asarray([3], dtype=np.int64),
            load_block_gvas_np=np.asarray([0x2000], dtype=np.int64),
        )
        task = LayerTransferTask(0, [LayerBlockRange(request, 0, 1)])
        task.shared_block_data = thread.build_shared_data(task)
        return thread, LayerLoadTask(None, [task], 0), invalid_block_ids, get_event

    def test_save_exception_finishes_queue_and_request_accounting(self):
        thread, tasks = self._make_send_thread()
        thread.add_stored_request("r1")
        thread.request_queue.put(tasks)
        thread.layer_batch_builder.build_addrs = MagicMock(side_effect=RuntimeError("probe failed"))

        with patch.object(thread.request_queue, "task_done", wraps=thread.request_queue.task_done) as task_done:
            thread._handle_request(tasks)

        self.assertEqual(task_done.call_count, 1)
        self.assertTrue(thread.layer_save_finished_events[0].is_set())
        self.assertEqual(dict(thread.stored_requests), {})
        self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})
        self.assertEqual(thread.get_kv_events(), [])

    def test_load_exception_finishes_queue_and_marks_blocks_invalid(self):
        thread, data, invalid_block_ids, get_event = self._make_recv_thread()
        thread.request_queue.put(data)
        thread.layer_batch_builder.build_addrs = MagicMock(side_effect=RuntimeError("probe failed"))

        with patch.object(thread.request_queue, "task_done", wraps=thread.request_queue.task_done) as task_done:
            thread._handle_request(data)

        self.assertEqual(task_done.call_count, 1)
        self.assertTrue(thread.layer_load_finished_events[0].is_set())
        self.assertTrue(get_event.is_set())
        self.assertEqual(invalid_block_ids, {3})


class TestKVTransferTpMismatchDispatch(unittest.TestCase):
    """TP-mismatch worker dispatch wiring for Sending/Recving threads."""

    def _make_sending(self, worker=None, exists_result=None):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=False,
            worker=worker,
        )
        return t, store

    def _make_recving(self, worker=None):
        store = FakeStore([0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
            worker=worker,
        )
        return t, store

    def test_sending_dispatches_to_worker_when_tp_mismatch(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_sending(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._store_kv_tp_mismatch.assert_called_once_with(req)

    def test_sending_normal_path_when_worker_none(self):
        # worker=None -> tp_mismatch dispatch skipped, normal store path runs.
        t, store = self._make_sending(worker=None, exists_result=[1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)  # normal path executed

    def test_recving_dispatches_to_worker_when_tp_mismatch(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        req.load_spec = MagicMock()
        req.load_spec.token_len = 16
        req.load_spec.vllm_cached_tokens = 0
        t.request_queue.put(req)
        t._handle_request(req)
        worker._load_kv_tp_mismatch.assert_called_once()
        args = worker._load_kv_tp_mismatch.call_args.args
        # (block_hashes, block_ids, token_len, mask_num)
        self.assertEqual(args[2], 16)  # token_len
        self.assertEqual(args[3], 0)  # mask_num

    def test_recving_tp_mismatch_missing_load_spec_finishes(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._load_kv_tp_mismatch.assert_not_called()
        self.assertEqual(t.get_and_clear_finished_requests(), {"r1"})
        self.assertEqual(t.request_queue.unfinished_tasks, 0)

    def test_recving_tp_mismatch_task_done_on_exception(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        worker._load_kv_tp_mismatch.side_effect = RuntimeError("load failed")
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        req.load_spec = MagicMock()
        req.load_spec.token_len = 16
        req.load_spec.vllm_cached_tokens = 0
        t.request_queue.put(req)
        with self.assertRaises(RuntimeError):
            t._handle_request(req)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
