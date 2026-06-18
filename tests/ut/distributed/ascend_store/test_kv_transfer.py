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
from unittest.mock import MagicMock

import numpy as np
from vllm.distributed.kv_events import BlockStored

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    LoadSpec,
    PoolKey,
    ReqMeta,
    SharedBlockData,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreKeyLayerRecvingThread,
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
    _circular_shift,
    _circular_shift_array,
    record_failed_blocks,
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
        self.group_block_len = {0: [block_size]}
        self.group_kv_caches_base_addr = {0: [1000]}

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


class _FakeTokenDB:
    """Minimal ChunkedTokenDatabase-like object for LayerBatchBuilder.

    ``block_lens`` describes a single layer's caches and is tiled ``num_layers``
    times to match the flat [layer0_caches..., layer1_caches...] layout that
    ``_infer_cache_group_metadata`` produces in production. ``base_addrs`` is
    already expected flat (one entry per cache per layer).
    """

    def __init__(self, block_lens=None, base_addrs=None, num_layers=1, block_strides=None):
        per_layer = block_lens or [128, 256]
        flat_block_len = list(per_layer) * num_layers
        self.group_block_len = {0: flat_block_len}
        self.group_kv_caches_base_addr = {0: base_addrs or [1000, 2000]}
        # group_block_stride mirrors group_block_len when not provided.
        self.group_block_stride = {0: block_strides or flat_block_len}


class _FakePoolKey:
    def __init__(self, chunk_hash):
        self.chunk_hash = chunk_hash

    def split_layers(self, num_layers):
        return [_FakePoolKey(f"{self.chunk_hash}_L{i}") for i in range(num_layers)]

    def to_string(self):
        return f"key_{self.chunk_hash}"


class _FakeStore:
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


class _FakeDB:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self._hash_block_size = block_size
        self.group_block_len = {0: [block_size]}
        self.group_kv_caches_base_addr = {0: [1000]}
        # LayerBatchBuilder reads group_block_stride; mirror group_block_len.
        self.group_block_stride = {0: [block_size]}

    def _make_key_by_hash(self, chunk_hash, **kwargs):
        return _FakePoolKey(chunk_hash)

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

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = block_ids[start // self.block_size]
        return [2000 + layer_id * 100 + block_id], [end - start], block_id


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


# ===========================================================================
# _circular_shift / _circular_shift_array
# ===========================================================================
class TestCircularShift(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_circular_shift([], 3), [])

    def test_zero_offset(self):
        self.assertEqual(_circular_shift([1, 2, 3], 0), [1, 2, 3])

    def test_positive_offset(self):
        self.assertEqual(_circular_shift([1, 2, 3, 4], 2), [3, 4, 1, 2])

    def test_offset_equals_length(self):
        self.assertEqual(_circular_shift([1, 2, 3], 3), [1, 2, 3])

    def test_single_element(self):
        self.assertEqual(_circular_shift([42], 1), [42])


class TestCircularShiftArray(unittest.TestCase):
    def test_empty(self):
        result = _circular_shift_array(np.array([], dtype=np.int64), 3)
        self.assertEqual(len(result), 0)

    def test_zero_offset(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = _circular_shift_array(arr, 0)
        np.testing.assert_array_equal(result, arr)

    def test_positive_offset(self):
        arr = np.array([10, 20, 30, 40], dtype=np.int64)
        result = _circular_shift_array(arr, 2)
        np.testing.assert_array_equal(result, [30, 40, 10, 20])

    def test_offset_mod(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        # offset=5 => 5%3=2
        result = _circular_shift_array(arr, 5)
        np.testing.assert_array_equal(result, [3, 1, 2])

    def test_offset_equals_length(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        # offset=3 => 3%3=0, should return original
        result = _circular_shift_array(arr, 3)
        np.testing.assert_array_equal(result, arr)


# ===========================================================================
# record_failed_blocks
# ===========================================================================
class TestRecordFailedBlocks(unittest.TestCase):
    def test_all_success(self):
        result = record_failed_blocks([10, 20, 30], [0, 0, 0])
        self.assertEqual(result, set())

    def test_some_failed(self):
        result = record_failed_blocks([10, 20, 30], [0, -1, 0])
        self.assertEqual(result, {20})

    def test_all_failed(self):
        result = record_failed_blocks([5, 6], [-1, -1])
        self.assertEqual(result, {5, 6})

    def test_empty(self):
        result = record_failed_blocks([], [])
        self.assertEqual(result, set())


# ===========================================================================
# LayerBatchBuilder
# ===========================================================================
class TestLayerBatchBuilderBuildTransferArrays(unittest.TestCase):
    """Test _build_transfer_arrays which computes per-layer addresses."""

    def _make_builder(self, block_lens=None, base_addrs=None, num_ranks=1, page_size=4096, num_layers=1):
        db = _FakeTokenDB(block_lens=block_lens, base_addrs=base_addrs, num_layers=num_layers)
        return LayerBatchBuilder(
            token_database=db,
            my_key_index=0,
            num_ranks_per_layer=num_ranks,
            page_size_bytes=page_size,
            num_layers=num_layers,
        )

    def test_single_block_single_rank(self):
        builder = self._make_builder(block_lens=[100], base_addrs=[1000], num_ranks=1, page_size=4096)
        block_ids = np.array([5], dtype=np.int64)
        base_gvas = np.array([0x10000], dtype=np.int64)
        addrs, sizes, gvas = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=0)
        # 1 block * 1 sub-block per block = 1 entry
        self.assertEqual(len(addrs), 1)
        self.assertEqual(len(sizes), 1)
        self.assertEqual(len(gvas), 1)
        self.assertEqual(addrs[0], 1000 + 5 * 100)
        self.assertEqual(sizes[0], 100)
        # gva = base_gva + (layer_id * num_ranks + my_key_index) * page_size + inner_offset
        # layer_id=0, inner_offset=0
        self.assertEqual(gvas[0], 0x10000 + 0 * 4096 + 0)

    def test_two_sub_blocks_per_block(self):
        builder = self._make_builder(block_lens=[50, 60], base_addrs=[1000, 2000])
        block_ids = np.array([3], dtype=np.int64)
        base_gvas = np.array([0x20000], dtype=np.int64)
        addrs, sizes, gvas = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=0)
        # 1 block * 2 sub-blocks = 2 entries
        self.assertEqual(len(addrs), 2)
        self.assertEqual(sizes[0], 50)
        self.assertEqual(sizes[1], 60)

    def test_multi_block(self):
        builder = self._make_builder(block_lens=[100], base_addrs=[1000])
        block_ids = np.array([1, 2, 3], dtype=np.int64)
        base_gvas = np.array([0x10000, 0x20000, 0x30000], dtype=np.int64)
        addrs, sizes, gvas = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=0)
        self.assertEqual(len(addrs), 3)
        self.assertEqual(addrs[0], 1000 + 1 * 100)
        self.assertEqual(addrs[1], 1000 + 2 * 100)
        self.assertEqual(addrs[2], 1000 + 3 * 100)

    def test_layer_offset(self):
        builder = self._make_builder(
            block_lens=[100, 200],
            base_addrs=[1000, 2000, 3000, 4000],  # 2 caches * 2 layers (flat)
            num_ranks=2,
            page_size=4096,
            num_layers=2,
        )
        block_ids = np.array([0], dtype=np.int64)
        base_gvas = np.array([0x10000], dtype=np.int64)
        addrs, sizes, gvas = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=1)
        # layer_id=1 => base_offset=1*2=2, so uses addrs[2], addrs[3]
        self.assertEqual(addrs[0], 3000)
        self.assertEqual(addrs[1], 4000)
        # gva: rank_layer_offset = (1*2+0)*4096 = 8192
        expected_gva_offset = (1 * 2 + 0) * 4096
        self.assertEqual(gvas[0], 0x10000 + expected_gva_offset)

    def test_non_zero_my_key_index(self):
        """GVA offset uses my_key_index correctly."""
        db = _FakeTokenDB(block_lens=[100], base_addrs=[1000])
        builder = LayerBatchBuilder(
            token_database=db,
            my_key_index=2,
            num_ranks_per_layer=4,
            page_size_bytes=4096,
            num_layers=1,
        )
        block_ids = np.array([0], dtype=np.int64)
        base_gvas = np.array([0x50000], dtype=np.int64)
        addrs, sizes, gvas = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=1)
        # rank_layer_offset = (layer_id * num_ranks + my_key_index) * page_size
        # = (1 * 4 + 2) * 4096 = 6 * 4096 = 24576
        expected_gva_offset = (1 * 4 + 2) * 4096
        self.assertEqual(gvas[0], 0x50000 + expected_gva_offset)


class TestLayerBatchBuilderDedupe(unittest.TestCase):
    def test_single_element(self):
        ids = np.array([1], dtype=np.int64)
        gvas = np.array([100], dtype=np.int64)
        out_ids, out_gvas = LayerBatchBuilder._dedupe_transfer_blocks(ids, gvas)
        np.testing.assert_array_equal(out_ids, ids)
        np.testing.assert_array_equal(out_gvas, gvas)

    def test_no_duplicates(self):
        ids = np.array([1, 2, 3], dtype=np.int64)
        gvas = np.array([100, 200, 300], dtype=np.int64)
        out_ids, out_gvas = LayerBatchBuilder._dedupe_transfer_blocks(ids, gvas)
        self.assertEqual(len(out_ids), 3)

    def test_with_duplicates(self):
        ids = np.array([1, 2, 1], dtype=np.int64)
        gvas = np.array([100, 200, 100], dtype=np.int64)
        out_ids, out_gvas = LayerBatchBuilder._dedupe_transfer_blocks(ids, gvas)
        self.assertEqual(len(out_ids), 2)


class TestLayerBatchBuilderBuildShared(unittest.TestCase):
    """Test build_shared which pre-computes SharedBlockData."""

    def _make_builder(self):
        db = _FakeTokenDB(block_lens=[100], base_addrs=[1000])
        return LayerBatchBuilder(
            token_database=db, my_key_index=0, num_ranks_per_layer=1, page_size_bytes=4096, num_layers=1
        )

    def _make_req(self, req_id="r1", block_ids=None, gvas=None, is_last_chunk=True, last_block_gva=None):
        block_ids = block_ids or [5, 6]
        gvas = gvas or [0x1000, 0x2000]
        return ReqMeta(
            req_id=req_id,
            block_ids_by_group=[block_ids],
            block_hashes=[b"\xaa", b"\xbb"],
            can_save=True,
            is_last_chunk=is_last_chunk,
            block_ids_np=np.asarray(block_ids, dtype=np.int64),
            block_gvas_np=np.asarray(gvas, dtype=np.int64),
            gva_block_offset=0,
            last_block_gva=last_block_gva,
        )

    def test_empty_block_ranges(self):
        builder = self._make_builder()
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        result = builder.build_shared(task)
        self.assertIsNone(result)

    def test_single_range_two_blocks(self):
        builder = self._make_builder()
        req = self._make_req()
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=2)],
        )
        result = builder.build_shared(task)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SharedBlockData)
        self.assertEqual(result.req_ids, ["r1"])
        self.assertEqual(result.is_last_chunks, [True])
        np.testing.assert_array_equal(result.block_ids_arr, [5, 6])
        np.testing.assert_array_equal(result.block_gvas_arr, [0x1000, 0x2000])

    def test_partial_block(self):
        builder = self._make_builder()
        req = self._make_req(block_ids=[5, 6, 7], gvas=[0x1000, 0x2000, 0x3000], last_block_gva=0x9999)
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[
                LayerBlockRange(request=req, start_block=0, end_block=2, partial_block_index=2),
            ],
        )
        result = builder.build_shared(task)
        self.assertIsNotNone(result)
        # Should have 2 full blocks + 1 partial block
        self.assertEqual(len(result.block_ids_arr), 3)
        self.assertEqual(result.block_ids_arr[2], 7)
        self.assertEqual(result.block_gvas_arr[2], 0x9999)

    def test_multiple_ranges_dedup(self):
        builder = self._make_builder()
        req1 = self._make_req(req_id="r1", block_ids=[5, 6], gvas=[0x1000, 0x2000])
        req2 = self._make_req(req_id="r2", block_ids=[5, 7], gvas=[0x1000, 0x3000])
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[
                LayerBlockRange(request=req1, start_block=0, end_block=2),
                LayerBlockRange(request=req2, start_block=0, end_block=2),
            ],
        )
        result = builder.build_shared(task)
        # block_id=5 appears in both ranges (with same gva), should be deduped
        self.assertIsNotNone(result)
        unique_block_ids = set(result.block_ids_arr.tolist())
        self.assertIn(5, unique_block_ids)

    def test_gva_offset_out_of_range_raises(self):
        """build_shared raises RuntimeError when gva offset exceeds block_gvas_np."""
        builder = self._make_builder()
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            can_save=True,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
            gva_block_offset=1,  # offset=1 but only 1 element => gva_start=1 > len=1
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        with self.assertRaises(RuntimeError):
            builder.build_shared(task)

    def test_gva_negative_offset_raises(self):
        """build_shared raises RuntimeError when computed gva_start is negative."""
        builder = self._make_builder()
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            can_save=True,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
            gva_block_offset=-1,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        with self.assertRaises(RuntimeError):
            builder.build_shared(task)


class TestLayerBatchBuilderBuildAddrs(unittest.TestCase):
    """Test build_addrs which computes per-layer addresses from shared data."""

    def _make_builder(self):
        db = _FakeTokenDB(block_lens=[100], base_addrs=[1000])
        return LayerBatchBuilder(
            token_database=db, my_key_index=0, num_ranks_per_layer=1, page_size_bytes=4096, num_layers=1
        )

    def test_basic(self):
        builder = self._make_builder()
        shared = SharedBlockData(
            block_ids_arr=np.array([5], dtype=np.int64),
            block_gvas_arr=np.array([0x10000], dtype=np.int64),
            req_ids=["r1"],
            is_last_chunks=[True],
        )
        result = builder.build_addrs(shared, layer_id=0)
        self.assertEqual(result.layer_id, 0)
        self.assertEqual(result.req_ids, ["r1"])
        self.assertEqual(len(result.addr_array), 1)
        self.assertEqual(result.addr_array[0], 1000 + 5 * 100)


class TestLayerBatchBuilderBuild(unittest.TestCase):
    """Test full build method (build_shared + build_addrs)."""

    def _make_builder(self):
        db = _FakeTokenDB(block_lens=[100], base_addrs=[1000])
        return LayerBatchBuilder(
            token_database=db, my_key_index=0, num_ranks_per_layer=1, page_size_bytes=4096, num_layers=1
        )

    def test_empty_task(self):
        builder = self._make_builder()
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        result = builder.build(task)
        self.assertIsNone(result)

    def test_full_build(self):
        builder = self._make_builder()
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            can_save=True,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x10000], dtype=np.int64),
            gva_block_offset=0,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        result = builder.build(task)
        self.assertIsNotNone(result)
        self.assertEqual(result.layer_id, 0)
        self.assertEqual(len(result.addr_array), 1)


class TestLayerBatchBuilderRequireRequestArrays(unittest.TestCase):
    def test_missing_block_ids_np(self):
        db = _FakeTokenDB()
        _builder = LayerBatchBuilder(
            token_database=db, my_key_index=0, num_ranks_per_layer=1, page_size_bytes=4096, num_layers=1
        )
        req = ReqMeta(req_id="r1")
        block_range = LayerBlockRange(request=req, start_block=0, end_block=1)
        with self.assertRaises(RuntimeError):
            LayerBatchBuilder._require_request_arrays(block_range)

    def test_valid_arrays(self):
        req = ReqMeta(
            req_id="r1",
            block_ids_np=np.asarray([1, 2], dtype=np.int64),
            block_gvas_np=np.asarray([100, 200], dtype=np.int64),
        )
        block_range = LayerBlockRange(request=req, start_block=0, end_block=1)
        ids, gvas = LayerBatchBuilder._require_request_arrays(block_range)
        np.testing.assert_array_equal(ids, [1, 2])
        np.testing.assert_array_equal(gvas, [100, 200])


class TestLayerBatchBuilderScratchArray(unittest.TestCase):
    def _make_builder(self):
        db = _FakeTokenDB()
        return LayerBatchBuilder(
            token_database=db, my_key_index=0, num_ranks_per_layer=1, page_size_bytes=4096, num_layers=1
        )

    def test_ensure_creates_new(self):
        builder = self._make_builder()
        ids, gvas = builder._ensure_buf(10)
        self.assertEqual(ids.shape[0], 10)
        self.assertEqual(gvas.shape[0], 10)

    def test_ensure_reuses(self):
        builder = self._make_builder()
        ids1, _ = builder._ensure_buf(10)
        ids1[0] = 42
        # Requesting a smaller view reuses the existing buffer.
        ids2, _ = builder._ensure_buf(5)
        self.assertEqual(ids2[0], 42)

    def test_ensure_grows(self):
        builder = self._make_builder()
        builder._ensure_buf(5)
        ids, _ = builder._ensure_buf(20)
        self.assertEqual(ids.shape[0], 20)


# ===========================================================================
# _split_transfer_packets
# ===========================================================================
class TestSplitTransferPackets(unittest.TestCase):
    def _call(self, sizes, max_bytes):
        gvas: np.ndarray = np.arange(len(sizes), dtype=np.int64)
        addrs: np.ndarray = np.arange(len(sizes), dtype=np.int64) * 100
        sizes_arr = np.array(sizes, dtype=np.int64)
        out_gvas, out_addrs, out_sizes = KVTransferThread._split_transfer_packets(gvas, addrs, sizes_arr, max_bytes)
        return out_gvas, out_addrs, out_sizes

    def test_no_split_when_disabled(self):
        gvas, addrs, sizes = self._call([100, 200], max_bytes=0)
        self.assertEqual(len(gvas), 2)

    def test_no_split_when_sizes_fit(self):
        gvas, addrs, sizes = self._call([50, 50], max_bytes=100)
        self.assertEqual(len(gvas), 2)

    def test_split_large_entry(self):
        gvas, addrs, sizes = self._call([100], max_bytes=30)
        # 100/30=4 splits => 30+30+30+10
        self.assertEqual(len(gvas), 4)
        self.assertEqual(sizes[0], 30)
        self.assertEqual(sizes[-1], 10)

    def test_split_preserves_gvas_and_addrs(self):
        gvas, addrs, sizes = self._call([100], max_bytes=50)
        # 100/50=2 splits
        self.assertEqual(len(gvas), 2)
        self.assertEqual(gvas[0], 0)
        self.assertEqual(gvas[1], 50)
        self.assertEqual(addrs[0], 0)
        self.assertEqual(addrs[1], 50)


# ===========================================================================
# KVCacheStoreKeyLayerSendingThread
# ===========================================================================
class TestKVCacheStoreKeyLayerSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, num_layers=2):
        store = _FakeStore(exists_result or [0, 0])
        db = _FakeDB()
        t = KVCacheStoreKeyLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=num_layers,
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            sync_save_events=[MagicMock() for _ in range(num_layers)],
        )
        return t, store

    def test_empty_tasks(self):
        t, store = self._make_thread()
        t.request_queue.put([])
        t._handle_request([])
        self.assertEqual(len(store.put_calls), 0)

    def test_too_many_tasks_raises(self):
        t, store = self._make_thread()
        task1 = LayerTransferTask(layer_id=0, block_ranges=[])
        task2 = LayerTransferTask(layer_id=1, block_ranges=[])
        with self.assertRaises(ValueError):
            t._handle_request([task1, task2])

    def test_puts_missing_keys(self):
        t, store = self._make_thread(exists_result=[1, 0], num_layers=2)
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[0, 1]],
            block_hashes=[b"\xaa", b"\xbb"],
            can_save=True,
            save_end_token=32,
            save_start_token=0,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=2)],
        )
        t.add_stored_request("r1")
        t.request_queue.put([task])
        t._handle_request([task])
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)

    def test_final_layer_marks_finished(self):
        t, store = self._make_thread(exists_result=[0], num_layers=2)
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[0]],
            block_hashes=[b"\xaa"],
            can_save=True,
            save_end_token=16,
            save_start_token=0,
            is_last_chunk=True,
        )
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        t.add_stored_request("r1")
        t.request_queue.put([task])
        t._handle_request([task])
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_non_final_layer_no_finish(self):
        t, store = self._make_thread(exists_result=[0], num_layers=2)
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[0]],
            block_hashes=[b"\xaa"],
            can_save=True,
            save_end_token=16,
            save_start_token=0,
            is_last_chunk=True,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        t.add_stored_request("r1")
        t.request_queue.put([task])
        t._handle_request([task])
        finished = t.get_and_clear_finished_requests()
        self.assertNotIn("r1", finished)

    def test_cached_process_tokens(self):
        t, store = self._make_thread(exists_result=[0], num_layers=2)
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[0]],
            block_hashes=[b"\xaa"],
            can_save=True,
            save_end_token=16,
            save_start_token=0,
        )
        task = LayerTransferTask(layer_id=0, block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)])
        cached = t.build_cached_process_tokens(task)
        self.assertIsNotNone(cached)
        self.assertIn(0, cached)


# ===========================================================================
# KVCacheStoreKeyLayerRecvingThread
# ===========================================================================
class TestKVCacheStoreKeyLayerRecvingThread(unittest.TestCase):
    def _make_thread(self, num_layers=2):
        store = _FakeStore()
        db = _FakeDB()
        get_event = threading.Event()
        t = KVCacheStoreKeyLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
            layer_load_finished_events=[threading.Event() for _ in range(num_layers)],
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            num_layers=num_layers,
        )
        return t, store, get_event

    def test_empty_transfer_tasks(self):
        t, store, get_event = self._make_thread()
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[],
            layer_id=0,
        )
        t.request_queue.put(data)
        t._handle_request(data)
        self.assertTrue(t.layer_load_finished_events[0].is_set())
        # get_event is always set at the end of _handle_request
        self.assertTrue(get_event.is_set())

    def test_basic_recv(self):
        t, store, get_event = self._make_thread()
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[0]],
            block_hashes=[b"\xaa"],
            load_spec=MagicMock(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=16,
                can_load=True,
                token_len=16,
            ),
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task],
            layer_id=0,
        )
        t.request_queue.put(data)
        t._handle_request(data)
        self.assertEqual(len(store.get_calls), 1)
        self.assertTrue(get_event.is_set())
        self.assertTrue(t.layer_load_finished_events[0].is_set())

    def test_final_layer_marks_finished(self):
        t, store, get_event = self._make_thread(num_layers=2)
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[0]],
            block_hashes=[b"\xaa"],
            is_last_chunk=True,
        )
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task],
            layer_id=1,
        )
        t.request_queue.put(data)
        t._handle_request(data)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_too_many_tasks_raises(self):
        t, store, get_event = self._make_thread()
        task1 = LayerTransferTask(layer_id=0, block_ranges=[])
        task2 = LayerTransferTask(layer_id=1, block_ranges=[])
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task1, task2],
            layer_id=0,
        )
        with self.assertRaises(ValueError):
            t._handle_request(data)


# ===========================================================================
# KVCacheStoreLayerSendingThread (GVA path additional tests)
# ===========================================================================
class TestKVCacheStoreLayerSendingThreadGVA(unittest.TestCase):
    def _make_thread(self, num_layers=2, max_transfer_blocks=0, max_transfer_bytes=0):
        store = _FakeStore([0, 0])
        db = _FakeDB()
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
            page_size_bytes=4096,
            ready_event=threading.Event(),
            num_layers=num_layers,
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            sync_save_events=[MagicMock() for _ in range(num_layers)],
            max_transfer_blocks=max_transfer_blocks,
            max_transfer_bytes=max_transfer_bytes,
        )
        return t, store

    def test_build_shared_data(self):
        t, store = self._make_thread()
        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            can_save=True,
            save_end_token=16,
            save_start_token=0,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
            gva_block_offset=0,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        shared = t.build_shared_data(task)
        self.assertIsNotNone(shared)
        np.testing.assert_array_equal(shared.block_ids_arr, [5])

    def test_stored_request_management(self):
        t, _ = self._make_thread()
        t.add_stored_request("r1")
        t.add_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 2)
        t.dec_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 1)
        t.delete_finished_stored_request("r1")
        self.assertNotIn("r1", t.stored_requests)

    def test_empty_tasks_gva_path(self):
        t, store = self._make_thread()
        t.request_queue.put([])
        t._handle_request([])
        self.assertEqual(len(store.put_calls), 0)
        # Empty task path returns immediately without setting save finished event
        self.assertFalse(t.layer_save_finished_events[0].is_set())

    def test_too_many_tasks_raises(self):
        t, store = self._make_thread()
        task1 = LayerTransferTask(layer_id=0, block_ranges=[])
        task2 = LayerTransferTask(layer_id=1, block_ranges=[])
        with self.assertRaises(ValueError):
            t._handle_request([task1, task2])

    def test_gva_path_with_shared_block_data(self):
        """Test the GVA batch_copy path when shared_block_data is provided."""
        t, store = self._make_thread(num_layers=2)
        t.m_store.store = MagicMock()
        t.m_store.store.batch_copy.return_value = 0

        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            can_save=True,
            save_end_token=16,
            save_start_token=0,
            is_last_chunk=True,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
            gva_block_offset=0,
        )
        shared = SharedBlockData(
            block_ids_arr=np.array([5], dtype=np.int64),
            block_gvas_arr=np.array([0x1000], dtype=np.int64),
            req_ids=["r1"],
            is_last_chunks=[True],
        )
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
            shared_block_data=shared,
        )
        t.add_stored_request("r1")
        t.request_queue.put([task])
        t._handle_request([task])
        # GVA path calls batch_copy
        t.m_store.store.batch_copy.assert_called_once()
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_gva_path_batch_copy_failure(self):
        """Test GVA path when batch_copy returns non-zero."""
        t, store = self._make_thread(num_layers=2)
        t.m_store.store = MagicMock()
        t.m_store.store.batch_copy.return_value = -1

        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            can_save=True,
            save_end_token=16,
            save_start_token=0,
            is_last_chunk=True,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
            gva_block_offset=0,
        )
        shared = SharedBlockData(
            block_ids_arr=np.array([5], dtype=np.int64),
            block_gvas_arr=np.array([0x1000], dtype=np.int64),
            req_ids=["r1"],
            is_last_chunks=[True],
        )
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
            shared_block_data=shared,
        )
        t.add_stored_request("r1")
        t.request_queue.put([task])
        t._handle_request([task])
        # batch_copy failure is logged but the request is still reported as
        # finished (best-effort transfer) so the pipeline is not stalled.
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)


# ===========================================================================
# KVCacheStoreLayerRecvingThread (GVA path additional tests)
# ===========================================================================
class TestKVCacheStoreLayerRecvingThreadGVA(unittest.TestCase):
    def _make_thread(self, num_layers=2, stagger_us=0):
        store = _FakeStore()
        db = _FakeDB()
        get_event = threading.Event()
        t = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            my_key_index=0,
            num_ranks_per_layer=1,
            page_size_bytes=4096,
            ready_event=threading.Event(),
            get_event=get_event,
            layer_load_finished_events=[threading.Event() for _ in range(num_layers)],
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            num_layers=num_layers,
            h2d_stagger_us=stagger_us,
        )
        return t, store, get_event

    def test_empty_transfer_tasks(self):
        t, store, get_event = self._make_thread()
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[],
            layer_id=0,
        )
        t.request_queue.put(data)
        t._handle_request(data)
        self.assertTrue(t.layer_load_finished_events[0].is_set())

    def test_too_many_tasks_raises(self):
        t, store, get_event = self._make_thread()
        task1 = LayerTransferTask(layer_id=0, block_ranges=[])
        task2 = LayerTransferTask(layer_id=1, block_ranges=[])
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task1, task2],
            layer_id=0,
        )
        with self.assertRaises(ValueError):
            t._handle_request(data)

    def test_stagger_delay_zero(self):
        t, _, _ = self._make_thread(stagger_us=0)
        delay = t._get_h2d_stagger_delay_us(layer_id=0)
        self.assertEqual(delay, 0)

    def test_stagger_delay_basic(self):
        t, _, _ = self._make_thread(stagger_us=100)
        # tp_rank=0, layer_id=0, tp_size=1 => slot=0 => delay=0
        delay = t._get_h2d_stagger_delay_us(layer_id=0)
        self.assertEqual(delay, 0)
        # tp_rank=0, layer_id=1, tp_size=4 => slot=(0+1)%4=1 => delay=1*100
        t.tp_size = 4
        delay = t._get_h2d_stagger_delay_us(layer_id=1)
        self.assertEqual(delay, 100)

    def test_gva_recv_with_task(self):
        """Test GVA recv path with a transfer task that has shared_block_data."""
        t, store, get_event = self._make_thread(num_layers=2)
        t.m_store.store = MagicMock()
        t.m_store.store.batch_copy.return_value = 0

        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            is_last_chunk=True,
            save_end_token=16,
            save_start_token=0,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
            gva_block_offset=0,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task],
            layer_id=0,
        )
        t.request_queue.put(data)
        t._handle_request(data)
        t.m_store.store.batch_copy.assert_called_once()
        self.assertTrue(get_event.is_set())
        self.assertTrue(t.layer_load_finished_events[0].is_set())

    def test_gva_recv_final_layer_marks_finished(self):
        t, store, get_event = self._make_thread(num_layers=2)
        t.m_store.store = MagicMock()
        t.m_store.store.batch_copy.return_value = 0

        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            is_last_chunk=True,
            save_end_token=16,
            save_start_token=0,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
            gva_block_offset=0,
        )
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task],
            layer_id=1,
        )
        t.request_queue.put(data)
        t._handle_request(data)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_gva_recv_batch_copy_failure(self):
        t, store, get_event = self._make_thread(num_layers=2)
        t.m_store.store = MagicMock()
        t.m_store.store.batch_copy.return_value = -1

        req = ReqMeta(
            req_id="r1",
            block_ids_by_group=[[5]],
            block_hashes=[b"\xaa"],
            is_last_chunk=True,
            save_end_token=16,
            save_start_token=0,
            block_ids_np=np.asarray([5], dtype=np.int64),
            block_gvas_np=np.asarray([0x1000], dtype=np.int64),
            gva_block_offset=0,
        )
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[LayerBlockRange(request=req, start_block=0, end_block=1)],
        )
        data = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task],
            layer_id=1,
        )
        t.request_queue.put(data)
        t._handle_request(data)
        # batch_copy failure is logged but the request is still reported as
        # finished (best-effort transfer) so the pipeline is not stalled.
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)


# ===========================================================================
# KVTransferThread base class additional tests
# ===========================================================================
class TestKVTransferThreadBase(unittest.TestCase):
    def test_discard_finished_requests(self):
        store = _FakeStore()
        db = _FakeDB()
        t = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        t.set_finished_request("r1")
        t.set_finished_request("r2")
        t.discard_finished_requests({"r1"})
        finished = t.get_and_clear_finished_requests()
        self.assertNotIn("r1", finished)
        self.assertIn("r2", finished)

    def test_try_finish_and_delete_stored_request(self):
        store = _FakeStore()
        db = _FakeDB()
        t = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        # Not yet stored
        self.assertFalse(t.try_finish_and_delete_stored_request("r1"))
        t.add_stored_request("r1")
        # Count is 1, not 0
        self.assertFalse(t.try_finish_and_delete_stored_request("r1"))
        t.dec_stored_request("r1")
        # Count is 0 now
        self.assertTrue(t.try_finish_and_delete_stored_request("r1"))

    def test_get_block_size_list(self):
        store = _FakeStore()
        db = _FakeDB()
        t = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=[16, 32],
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        self.assertEqual(t._get_block_size(0), 16)
        self.assertEqual(t._get_block_size(1), 32)
        # Out of range returns first
        self.assertEqual(t._get_block_size(5), 16)

    def test_skip_null_blocks(self):
        store = _FakeStore()
        db = _FakeDB()
        KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        req = ReqMeta(req_id="r1", skip_null_blocks_by_group=[True])
        self.assertTrue(KVTransferThread._skip_null_blocks(req, 0))
        self.assertFalse(KVTransferThread._skip_null_blocks(req, 1))  # out of range

        req2 = ReqMeta(req_id="r2")
        self.assertFalse(KVTransferThread._skip_null_blocks(req2, 0))

    def test_skip_null_blocks_non_kv_role(self):
        store = _FakeStore()
        db = _FakeDB()
        KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        req = ReqMeta(req_id="r1", skip_null_blocks_by_group=[True])
        self.assertFalse(KVTransferThread._skip_null_blocks(req, 0, cache_role="state"))


if __name__ == "__main__":
    unittest.main()
