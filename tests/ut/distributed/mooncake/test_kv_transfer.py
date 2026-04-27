import threading
import unittest
from types import SimpleNamespace

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LayerMultiBlockReqMeta,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    KVCacheStoreSendingThread,
)


class _FakeKey:
    def __init__(self, value: str):
        self._value = value

    def to_string(self) -> str:
        return self._value


class _FakeStore:
    def __init__(self, exists_result: list[int]):
        self.exists_result = exists_result
        self.put_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []

    def set_device(self):
        return None

    def exists(self, keys: list[str]) -> list[int]:
        # Return exact number of states for requested keys.
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))


class _FakeTokenDatabase:
    def process_tokens(self, token_len, block_hashes):
        for i, _ in enumerate(block_hashes):
            yield i * 16, (i + 1) * 16, _FakeKey(f"k{i}")

    def process_tokens_with_block_ids(
        self,
        token_len,
        block_hashes,
        block_ids,
        mask_num=0,
        kv_cache_group_id=0,
        skip_null_blocks=False,
    ):
        for i, _ in enumerate(block_hashes):
            block_id = block_ids[i]
            if skip_null_blocks and block_id <= 0:
                continue
            yield i * 16, (i + 1) * 16, _FakeKey(f"k{i}"), block_id

    def prepare_value(self, start, end, block_ids, kv_cache_group_id=0):
        block_id = start // 16
        return [1000 + block_id], [end - start], block_id

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = start // 16
        return [2000 + layer_id * 100 + block_id], [end - start]


class TestKVTransferMissingKeyPut(unittest.TestCase):
    def test_sending_thread_only_puts_missing_keys(self):
        store = _FakeStore(exists_result=[1, 0, 1, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            enable_kv_event=False,
        )

        req_meta = ReqMeta(
            req_id="req-1",
            token_len_chunk=64,
            block_ids_by_group=[[0, 1, 2, 3]],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            kv_cache_group_ids=[0],
            current_event=None,
        )
        thread.add_stored_request("req-1")
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 1)
        put_keys, put_addrs, put_sizes = store.put_calls[0]
        self.assertEqual(put_keys, ["k1", "k3"])
        self.assertEqual(put_addrs, [[1001], [1003]])
        self.assertEqual(put_sizes, [[16], [16]])

    def test_layer_sending_thread_only_puts_missing_keys(self):
        store = _FakeStore(exists_result=[1, 0, 1, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=2,
            enable_kv_event=False,
        )

        req_meta = LayerMultiBlockReqMeta(
            req_id="req-2",
            keys=[_FakeKey("k0"), _FakeKey("k1"), _FakeKey("k2"), _FakeKey("k3")],  # type: ignore[arg-type]
            starts=[0, 16, 32, 48],
            ends=[16, 32, 48, 64],
            block_ids_by_group=[[0, 1, 2, 3]],
            layer_id=1,
            is_last_chunk=False,
            current_event=None,
        )
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 1)
        put_keys, put_addrs, put_sizes = store.put_calls[0]
        self.assertEqual(put_keys, ["k1", "k3"])
        self.assertEqual(put_addrs, [[2101], [2103]])
        self.assertEqual(put_sizes, [[16], [16]])


if __name__ == "__main__":
    unittest.main()
