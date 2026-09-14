# SPDX-License-Identifier: Apache-2.0
import copy
import ctypes
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from tests.ut.kvpp_utils import indexer_name, layer_name, make_cache_config
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata, LoadSpec, PoolKey, ReqMeta


class ByteStore:
    """Store complete objects using the same address lists as the real backend."""

    requires_exists_before_put = False

    def __init__(self):
        self.objects = {}
        self.get_keys = []

    def exists(self, keys):
        return [int(key in self.objects) for key in keys]

    def put(self, keys, addrs, sizes):
        for key, pointers, lengths in zip(keys, addrs, sizes):
            self.objects[key] = b"".join(ctypes.string_at(int(p), int(n)) for p, n in zip(pointers, lengths))

    def get(self, keys, addrs, sizes):
        self.get_keys.extend(keys)
        results = []
        for key, pointers, lengths in zip(keys, addrs, sizes):
            value = self.objects.get(key)
            if value is None:
                results.append(1)
                continue
            assert len(value) == sum(lengths)
            cursor = 0
            for pointer, length in zip(pointers, lengths):
                ctypes.memmove(int(pointer), value[cursor : cursor + int(length)], int(length))
                cursor += int(length)
            results.append(0)
        return results


class TestKVPPPool(unittest.TestCase):
    def test_unsupported_groups_are_rejected_before_backend_initialization(self):
        for groups in (None, [], [object(), object()]):
            kv_cache_config = None if groups is None else SimpleNamespace(kv_cache_groups=groups)
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, "one logical full-attention"):
                make_worker(self, tp_size=2, kvpp=True, kv_cache_config=kv_cache_config)

    def make_registered_worker(self, rank=0, tp=2, targets=3, mtp=True, blocks=3, reverse=False):
        specs = {
            layer_name(i): AscendMLAAttentionSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.int8)
            for i in range(targets + int(mtp))
        }
        worker = make_worker(
            self,
            tp_rank=rank,
            tp_size=tp,
            num_layers=targets,
            num_hidden_layers=targets,
            use_mla=True,
            kvpp=True,
            kv_cache_config=make_cache_config(specs, num_blocks=blocks),
            extra_config={"backend": "memcache", "load_async": True},
        )
        worker.vllm_config.model_config.hf_config.num_nextn_predict_layers = 1
        worker.max_model_len = 8192
        if mtp:
            worker.vllm_config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=1)
        caches = {}
        # Each layer has one raw allocation with attention, index and scale views.
        for i in range(targets + int(mtp)):
            raw = torch.full((blocks * 20,), i + 1, dtype=torch.uint8)
            caches[layer_name(i)] = (raw[: blocks * 8].view(blocks, 2, 4),)
            caches[indexer_name(i)] = (
                raw[blocks * 8 : blocks * 16].view(blocks, 2, 4),
                raw[blocks * 16 :].view(torch.float16).view(blocks, 2, 1),
            )
        if reverse:
            caches = dict(reversed(list(caches.items())))
        worker._start_kv_transfer_threads = MagicMock()
        with patch("torch.distributed.all_gather_object") as gather:
            worker.register_kv_caches(caches)
            gather.assert_not_called()
        pointers, lengths = worker.m_store.register_buffer.call_args.args
        owned_storages = {
            part.untyped_storage().data_ptr(): part.untyped_storage().nbytes()
            for parts in worker.kv_caches.values()
            for part in parts
        }
        self.assertEqual(dict(zip(pointers, lengths)), owned_storages)
        return worker, caches

    def request(self, worker, cached=0):
        return ReqMeta(
            req_id="r",
            token_len_chunk=6,
            block_ids_by_group=[[0, 1, 2]],
            block_hashes=["aa", "bb", "cc"],
            current_event=MagicMock(),
            num_prompt_tokens=6,
            load_spec=LoadSpec(vllm_cached_tokens=cached, kvpool_cached_tokens=6, can_load=True, token_len=6),
        )

    def transfer(self, worker, store, request, receive=False):
        cls = KVCacheStoreRecvingThread if receive else KVCacheStoreSendingThread
        kwargs = {} if receive else {"put_step": worker.put_step}
        thread = cls(
            store,
            worker.token_database,
            worker.block_size,
            worker.tp_rank,
            tp_size=worker.tp_size,
            worker=worker,
            **kwargs,
        )
        thread.request_queue.put(request)
        if not receive:
            thread.stored_requests[request.req_id] = 1
        thread._handle_request(thread.request_queue.get())
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)
        self.assertEqual(thread.get_and_clear_finished_requests({request.req_id}), {request.req_id})
        self.assertEqual(thread.get_and_clear_finished_requests({request.req_id}), set())
        return thread

    def test_wire_format_off_is_unchanged(self):
        metadata = KeyMetadata("m", 0, 0, 0, 0)
        old = "m@pcp:0@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@aa"
        self.assertEqual(PoolKey(metadata, "aa").to_string(), old)
        metadata.kvpp_layout = "1234"
        self.assertEqual(PoolKey(metadata, "aa").to_string(), old[:-3] + "@kvpp:1234@aa")

    def test_owner_and_mtp_roundtrip_never_write_foreign_views(self):
        store = ByteStore()
        workers = [self.make_registered_worker(rank) for rank in range(2)]
        self.assertEqual(workers[0][0].metadata[0].kvpp_layout, workers[1][0].metadata[0].kvpp_layout)
        for worker, caches in workers:
            self.assertEqual(worker.head_or_tp_rank, worker.tp_rank)
            self.assertEqual(worker.put_step, 1)
            self.assertIn(layer_name(3), worker.kv_caches)
            request = self.request(worker)
            request.load_spec = None
            self.transfer(worker, store, request)
            self.assertEqual(len(store.objects), 3 * (worker.tp_rank + 1))
            expected = {name: [part.clone() for part in parts] for name, parts in caches.items()}
            for parts in worker.kv_caches.values():
                for part in parts:
                    part.zero_()
            self.transfer(worker, store, self.request(worker), receive=True)
            for name, parts in caches.items():
                for actual, wanted in zip(parts, expected[name]):
                    self.assertTrue(torch.equal(actual, wanted), name)
            self.assertEqual(worker.lookup_scheduler(6, ["aa", "bb", "cc"]), 0)  # mock backend
        for worker, _ in workers:
            worker.m_store = store
            self.assertEqual(worker.lookup_scheduler(6, ["aa", "bb", "cc"]), 6)
        # A missing owner's middle object cuts the common contiguous prefix.
        middle = next(key for key in store.objects if "@head_or_tp_rank:1@" in key and key.endswith("@bb"))
        del store.objects[middle]
        self.assertEqual(workers[0][0].lookup_scheduler(6, ["aa", "bb", "cc"]), 2)

    def test_layout_is_independent_of_address_capacity_and_dictionary_order(self):
        first, _ = self.make_registered_worker()
        other, _ = self.make_registered_worker(rank=1, blocks=5, reverse=True)
        self.assertEqual(first.metadata[0].kvpp_layout, other.metadata[0].kvpp_layout)
        changed, _ = self.make_registered_worker(tp=3)
        self.assertNotEqual(first.metadata[0].kvpp_layout, changed.metadata[0].kvpp_layout)
        changed, _ = self.make_registered_worker(mtp=False)
        self.assertNotEqual(first.metadata[0].kvpp_layout, changed.metadata[0].kvpp_layout)

    def test_empty_owner_participates_without_pool_io(self):
        worker, _ = self.make_registered_worker(rank=3, tp=4, targets=2, mtp=False)
        self.assertEqual(worker.group_block_len, {0: []})
        self.assertEqual(worker.kvpp_shard_ranks, {(0, 0): (0, 1)})
        store = ByteStore()
        request = self.request(worker)
        request.load_spec = None
        self.transfer(worker, store, request)
        self.transfer(worker, store, self.request(worker), receive=True)
        self.assertEqual(store.objects, {})
        self.assertEqual(store.get_keys, [])

    def test_mtp_only_owner_has_an_object(self):
        worker, _ = self.make_registered_worker(rank=3, tp=4, targets=2)
        self.assertEqual(set(worker.kv_caches), {layer_name(2), indexer_name(2)})
        self.assertEqual(worker.kvpp_shard_ranks, {(0, 0): (0, 1, 2, 3)})

    def test_get_count_mismatch_invalidates_all_requested_blocks(self):
        worker, _ = self.make_registered_worker()
        for result in (None, [], [0], [0, 0, 0, 0]):
            with self.subTest(result=result):
                store = MagicMock()
                store.get.return_value = result
                thread = self.transfer(worker, store, self.request(worker), receive=True)
                self.assertEqual(thread._invalid_block_ids, {0, 1, 2})

    def test_hbm_prefix_is_not_loaded_again(self):
        worker, _ = self.make_registered_worker()
        store = MagicMock()
        store.get.return_value = [0, 0]
        self.transfer(worker, store, self.request(worker, cached=2), receive=True)
        keys = store.get.call_args.args[0]
        self.assertEqual(len(keys), 2)
        self.assertTrue(all(not key.endswith("@aa") for key in keys))

    def test_receiving_completion_waits_for_actual_io(self):
        worker, _ = self.make_registered_worker()
        entered, release = threading.Event(), threading.Event()

        def delayed_get(*_args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release the get operation")
            return [0, 0, 0]

        store = MagicMock()
        store.get.side_effect = delayed_get
        receiver = KVCacheStoreRecvingThread(
            store, worker.token_database, worker.block_size, worker.tp_rank, worker=worker
        )
        receiver.request_queue.put(self.request(worker))
        io = threading.Thread(target=lambda: receiver._handle_request(receiver.request_queue.get()))
        io.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertEqual(receiver.get_and_clear_finished_requests({"r"}), set())
            self.assertEqual(receiver.request_queue.unfinished_tasks, 1)
        finally:
            release.set()
            io.join(5)
        self.assertFalse(io.is_alive())
        self.assertEqual(receiver.get_and_clear_finished_requests({"r"}), {"r"})
        self.assertEqual(receiver.get_and_clear_finished_requests({"r"}), set())
        self.assertEqual(receiver.request_queue.unfinished_tasks, 0)

    def test_pp_manifest_collects_once_and_lookup_includes_every_stage(self):
        worker, caches = self.make_registered_worker()
        worker.pp_size = 2

        def gather(output, local, group):
            remote = copy.deepcopy(local)
            remote["pp_rank"] = 1
            for component in remote["groups"][0]["components"]:
                component["physical_layer_id"] += 3
            output[:] = [local, remote]

        with (
            patch("torch.distributed.all_gather_object", side_effect=gather) as collective,
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_pp_group"),
        ):
            worker._initialize_kvpp_pool_layout(caches)
        collective.assert_called_once()
        key = PoolKey(worker.metadata[0], "aa").to_string()
        keys = worker._expand_lookup_keys_by_rank([key], 0)
        self.assertEqual(len(keys), 4)
        self.assertEqual(sum("@pp_rank:1@" in key for key in keys), 2)
