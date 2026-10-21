# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class TestBulkFullBlocks(unittest.TestCase):
    def setUp(self):
        self.hashes = [i.to_bytes(32, "big") for i in range(6)]
        self.db = ChunkedTokenDatabase([KeyMetadata("test", 0, 0, 0, g) for g in range(2)], [896, 896], None)
        self.attention = ctypes.create_string_buffer(bytes(i % 251 for i in range(6 * 896)))
        self.state = ctypes.create_string_buffer(bytes(i % 253 for i in range(6 * 73)))
        self.db.set_group_buffers(
            {0: [ctypes.addressof(self.attention)], 1: [ctypes.addressof(self.state)]},
            {0: [896], 1: [73]},
        )
        self.objects = {}
        self.store = MagicMock()
        self.store.requires_exists_before_put = True
        self.store.exists.side_effect = lambda keys: [int(key in self.objects) for key in keys]
        self.store.put.side_effect = self.put
        self.sender = KVCacheStoreSendingThread(
            self.store, self.db, [896, 896], tp_rank=0, group_uses_align_state=[False, True]
        )
        self.worker = KVPoolWorker.__new__(KVPoolWorker)
        self.worker.cache_coordinator = None
        self.worker.token_database = self.db
        self.worker.max_model_len = 8192
        self.worker.group_uses_align_state = [False, True]
        self.worker.cache_transfer_granularity = 896
        self.worker.m_store = self.store
        self.worker._expand_lookup_keys_by_rank = lambda keys, group: keys

    def put(self, keys, addrs, sizes):
        for key, addresses, lengths in zip(keys, addrs, sizes, strict=True):
            value = b"".join(ctypes.string_at(addr, size) for addr, size in zip(addresses, lengths, strict=True))
            self.objects.setdefault(key, value)
        return True

    def request(self, end, target=None, can_save=True):
        return ReqMeta(
            req_id="req",
            token_len_chunk=end,
            target_token_len=end if target is None else target,
            block_hashes=self.hashes,
            block_ids_by_group=[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            kv_cache_group_ids=[0, 1],
            can_save=can_save,
        )

    def lookup(self, end):
        return self.worker.lookup_scheduler(end, self.hashes, [0, 1])

    def scheduler(self, discard):
        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.kv_role = "kv_both"
        scheduler.use_block_key_layerwise = False
        scheduler.use_layerwise_transfer = False
        scheduler.use_layerwise = False
        scheduler._discard_partial_chunks = discard
        scheduler.cache_transfer_granularity = 896
        scheduler.client = SimpleNamespace(lookup=self.worker.lookup_scheduler)
        scheduler.kv_cache_group_ids = [0, 1]
        scheduler.use_eagle = False
        scheduler.cacheable_group_ids = [0, 1]
        scheduler.original_block_size = [896, 896]
        scheduler.load_async = False
        scheduler.load_specs = {}
        return scheduler

    def test_scheduler_queries_complete_checkpoint_before_final_prompt_token(self):
        self.sender._handle_stored_request(self.request(896))
        self.sender._handle_stored_request(self.request(1792))
        for discard in (False, True):
            for end, expected in ((896, 0), (1792, 896), (1921, 1792)):
                with self.subTest(discard=discard, end=end):
                    scheduler = self.scheduler(discard)
                    request = SimpleNamespace(
                        request_id="req", prompt_token_ids=list(range(end)), num_tokens=end, block_hashes=self.hashes
                    )
                    self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (expected, False))
                    if expected:
                        self.assertEqual(scheduler.load_specs["req"].kvpool_cached_tokens, expected)

    def test_prompt_end_state_does_not_create_nonexistent_earlier_checkpoint(self):
        self.sender._handle_stored_request(self.request(1792))
        request = SimpleNamespace(
            request_id="req", prompt_token_ids=list(range(1792)), num_tokens=1792, block_hashes=self.hashes
        )
        scheduler = self.scheduler(False)
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))
        self.assertFalse(scheduler.load_specs)

    def test_bulk_read_and_write_iterators_exclude_partial_tails(self):
        for end in (129, 896, 1921, 2946):
            with self.subTest(end=end):
                read = list(self.db.process_token_key_strings(end, self.hashes))
                write = list(self.db.process_token_key_strings_with_block_ids(end, self.hashes, [1, 2, 3, 4, 5]))
                self.assertEqual(len(read), end // 896)
                self.assertEqual([item[:4] for item in write], read)
                self.assertTrue(all(stop - start == 896 for start, stop, *_ in read))

    def test_partial_save_cannot_block_later_full_payload(self):
        self.sender._handle_stored_request(self.request(1921))
        third_key = list(self.db.process_token_key_strings(2688, self.hashes))[2][2]
        self.assertNotIn(third_key, self.objects)
        self.sender._handle_stored_request(self.request(2688))
        self.assertEqual(len(self.objects[third_key]), 896)
        self.assertEqual(self.objects[third_key], ctypes.string_at(ctypes.addressof(self.attention) + 3 * 896, 896))
        self.assertEqual(self.lookup(3713), 2688)

    def test_unaligned_state_cannot_impersonate_earlier_boundary(self):
        for saved_end in (1792, 1921):
            with self.subTest(saved_end=saved_end):
                self.objects.clear()
                self.sender._handle_stored_request(self.request(saved_end, target=1921))
                self.assertEqual(len(self.objects), 2)
                self.assertFalse(any("@group:1@" in key for key in self.objects))
                self.assertEqual(self.lookup(3713), 0)

    def test_aligned_state_saves_only_the_actual_checkpoint(self):
        self.sender._handle_stored_request(self.request(2688))
        state_keys = [key for key in self.objects if "@group:1@" in key]
        self.assertEqual(len(state_keys), 1)
        self.assertEqual(self.objects[state_keys[0]], ctypes.string_at(ctypes.addressof(self.state) + 3 * 73, 73))
        self.assertEqual(self.lookup(3713), 2688)

    def test_unaligned_progress_keeps_previous_full_checkpoint_reusable(self):
        self.sender._handle_stored_request(self.request(896))
        self.sender._handle_stored_request(self.request(1921))
        self.assertEqual(self.lookup(3713), 896)
        self.store.get_size.assert_not_called()

    def test_full_hit_restores_matching_payload_sizes_and_bytes(self):
        self.sender._handle_stored_request(self.request(1921))
        self.sender._handle_stored_request(self.request(2688))
        attention = ctypes.string_at(ctypes.addressof(self.attention) + 896, 2688)
        state = ctypes.string_at(ctypes.addressof(self.state) + 3 * 73, 73)
        ctypes.memset(ctypes.addressof(self.attention), 0, len(self.attention))
        ctypes.memset(ctypes.addressof(self.state), 0, len(self.state))

        def get(keys, addrs, sizes):
            for key, addresses, lengths in zip(keys, addrs, sizes, strict=True):
                payload = self.objects[key]
                self.assertEqual(sum(lengths), len(payload))
                offset = 0
                for addr, size in zip(addresses, lengths, strict=True):
                    ctypes.memmove(addr, payload[offset : offset + size], size)
                    offset += size
            return [0] * len(keys)

        self.store.get.side_effect = get
        self.worker.use_layerwise = False
        self.worker.load_async = False
        self.worker.tp_mismatch = False
        self.worker.grouped_block_size = [896, 896]
        self.worker.tp_rank = 0
        self.worker.backend_name = "mooncake"
        self.worker._record_kv_connector_operation = MagicMock()
        request = self.request(3713)
        request.block_ids_by_group[1] = [0, 0, 3, 0, 0]
        request.load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=2688, can_load=True)
        self.worker.start_load_kv(SimpleNamespace(requests=[request]))
        self.store.get.assert_called_once()
        self.assertEqual(ctypes.string_at(ctypes.addressof(self.attention) + 896, 2688), attention)
        self.assertEqual(ctypes.string_at(ctypes.addressof(self.state) + 3 * 73, 73), state)

    def test_bulk_keys_preserve_existing_key_format(self):
        for group in (0, 1):
            with self.subTest(group=group):
                expected = [self.db._get_key_prefix(group) + value.hex() for value in self.hashes[:2]]
                read = list(self.db.process_token_key_strings(1921, self.hashes, kv_cache_group_id=group))
                write = list(
                    self.db.process_token_key_strings_with_block_ids(
                        1921, self.hashes, [1, 2, 3, 4, 5], kv_cache_group_id=group
                    )
                )
                self.assertEqual([row[2] for row in read], expected)
                self.assertEqual([row[2] for row in write], expected)

    def test_key_layerwise_iterator_and_partial_descriptor_are_unchanged(self):
        chunks = list(self.db.process_tokens(1921, self.hashes))
        self.assertEqual(chunks[-1][:2], (1792, 1921))
        _, sizes, _ = self.db.prepare_value(1792, 1921, [1, 2, 3, 4, 5])
        self.assertEqual(sum(sizes), 129)

    def test_group_sizes_and_tail_clipped_block_ids_are_preserved(self):
        db = ChunkedTokenDatabase(
            [KeyMetadata("test", 0, 0, 0, g) for g in range(2)], [128, 896], None, hash_block_size=128
        )
        hashes = [i.to_bytes(32, "big") for i in range(28)]
        for group, block_size in ((0, 128), (1, 896)):
            with self.subTest(group=group):
                # The physical table covers the last two logical chunks,
                # including the partial tail. Do not rebase it by flooring.
                result = list(
                    db.process_token_key_strings_with_block_ids(1921, hashes, [7, 8], kv_cache_group_id=group)
                )
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0][4], 7)
                self.assertEqual(result[0][1] - result[0][0], block_size)

    def test_can_save_false_does_not_publish(self):
        self.sender._handle_stored_request(self.request(896, can_save=False))
        self.assertFalse(self.objects)
        self.store.put.assert_not_called()

    def test_layerwise_recurrent_lookup_selects_checkpoint_before_final_token(self):
        for mode in ("gva", "block_key", "legacy"):
            for length in (896, 1792, 1921, 2817):
                with self.subTest(mode=mode, length=length):
                    scheduler = self.scheduler(False)
                    scheduler.use_layerwise = True
                    scheduler.use_layerwise_transfer = mode == "gva"
                    scheduler.use_block_key_layerwise = mode == "block_key"
                    scheduler.num_speculative_blocks_by_group = {1: 0}
                    available = (896, 1792, 2688)

                    def lookup(request, limit, local, available=available, **kwargs):
                        return max((end for end in available if end <= limit), default=0)

                    probe = MagicMock(side_effect=lookup)
                    scheduler._get_layerwise_hit_tokens = probe
                    scheduler._get_block_key_layerwise_hit_tokens = probe
                    scheduler._get_store_lookup_hit_tokens = probe
                    request = SimpleNamespace(request_id="req", prompt_token_ids=list(range(length)), num_tokens=length)
                    expected = (length - 1) // 896 * 896
                    self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (expected, False))
                    if probe.called:
                        self.assertLess(probe.call_args.args[1], length)
                    else:
                        self.assertEqual(expected, 0)
                    if expected:
                        spec = scheduler.load_specs["req"]
                        self.assertEqual(spec.kvpool_cached_tokens, expected)
                        self.assertEqual(spec.kvpool_store_skip_tokens, expected)
                        self.assertTrue(spec.can_load)

    def test_layerwise_cannot_rewind_the_only_available_recurrent_checkpoint(self):
        scheduler = self.scheduler(False)
        scheduler.use_layerwise = scheduler.use_layerwise_transfer = True
        scheduler.num_speculative_blocks_by_group = {1: 0}
        scheduler._get_layerwise_hit_tokens = lambda request, limit, local: 1792 if limit >= 1792 else 0
        request = SimpleNamespace(request_id="req", prompt_token_ids=list(range(1792)), num_tokens=1792)
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))
        self.assertFalse(scheduler.load_specs)

    def test_attention_only_layerwise_keeps_full_hit_replay_behavior(self):
        scheduler = self.scheduler(False)
        scheduler.use_layerwise = scheduler.use_layerwise_transfer = True
        scheduler.num_speculative_blocks_by_group = {}
        scheduler._get_layerwise_hit_tokens = MagicMock(return_value=1792)
        request = SimpleNamespace(request_id="req", prompt_token_ids=list(range(1792)), num_tokens=1792)
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (1791, False))
        self.assertEqual(scheduler._get_layerwise_hit_tokens.call_args.args[1], 1792)
        self.assertEqual(scheduler.load_specs["req"].kvpool_store_skip_tokens, 1792)
