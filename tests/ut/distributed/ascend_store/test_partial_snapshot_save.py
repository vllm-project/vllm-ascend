# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401

# isort: split
from tests.ut.distributed.ascend_store import test_bulk_full_blocks as bulk_tests
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
)


class TestPartialSnapshotSave(unittest.TestCase):
    def setUp(self):
        self.fixture = bulk_tests.TestBulkFullBlocks()
        self.fixture.setUp()
        self.worker = self.fixture.worker
        self.worker._partial_snapshot_namespace = "writer-epoch"
        self.worker.kv_role = "kv_both"
        self.worker.consumer_is_to_put = False
        self.worker._save_disabled = threading.Event()
        self.fixture.sender.worker = self.worker

    def snapshots(self):
        return {key: value for key, value in self.fixture.objects.items() if "@snapshot:" in key}

    def test_bulk_partial_then_full_in_mixed_and_separate_pd(self):
        for role in ("kv_both", "kv_producer"):
            with self.subTest(role=role):
                self.fixture.objects.clear()
                self.worker.kv_role = role
                self.fixture.sender._handle_stored_request(self.fixture.request(1792))
                committed = dict(self.fixture.objects)
                self.fixture.sender._handle_stored_request(self.fixture.request(1921))
                snapshots = self.snapshots()
                self.assertEqual(len(snapshots), 2)
                attention = next(value for key, value in snapshots.items() if "@group:0@" in key)
                state = next(value for key, value in snapshots.items() if "@group:1@" in key)
                self.assertEqual(attention, ctypes.string_at(ctypes.addressof(self.fixture.attention) + 3 * 896, 129))
                self.assertEqual(state, ctypes.string_at(ctypes.addressof(self.fixture.state) + 3 * 73, 73))
                self.assertTrue(all(key.endswith(":1792:1921") for key in snapshots))
                self.assertEqual({key: self.fixture.objects[key] for key in committed}, committed)
                # The consumer probes the same shared pool, using only prefix
                # hashes. P's request-private tail cannot increase the hit.
                reader = self.fixture.scheduler(False)
                reader.kv_role = "kv_consumer" if role == "kv_producer" else "kv_both"
                reader.consumer_is_to_load = True
                request = SimpleNamespace(
                    request_id="different-reader",
                    prompt_token_ids=list(range(2817)),
                    num_tokens=2817,
                    block_hashes=self.fixture.hashes,
                )
                self.assertEqual(reader.get_num_new_matched_tokens(request, 0), (1792, False))
                self.assertEqual(reader.load_specs[request.request_id].kvpool_cached_tokens, 1792)
                next_state = 101 if role == "kv_both" else 102
                ctypes.memset(ctypes.addressof(self.fixture.state) + 3 * 73, next_state, 73)
                ctypes.memset(ctypes.addressof(self.fixture.attention) + 3 * 896, next_state + 1, 896)
                self.fixture.sender._handle_stored_request(self.fixture.request(2688))
                self.assertEqual(self.fixture.lookup(3713), 2688)
                third_key = list(self.fixture.db.process_token_key_strings(2688, self.fixture.hashes))[2][2]
                state_key = list(
                    self.fixture.db.process_token_key_strings(2688, self.fixture.hashes, kv_cache_group_id=1)
                )[2][2]
                self.assertEqual(self.fixture.objects[third_key], bytes([next_state + 1]) * 896)
                self.assertEqual(self.fixture.objects[state_key], bytes([next_state]) * 73)
                self.assertEqual(self.snapshots(), snapshots)
                for call in self.fixture.store.exists.call_args_list:
                    self.assertTrue(all("@snapshot:" not in key for key in call.args[0]))

    def test_partial_only_is_never_a_hit_even_if_snapshot_put_succeeds(self):
        self.fixture.sender._handle_stored_request(self.fixture.request(1921))
        self.assertEqual(len(self.snapshots()), 2)
        self.assertEqual(self.fixture.lookup(3713), 0, "Attention without a matching state checkpoint cannot hit")
        self.assertFalse(any("@group:1@" in key for key in self.fixture.objects if "@snapshot:" not in key))

    def test_state_uses_actual_forward_end_not_floored_save_extent(self):
        self.fixture.sender._handle_stored_request(self.fixture.request(1792, target=1921))
        states = {key: value for key, value in self.fixture.objects.items() if "@group:1@" in key}
        self.assertEqual(len(states), 1)
        key, state = next(iter(states.items()))
        self.assertIn("@snapshot:", key)
        self.assertTrue(key.endswith(":1792:1921"))
        self.assertEqual(state, ctypes.string_at(ctypes.addressof(self.fixture.state) + 3 * 73, 73))
        self.assertEqual(self.fixture.lookup(3713), 0)

    def test_no_tail_hash_is_needed_and_future_speculative_slots_are_not_copied(self):
        request = self.fixture.request(1921)
        request.block_hashes = self.fixture.hashes[:2]
        self.worker._save_partial_snapshots([request])
        self.assertEqual(len(self.snapshots()), 2)
        state = next(value for key, value in self.snapshots().items() if "@group:1@" in key)
        self.assertEqual(state, ctypes.string_at(ctypes.addressof(self.fixture.state) + 3 * 73, 73))

    def test_null_state_is_not_saved(self):
        request = self.fixture.request(1921)
        request.block_ids_by_group[1][2] = 0
        self.worker._save_partial_snapshots([request])
        self.assertEqual(len(self.snapshots()), 1)
        self.assertTrue(all("@group:0@" in key for key in self.snapshots()))

    def test_partial_does_not_emit_reusable_block_events(self):
        self.fixture.sender.enable_kv_event = True
        self.fixture.sender.update_kv_event = MagicMock()
        self.fixture.sender._handle_stored_request(self.fixture.request(129))
        self.assertEqual(len(self.snapshots()), 2)
        self.fixture.sender.update_kv_event.assert_not_called()

    def test_private_circular_groups_are_not_addressed_as_token_blocks(self):
        self.fixture.db.cache_coordinator = SimpleNamespace(cacheable_group_ids=[0])
        self.worker._save_partial_snapshots([self.fixture.request(1921)])
        self.assertEqual(len(self.snapshots()), 1)
        self.assertTrue(all("@group:0@" in key for key in self.snapshots()))

    def test_configuration_roles_and_failure_do_not_publish_hits(self):
        for reason in ("discard", "consumer", "disabled", "cannot_save", "aligned", "empty", "put_failure"):
            with self.subTest(reason=reason):
                self.fixture.objects.clear()
                self.worker._partial_snapshot_namespace = None if reason == "discard" else "writer-epoch"
                self.worker.kv_role = "kv_consumer" if reason == "consumer" else "kv_both"
                self.worker._save_disabled.clear()
                if reason == "disabled":
                    self.worker._save_disabled.set()
                end = 1792 if reason == "aligned" else 0 if reason == "empty" else 1921
                request = self.fixture.request(end, can_save=reason != "cannot_save")
                if reason == "put_failure":
                    with patch.object(self.fixture.store, "put", return_value=False):
                        self.worker._save_partial_snapshots([request])
                else:
                    self.worker._save_partial_snapshots([request])
                self.assertFalse(self.fixture.objects)

    def test_snapshot_identity_separates_requests_workers_and_boundaries(self):
        self.worker._save_partial_snapshots([self.fixture.request(1921)])
        request = self.fixture.request(1921)
        request.req_id = "another-request"
        self.worker._save_partial_snapshots([request])
        self.worker._partial_snapshot_namespace = "another-writer-epoch"
        self.worker._save_partial_snapshots([request])
        self.worker._save_partial_snapshots([self.fixture.request(1922)])
        self.assertEqual(len(self.snapshots()), 8)

    def configure_layerwise(self, mode):
        worker = self.worker
        worker.num_layers = 2
        worker.use_block_key_layerwise = mode == "block_key"
        worker.use_layerwise_transfer = mode == "gva"
        worker.use_layerwise = True
        worker.block_key_hybrid = mode == "block_key"
        worker.current_layer = 0
        # The last layer has already recorded its attention-entry event. A
        # fresh post-compute event must fence the whole-state snapshot instead.
        worker._attention_saved_layers = {1}
        worker.sync_save_events = [MagicMock(), MagicMock()]
        worker.layer_save_finished_events = [threading.Event(), threading.Event()]
        worker.layer_save_tasks = [[], []]
        worker.kv_send_thread = MagicMock()
        worker._finish_attention_window = MagicMock()
        worker._wait_for_final_layer_save = MagicMock()

    def test_layerwise_snapshots_wait_for_all_layers_in_mixed_and_pd(self):
        for mode in ("gva", "block_key", "legacy"):
            for role in ("kv_both", "kv_producer"):
                with self.subTest(mode=mode, role=role):
                    self.fixture.objects.clear()
                    self.configure_layerwise(mode)
                    self.worker.kv_role = role
                    meta = AscendConnectorMetadata(set())
                    meta.add_request(self.fixture.request(1921))
                    event = MagicMock()
                    order = []
                    event.record.side_effect = lambda order=order: order.append("record")
                    event.synchronize.side_effect = lambda order=order: order.append("synchronize")

                    def put(*args, order=order):
                        order.append("put")
                        return self.fixture.put(*args)

                    with (
                        patch(
                            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.torch.npu.Event",
                            return_value=event,
                        ),
                        patch.object(self.fixture.store, "put", side_effect=put),
                    ):
                        self.worker.save_kv_layer(meta)
                        self.assertFalse(self.snapshots(), "An earlier layer cannot publish the snapshot")
                        self.worker.save_kv_layer(meta)
                    self.assertEqual(order, ["record", "synchronize", "put"])
                    self.assertEqual(len(self.snapshots()), 2)
                    self.assertEqual(self.fixture.lookup(3713), 0)
                    self.assertEqual(self.worker.current_layer, 2)

    def test_layerwise_failure_never_creates_reusable_keys(self):
        self.configure_layerwise("gva")
        self.worker.current_layer = 1
        meta = AscendConnectorMetadata(set())
        meta.add_request(self.fixture.request(1921))
        with patch.object(self.fixture.store, "put", return_value=False):
            self.worker.save_kv_layer(meta)
        self.assertFalse(self.fixture.objects)
        self.assertEqual(self.fixture.lookup(3713), 0)

    def test_segmented_attention_and_whole_state_keep_distinct_payload_lengths(self):
        db = ChunkedTokenDatabase([KeyMetadata("test", 0, 0, 0, group) for group in range(2)], [16, 32], None)
        first = ctypes.create_string_buffer(bytes(range(96)))
        second = ctypes.create_string_buffer(bytes(range(100, 196)))
        state = ctypes.create_string_buffer(bytes(range(21)))
        db.set_group_buffers(
            {0: [ctypes.addressof(first), ctypes.addressof(second)], 1: [ctypes.addressof(state)]},
            {0: [8, 8], 1: [7]},
            group_block_stride={0: [16, 16]},
            group_cache_entry_segments={0: [(0, 2), (1, 2)]},
        )
        request = self.fixture.request(29)
        request.block_ids_by_group = [[2], [1]]
        rows = list(db.prepare_partial_snapshots(request, [False, True], "epoch"))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][2], [8, 5])
        self.assertEqual(rows[0][1], [ctypes.addressof(first) + 32, ctypes.addressof(second) + 32])
        self.assertEqual(rows[1][2], [7], "State byte length is independent of the partial token count")


if __name__ == "__main__":
    unittest.main()
