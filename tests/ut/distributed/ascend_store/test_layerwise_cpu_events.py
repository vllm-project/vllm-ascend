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
from types import SimpleNamespace
from unittest.mock import patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
# isort: on

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cpu_events import LayerwiseCpuEvents


class Readability:
    def __init__(self):
        self.missing = set()
        self.queries = []

    def batch_is_readable(self, keys):
        self.queries.extend(keys)
        return [key not in self.missing for key in keys]


class Coordinator:
    def find_reachable_hit_tokens(self, hashes, tokens, query, **kwargs):
        hits = query(0, hashes, None)
        count = 0
        for block_hash in hashes:
            if block_hash not in hits:
                break
            count += 1
        return min(tokens, count * 128)


class TestLayerwiseCpuEvents(unittest.TestCase):
    def setUp(self):
        self.store = Readability()
        self.worker = SimpleNamespace(
            kv_send_thread=SimpleNamespace(save_disabled=threading.Event()),
            cache_coordinator=Coordinator(),
            layerwise_keys=SimpleNamespace(make_hit_check_keys=lambda group, hash_hex, ranks: [f"target@{hash_hex}"]),
            m_store=self.store,
            kv_cache_config=SimpleNamespace(dspark_draft_only_group_ids=()),
            get_group_tp_size=lambda group: 1,
            cacheable_group_ids={0},
            grouped_block_size=[128],
            hash_block_size=128,
        )
        self.request = SimpleNamespace(
            req_id="request",
            can_save=True,
            save_end_token=256,
            token_ids=list(range(256)),
            block_hashes=[b"a" * 32, b"b" * 32],
        )
        self.publisher = LayerwiseCpuEvents(self.worker)

    def test_complete_commit_publishes_cpu_grains_and_parent_chain(self):
        self.publisher.observe([self.request], set())
        events = self.publisher.drain()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].medium, "cpu")
        self.assertEqual(events[0].block_size, 128)
        self.assertIsNone(events[0].parent_block_hash)
        self.assertEqual(events[1].parent_block_hash, events[0].block_hashes[0])
        self.assertEqual(events[1].token_ids, list(range(128, 256)))
        self.assertFalse(self.publisher.drain())

    def test_incomplete_target_does_not_publish_unfinished_grain(self):
        self.store.missing.add(f"target@{(b'b' * 32).hex()}")
        self.publisher.observe([self.request], set())
        self.assertEqual(len(self.publisher.drain()), 1)
        self.assertFalse(self.publisher.drain())
        self.store.missing.clear()
        self.assertEqual(len(self.publisher.drain()), 1)

    def test_draft_state_is_required(self):
        self.worker._dspark_prefix_keys = SimpleNamespace(make_hit_check_keys=lambda hash_hex: [f"draft@{hash_hex}"])
        self.store.missing.add(f"draft@{(b'a' * 32).hex()}")
        self.publisher.observe([self.request], set())
        self.assertFalse(self.publisher.drain())
        self.store.missing.clear()
        self.assertEqual(len(self.publisher.drain()), 2)

    def test_failed_sender_does_not_publish(self):
        self.publisher.observe([self.request], set())
        self.worker.kv_send_thread.save_disabled.set()
        self.assertFalse(self.publisher.drain())
        self.assertFalse(self.publisher.pending)

    def test_failed_request_is_not_queued(self):
        self.request.can_save = False
        self.publisher.observe([self.request], set())
        self.assertFalse(self.publisher.pending)

    def test_incomplete_token_grain_is_not_published(self):
        self.request.save_end_token = 127
        self.request.token_ids = self.request.token_ids[:127]
        self.publisher.observe([self.request], set())
        self.assertFalse(self.publisher.drain())

    def test_stale_metadata_expires(self):
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cpu_events.time.monotonic",
            return_value=0,
        ):
            self.publisher.observe([self.request], set())
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cpu_events.time.monotonic",
            return_value=301,
        ):
            self.assertFalse(self.publisher.drain())
        self.assertFalse(self.publisher.pending)


if __name__ == "__main__":
    unittest.main()
