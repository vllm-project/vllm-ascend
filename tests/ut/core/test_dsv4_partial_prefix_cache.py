#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Unit tests for the DSv4 partial compressed prefix-cache helpers.

These cover the pure-Python pieces that back ``CompressAttentionManager``'s
partial-hit support: range hashing and the per-``BlockPool`` partial-cache maps.
The maps must live on the ``BlockPool`` instance (not module globals), so the
isolation test below is a regression guard for that requirement.
"""
from types import SimpleNamespace

from tests.ut.base import TestBase
from vllm_ascend.core.single_type_kv_cache_manager import (
    _hash_range,
    _insert_partial_cache,
    get_partial_cached_block,
    remove_partial_cache_entries_for_block,
)


def _block(block_id: int) -> SimpleNamespace:
    # _insert/get/remove only touch ``block_id``.
    return SimpleNamespace(block_id=block_id)


def _pool() -> SimpleNamespace:
    # A bare object is enough: the partial-cache maps are attached lazily.
    return SimpleNamespace()


class TestHashRange(TestBase):

    def setUp(self):
        # 4 hash-blocks of 8 tokens each -> hashes for tokens [0,8,16,24).
        self.block_hashes = [b"h0", b"h1", b"h2", b"h3"]
        self.hash_block_size = 8

    def test_unaligned_returns_none(self):
        self.assertIsNone(_hash_range(self.block_hashes, self.hash_block_size, 0, 12))
        self.assertIsNone(_hash_range(self.block_hashes, self.hash_block_size, 4, 16))

    def test_empty_or_inverted_range_returns_none(self):
        self.assertIsNone(_hash_range(self.block_hashes, self.hash_block_size, 8, 8))
        self.assertIsNone(_hash_range(self.block_hashes, self.hash_block_size, 16, 8))

    def test_out_of_range_returns_none(self):
        # end token 40 -> needs 5 hashes, only 4 available.
        self.assertIsNone(_hash_range(self.block_hashes, self.hash_block_size, 0, 40))

    def test_valid_range_concatenates_hashes(self):
        self.assertEqual(_hash_range(self.block_hashes, self.hash_block_size, 0, 16), b"h0h1")
        self.assertEqual(_hash_range(self.block_hashes, self.hash_block_size, 8, 24), b"h1h2")


class TestPartialPrefixCache(TestBase):

    def setUp(self):
        self.pool = _pool()
        self.group_id = 0

    def test_insert_and_get(self):
        block = _block(7)
        _insert_partial_cache(self.pool, b"abc", self.group_id, block)
        self.assertIs(get_partial_cached_block(self.pool, b"abc", self.group_id), block)

    def test_miss_returns_none(self):
        self.assertIsNone(get_partial_cached_block(self.pool, b"missing", self.group_id))

    def test_group_id_scopes_lookup(self):
        block = _block(7)
        _insert_partial_cache(self.pool, b"abc", 0, block)
        self.assertIsNone(get_partial_cached_block(self.pool, b"abc", 1))

    def test_reinsert_updates_block_and_drops_old_tracking(self):
        old, new = _block(1), _block(2)
        _insert_partial_cache(self.pool, b"abc", self.group_id, old)
        _insert_partial_cache(self.pool, b"abc", self.group_id, new)
        self.assertIs(get_partial_cached_block(self.pool, b"abc", self.group_id), new)
        # Evicting the old block must not drop the entry now owned by the new block.
        remove_partial_cache_entries_for_block(self.pool, old.block_id)
        self.assertIs(get_partial_cached_block(self.pool, b"abc", self.group_id), new)

    def test_eviction_removes_entries_for_block(self):
        block = _block(5)
        _insert_partial_cache(self.pool, b"abc", self.group_id, block)
        _insert_partial_cache(self.pool, b"def", self.group_id, block)
        remove_partial_cache_entries_for_block(self.pool, 5)
        self.assertIsNone(get_partial_cached_block(self.pool, b"abc", self.group_id))
        self.assertIsNone(get_partial_cached_block(self.pool, b"def", self.group_id))

    def test_state_is_isolated_per_block_pool(self):
        pool_a, pool_b = _pool(), _pool()
        _insert_partial_cache(pool_a, b"abc", self.group_id, _block(1))
        # A second pool must not see the first pool's entries (no module globals).
        self.assertIsNone(get_partial_cached_block(pool_b, b"abc", self.group_id))
