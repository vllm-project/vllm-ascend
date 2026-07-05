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

import unittest
from dataclasses import dataclass, replace
from unittest.mock import patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, SlidingWindowSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import get_block_hashes
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import (
    AscendStoreCoordinator,
    ExternalCachedBlockPool,
)

# isort: on


def _hashes(num_blocks: int) -> list[bytes]:
    return [bytes([idx % 251]) * 32 for idx in range(num_blocks)]


def _full_spec(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )


def _sliding_spec(block_size: int, sliding_window: int) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )


@dataclass(frozen=True)
class _FakeCompressedSpec:
    block_size: int
    compress_ratio: int

    def copy_with_new_block_size(self, block_size):
        return replace(self, block_size=block_size)


class _FakeCompressedManager:
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block=False,
        alignment_tokens=16,
        **kwargs,
    ):
        computed: tuple[list[object], ...] = tuple([] for _ in kv_cache_group_ids)
        logical_block_size = kv_cache_spec.block_size * kv_cache_spec.compress_ratio
        max_blocks = max_length // logical_block_size
        for block_hash in list(block_hashes)[:max_blocks]:
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for blocks, block in zip(computed, cached):
                blocks.append(block)
        return computed


class TestAscendStoreCoordinator(unittest.TestCase):
    def test_compressed_group_hits_on_effective_granularity(self):
        block_hashes = _hashes(128)
        grouped_hash = get_block_hashes(block_hashes, group_block_size=128 * 128, hash_block_size=128)[0]
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _full_spec(128))],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c128"],
        )

        _, hit_length = coord.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool({(0, bytes(grouped_hash))}),
        )

        self.assertEqual(hit_length, 128 * 128)

    def test_compressed_spec_does_not_apply_ratio_twice(self):
        block_hashes = _hashes(128)
        grouped_hash = get_block_hashes(block_hashes, group_block_size=128 * 128, hash_block_size=128)[0]

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            return_value=_FakeCompressedManager,
        ):
            coord = AscendStoreCoordinator(
                [KVCacheGroupSpec(["layer.0"], _FakeCompressedSpec(block_size=128, compress_ratio=128))],
                scheduler_block_size=128 * 128,
                hash_block_size=128,
                group_block_sizes=[128],
                group_cache_families=["c128"],
            )

            _, hit_length = coord.find_longest_cache_hit(
                block_hashes,
                128 * 128,
                ExternalCachedBlockPool({(0, bytes(grouped_hash))}),
            )

        self.assertEqual(coord.group_effective_specs[0].compress_ratio, 1)
        self.assertEqual(hit_length, 128 * 128)

    def test_missing_required_group_returns_zero(self):
        block_hashes = _hashes(128)
        c1_exists = {(0, block_hash) for block_hash in block_hashes}
        coord = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["layer.0"], _full_spec(128)),
                KVCacheGroupSpec(["layer.1"], _full_spec(128)),
            ],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["c1", "c128"],
        )

        _, hit_length = coord.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool(c1_exists),
        )

        self.assertEqual(hit_length, 0)

    def test_store_mask_uses_manager_reachability(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _sliding_spec(block_size=128, sliding_window=256))],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
        )

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            return_value=[False, False, False, True],
        ):
            masks = coord.store_mask(512)

        self.assertEqual(masks, ([False, False, False, True],))

    def test_store_mask_propagates_eagle_to_same_spec_siblings(self):
        calls = []

        def fake_reachable_block_mask(*args, **kwargs):
            calls.append(kwargs["use_eagle"])
            return [True, False, True, False]

        shared_spec = _sliding_spec(block_size=128, sliding_window=256)
        coord = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["layer.0"], shared_spec),
                KVCacheGroupSpec(["layer.mtp"], shared_spec, is_eagle_group=True),
            ],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128, 128],
            group_cache_families=["c1", "c1"],
        )

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            side_effect=fake_reachable_block_mask,
        ):
            masks = coord.store_mask(512)

        self.assertEqual(calls, [True, True])
        self.assertEqual(masks, ([True, False, True, False], [True, False, True, False]))

    def test_compressed_masks_stay_unmasked(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _sliding_spec(block_size=128, sliding_window=512))],
            scheduler_block_size=2048,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c4"],
        )

        self.assertEqual(coord.store_mask(2048, num_prompt_tokens=2048), ([True] * 4,))
        with patch.object(
            coord,
            "find_longest_cache_hit",
            return_value=(([False, False, False, True],), 2048),
        ):
            self.assertEqual(coord.load_mask(_hashes(16), 2048), ([True] * 4,))


if __name__ == "__main__":
    unittest.main()
