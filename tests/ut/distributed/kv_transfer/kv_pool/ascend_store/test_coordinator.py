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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# isort: off
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import coordinator as module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import get_block_hashes
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


@dataclass(frozen=True)
class _CompressedFullSpec(FullAttentionSpec):
    compress_ratio: int = 1


def test_empty_eagle_groups_and_legacy_spec_copy():
    coordinator = AscendStoreCoordinator([], 4, 4, [], [], use_eagle=True)
    assert coordinator.attention_groups == []
    assert coordinator.eagle_attn_group_indices == set()
    assert coordinator.eagle_reachable_group_ids == set()

    @dataclass(frozen=True)
    class LegacySpec:
        block_size: int
        other: str

    original = LegacySpec(4, "preserved")
    assert module._copy_spec_with_block_size(original, 8) == LegacySpec(8, "preserved")
    assert original.block_size == 4


class TestAscendStoreCoordinator(unittest.TestCase):
    def test_compressed_group_hits_on_effective_granularity(self):
        block_hashes = _hashes(128)
        grouped_hash = get_block_hashes(block_hashes, group_block_size=128 * 128, hash_block_size=128)[0]
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _full_spec(128 * 128))],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128 * 128],
            group_cache_families=["c128"],
        )

        _, hit_length = coord.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool(128, {(0, bytes(grouped_hash))}),
        )

        self.assertEqual(hit_length, 128 * 128)

    def test_compressed_spec_does_not_apply_ratio_twice(self):
        block_hashes = _hashes(128)
        grouped_hash = get_block_hashes(block_hashes, group_block_size=128 * 128, hash_block_size=128)[0]

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._get_manager_class",
            return_value=FullAttentionManager,
        ):
            coord = AscendStoreCoordinator(
                [
                    KVCacheGroupSpec(
                        ["layer.0"],
                        _CompressedFullSpec(
                            block_size=128 * 128,
                            num_kv_heads=1,
                            head_size=1,
                            dtype=torch.float32,
                            compress_ratio=128,
                        ),
                    )
                ],
                scheduler_block_size=128 * 128,
                hash_block_size=128,
                group_block_sizes=[128 * 128],
                group_cache_families=["c128"],
            )

            _, hit_length = coord.find_longest_cache_hit(
                block_hashes,
                128 * 128,
                ExternalCachedBlockPool(128, {(0, bytes(grouped_hash))}),
            )

        self.assertEqual(coord.group_effective_specs[0].compress_ratio, 128)
        self.assertEqual(hit_length, 128 * 128)

    def test_missing_required_group_returns_zero(self):
        block_hashes = _hashes(128)
        c1_exists = {(0, block_hash) for block_hash in block_hashes}
        coord = AscendStoreCoordinator(
            [
                KVCacheGroupSpec(["layer.0"], _full_spec(128)),
                KVCacheGroupSpec(["layer.1"], _full_spec(128 * 128)),
            ],
            scheduler_block_size=128 * 128,
            hash_block_size=128,
            group_block_sizes=[128, 128 * 128],
            group_cache_families=["c1", "c128"],
        )

        _, hit_length = coord.find_longest_cache_hit(
            block_hashes,
            128 * 128,
            ExternalCachedBlockPool(128, c1_exists),
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

    def test_lookup_mask_uses_reachability_without_retention(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _sliding_spec(block_size=128, sliding_window=256))],
            scheduler_block_size=512,
            hash_block_size=128,
            group_block_sizes=[128],
            group_cache_families=["c1"],
            retention_interval=256,
        )
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator._reachable_block_mask",
            return_value=[False, False, False, True],
        ) as reachable:
            masks = coord.lookup_mask(512)

        self.assertEqual(masks, ([False, False, False, True],))
        self.assertIsNone(reachable.call_args.kwargs["retention_interval"])

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

    def test_compressed_masks_use_full_attention_reachability(self):
        coord = AscendStoreCoordinator(
            [KVCacheGroupSpec(["layer.0"], _full_spec(block_size=512))],
            scheduler_block_size=2048,
            hash_block_size=128,
            group_block_sizes=[512],
            group_cache_families=["c4"],
        )

        self.assertEqual(
            coord.store_mask(2048, num_prompt_tokens=2048),
            ([True, True, True, True],),
        )
        cached_mask = [True, True, False, False]
        with patch.object(
            coord,
            "find_longest_cache_hit",
            return_value=((cached_mask,), 1024),
        ):
            self.assertEqual(
                coord.load_mask(_hashes(16), 2048),
                (cached_mask,),
            )


if __name__ == "__main__":
    unittest.main()


def test_external_pool_requires_every_requested_group():
    pool = ExternalCachedBlockPool(4, {(0, b"a"), (1, b"a")})
    assert pool.get_cached_block(b"a", [0, 1]) == [pool._present_block, pool._present_block]
    assert pool.get_cached_block(b"a", [0, 2]) is None
    assert pool.get_cached_block(b"b", [0]) is None
    assert pool.get_cached_block(b"b", []) == []


def test_spec_unwrapping_and_copy_preserve_original():
    spec = _full_spec(4)
    wrapped = UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"a": spec})
    assert module._unwrap_spec(wrapped) is spec
    assert module._copy_spec_with_block_size(spec, 4) is spec
    assert module._copy_spec_with_block_size(spec, 8).block_size == 8
    legacy = _FakeCompressedSpec(4, 8)
    assert module._copy_spec_with_block_size(legacy, 8) == _FakeCompressedSpec(8, 8)
    assert legacy.block_size == 4


@pytest.mark.parametrize("registry_status", ["missing_module", "missing_registry", "miss"])
def test_manager_registry_falls_back_to_legacy_map_and_caches(monkeypatch, registry_status):
    spec = _full_spec(4)
    manager = object()
    registry = SimpleNamespace(get_manager_class=MagicMock(return_value=None))

    def import_dependency(name):
        if name == "vllm.v1.kv_cache_spec_registry":
            if registry_status == "missing_module":
                raise ImportError(name)
            return SimpleNamespace(KVCacheSpecRegistry=registry if registry_status == "miss" else None)
        assert name == "vllm.v1.core.single_type_kv_cache_manager"
        return SimpleNamespace(spec_manager_map={type(spec): manager})

    importer = MagicMock(side_effect=import_dependency)
    monkeypatch.setattr(module, "import_module", importer)
    monkeypatch.setattr(module._get_manager_class, "_manager_class_cache", {}, raising=False)
    assert module._get_manager_class(spec) is manager
    assert module._get_manager_class(spec) is manager
    assert importer.call_count == 2


@pytest.mark.parametrize("failure", ["import", "lookup"])
def test_manager_registry_reports_missing_spec_with_original_cause(monkeypatch, failure):
    monkeypatch.setattr(module._get_manager_class, "_manager_class_cache", {"registry": None}, raising=False)
    importer = MagicMock(return_value=SimpleNamespace(spec_manager_map={}))
    if failure == "import":
        importer.side_effect = ImportError("old vllm")
    monkeypatch.setattr(module, "import_module", importer)
    with pytest.raises(AssertionError, match="No manager registered") as error:
        module._get_manager_class(_full_spec(4))
    assert isinstance(error.value.__cause__, (ImportError, KeyError))


@pytest.mark.parametrize("unsupported", ["retention_interval", "num_prompt_tokens", "other_option"])
def test_reachability_wrapper_retries_legacy_signature_with_required_arguments(unsupported):
    reach = MagicMock(side_effect=[TypeError(f"unexpected keyword {unsupported}"), [False, True]])
    manager = SimpleNamespace(reachable_block_mask=reach)
    spec = _sliding_spec(4, 8)
    kwargs = dict(
        start_block=0,
        end_block=2,
        alignment_tokens=4,
        kv_cache_spec=spec,
        use_eagle=True,
        retention_interval=8,
        num_prompt_tokens=16,
    )
    assert module._reachable_block_mask(manager, **kwargs) == [False, True]
    assert reach.call_args_list[0].kwargs == kwargs
    assert reach.call_args_list[1].kwargs == dict(
        start_block=0, end_block=2, alignment_tokens=4, kv_cache_spec=spec, use_eagle=True
    )
    assert kwargs["retention_interval"] == 8
    assert module._reachable_block_mask(SimpleNamespace(), **kwargs) is None


@pytest.mark.parametrize("with_full", [False, True])
@pytest.mark.parametrize("eagle", [False, True])
def test_hybrid_hit_converges_across_real_cache_managers(with_full, eagle):
    specs = [_sliding_spec(4, 8), _sliding_spec(4, 12)]
    if with_full:
        specs.insert(0, _full_spec(4))
    groups = [KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)]
    coord = AscendStoreCoordinator(groups, 4, 4, [4] * len(groups), ["c1"] * len(groups), use_eagle=eagle)
    hashes = _hashes(6)
    exists = {(g, h) for g in range(len(groups)) for h in hashes[:4]}
    masks, length = coord.find_longest_cache_hit(hashes, 24, ExternalCachedBlockPool(4, exists))
    assert length == (12 if eagle else 16)
    assert len(masks) == len(groups)
    assert all(len(mask) == length // 4 for mask in masks)
    load_masks = coord.load_mask(hashes, 16)
    assert len(load_masks) == len(groups)
    assert all(len(mask) == 4 and mask[-1] for mask in load_masks)
