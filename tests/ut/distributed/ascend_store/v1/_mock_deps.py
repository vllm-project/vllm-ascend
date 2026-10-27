"""Extend the legacy CPU bootstrap with contracts exercised only by v1."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

from tests.ut.distributed.ascend_store import _mock_deps as shared
from tests.ut.distributed.ascend_store._mock_deps import (
    _FakeFullAttentionSpec,
    _FakeKVCacheSpec,
    _FakeMambaSpec,
    _FakeSlidingWindowSpec,
)


class _FakeCircularBufferSpec(_FakeKVCacheSpec):
    prefix_cacheable = False
    participates_in_prefix_caching = False
    is_circular = True


class _FakeSingleTypeKVCacheManager:
    def __init__(self, *args, **kwargs):
        self._mock = MagicMock()

    def __getattr__(self, name):
        return getattr(self._mock, name)

    @classmethod
    def reachable_block_mask(
        cls,
        start_block,
        end_block,
        alignment_tokens,
        kv_cache_spec,
        use_eagle,
        retention_interval=None,
        reachable_boundaries=(),
        dcp_world_size=1,
        final_segment_end_block=None,
    ):
        return None

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
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        computed: tuple[list[object], ...] = tuple([] for _ in kv_cache_group_ids)
        max_blocks = max_length // kv_cache_spec.block_size
        for block_hash in list(block_hashes)[:max_blocks]:
            cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
            if not cached:
                break
            for blocks, block in zip(computed, cached):
                blocks.append(block)
        if drop_eagle_block and computed and computed[0]:
            for blocks in computed:
                blocks.pop()
        hit_length = len(computed[0]) * kv_cache_spec.block_size if computed else 0
        return computed, hit_length


class _FakeFullAttentionManager(_FakeSingleTypeKVCacheManager):
    supports_fine_grained_hash_lookup = True

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
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        del dcp_world_size, pcp_world_size
        block_size = kv_cache_spec.block_size
        if alignment_tokens >= block_size:
            return super().find_longest_cache_hit(
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                kv_cache_spec,
                drop_eagle_block,
                alignment_tokens,
            )

        computed: tuple[list[object], ...] = tuple([] for _ in kv_cache_group_ids)
        scale = block_size // alignment_tokens
        full_block_count = 0
        for hash_index in range(scale - 1, min(max_length // alignment_tokens, len(block_hashes)), scale):
            cached = block_pool.get_cached_block(block_hashes[hash_index], kv_cache_group_ids)
            if not cached:
                break
            for blocks, block in zip(computed, cached):
                blocks.append(block)
            full_block_count += 1

        hit_length = full_block_count * block_size
        first_partial_index = full_block_count * scale
        last_partial_index = min(
            first_partial_index + scale,
            max_length // alignment_tokens,
            len(block_hashes),
        )
        for hash_index in range(last_partial_index - 1, first_partial_index - 1, -1):
            cached = block_pool.get_cached_block(block_hashes[hash_index], kv_cache_group_ids)
            if not cached:
                continue
            for blocks, block in zip(computed, cached):
                blocks.append(block)
            hit_length = (hash_index + 1) * alignment_tokens
            break
        if drop_eagle_block and hit_length:
            hit_length -= alignment_tokens
        return computed, hit_length


class _FakeMambaManager(_FakeSingleTypeKVCacheManager):
    supports_fine_grained_hash_lookup = True

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
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        del drop_eagle_block, dcp_world_size, pcp_world_size
        block_size = kv_cache_spec.block_size
        if alignment_tokens >= block_size:
            return super().find_longest_cache_hit(
                block_hashes,
                max_length,
                kv_cache_group_ids,
                block_pool,
                kv_cache_spec,
                False,
                alignment_tokens,
            )

        computed: tuple[list[object], ...] = tuple([] for _ in kv_cache_group_ids)
        scale = block_size // alignment_tokens
        last_index = min(max_length // alignment_tokens, len(block_hashes)) - 1
        for hash_index in range(last_index, -1, -1):
            cached = block_pool.get_cached_block(block_hashes[hash_index], kv_cache_group_ids)
            if not cached:
                continue
            block_index = hash_index // scale
            for blocks, block in zip(computed, cached):
                blocks.extend([block_pool.null_block] * block_index)
                blocks.append(block)
            return computed, (hash_index + 1) * alignment_tokens
        return computed, 0


class _FakeSlidingWindowManager(_FakeSingleTypeKVCacheManager):
    @classmethod
    def reachable_block_mask(
        cls,
        start_block,
        end_block,
        alignment_tokens,
        kv_cache_spec,
        use_eagle,
        retention_interval=None,
        reachable_boundaries=(),
        dcp_world_size=1,
        final_segment_end_block=None,
    ):
        if alignment_tokens is None:
            return None
        block_size = kv_cache_spec.block_size * dcp_world_size
        if alignment_tokens % block_size:
            return None
        need = (kv_cache_spec.sliding_window - 2 + block_size) // block_size + int(use_eagle)
        per_segment = max(alignment_tokens // block_size, 1)
        return [(idx - int(use_eagle)) % per_segment >= per_segment - need for idx in range(start_block, end_block)]

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
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        block_size = kv_cache_spec.block_size
        need = max(1, (kv_cache_spec.sliding_window - 2 + block_size) // block_size) + int(drop_eagle_block)
        maximum = min(max_length // block_size, len(block_hashes))
        for end in range(maximum, 0, -1):
            hit_end = end - int(drop_eagle_block)
            if hit_end * block_size % alignment_tokens:
                continue
            start = max(0, end - need)
            tail = [block_pool.get_cached_block(block_hashes[index], kv_cache_group_ids) for index in range(start, end)]
            if not all(tail):
                continue
            computed = tuple([block_pool.null_block] * start for _ in kv_cache_group_ids)
            for cached in tail[: hit_end - start]:
                for blocks, block in zip(computed, cached):
                    blocks.append(block)
            return computed, hit_end * block_size
        return tuple([] for _ in kv_cache_group_ids), 0


class _FakeKVCacheSpecRegistry:
    @classmethod
    def get_manager_class(cls, kv_cache_spec):
        if isinstance(kv_cache_spec, _FakeSlidingWindowSpec):
            return _FakeSlidingWindowManager
        if isinstance(kv_cache_spec, _FakeFullAttentionSpec):
            return _FakeFullAttentionManager
        if isinstance(kv_cache_spec, _FakeMambaSpec):
            return _FakeMambaManager
        return _FakeSingleTypeKVCacheManager


if shared._MOCK_VLLM_DEPS:
    _FakeKVCacheSpec.prefix_cacheable = True
    managers = sys.modules["vllm.v1.core.single_type_kv_cache_manager"]
    managers.SingleTypeKVCacheManager = _FakeSingleTypeKVCacheManager
    managers.FullAttentionManager = _FakeFullAttentionManager
    managers.SlidingWindowManager = _FakeSlidingWindowManager
    managers.MambaManager = _FakeMambaManager
    managers.spec_manager_map = {
        _FakeFullAttentionSpec: _FakeFullAttentionManager,
        _FakeSlidingWindowSpec: _FakeSlidingWindowManager,
        _FakeMambaSpec: _FakeMambaManager,
    }
    sys.modules["vllm.v1.kv_cache_interface"].CircularBufferSpec = _FakeCircularBufferSpec
    sys.modules["vllm.v1.kv_cache_spec_registry"].KVCacheSpecRegistry = _FakeKVCacheSpecRegistry

# Legacy tests mock the shared module wholesale. v1 uses its real initialization
# policy, while the legacy global_te mock must remain available to legacy tests.
shared_engine = sys.modules.get("vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine")
if isinstance(shared_engine, Mock):
    source = (
        Path(__file__).resolve().parents[5] / "vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py"
    )
    spec = importlib.util.spec_from_file_location("_ascend_store_v1_upstream_transfer_engine", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shared_engine.GlobalTE = module.GlobalTE
