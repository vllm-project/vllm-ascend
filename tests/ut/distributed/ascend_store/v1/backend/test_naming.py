"""Compile keys once and preserve Backend wire identity across configuration representations."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from tests.ut.distributed.ascend_store import test_layerwise_keys as legacy_cases
from tests.ut.distributed.ascend_store.v1.helpers import make_pool_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import memcache, mooncake
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import map_keys


@pytest.fixture
def naming_case():
    config = legacy_cases.TestBoundLayerwiseKeys.make_config(2)
    cache = legacy_cases.TestBoundLayerwiseKeys.make_cache_config(2)
    pool = make_pool_config(
        tp_size=config.parallel_config.tensor_parallel_size,
        pp_size=2,
        use_hybrid=True,
        group_specs=tuple(group.kv_cache_spec for group in cache.kv_cache_groups),
    )
    return config, cache, pool


@pytest.mark.parametrize("backend,legacy", ((memcache, MemcacheBackend), (mooncake, MooncakeBackend)))
@pytest.mark.parametrize("pp_size", (1, 2))
@pytest.mark.parametrize("group_count", (1, 2))
def test_compiled_keys_match_legacy_and_do_not_reread_inputs(backend, legacy, pp_size, group_count, monkeypatch):
    config = legacy_cases.TestBoundLayerwiseKeys.make_config(pp_size)
    cache = legacy_cases.TestBoundLayerwiseKeys.make_cache_config(group_count)
    sizes = tuple(16 * (group + 1) for group in range(group_count))
    pool = make_pool_config(
        tp_size=config.parallel_config.tensor_parallel_size,
        pp_size=pp_size,
        use_hybrid=group_count > 1,
        group_specs=tuple(SimpleNamespace(block_size=size) for size in sizes),
    )
    expected = legacy.bind_layerwise_keys(
        vllm_config=config,
        kv_cache_config=cache,
        model_name="model",
        use_hybrid=group_count > 1,
        grouped_block_size=list(sizes),
    )
    # The new adapter must compile its own keys, not delegate to the legacy one.
    fail = Mock(side_effect=AssertionError("runtime key compilation or legacy adapter"))
    monkeypatch.setattr(legacy, "bind_layerwise_keys", fail)
    builder = backend.create_key_builder(config, cache, pool)
    config.parallel_config.pipeline_parallel_size = 99
    config.model_config.compute_hash = fail
    cache.kv_cache_groups[0].layer_names.append("changed")
    for name in ("_full_key_prefix",) if backend is memcache else ("_block_key_prefix", "_hybrid_key_prefix"):
        monkeypatch.setattr(backend, name, fail)
    monkeypatch.setattr(backend, "create_key_builder", fail)
    for group in range(group_count):
        for lookup_heads in (0, 1, 2):
            assert tuple(map_keys(parts, (b"a",)) for parts in builder.lookup_parts(group, lookup_heads)) == tuple(
                (key,) for key in expected.make_hit_check_keys(group, "61", lookup_heads)
            )
        for stage in reversed(range(pp_size)):
            for head in (1, 0):
                for dcp_rank in (0, 3):
                    metadata = KeyMetadata("model", head, dcp_rank, stage, group)
                    assert map_keys(builder(metadata), (b"a", "b")) == tuple(
                        expected.make_full_key(group, value, head, stage) for value in ("61", "b")
                    )
    fail.assert_not_called()


def test_mooncake_identity_normalizes_cache_size_and_group_representation(naming_case):
    class CacheConfig(SimpleNamespace):
        def compute_hash(self):
            return f"block-size:{self.block_size}"

    config, cache, pool = naming_case
    config.cache_config = CacheConfig(block_size=128, cache_dtype="auto")
    config.speculative_config = SimpleNamespace(compute_hash=lambda: "draft-model")
    expected = MooncakeBackend.bind_layerwise_keys(
        vllm_config=config,
        kv_cache_config=cache,
        model_name="model",
        use_hybrid=True,
        grouped_block_size=[spec.block_size for spec in pool.group_specs],
    )
    builder = mooncake.create_key_builder(config, cache, pool)
    assert config.cache_config.block_size == 128
    for group in cache.kv_cache_groups:
        group.kv_cache_spec = UniformTypeKVCacheSpecs(
            block_size=group.kv_cache_spec.block_size,
            kv_cache_specs={name: group.kv_cache_spec for name in group.layer_names},
        )
    config.cache_config.block_size = 16
    represented = mooncake.create_key_builder(config, cache, pool)
    assert represented.prefixes_by_group_stage == builder.prefixes_by_group_stage
    for group in (0, 1):
        assert map_keys(builder(KeyMetadata("model", 1, 0, 1, group)), ("a",)) == (
            expected.make_full_key(group, "a", 1, 1),
        )
    config.speculative_config.compute_hash = lambda: "different-draft"
    changed = mooncake.create_key_builder(config, cache, pool)
    assert changed.prefixes_by_group_stage != builder.prefixes_by_group_stage


def test_memcache_does_not_read_unused_layout_identity(naming_case):
    _, _, pool = naming_case
    builder = memcache.create_key_builder(object(), object(), pool)
    assert map_keys(builder(KeyMetadata("model", 1, 0, 1, 1)), ("a",)) == ("model@1@pp1@a@1",)
    assert tuple(map_keys(parts, ("a",)) for parts in builder.lookup_parts(1, 1)) == (
        ("model@1@pp0@a@0",),
        ("model@1@pp1@a@0",),
    )


@pytest.mark.parametrize(
    "invalid,message", (("empty", "empty groups"), ("gap", "contiguous"), ("coverage", "cover every model layer"))
)
def test_mooncake_rejects_unrepresentable_partitions(naming_case, invalid, message):
    config, cache, pool = naming_case
    if invalid == "empty":
        cache.kv_cache_groups[1].layer_names = []
    else:
        ranges = ((0, 3), (4, 7)) if invalid == "gap" else ((0, 3), (3, 6))
        config.model_config.get_layers_start_end_indices = lambda parallel: ranges[parallel.rank // pool.tp_size]
    with pytest.raises(ValueError, match=message):
        mooncake.create_key_builder(config, cache, pool)
