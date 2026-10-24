"""Backend codecs consume immutable facts and preserve the existing wire identity."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from tests.ut.distributed.ascend_store import test_layerwise_keys as legacy_cases
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import memcache_backend, mooncake_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import memcache, mooncake
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.naming import (
    resolve_key_naming_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping.local_layout import (
    ObjectCoordinate,
)


@pytest.mark.parametrize("backend,legacy", ((memcache, memcache_backend), (mooncake, mooncake_layerwise)))
@pytest.mark.parametrize("pp_size", (1, 2))
@pytest.mark.parametrize("group_count", (1, 2))
def test_resolved_codec_matches_legacy_and_snapshots_all_inputs(backend, legacy, pp_size, group_count):
    config = legacy_cases.TestBoundLayerwiseKeys.make_config(pp_size)
    cache = legacy_cases.TestBoundLayerwiseKeys.make_cache_config(group_count)
    sizes = tuple(16 * (group + 1) for group in range(group_count))
    expected = legacy.bind_layerwise_keys(
        vllm_config=config,
        kv_cache_config=cache,
        model_name="model",
        use_hybrid=group_count > 1,
        grouped_block_size=list(sizes),
    )
    facts = resolve_key_naming_config(
        config,
        cache,
        model_name="model",
        use_hybrid=group_count > 1,
        group_block_sizes=sizes,
        requires_layout_identity=backend is mooncake,
    )
    # The Backend no longer has access to either mutable upstream config.
    config.parallel_config.pipeline_parallel_size = 99
    config.model_config.compute_hash = Mock(side_effect=AssertionError("late config access"))
    cache.kv_cache_groups[0].layer_names.append("changed")
    with patch.object(legacy, "bind_layerwise_keys", side_effect=AssertionError("legacy config adapter")):
        builder = backend.create_key_builder(facts)
        for group in range(group_count):
            coordinates = tuple(
                ObjectCoordinate(stage, 0, head) for stage in reversed(range(pp_size)) for head in (1, 0)
            )
            for lookup_heads in (0, 1, 2):
                naming = builder.bind(group, coordinates, lookup_head_count=lookup_heads)
                assert naming.keys((b"a", "b")) == tuple(
                    tuple(expected.make_full_key(group, value, pos.head_rank, pos.pp_rank) for value in ("61", "b"))
                    for pos in coordinates
                )
                assert naming.lookup_keys((b"a",)) == tuple(
                    (key,) for key in expected.make_hit_check_keys(group, "61", lookup_heads)
                )


@pytest.mark.parametrize("backend,legacy", ((memcache, memcache_backend), (mooncake, mooncake_layerwise)))
@pytest.mark.parametrize("pp_size,group_count", ((1, 1), (1, 2), (2, 1), (2, 2)))
def test_runtime_only_joins_precompiled_parts(backend, legacy, pp_size, group_count, monkeypatch):
    config = legacy_cases.TestBoundLayerwiseKeys.make_config(pp_size)
    cache = legacy_cases.TestBoundLayerwiseKeys.make_cache_config(group_count)
    sizes = tuple(16 * (group + 1) for group in range(group_count))
    expected = legacy.bind_layerwise_keys(
        vllm_config=config,
        kv_cache_config=cache,
        model_name="model",
        use_hybrid=group_count > 1,
        grouped_block_size=list(sizes),
    )
    facts = resolve_key_naming_config(
        config,
        cache,
        model_name="model",
        use_hybrid=group_count > 1,
        group_block_sizes=sizes,
        requires_layout_identity=backend is mooncake,
    )
    group, stage = group_count - 1, pp_size - 1
    key = expected.make_full_key(group, "61", 1, stage)
    lookup = tuple((key,) for key in expected.make_hit_check_keys(group, "61", 2))
    builder = backend.create_key_builder(facts)
    fail = Mock(side_effect=AssertionError("runtime wire recompilation"))
    prefixes = ("_full_key_prefix",) if backend is memcache else ("_block_key_prefix", "_hybrid_key_prefix")
    for name in prefixes:
        monkeypatch.setattr(backend, name, fail)
    monkeypatch.setattr(backend, "create_key_builder", fail)
    monkeypatch.setattr(legacy, "bind_layerwise_keys", fail)
    bound = builder.bind(group, (ObjectCoordinate(stage, 0, 1),), lookup_head_count=2)
    assert bound.keys((b"a", "61")) == ((key, key),)
    assert bound.lookup_keys((b"a",)) == lookup
    fail.assert_not_called()


def test_mooncake_identity_normalizes_cache_size_and_group_representation():
    class CacheConfig(SimpleNamespace):
        def compute_hash(self):
            return f"block-size:{self.block_size}"

    config = legacy_cases.TestBoundLayerwiseKeys.make_config(2)
    config.cache_config = CacheConfig(block_size=128, cache_dtype="auto")
    config.speculative_config = SimpleNamespace(compute_hash=lambda: "draft-model")
    cache = legacy_cases.TestBoundLayerwiseKeys.make_cache_config(2)
    expected = mooncake_layerwise.bind_layerwise_keys(
        vllm_config=config,
        kv_cache_config=cache,
        model_name="model",
        use_hybrid=True,
        grouped_block_size=[16, 32],
    )
    facts = resolve_key_naming_config(
        config,
        cache,
        model_name="model",
        use_hybrid=True,
        group_block_sizes=(16, 32),
        requires_layout_identity=True,
    )
    assert config.cache_config.block_size == 128
    assert facts.cache_block_size == 16 and facts.cache_hash == "block-size:16"
    for group in cache.kv_cache_groups:
        group.kv_cache_spec = UniformTypeKVCacheSpecs(
            block_size=group.kv_cache_spec.block_size,
            kv_cache_specs={name: group.kv_cache_spec for name in group.layer_names},
        )
    config.cache_config.block_size = 16
    represented = resolve_key_naming_config(
        config,
        cache,
        model_name="model",
        use_hybrid=True,
        group_block_sizes=(16, 32),
        requires_layout_identity=True,
    )
    assert represented == facts
    builder = mooncake.create_key_builder(facts)
    for group in (0, 1):
        naming = builder.bind(group, (ObjectCoordinate(1, 0, 1),), lookup_head_count=2)
        assert naming.keys(("a",)) == ((expected.make_full_key(group, "a", 1, 1),),)
    changed = mooncake.create_key_builder(replace(facts, speculative_hash="different-draft"))
    assert changed.prefixes_by_group_stage != builder.prefixes_by_group_stage


def test_memcache_does_not_read_unused_layout_identity():
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=2))
    facts = resolve_key_naming_config(
        config,
        object(),
        model_name="model",
        use_hybrid=True,
        group_block_sizes=(16, 32),
        requires_layout_identity=False,
    )
    naming = memcache.create_key_builder(facts).bind(1, (ObjectCoordinate(1, 0, 1),), lookup_head_count=1)
    assert naming.keys(("a",)) == (("model@1@pp1@a@1",),)
    assert naming.lookup_keys(("a",)) == (("model@1@pp0@a@0",), ("model@1@pp1@a@0",))


@pytest.mark.parametrize(
    "changes,message",
    (
        ({"group_layer_names": (("layer",), ())}, "empty groups"),
        ({"layer_ranges": ((0, 3), (4, 7))}, "contiguous"),
        ({"layer_ranges": ((0, 3), (3, 6))}, "cover every model layer"),
    ),
)
def test_mooncake_validates_protocol_requirements_from_facts(changes, message):
    facts = resolve_key_naming_config(
        legacy_cases.TestBoundLayerwiseKeys.make_config(2),
        legacy_cases.TestBoundLayerwiseKeys.make_cache_config(2),
        model_name="model",
        use_hybrid=True,
        group_block_sizes=(16, 32),
        requires_layout_identity=True,
    )
    with pytest.raises(ValueError, match=message):
        mooncake.create_key_builder(replace(facts, **changes))
