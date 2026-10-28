"""Configuration normalization and support boundaries, before process-local binding."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import configuration as resolve


@pytest.mark.parametrize(
    "name",
    (
        "use_layerwise",
        "load_async",
        "consumer_is_to_load",
        "consumer_is_to_put",
        "save_decode_cache",
        "discard_partial_chunks",
    ),
)
@pytest.mark.parametrize("value", ("false", "true", 0, 1, None))
def test_boolean_options_reject_truthiness_coercion(configuration_case, name, value):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config[name] = value
    with pytest.raises(ValueError, match=f"{name} must be a boolean"):
        resolve.resolve_kv_pool_config(config, cache)


@pytest.mark.parametrize("name", ("layerwise_prefetch_layers", "prefill_pp_size"))
@pytest.mark.parametrize("value", (0, -1, True, 2.0, 2.9, "2.9", "invalid", None))
def test_positive_integer_options_reject_invalid_values(configuration_case, name, value):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = "kv_consumer"
    extra = config.kv_transfer_config.kv_connector_extra_config
    extra.update(use_layerwise=name == "layerwise_prefetch_layers", consumer_is_to_put=True)
    extra[name] = value
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        resolve.resolve_kv_pool_config(config, cache)


@pytest.mark.parametrize("value", (2, "2"))
@pytest.mark.parametrize("layerwise", (False, True))
def test_normalization_preserves_integer_strings_defaults_and_input(configuration_case, value, layerwise):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = "kv_consumer"
    extra = config.kv_transfer_config.kv_connector_extra_config
    extra.update(
        backend=" MoonCake ",
        use_layerwise=layerwise,
        consumer_is_to_put=True,
        layerwise_prefetch_layers=value if layerwise else "unused",
        prefill_pp_size=1 if layerwise else value,
    )
    before = dict(extra)
    result = resolve.resolve_kv_pool_config(config, cache)
    assert result.backend_name == "mooncake"
    assert result.layerwise_prefetch_layers == 2
    assert result.pp_partition.target_layer_counts == ((4,) if layerwise else (2, 2))
    assert not result.load_enabled and result.store_enabled
    assert extra == before


@pytest.mark.parametrize("prefix_caching", (False, True))
@pytest.mark.parametrize("hash_size", (8, 16, 32))
def test_request_hash_granularity_comes_from_vllm(configuration_case, monkeypatch, prefix_caching, hash_size):
    config, cache = configuration_case
    config.cache_config.enable_prefix_caching = prefix_caching
    config.cache_config.prefix_match_unit = 4
    config.parallel_config.decode_context_parallel_size = 2
    original = cache.kv_cache_groups[0].kv_cache_spec
    specs = tuple(replace(original, block_size=size) for size in (16, 32))
    cache.kv_cache_groups = [SimpleNamespace(kv_cache_spec=spec, is_eagle_group=False) for spec in specs]
    effective_specs = tuple(resolve.resolve_dcp_kv_cache_spec(spec, 2) for spec in specs)
    upstream_resolver = Mock(return_value=(64, hash_size))
    monkeypatch.setattr(resolve.kv_cache_utils, "resolve_kv_cache_block_sizes", upstream_resolver)

    # Neither prefix caching nor local group sizes override the API's result.
    cache.transfer_group_ids = (0, 1)
    result = resolve.resolve_kv_pool_config(config, cache)
    assert (result.cache_transfer_granularity, result.hash_block_size) == (64, hash_size)
    assert result.group_specs == effective_specs
    upstream_resolver.assert_called_once_with(cache, config)
    assert config.cache_config.enable_prefix_caching is prefix_caching
    assert config.cache_config.prefix_match_unit == 4


@pytest.mark.parametrize("selected,expected", (((3, 1, 2), (3, 2)), ((1,), ()), ((), ())))
def test_transfer_projection_preserves_original_ids_and_order(configuration_case, selected, expected):
    _, cache = configuration_case
    attention = cache.kv_cache_groups[0].kv_cache_spec
    private = CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=1, head_size_v=0, dtype=attention.dtype)
    private = UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={"layers.1": private})
    cache.kv_cache_groups = [
        KVCacheGroupSpec([f"layers.{index}"], spec)
        for index, spec in enumerate((attention, private, attention, attention))
    ]
    cache.transfer_group_ids = selected
    if expected:
        assert resolve._resolve_transfer_group_ids(cache) == expected
    else:
        with pytest.raises(ValueError, match="at least one transferable prefix-cacheable"):
            resolve._resolve_transfer_group_ids(cache)


@pytest.mark.parametrize("layers,partitions,expected", ((7, 3, (2, 3, 2)), (8, 3, (3, 3, 2)), (4, 1, (4,))))
def test_default_consumer_partitions_preserve_vllm_distribution(configuration_case, layers, partitions, expected):
    config, _ = configuration_case
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.kv_connector_extra_config["prefill_pp_size"] = partitions
    config.model_config.hf_text_config.num_hidden_layers = layers
    assert resolve.resolve_consumer_pipeline_partitions(config, store_enabled=True) == expected
