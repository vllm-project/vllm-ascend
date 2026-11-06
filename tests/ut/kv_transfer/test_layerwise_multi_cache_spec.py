"""Every cache component participates by its own reuse intent and backing."""

from dataclasses import replace

import pytest
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from tests.ut.distributed.ascend_store.test_layerwise_cache_layout import cache, runtime, spec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)


@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_cache_components_are_preserved_without_main_indexer_classification(reverse):
    suffixes = ("attn", "indexer.k_cache", "indexer.compressor.state_cache", "compressor.state_cache", "swa_cache")
    domains = [[f"model.layers.{layer}.self_attn.{suffix}" for layer in range(6)] for suffix in suffixes]
    if reverse:
        domains.reverse()
    specs = {name: spec(reuse=i < 2) for i, domain in enumerate(domains) for name in domain}
    config = cache(specs, domains)
    config.kv_cache_groups = [
        KVCacheGroupSpec(
            domain, UniformTypeKVCacheSpecs(block_size=2, kv_cache_specs={name: specs[name] for name in domain})
        )
        for domain in domains
    ]
    original_names = set(specs)
    apply_layerwise_kv_cache_plan(config, runtime())
    assert {name for tensor in config.kv_cache_tensors for name in tensor.layers} == original_names
    for domain in domains[2:]:
        assert next(t for t in config.kv_cache_tensors if domain[0] in t.layers).layers == domain


def test_indexer_only_cache_does_not_require_a_main_cache():
    names = [f"model.layers.{layer}.a.indexer.k_cache" for layer in range(6)]
    config = cache(dict.fromkeys(names, spec()))
    apply_layerwise_kv_cache_plan(config, runtime())
    assert [tensor.layers for tensor in config.kv_cache_tensors] == [names[::2], names[1::2]]


def test_residency_does_not_propagate_to_another_component_in_same_group():
    main = [f"model.layers.{layer}.attn" for layer in range(6)]
    index = [f"model.layers.{layer}.indexer.k_cache" for layer in range(6)]
    specs = dict.fromkeys(main + index, spec())
    specs[index[2]] = replace(specs[index[2]], reuse=False)
    config = cache(specs, [main, index])
    original = list(config.kv_cache_tensors)
    with pytest.raises(NotImplementedError, match="Mixed reuse intent"):
        apply_layerwise_kv_cache_plan(config, runtime())
    assert config.kv_cache_tensors == original
    assert specs[main[2]].reuse
    assert not specs[index[2]].reuse
