# SPDX-License-Identifier: Apache-2.0
from dataclasses import fields, replace
from sys import modules
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.v1.core import kv_cache_utils, single_type_kv_cache_manager
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.engine import core as engine_core
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.patch.patch_layerwise_connector_specs import register_layerwise_connector_patches
from vllm_ascend.patch.platform import patch_layerwise_cache_planning as patch


def full_spec():
    return FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)


def state_spec():
    return MambaSpec(block_size=32, shapes=((2, 4),), dtypes=(torch.float32,), mamba_cache_mode="align")


def mixed_group(spec):
    members = {"layer.0": spec, "layer.1": replace(spec, reuse=True)}
    wrapped = UniformTypeKVCacheSpecs(block_size=spec.block_size, kv_cache_specs=members)
    return KVCacheGroupSpec(list(members), wrapped)


def test_engine_rejects_different_rank_intent_before_merging():
    spec = full_spec()
    ranks = [{"layer.0": spec}, {"layer.0": replace(spec, reuse=True)}]
    with pytest.raises(ValueError, match=r"across workers for layer\.0"):
        if hasattr(engine_core, "get_global_kv_cache_groups"):
            engine_core.get_global_kv_cache_groups(None, ranks)
        else:
            engine_core.get_kv_cache_configs(None, ranks, [1024, 512])


def test_scheduler_preserves_named_specs_when_all_layers_are_resident():
    spec = full_spec()
    members = {"first": spec, "second": replace(spec, head_size=16)}
    wrapped = UniformTypeKVCacheSpecs(block_size=spec.block_size, kv_cache_specs=members)
    config = KVCacheConfig(16, [], [KVCacheGroupSpec(list(members), wrapped)])
    scheduler = engine_core.generate_scheduler_kv_cache_config([config])
    actual = scheduler.kv_cache_groups[0].kv_cache_spec.kv_cache_specs
    assert actual is not members
    assert actual == members
    assert not any(s.reuse for s in actual.values())


def test_scalar_scheduler_readers_keep_group_geometry(monkeypatch):
    group = mixed_group(state_spec())
    config = KVCacheConfig(16, [], [group])
    # A connector can have populated cached group lists before scheduler init.
    assert config.transfer_groups[0] is group
    scheduler = SimpleNamespace(has_mamba_layers=True, mamba_prefill_checkpoint_alignment=None)

    def scalar_reader(self, vllm_config, geometry):
        self.manager_config = geometry
        assert isinstance(geometry.transfer_groups[0].kv_cache_spec, MambaSpec)
        assert geometry.layerwise_scheduler_config is config

    monkeypatch.setattr(patch, "_orig_scheduler_init", scalar_reader)
    patch._scheduler_init(scheduler, None, config)
    assert scheduler.kv_cache_config is config
    assert scheduler.manager_config.kv_cache_groups[0] is not group
    assert group.kv_cache_spec.kv_cache_specs["layer.1"].reuse


def test_worker_connector_type_checks_on_per_layer_groups(monkeypatch):
    monkeypatch.setattr(KVConnectorFactory, "_registry", KVConnectorFactory._registry.copy())
    register_layerwise_connector_patches()
    load = KVConnectorFactory.get_connector_class_by_name
    hidden = HiddenStateCacheSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)
    cfg = KVCacheConfig(16, [], [mixed_group(full_spec()), mixed_group(state_spec())])
    hidden_cfg = replace(cfg, kv_cache_groups=[cfg.kv_cache_groups[0], mixed_group(hidden)])
    assert load("ExampleHiddenStatesConnector")._find_cache_kv_group_id(hidden_cfg) == 1
    worker = modules[load("MooncakeConnector").__module__].MooncakeConnectorWorker
    sender = SimpleNamespace(kv_cache_config=cfg, _physical_blocks_per_logical_kv_block=2)
    assert worker._logical_to_kernel_block_ids(sender, [[2], [3]]) == [[4, 5], [3]]
    fill = modules[load("DecodeBenchConnector").__module__]._get_startup_fill_group_ids
    assert fill(SimpleNamespace(kv_cache_groups=cfg.kv_cache_groups, needs_kv_cache_zeroing=True)) == {1}
    invalid = replace(cfg, kv_cache_groups=[mixed_group(replace(state_spec(), mamba_cache_mode="all"))])
    settings = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
    with pytest.raises(ValueError, match="mamba_cache_mode"):
        load("MooncakeStoreConnector")._validate_kv_cache_config(settings, invalid)


def test_hybrid_prefix_alignment_and_scheduler_copy():
    if not hasattr(engine_core, "resolve_cache_hit_alignment_tokens"):
        pytest.skip("vLLM uses fixed cache hit alignment on this release")
    state = state_spec()
    groups = [KVCacheGroupSpec(["attention"], full_spec()), mixed_group(state)]
    config = KVCacheConfig(16, [], groups)
    scheduler = engine_core.generate_scheduler_kv_cache_config([config, config])
    wrapped = scheduler.kv_cache_groups[1].kv_cache_spec
    assert isinstance(wrapped, UniformTypeKVCacheSpecs)
    assert wrapped.kv_cache_specs is not groups[1].kv_cache_spec.kv_cache_specs
    assert {name: spec.reuse for name, spec in wrapped.kv_cache_specs.items()} == {"layer.0": False, "layer.1": True}

    reference = replace(config, kv_cache_groups=[groups[0], replace(groups[1], kv_cache_spec=state)])
    settings = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    assert engine_core.resolve_cache_hit_alignment_tokens(scheduler, settings, 32, 16) == (
        kv_cache_utils.resolve_cache_hit_alignment_tokens(reference, settings, 32, 16)
    )
    with pytest.raises(ValueError, match="block counts"):
        engine_core.generate_scheduler_kv_cache_config([config, replace(config, num_blocks=8)])


@pytest.mark.parametrize("make_spec", [full_spec, state_spec])
def test_manager_allocation_with_per_layer_reuse(make_spec):
    spec = make_spec()
    reuse = next(f for f in fields(spec) if f.name == "reuse")
    assert reuse.kw_only and not reuse.compare
    assert hash(spec) == hash(replace(spec, reuse=True))
    wrapped = mixed_group(spec).kv_cache_spec
    managers = [
        single_type_kv_cache_manager.get_manager_for_kv_cache_spec(
            group_spec,
            max_in_flight_tokens=32,
            max_model_len=128,
            block_pool=BlockPool(32, enable_caching=False, hash_block_size=16),
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=spec.block_size,
        )
        for group_spec in (spec, wrapped)
    ]
    assert type(managers[0]) is type(managers[1])
    blocks = [[b.block_id for b in manager.allocate_new_blocks("request", 32, 32)] for manager in managers]
    assert blocks[0] == blocks[1]
