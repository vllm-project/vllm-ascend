"""Configuration normalization and support boundaries, before process-local binding."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import resolve
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.resolve import (
    resolve_group_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.topology import (
    KVPoolLayerTopology,
)


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
    assert result.consumer_pipeline_partitions == ((4,) if layerwise else (2, 2))
    assert not result.load_enabled and result.store_enabled
    assert extra == before


@pytest.mark.parametrize(
    "extra,parallel,model,groups,message",
    (
        ({"backend": 1}, {}, {}, "attention", "backend must be a string"),
        ({"backend": "unknown"}, {}, {}, "attention", "Unsupported AscendStore"),
        ({"backend": "yuanrong"}, {}, {}, "attention", "Yuanrong Backend"),
        (
            {"use_layerwise": True, "consumer_is_to_put": True, "prefill_pp_size": 2},
            {},
            {},
            "attention",
            "Consumer pipeline Store",
        ),
        ({"use_layerwise": True, "prefill_tp_size": 2}, {}, {}, "attention", "does not support TP mismatch"),
        ({"use_layerwise": True}, {}, {}, "align", "Mamba align-state groups"),
        ({"use_layerwise": True}, {}, {}, "private", "private KV state requires non-layerwise"),
        ({}, {}, {}, "private_only", "at least one prefix-cacheable"),
        ({}, {}, {}, "mamba", "mamba_cache_mode='align'"),
        (
            {"use_layerwise": True},
            {"decode_context_parallel_size": 2},
            {},
            "attention",
            "does not support context parallelism",
        ),
        ({"discard_partial_chunks": False}, {}, {}, "attention", "discard_partial_chunks=False"),
        ({"use_layerwise": True, "discard_partial_chunks": False}, {}, {}, "attention", "discard_partial_chunks=False"),
        (
            {"prefill_tp_size": 4},
            {"tensor_parallel_size": 2},
            {"use_mla": True, "heads": 1},
            "attention",
            "cannot represent the configured TP mismatch",
        ),
        (
            {"prefill_tp_size": 4},
            {"tensor_parallel_size": 2},
            {"heads": 6},
            "attention",
            "cannot represent the configured TP mismatch",
        ),
        (
            {"prefill_tp_size": 4},
            {"tensor_parallel_size": 2},
            {"heads": 8, "index_topk": 32},
            "attention",
            "sparse KV layouts",
        ),
        (
            {"prefill_tp_size": 4},
            {"tensor_parallel_size": 2},
            {"heads": 8},
            "two_attention",
            "one dense transferable KV cache group",
        ),
        (
            {"prefill_tp_size": 4, "consumer_is_to_put": True, "prefill_pp_size": 2},
            {"tensor_parallel_size": 2},
            {"heads": 8},
            "attention",
            "Consumer pipeline Store",
        ),
        (
            {},
            {"tensor_parallel_size": 2, "prefill_context_parallel_size": 2, "decode_context_parallel_size": 2},
            {"heads": 8},
            "attention",
            "combined DCP and PCP",
        ),
        (
            {},
            {"tensor_parallel_size": 4, "decode_context_parallel_size": 2},
            {"heads": 1},
            "attention",
            "remaining same-key TP writer replicas",
        ),
    ),
)
def test_configuration_rejects_unsupported_compositions(configuration_case, extra, parallel, model, groups, message):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.kv_connector_extra_config.update(extra)
    for name, value in parallel.items():
        setattr(config.parallel_config, name, value)
    config.model_config.use_mla = model.get("use_mla", False)
    config.model_config.get_total_num_kv_heads.return_value = model.get("heads", 4)
    if "index_topk" in model:
        config.model_config.hf_text_config.index_topk = model["index_topk"]
    attention = cache.kv_cache_groups[0].kv_cache_spec
    state = MambaSpec(
        block_size=4, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align" if groups == "align" else "none"
    )
    private = CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=1, head_size_v=0, dtype=torch.float32)
    specs = {
        "attention": [attention],
        "two_attention": [attention, attention],
        "align": [attention, state],
        "mamba": [state],
        "private": [attention, private],
        "private_only": [private],
    }[groups]
    cache.kv_cache_groups = [SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    cache.transfer_group_ids = tuple(range(len(specs)))
    with pytest.raises(ValueError, match=message):
        resolve.resolve_kv_pool_config(config, cache)


@pytest.mark.parametrize(
    "scheduler_size,hash_size,group_size", ((6, 4, 4), (12, 4, 6), (12, 4, 8), (0, 4, 4), (4, 0, 4))
)
def test_invalid_block_geometry(configuration_case, monkeypatch, scheduler_size, hash_size, group_size):
    config, cache = configuration_case
    cache.kv_cache_groups[0].kv_cache_spec = replace(cache.kv_cache_groups[0].kv_cache_spec, block_size=group_size)
    monkeypatch.setattr(resolve.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (scheduler_size, hash_size))
    with pytest.raises(ValueError, match="block size"):
        resolve.resolve_kv_pool_config(config, cache)


@pytest.mark.parametrize("prefix_caching,hash_size", ((False, 64), (True, 16), (False, 8)))
@pytest.mark.parametrize("raw_sizes", ((16, 32), (32, 32)))
def test_multigroup_dcp_keeps_v1_hash_granularity_when_prefix_caching_is_disabled(
    configuration_case, monkeypatch, prefix_caching, hash_size, raw_sizes
):
    config, cache = configuration_case
    config.cache_config.enable_prefix_caching = prefix_caching
    config.parallel_config.decode_context_parallel_size = 2
    original = cache.kv_cache_groups[0].kv_cache_spec
    specs = tuple(replace(original, block_size=size) for size in raw_sizes)
    cache.kv_cache_groups = [SimpleNamespace(kv_cache_spec=spec) for spec in specs]
    effective_specs = tuple(resolve.resolve_dcp_kv_cache_spec(spec, 2) for spec in specs)
    engine_resolver = Mock(return_value=(64, hash_size))
    monkeypatch.setattr(resolve.kv_cache_utils, "resolve_kv_cache_block_sizes", engine_resolver)
    expected_hash_size = hash_size if prefix_caching else min(hash_size, raw_sizes[0])
    assert resolve._resolve_kv_pool_block_sizes(config, cache, effective_specs, (0, 1)) == (
        64,
        hash_size,
        expected_hash_size,
    )
    assert config.cache_config.enable_prefix_caching is prefix_caching
    engine_resolver.assert_called_once_with(cache, config)


@pytest.mark.parametrize("role,peer", (("kv_producer", "decode"), ("kv_consumer", "prefill")))
@pytest.mark.parametrize("axis", ("dcp", "pcp"))
def test_layerwise_peer_context_parallel_must_match(configuration_case, role, peer, axis):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = role
    config.parallel_config.decode_context_parallel_size = 2
    config.kv_transfer_config.kv_connector_extra_config.update(
        backend="memcache", use_layerwise=True, **{f"{peer}_{axis}_size": 1 if axis == "dcp" else 2}
    )
    with pytest.raises(ValueError, match="same dcp_size/pcp_size"):
        resolve.resolve_kv_pool_config(config, cache)


@pytest.mark.parametrize("attention_companion", (False, True))
@pytest.mark.parametrize("role,peer", (("kv_producer", "decode"), ("kv_consumer", "prefill")))
def test_align_state_tp_mismatch_is_rejected(configuration_case, role, peer, attention_companion):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = role
    config.kv_transfer_config.kv_connector_extra_config[f"{peer}_tp_size"] = 4
    config.parallel_config.tensor_parallel_size = 2
    config.model_config.get_total_num_kv_heads.return_value = 8
    state = MambaSpec(block_size=4, shapes=((2, 3),), dtypes=(torch.float32,), mamba_cache_mode="align")
    cache.kv_cache_groups = (cache.kv_cache_groups if attention_companion else []) + [
        SimpleNamespace(kv_cache_spec=state)
    ]
    cache.transfer_group_ids = tuple(range(len(cache.kv_cache_groups)))
    with pytest.raises(ValueError, match="one dense transferable KV cache group"):
        resolve.resolve_kv_pool_config(config, cache)


@pytest.mark.parametrize("kvpp_size", (1, 2))
def test_kvpp_uses_authoritative_size(configuration_case, monkeypatch, kvpp_size):
    config, cache = configuration_case
    config.additional_config = {"enable_kvpp": True}
    config_type = SimpleNamespace(from_vllm_config=lambda observed: SimpleNamespace(size=kvpp_size))
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_config", SimpleNamespace(KVPPConfig=config_type))
    if kvpp_size > 1:
        with pytest.raises(ValueError, match="active KVPP"):
            resolve.resolve_kv_pool_config(config, cache)
    else:
        assert resolve.resolve_kv_pool_config(config, cache).transfer_group_ids == (0,)


def test_layerwise_buffer_reuse_requires_partial_object_support(configuration_case, monkeypatch):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
    monkeypatch.setattr(resolve, "_uses_layerwise_buffer_reuse", lambda *_: True)
    with pytest.raises(ValueError, match="buffer reuse requires native partial-object support"):
        resolve.resolve_kv_pool_config(config, cache)


def test_layer_topology_groups_shared_entries_and_offsets_mtp_layers() -> None:
    layers = resolve_group_layers(["model.layers.1.v", "mtp.layers.0.attn", "model.layers.1.k"], base_layer_count=4)
    assert layers == (
        KVPoolLayerTopology(1, ("model.layers.1.k", "model.layers.1.v")),
        KVPoolLayerTopology(4, ("mtp.layers.0.attn",)),
    )


@pytest.mark.parametrize(
    "kv_role, extra_config, use_layerwise",
    (
        ("kv_producer", {"decode_dcp_size": 2, "decode_pcp_size": 1}, True),
        ("kv_consumer", {"prefill_dcp_size": 2, "prefill_pcp_size": 1}, True),
        ("kv_producer", {}, True),
        ("kv_consumer", {}, True),
        ("kv_both", {"prefill_dcp_size": 1, "decode_pcp_size": 2}, True),
        ("kv_producer", {"decode_dcp_size": 1}, False),
        ("kv_consumer", {"prefill_pcp_size": 2}, False),
    ),
)
def test_peer_context_parallel_check_preserves_production_scope(kv_role, extra_config, use_layerwise) -> None:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=kv_role, kv_connector_extra_config=extra_config),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2, prefill_context_parallel_size=1),
        speculative_config=None,
    )

    resolve._validate_kv_pool_preflight(config, "memcache", use_layerwise=use_layerwise)
