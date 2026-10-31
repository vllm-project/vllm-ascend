"""Configuration normalization and support boundaries, before process-local binding."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec

from tests.ut.distributed.ascend_store.v1.construction_fixtures import create_scheduler, create_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import (
    configuration as configuration_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import configuration as resolve


@pytest.mark.parametrize(
    "extra,parallel,model,groups,message",
    (
        ({"backend": 1}, {}, {}, "attention", "backend must be a string"),
        ({"backend": "unknown"}, {}, {}, "attention", "Unsupported AscendStore"),
        ({"backend": "yuanrong"}, {}, {}, "attention", "Unsupported AscendStore"),
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
        ({}, {}, {}, "private_only", "at least one transferable prefix-cacheable"),
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
    cache.kv_cache_groups = [SimpleNamespace(kv_cache_spec=spec, is_eagle_group=False) for spec in specs]
    cache.transfer_group_ids = tuple(range(len(specs)))
    with pytest.raises(ValueError, match=message):
        create_scheduler(config, cache, "ipc://unused")


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
        create_scheduler(config, cache, "ipc://unused")


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
        SimpleNamespace(kv_cache_spec=state, is_eagle_group=False)
    ]
    cache.transfer_group_ids = tuple(range(len(cache.kv_cache_groups)))
    with pytest.raises(ValueError, match="one dense transferable KV cache group"):
        create_scheduler(config, cache, "ipc://unused")


@pytest.mark.parametrize("kvpp_size", (1, 2))
def test_kvpp_uses_authoritative_size(configuration_case, monkeypatch, kvpp_size):
    config, cache = configuration_case
    config.additional_config = {"enable_kvpp": True}
    config_type = SimpleNamespace(from_vllm_config=lambda observed: SimpleNamespace(size=kvpp_size))
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_config", SimpleNamespace(KVPPConfig=config_type))
    if kvpp_size > 1:
        with pytest.raises(ValueError, match="active KVPP"):
            create_scheduler(config, cache, "ipc://unused")
    else:
        assert create_scheduler(config, cache, "ipc://unused")._config.transfer_group_ids == (0,)


def test_layerwise_buffer_reuse_requires_partial_object_support(configuration_case, monkeypatch):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = True
    monkeypatch.setattr(configuration_module, "_uses_layerwise_buffer_reuse", lambda *_: True)
    with pytest.raises(ValueError, match="buffer reuse requires native partial-object support"):
        create_scheduler(config, cache, "ipc://unused")


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
def test_peer_context_parallel_check_preserves_production_scope(
    configuration_case, kv_role, extra_config, use_layerwise
) -> None:
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.kv_connector_extra_config.update(
        extra_config, backend="memcache", use_layerwise=use_layerwise
    )
    config.parallel_config.decode_context_parallel_size = 2
    admitted = resolve.resolve_kv_pool_config(config, cache)
    configuration_module.validate_kv_pool_config(config, cache, admitted)


@pytest.mark.parametrize("side", ("scheduler", "worker"))
def test_missing_logical_block_hash_boundary_is_rejected_before_binding(configuration_case, monkeypatch, side):
    config, cache = configuration_case
    incompatible = replace(resolve.resolve_kv_pool_config(config, cache), hash_block_size=8)
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_args: incompatible)
    bind_topology = Mock(side_effect=AssertionError("Topology must not be bound"))
    lookup_client = Mock(side_effect=AssertionError("Lookup must not be connected"))
    monkeypatch.setattr(connector_module, "resolve_worker_topology", bind_topology)
    monkeypatch.setattr(connector_module, "LookupClient", lookup_client)
    with pytest.raises(ValueError, match="group 0 block size \\(4\\) must be a multiple of hash block size \\(8\\)"):
        if side == "scheduler":
            create_scheduler(config, cache, "unused")
        else:
            create_worker(config, cache)
    bind_topology.assert_not_called()
    lookup_client.assert_not_called()
