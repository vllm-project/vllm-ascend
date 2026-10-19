"""Partition contracts are rank-free; local topology adds coordinates, not writer policy."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import resolve
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.topology import (
    CPPartitionSpec,
    PPPartitionSpec,
    context_replica_groups,
)


@pytest.mark.parametrize(
    "backend,layerwise,dcp,heads,expected_pp,expected_cp",
    (
        ("mooncake", True, 1, 1, PPPartitionSpec(), CPPartitionSpec()),
        ("memcache", False, 2, 1, PPPartitionSpec(), CPPartitionSpec(1, 2)),
        ("memcache", True, 2, 4, PPPartitionSpec(None, 4), CPPartitionSpec(1, 2)),
        ("memcache", True, 2, 1, PPPartitionSpec(None, 4), CPPartitionSpec(1, 2, 2, 2 * 1024 * 1024)),
        ("memcache", True, 4, 2, PPPartitionSpec(None, 4), CPPartitionSpec(1, 4, 4, 2 * 1024 * 1024)),
    ),
)
def test_admission_resolves_specs_without_rank_or_native_client(
    configuration_case, monkeypatch, backend, layerwise, dcp, heads, expected_pp, expected_cp
):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config.update(backend=backend, use_layerwise=layerwise)
    config.parallel_config.tensor_parallel_size = 4 if dcp == 4 else 2
    config.parallel_config.decode_context_parallel_size = dcp
    config.model_config.get_total_num_kv_heads.return_value = heads
    for name in ("get_tp_group", "get_pp_group", "get_dcp_group", "get_pcp_group"):
        monkeypatch.setattr(resolve, name, Mock(side_effect=AssertionError("rank during admission")))
    admitted = resolve.resolve_kv_pool_config(config, cache)
    assert admitted.pp_partition == expected_pp
    assert admitted.cp_partition == expected_cp
    with pytest.raises(FrozenInstanceError):
        admitted.cp_partition.object_shard_count = 99  # type: ignore[misc]


def test_target_pp_counts_have_one_source(configuration_case):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.kv_connector_extra_config.update(consumer_is_to_put=True, prefill_pp_size=2)
    admitted = resolve.resolve_kv_pool_config(config, cache)
    assert admitted.pp_partition == PPPartitionSpec((2, 2))
    assert admitted.consumer_pipeline_partitions is admitted.pp_partition.target_layer_counts


def test_pcp_does_not_multiply_the_wire_formats_reserved_dcp_slots(configuration_case):
    config, _ = configuration_case
    config.parallel_config.prefill_context_parallel_size = 3
    config.parallel_config.decode_context_parallel_size = 2
    pp, cp = resolve.resolve_object_partitions(
        config, "memcache", use_layerwise=True, put_step=2, target_layer_counts=None
    )
    assert pp == PPPartitionSpec(global_layer_count=4)
    assert cp == CPPartitionSpec(3, 2, 2, 2 * 1024 * 1024)


def test_local_topology_binds_stage_context_and_explicit_collective_members(configuration_case, monkeypatch):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config.update(backend="memcache", use_layerwise=True)
    config.parallel_config.tensor_parallel_size = 4
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = 2
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_total_num_hidden_layers = lambda: 7
    config.model_config.get_layers_start_end_indices = Mock(return_value=(3, 7))
    config.model_config.get_num_layers = Mock(return_value=4)
    cache.kv_cache_groups[0].layer_names = ["model.layers.3", "model.layers.4"]
    admitted = resolve.resolve_kv_pool_config(config, cache)
    monkeypatch.setattr(resolve, "get_tp_group", lambda: SimpleNamespace(rank_in_group=3, ranks=[8, 9, 10, 11]))
    monkeypatch.setattr(resolve, "get_pp_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(resolve, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=1))
    topology = resolve.resolve_worker_topology(config, cache, admitted)
    assert topology.tp_partition is admitted.tp_partition
    assert topology.pp_partition is admitted.pp_partition
    assert topology.cp_partition is admitted.cp_partition
    assert (topology.stage_layer_start, topology.stage_layer_count, topology.object_shard_index) == (3, 4, 1)
    assert topology.tp_global_ranks == (8, 9, 10, 11)
    assert topology.context_replica_groups == ((0, 2), (1, 3))
    config.model_config.get_layers_start_end_indices.assert_called_once_with(config.parallel_config)
    config.model_config.get_num_layers.assert_called_once_with(config.parallel_config)


@pytest.mark.parametrize(
    "rank,put_step,dcp,expected",
    ((4, 4, 2, ((4, 6), (5, 7))), (2, 2, 4, ((2,), (3,))), (3, 4, 1, ((0, 1, 2, 3),))),
)
def test_context_membership_is_not_the_number_of_reserved_slots(rank, put_step, dcp, expected):
    assert context_replica_groups(rank, put_step, dcp) == expected
