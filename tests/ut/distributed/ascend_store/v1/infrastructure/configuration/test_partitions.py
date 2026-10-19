"""Partition contracts are rank-free; local topology adds coordinates, not writer policy."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration import resolve
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.topology import (
    ContextRank,
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
    tp = SimpleNamespace(rank_in_group=3, ranks=[8, 9, 10, 11], cpu_group=object())
    monkeypatch.setattr(resolve, "get_tp_group", lambda: tp)
    monkeypatch.setattr(resolve, "get_pp_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(resolve, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=1))
    gather = Mock(
        side_effect=lambda output, _local, **_kw: output.__setitem__(
            slice(None), [(ContextRank(8 + rank, rank, rank % 2),) for rank in range(4)]
        )
    )
    monkeypatch.setattr(resolve.dist, "all_gather_object", gather)
    topology = resolve.resolve_worker_topology(config, cache, admitted)
    assert topology.tp_partition is admitted.tp_partition
    assert topology.pp_partition is admitted.pp_partition
    assert topology.cp_partition is admitted.cp_partition
    assert (topology.stage_layer_start, topology.stage_layer_count, topology.object_shard_index) == (3, 4, 1)
    assert topology.global_rank == 11
    assert topology.context_global_ranks == (8, 9, 10, 11)
    assert topology.context_replica_groups == ((8, 10), (9, 11))
    gather.assert_called_once_with(
        [(ContextRank(8 + rank, rank, rank % 2),) for rank in range(4)], (ContextRank(11, 3, 1),), group=tp.cpu_group
    )
    config.model_config.get_layers_start_end_indices.assert_called_once_with(config.parallel_config)
    config.model_config.get_num_layers.assert_called_once_with(config.parallel_config)


@pytest.mark.parametrize(
    "rank,put_step,dcp,expected",
    ((4, 4, 2, ((4, 6), (5, 7))), (2, 2, 4, ((2,), (3,))), (3, 4, 1, ((0, 1, 2, 3),))),
)
def test_context_membership_is_not_the_number_of_reserved_slots(rank, put_step, dcp, expected):
    ranks = tuple(ContextRank(index, index, index % dcp) for index in range(8))
    assert context_replica_groups(ranks, rank, put_step) == expected


@pytest.mark.parametrize("dcp", (2, 4))
@pytest.mark.parametrize("put_step", (1, 2))
def test_pcp_dcp_uses_actual_coordinates_and_global_members(configuration_case, monkeypatch, dcp, put_step):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config.update(backend="memcache", use_layerwise=True)
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.prefill_context_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = dcp
    config.model_config.get_total_num_kv_heads.return_value = 2 // put_step
    config.model_config.get_layers_start_end_indices = lambda _: (0, 4)
    config.model_config.get_num_layers = lambda _: 4
    # Deliberately non-contiguous, non-monotone global ranks. PCP groups are
    # [40, 10] and [70, 20]; TP groups are [40, 70] and [10, 20].
    members = (
        ContextRank(40, 0, 0),
        ContextRank(70, 1, 0 if dcp == 2 else 2),
        ContextRank(10, 0, 1),
        ContextRank(20, 1, 1 if dcp == 2 else 3),
    )
    tp = SimpleNamespace(rank_in_group=1, ranks=[40, 70], cpu_group=object())
    pcp = SimpleNamespace(rank_in_group=0, ranks=[70, 20], cpu_group=object())
    monkeypatch.setattr(resolve, "get_tp_group", lambda: tp)
    monkeypatch.setattr(resolve, "get_pcp_group", lambda: pcp)
    monkeypatch.setattr(resolve, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(resolve, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=members[1].dcp_rank))
    calls = []

    def gather(output, local, *, group):
        calls.append(group)
        if group is pcp.cpu_group:
            assert local == (members[1],)
            output[:] = [(members[1],), (members[3],)]
        else:
            assert group is tp.cpu_group and local == (members[1], members[3])
            output[:] = [(members[0], members[2]), (members[1], members[3])]

    monkeypatch.setattr(resolve.dist, "all_gather_object", gather)
    admitted = resolve.resolve_kv_pool_config(config, cache)
    topology = resolve.resolve_worker_topology(config, cache, admitted)
    assert calls == [pcp.cpu_group, tp.cpu_group]
    assert topology.global_rank == 70
    assert topology.context_global_ranks == (10, 20, 40, 70)
    assert topology.cp_partition.object_shard_count == dcp
    assert topology.object_shard_index == members[1].dcp_rank
    expected = ((40, 70), (10, 20)) if dcp == 2 else ((40,), (10,), (70,), (20,))
    if put_step == 1:
        expected = ((70,), (20,))
    assert topology.context_replica_groups == expected


@pytest.mark.parametrize("layerwise,store,dcp", ((False, True, 2), (True, False, 2), (True, True, 1)))
def test_unused_context_exchange_is_skipped(configuration_case, monkeypatch, layerwise, store, dcp):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config.update(backend="memcache", use_layerwise=layerwise)
    config.kv_transfer_config.kv_role = "kv_both" if store else "kv_consumer"
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = dcp
    config.model_config.get_layers_start_end_indices = lambda _: (0, 4)
    config.model_config.get_num_layers = lambda _: 4
    monkeypatch.setattr(resolve, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, ranks=[4, 7]))
    monkeypatch.setattr(resolve, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(resolve, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(resolve.dist, "all_gather_object", Mock(side_effect=AssertionError("unused exchange")))
    admitted = resolve.resolve_kv_pool_config(config, cache)
    topology = resolve.resolve_worker_topology(config, cache, admitted)
    assert topology.global_rank == 4
    assert bool(topology.context_replica_groups) is (layerwise and store)
