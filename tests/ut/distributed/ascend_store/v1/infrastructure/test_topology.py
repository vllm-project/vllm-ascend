"""Partition contracts are rank-free; local topology adds coordinates, not writer policy."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import configuration
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import topology as topology_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.topology import (
    ContextRank,
    CPPartitionSpec,
    KVPoolLayerTopology,
    PPPartitionSpec,
    context_replica_groups,
    resolve_group_layers,
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
        monkeypatch.setattr(topology_module, name, Mock(side_effect=AssertionError("rank during admission")))
    admitted = configuration.resolve_kv_pool_config(config, cache)
    assert admitted.pp_partition == expected_pp
    assert admitted.cp_partition == expected_cp


def test_target_pp_counts_have_one_source(configuration_case):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.kv_connector_extra_config.update(consumer_is_to_put=True, prefill_pp_size=2)
    admitted = configuration.resolve_kv_pool_config(config, cache)
    assert admitted.pp_partition == PPPartitionSpec((2, 2))


def test_pcp_does_not_multiply_the_wire_formats_reserved_dcp_slots(configuration_case):
    config, _ = configuration_case
    config.parallel_config.prefill_context_parallel_size = 3
    config.parallel_config.decode_context_parallel_size = 2
    pp, cp = configuration.resolve_object_partitions(
        config, "memcache", use_layerwise=True, put_step=2, target_layer_counts=None
    )
    assert pp == PPPartitionSpec(global_layer_count=4)
    assert cp == CPPartitionSpec(3, 2, 2, 2 * 1024 * 1024)


@pytest.mark.parametrize(
    "layer_names,stage_count",
    (
        (["model.layers.3", "model.layers.4"], 4),
        (["model.layers.3.k", "model.layers.3.v", "model.layers.4", "model.layers.5", "model.layers.6", "mtp.0"], 5),
    ),
)
def test_local_topology_binds_stage_context_and_explicit_collective_members(
    configuration_case, monkeypatch, layer_names, stage_count
):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config.update(backend="memcache", use_layerwise=True)
    config.parallel_config.tensor_parallel_size = 4
    config.parallel_config.pipeline_parallel_size = 2
    config.parallel_config.decode_context_parallel_size = 2
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_total_num_hidden_layers = lambda: 7
    config.model_config.get_layers_start_end_indices = Mock(return_value=(3, 7))
    config.model_config.get_num_layers = Mock(side_effect=AssertionError("duplicate PP range query"))
    cache.kv_cache_groups[0].layer_names = layer_names
    admitted = configuration.resolve_kv_pool_config(config, cache)
    tp = SimpleNamespace(rank_in_group=3, ranks=[8, 9, 10, 11], cpu_group=object())
    monkeypatch.setattr(topology_module, "get_tp_group", lambda: tp)
    monkeypatch.setattr(topology_module, "get_pp_group", lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(topology_module, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=1))
    gather = Mock(
        side_effect=lambda output, _local, **_kw: output.__setitem__(
            slice(None), [(ContextRank(8 + rank, rank, rank % 2),) for rank in range(4)]
        )
    )
    monkeypatch.setattr(topology_module.dist, "all_gather_object", gather)
    topology = topology_module.resolve_worker_topology(config, cache, admitted)
    assert topology.config.tp_partition is admitted.tp_partition
    assert topology.config.pp_partition is admitted.pp_partition
    assert topology.config.cp_partition is admitted.cp_partition
    assert (topology.stage_layer_start, topology.stage_layer_count, topology.object_shard_index) == (3, stage_count, 1)
    assert topology.global_rank == 11
    assert topology.context_global_ranks == (8, 9, 10, 11)
    assert topology.context_replica_groups == ((8, 10), (9, 11))
    gather.assert_called_once_with(
        [(ContextRank(8 + rank, rank, rank % 2),) for rank in range(4)], (ContextRank(11, 3, 1),), group=tp.cpu_group
    )
    config.model_config.get_layers_start_end_indices.assert_called_once_with(config.parallel_config)
    config.model_config.get_num_layers.assert_not_called()


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
    monkeypatch.setattr(topology_module, "get_tp_group", lambda: tp)
    monkeypatch.setattr(topology_module, "get_pcp_group", lambda: pcp)
    monkeypatch.setattr(topology_module, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(topology_module, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=members[1].dcp_rank))
    calls = []

    def gather(output, local, *, group):
        calls.append(group)
        if group is pcp.cpu_group:
            assert local == (members[1],)
            output[:] = [(members[1],), (members[3],)]
        else:
            assert group is tp.cpu_group and local == (members[1], members[3])
            output[:] = [(members[0], members[2]), (members[1], members[3])]

    monkeypatch.setattr(topology_module.dist, "all_gather_object", gather)
    admitted = configuration.resolve_kv_pool_config(config, cache)
    topology = topology_module.resolve_worker_topology(config, cache, admitted)
    assert calls == [pcp.cpu_group, tp.cpu_group]
    assert topology.global_rank == 70
    assert topology.context_global_ranks == (10, 20, 40, 70)
    assert topology.config.cp_partition.object_shard_count == dcp
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
    monkeypatch.setattr(topology_module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, ranks=[4, 7]))
    monkeypatch.setattr(topology_module, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(topology_module, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(topology_module.dist, "all_gather_object", Mock(side_effect=AssertionError("unused exchange")))
    admitted = configuration.resolve_kv_pool_config(config, cache)
    topology = topology_module.resolve_worker_topology(config, cache, admitted)
    assert topology.global_rank == 4
    assert bool(topology.context_replica_groups) is (layerwise and store)


def test_layer_topology_groups_shared_entries_and_offsets_mtp_layers() -> None:
    layers = resolve_group_layers(["model.layers.1.v", "mtp.layers.0.attn", "model.layers.1.k"], base_layer_count=4)
    assert layers == (
        KVPoolLayerTopology(1, ("model.layers.1.k", "model.layers.1.v")),
        KVPoolLayerTopology(4, ("mtp.layers.0.attn",)),
    )


@pytest.mark.parametrize("ratios,expected", (((), "default"), ((4,), "c4"), ((4, 4), "c4"), ((1, 4), "mixed")))
def test_cache_family_uses_specs_without_model_fallback(configuration_case, monkeypatch, ratios, expected):
    config, cache = configuration_case
    attention = cache.kv_cache_groups[0].kv_cache_spec
    specs = {
        f"layers.{index}": AscendSlidingWindowMLASpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype=attention.dtype,
            sliding_window=16,
            compress_ratio=ratio,
        )
        for index, ratio in enumerate(ratios)
    }
    spec = (
        UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs)
        if len(specs) > 1
        else next(iter(specs.values()), attention)
    )
    cache.kv_cache_groups = [KVCacheGroupSpec(list(specs) or ["layers.0"], spec)]
    admitted = configuration.resolve_kv_pool_config(config, cache)
    del config.model_config.hf_text_config
    config.model_config.get_layers_start_end_indices = Mock(side_effect=AssertionError("unused global span"))
    monkeypatch.setattr(topology_module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, ranks=[0]))
    monkeypatch.setattr(topology_module, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))

    topology = topology_module.resolve_worker_topology(config, cache, admitted)

    assert topology.transfer_groups[0].key_metadata.cache_family == expected
    assert (topology.stage_layer_start, topology.stage_layer_count) == (0, 0)
