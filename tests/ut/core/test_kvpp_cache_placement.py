from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_placement import (
    KVPP_STAGING_ALIGNMENT_BYTES,
    KVPPCacheMemoryBudget,
    build_cache_allocation_groups,
    create_kvpp_cache_allocation_plan,
    kvpp_staging_size_bytes,
    map_kvpp_layers_to_owners,
)


def _config(*, mtp: bool = False, layers: int = 5):
    return SimpleNamespace(
        additional_config={"enable_kvpp": True},
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=SimpleNamespace(method="mtp") if mtp else None,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=layers,
                num_nextn_predict_layers=1,
            )
        ),
    )


def _spec():
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )


def _target(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _indexer(index: int) -> str:
    return f"model.layers.{index}.self_attn.indexer.k_cache"


def test_kvpp_01_owner_is_deterministic_and_bundles_share_owner():
    names = [name for i in range(5) for name in (_target(i), _indexer(i))]

    forward = map_kvpp_layers_to_owners(_config(), names)
    reverse = map_kvpp_layers_to_owners(_config(), reversed(names))

    assert forward == reverse
    assert all(forward[_target(i)] == forward[_indexer(i)] for i in range(5))


def test_kvpp_02_remainder_partition_is_three_plus_two():
    names = [_target(i) for i in range(5)]

    owners = map_kvpp_layers_to_owners(_config(), names)

    assert [owners[name] for name in names] == [0, 0, 0, 1, 1]


def test_kvpp_03_keeps_mtp_unpartitioned_and_allocates_it_on_each_rank():
    targets = [_target(i) for i in range(5)]
    mtp = "model.layers.5.mtp_block.self_attn.attn"
    worker_spec = dict.fromkeys([*targets, mtp], _spec())
    config = _config(mtp=True)

    owners = map_kvpp_layers_to_owners(config, worker_spec)
    plans = [create_kvpp_cache_allocation_plan(config, worker_spec, rank) for rank in range(2)]

    assert mtp not in owners
    assert all(mtp in plan.physical_cache_spec for plan in plans)


def test_kvpp_04_persistent_and_two_scratch_allocations_alias_round_robin():
    names = [_target(i) for i in range(5)]
    spec = _spec()
    worker_spec = dict.fromkeys(names, spec)
    group = KVCacheGroupSpec(names, spec)
    owners = map_kvpp_layers_to_owners(_config(), names)

    groups, scratch_layer_aliases = build_cache_allocation_groups([group], worker_spec, owners, kvpp_rank=1)

    # Rank 1 owns two persistent layers; three foreign layers share exactly
    # two alternating scratch tensors.
    # Projection preserves logical group order; the physical set contains the
    # two scratch tensors (0/1) plus this rank's persistent tensors (3/4).
    assert groups[0].layer_names == [names[0], names[1], names[3], names[4]]
    assert scratch_layer_aliases == {
        names[0]: [names[0], names[2]],
        names[1]: [names[1]],
    }


def test_kvpp_05_restores_logical_cache_view():
    names = [_target(i) for i in range(5)]
    worker_spec = dict.fromkeys(names, _spec())
    plan = create_kvpp_cache_allocation_plan(_config(), worker_spec, kvpp_rank=1)
    physical_names = list(plan.physical_cache_spec)
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[KVCacheTensor(size=1024, shared_by=[name]) for name in physical_names],
        kv_cache_groups=[KVCacheGroupSpec(physical_names, _spec())],
    )

    plan.restore_logical_cache_view(config)

    logical_tensor_names = {name for tensor in config.kv_cache_tensors for name in tensor.shared_by}
    assert logical_tensor_names == set(names)
    assert set(config.kv_cache_groups[0].layer_names) == set(names)


@pytest.mark.parametrize("available_bytes", [0, 1, (2 << 20) - 1, 2 << 20, 17 << 20, (1 << 40) + 17])
def test_staging_budget_finds_maximum_fitting_block_count(available_bytes):
    kv_page_bytes, staging_page_bytes = 57344, 12288
    budget = KVPPCacheMemoryBudget.create(available_bytes, kv_page_bytes, staging_page_bytes)
    count = budget.max_num_blocks
    assert count * kv_page_bytes + kvpp_staging_size_bytes(staging_page_bytes, count) <= available_bytes
    assert (count + 1) * kv_page_bytes + kvpp_staging_size_bytes(staging_page_bytes, count + 1) > available_bytes
    assert isinstance(count, int)


def test_staging_alignment_is_included_in_budget():
    alignment = KVPP_STAGING_ALIGNMENT_BYTES
    budget = KVPPCacheMemoryBudget.create(3 * alignment, alignment, alignment + 1)
    assert budget.max_num_blocks == 1
    assert kvpp_staging_size_bytes(alignment + 1, 1) == 2 * alignment


def test_staging_budget_counts_one_bundle_and_excludes_mtp():
    config = _config(mtp=True)
    config.cache_config = SimpleNamespace(num_gpu_blocks_override=None, enable_cross_layers=False)
    main_spec = _spec()
    indexer_spec = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16)
    worker_spec = {_target(i): main_spec for i in range(5)}
    worker_spec.update({_indexer(i): indexer_spec for i in range(5)})
    worker_spec["model.layers.5.mtp_block.self_attn.attn"] = main_spec
    plan = create_kvpp_cache_allocation_plan(config, worker_spec, kvpp_rank=1)
    assert plan.staging_page_size_bytes == main_spec.page_size_bytes + indexer_spec.page_size_bytes
    budget = plan.plan_memory(config, 1 << 30)
    assert budget.kv_page_size_bytes == sum(spec.page_size_bytes for spec in plan.physical_cache_spec.values())
    assert config.cache_config.num_gpu_blocks_override is None
    smaller_budget = plan.plan_memory(config, 1 << 28)
    assert smaller_budget.max_num_blocks < budget.max_num_blocks


def test_final_cross_rank_block_reduction_remains_inside_budget():
    budget = KVPPCacheMemoryBudget.create(64 << 20, 65536, 16384)
    for count in (1, budget.max_num_blocks // 2, budget.max_num_blocks):
        assert count * 65536 + kvpp_staging_size_bytes(16384, count) <= 64 << 20


def test_rank_local_scratch_specs_can_differ_without_changing_staging_requirement():
    config = _config()
    config.cache_config = SimpleNamespace(num_gpu_blocks_override=9, enable_cross_layers=False)
    large = _spec()
    small = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    worker_spec = {_target(i): large if i < 3 else small for i in range(5)}
    plans = [create_kvpp_cache_allocation_plan(config, worker_spec, rank) for rank in range(2)]
    allocations = [get_kv_cache_config_from_groups(config, plan.physical_cache_groups, 1 << 30) for plan in plans]
    for rank, plan in enumerate(plans):
        scratch_spec = small if rank == 0 else large
        assert len(plan.scratch_layer_aliases) == 2
        for name in plan.scratch_layer_aliases:
            assert plan.physical_cache_spec[name] == scratch_spec
            tensor = next(tensor for tensor in allocations[rank].kv_cache_tensors if name in tensor.shared_by)
            assert tensor.size == 9 * scratch_spec.page_size_bytes
        assert plan.staging_page_size_bytes == large.page_size_bytes
    assert sum(tensor.size for tensor in allocations[0].kv_cache_tensors) != sum(
        tensor.size for tensor in allocations[1].kv_cache_tensors
    )


def test_fractional_profile_budget_produces_integer_block_count():
    budget = KVPPCacheMemoryBudget.create((17 << 20) + 0.75, 57344, 12288)
    assert isinstance(budget.max_num_blocks, int)
    assert budget == KVPPCacheMemoryBudget.create(17 << 20, 57344, 12288)
