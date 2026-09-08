# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)

from vllm_ascend.core.typed_kv_cache import (
    TypedKVCachePlan,
    TypedPageSpec,
)
from vllm_ascend.core.typed_kv_cache_coordinator import (
    TypedKVCacheCoordinatorNoPrefixCache,
)


def test_real_managers_allocate_and_release_typed_superpages() -> None:
    attention = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
    )
    mamba = MambaSpec(
        block_size=640,
        shapes=((3, 6144), (16, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
    )
    typed_specs = (
        TypedPageSpec(0, attention.page_size_bytes, attention.block_size),
        TypedPageSpec(1, mamba.page_size_bytes, mamba.block_size),
    )
    plan = TypedKVCachePlan.exact_lcm(
        typed_specs,
        total_memory_bytes=6 * 69_468_160,
    )
    config = KVCacheConfig(
        num_blocks=plan.num_superpages,
        kv_cache_tensors=[KVCacheTensor(plan.total_managed_bytes, ["attn", "mamba"])],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["mamba"], mamba),
        ],
    )
    coordinator = TypedKVCacheCoordinatorNoPrefixCache(
        config,
        plan,
        max_model_len=8192,
        max_in_flight_tokens=8192,
        scheduler_block_size=640,
    )

    required_superpages = coordinator.get_num_blocks_to_allocate(
        "request-0",
        num_tokens=8192,
        new_computed_blocks=([], []),
        num_encoder_tokens=0,
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=8192,
    )
    blocks = coordinator.allocate_new_blocks("request-0", 8192, 8192)

    assert required_superpages == 2
    assert [sum(not block.is_null for block in group) for group in blocks] == [
        64,
        1,
    ]
    assert coordinator.block_pool.get_num_free_blocks() == 3

    coordinator.free("request-0")

    assert coordinator.block_pool.get_num_free_blocks() == 5
    assert coordinator.block_pool.get_usage() == 0.0
    assert coordinator.block_pool.take_events() == []


def test_real_managers_allocate_through_group_address_tables() -> None:
    attention = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
    )
    mamba = MambaSpec(
        block_size=640,
        shapes=((3, 6144), (16, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="align",
    )
    typed_specs = (
        TypedPageSpec(0, attention.page_size_bytes, attention.block_size),
        TypedPageSpec(1, mamba.page_size_bytes, mamba.block_size),
    )
    plan = TypedKVCachePlan.addressed(
        typed_specs,
        total_memory_bytes=(3 * mamba.page_size_bytes + 70 * attention.page_size_bytes),
    )
    config = KVCacheConfig(
        num_blocks=min(plan.num_blocks(0), plan.num_blocks(1)),
        kv_cache_tensors=[KVCacheTensor(plan.total_managed_bytes, ["attn", "mamba"])],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["mamba"], mamba),
        ],
    )
    coordinator = TypedKVCacheCoordinatorNoPrefixCache(
        config,
        plan,
        max_model_len=8192,
        max_in_flight_tokens=8192,
        scheduler_block_size=640,
    )

    assert (
        coordinator.get_num_blocks_to_allocate(
            "request-0",
            num_tokens=8192,
            new_computed_blocks=([], []),
            num_encoder_tokens=0,
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=8192,
        )
        == 0
    )
    blocks = coordinator.allocate_new_blocks("request-0", 8192, 8192)

    assert [sum(not block.is_null for block in group) for group in blocks] == [
        64,
        1,
    ]
    attention_interval = {plan.page_address_tables_bytes[0][block.block_id] for block in blocks[0] if not block.is_null}
    mamba_interval = {plan.page_address_tables_bytes[1][block.block_id] for block in blocks[1] if not block.is_null}
    assert attention_interval.isdisjoint(mamba_interval)
    assert coordinator.block_pool.get_usage() > 0

    coordinator.free("request-0")
    assert coordinator.block_pool.get_usage() == 0.0
