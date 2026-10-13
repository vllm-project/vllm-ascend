"""Source protection follows Store selection, not the request's whole block table."""

from dataclasses import replace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    RSWASpec,
    SlidingWindowSpec,
)

from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import FakeBlockPool, make_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    RangeStoreCommand,
    StateCheckpointSource,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.source_leases import StoreSourceLeases


@pytest.mark.parametrize("source_groups", ((0,), (1, 3), ()))
def test_stable_sources_need_neither_references_nor_release_jobs(source_groups) -> None:
    leases = StoreSourceLeases(make_config(groups=source_groups, store=True))
    pool = FakeBlockPool()
    leases.bind_block_pool(pool)
    command = RangeStoreCommand("request", TokenRange(4, 8), ((1, 2, 3),) * 4, (b"a", b"b"), 8)

    assert leases.acquire(command) is command
    assert not leases.has_pending()
    assert pool.touched == []


@pytest.mark.parametrize(
    "retention_interval,expected", ((None, (22, 23, 24, 25)), (0, (24, 25)), (64, (22, 23, 24, 25)))
)
def test_recycling_sources_respect_range_retention_and_original_group_ids(retention_interval, expected) -> None:
    attention = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
    window = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=33)
    config = replace(
        make_config(granularity=16, groups=(1, 3), store=True),
        hash_block_size=16,
        store_group_specs=((1, attention, False), (3, window, False)),
        retention_interval=retention_interval,
    )
    leases = StoreSourceLeases(config)
    pool = FakeBlockPool()
    leases.bind_block_pool(pool)
    command = RangeStoreCommand(
        "request",
        TokenRange(32, 96),
        ((40,), (1, 2, 3, 4, 5, 6), (41,), (20, 21, 22, 23, 24, 25, 26)),
        tuple(bytes([index]) for index in range(6)),
        97,
    )

    leased = leases.acquire(command)

    assert pool.touched == [expected]
    assert leased.store_job_id is not None
    leases.release({leased.store_job_id: 1})
    assert pool.freed == [tuple(reversed(expected))]


@pytest.mark.parametrize("published_end,expected", ((0, (10, 1, 2, 3)), (8, (10, 3)), (12, (10, 3))))
def test_partial_checkpoint_protects_only_state_and_required_companion_pages(published_end, expected) -> None:
    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    state = MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align")
    config = replace(
        make_config(granularity=16, groups=(0, 1), align_groups=frozenset((1,)), store=True),
        store_group_specs=((0, attention, False), (1, state, False)),
    )
    leases = StoreSourceLeases(config)
    pool = FakeBlockPool()
    leases.bind_block_pool(pool)
    command = CheckpointStoreCommand(
        "request",
        ((1, 2, 3, 4), (10, 11, 12)),
        (b"a", b"b", b"c"),
        published_end,
        (StateCheckpointSource(1, 10, 12),),
    )

    leased = leases.acquire(command)

    assert pool.touched == [expected]
    leases.release({leased.store_job_id: 1})
    assert pool.freed == [tuple(reversed(expected))]


def test_unselected_recycling_rows_create_no_lease() -> None:
    leases = StoreSourceLeases(make_config(store=True, recycling=True))
    pool = FakeBlockPool()
    leases.bind_block_pool(pool)
    # The selected source is null; neither the old prefix nor speculative suffix is read.
    command = RangeStoreCommand("request", TokenRange(4, 8), ((1, 0, 3),), (b"a", b"b"), 8)

    assert leases.acquire(command) is command
    assert not leases.has_pending()
    assert pool.touched == []


@pytest.mark.skipif(not hasattr(BlockPool, "free_blocks"), reason="Requires registered upstream recycling managers")
@pytest.mark.parametrize(
    "spec_type,options", ((ChunkedLocalAttentionSpec, {"attention_chunk_size": 4}), (RSWASpec, {"rswa_window": 4}))
)
def test_recycling_manager_keeps_sources_even_when_spec_inherits_full_attention(spec_type, options) -> None:
    spec = spec_type(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32, **options)
    config = replace(make_config(store=True), store_group_specs=((0, spec, False),))
    leases = StoreSourceLeases(config)
    pool = FakeBlockPool()
    leases.bind_block_pool(pool)
    command = RangeStoreCommand("request", TokenRange(4, 8), ((1, 2, 3),), (b"a", b"b"), 8)

    leased = leases.acquire(command)

    assert pool.touched == [(2,)]
    leases.release({leased.store_job_id: 1})
    assert pool.freed == [(2,)]
