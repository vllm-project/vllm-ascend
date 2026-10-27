"""Worker reads and Scheduler leases retain one shared source-interval rule."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk.fixtures import make_sparse_group_topology
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.configuration.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import TailKeyBoundary
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.reachability import (
    HybridReachability,
    KVCacheSpecRegistry,
    ReachablePrefix,
    UnitaryReachability,
    checkpoint_companion_indices,
    store_block_range,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.rows import lookup_chunk_ranges


@pytest.mark.parametrize(
    "start,end,hash_count,block_count,expected",
    ((0, 24, 6, 6, (0, 1, 2, 3, 4, 5)), (4, 24, 6, 2, (4, 5)), (8, 24, 4, 6, (2, 3)), (24, 24, 6, 6, ())),
)
def test_store_interval_intersects_hashes_and_retained_suffix(start, end, hash_count, block_count, expected) -> None:
    assert tuple(store_block_range(start, end, 4, hash_count, block_count)) == expected


@pytest.mark.parametrize(
    "boundary,published_end,block_count,expected",
    ((18, 0, 5, (0, 1, 2, 3, 4)), (18, 8, 2, (0, 1)), (18, 20, 5, (4,)), (16, 16, 4, (3,)), (18, 0, 0, ())),
)
def test_checkpoint_companions_keep_unpublished_pages_and_boundary(boundary, published_end, block_count, expected):
    assert tuple(checkpoint_companion_indices(boundary, published_end, 4, block_count)) == expected


@pytest.mark.parametrize("hybrid", (False, True), ids=("unitary", "hybrid"))
def test_reachability_rejects_runtime_observation_shape_mismatches(hybrid) -> None:
    topology = make_topology(group_ids=(0, 1))
    reachability = HybridReachability(topology.transfer_groups, 4, 4, 64) if hybrid else UnitaryReachability(0, 64, 4)
    with pytest.raises(ValueError, match="observation"):
        reachability.resolve_available_end(0, 4, (b"a",), ())

    rows = (np.asarray((0,)), np.asarray((4,)), (b"a",))
    observations = tuple((rows, ()) for _ in reachability.group_ids)
    with pytest.raises(ValueError, match="different lengths"):
        reachability.resolve_available_end(0, 4, (b"a",), observations)


def test_reachability_binds_registered_managers_once(monkeypatch) -> None:
    topology = make_topology(group_ids=(0, 1))
    reachability = HybridReachability(topology.transfer_groups, 4, 4, 64)
    monkeypatch.setattr(
        KVCacheSpecRegistry, "get_manager_class", lambda _spec: pytest.fail("Manager must already be bound")
    )
    assert reachability.select_for_lookup((b"a",), 4) == (None, None)
    assert reachability.select_for_store((b"a",), 4, 4) == (None, None)
    assert reachability.select_for_store((b"a",), 3, 4) == (None, None)


def test_reachability_rejects_unregistered_manager_during_binding(monkeypatch) -> None:
    monkeypatch.setattr(KVCacheSpecRegistry, "get_manager_class", lambda _spec: None)
    with pytest.raises(ValueError, match="No vLLM KV cache manager is registered"):
        HybridReachability(make_topology(group_ids=(0, 1)).transfer_groups, 4, 4, 64)


def test_reachability_matrix_preserves_contiguous_and_partial_tail_semantics() -> None:
    unitary = UnitaryReachability(0, max_model_len=64, cache_transfer_granularity=4)
    block_hashes = (b"a", b"b", b"c")
    start_token, end_token = 0, 12
    observations = (((np.asarray((0, 4, 8)), np.asarray((4, 4, 4)), block_hashes), (True, False, True)),)
    assert unitary.select_for_lookup(block_hashes, end_token) == (None,)
    assert unitary.resolve_available_end(start_token, end_token, block_hashes, observations) == ReachablePrefix(4)

    groups = (
        KVPoolGroupTopology(
            0,
            FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32),
            (KVPoolLayerTopology(0, ("layers.0",)),),
            make_topology().groups[0].key_metadata,
        ),
        KVPoolGroupTopology(
            1,
            MambaSpec(block_size=16, shapes=((1, 1),), dtypes=(torch.float32,), mamba_cache_mode="align"),
            (KVPoolLayerTopology(1, ("layers.1",)),),
            replace(make_topology().groups[0].key_metadata, kv_cache_group_id=1),
        ),
    )
    hybrid = HybridReachability(groups, 16, 4, 64)
    hybrid_observations = tuple(
        ((np.asarray((0, 0, 0)), np.asarray((4, 8, 12)), block_hashes), (False, False, True)) for _group_id in (0, 1)
    )
    assert hybrid.resolve_available_end(start_token, end_token, block_hashes, hybrid_observations) == ReachablePrefix(
        12,
        (TailKeyBoundary(0, 12), TailKeyBoundary(1, 12)),
    )


def test_sparse_lookup_keeps_upstream_alignment_and_window_requirements() -> None:
    topology = make_sparse_group_topology(block_size=16, sliding_window=33)
    reachability = HybridReachability(topology.transfer_groups, 64, 16, 256, retention_interval=0)
    block_hashes = tuple(bytes([index]) for index in range(1, 9))
    start_token, end_token = 0, 128
    masks = reachability.select_for_lookup(block_hashes, end_token)
    # The 64-token alignment still excludes blocks that cannot serve an aligned hit.
    assert masks[1] == (False, False, True, True, False, False, True, True)
    for window_boundaries, expected_end in (({48, 64, 112, 128}, 128), ({48, 64, 112}, 64)):
        observations = []
        for group, mask in zip(topology.transfer_groups, masks, strict=True):
            rows = lookup_chunk_ranges(128, block_hashes, block_size=16, hash_block_size=16, mask=mask)
            available = tuple(
                group.group_id == 1 or int.from_bytes(block_hash, "big") * 16 in window_boundaries
                for block_hash in rows[2]
            )
            observations.append((rows, available))
        assert (
            reachability.resolve_available_end(start_token, end_token, block_hashes, observations).end_token
            == expected_end
        )
