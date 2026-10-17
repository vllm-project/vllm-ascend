"""Reachable prefixes preserve contiguous hits and partial-tail semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.bulk_fixtures import make_sparse_group_topology
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection.reachability import (
    HybridReachability,
    ReachablePrefix,
    UnitaryReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.rows import lookup_chunk_rows


def test_reachability_matrix_preserves_contiguous_and_partial_tail_semantics() -> None:
    unitary = UnitaryReachability(0, max_model_len=64, cache_transfer_granularity=4)
    block_hashes = (b"a", b"b", b"c")
    query_range = TokenRange(0, 12)
    observations = (
        (
            (np.asarray((0, 4, 8)), np.asarray((4, 4, 4)), block_hashes),
            (True, False, True),
        ),
    )
    assert unitary.select_for_lookup(block_hashes, query_range) == (None,)
    assert unitary.resolve_available_end(query_range, block_hashes, observations) == ReachablePrefix(4)

    groups = (
        KVPoolGroupTopology(
            0,
            FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32),
            (KVPoolLayerTopology(0, ("layers.0",)),),
            make_topology().groups[0].key_metadata,
        ),
        KVPoolGroupTopology(
            1,
            MambaSpec(
                block_size=16,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
            (KVPoolLayerTopology(1, ("layers.1",)),),
            replace(make_topology().groups[0].key_metadata, kv_cache_group_id=1),
        ),
    )
    hybrid = HybridReachability(groups, 16, 4, 64)
    hybrid_observations = tuple(
        (
            (np.asarray((0, 0, 0)), np.asarray((4, 8, 12)), block_hashes),
            (False, False, True),
        )
        for _group_id in (0, 1)
    )
    assert hybrid.resolve_available_end(query_range, block_hashes, hybrid_observations) == ReachablePrefix(
        12,
        (TailKeyBoundary(0, 12), TailKeyBoundary(1, 12)),
    )


def test_sparse_lookup_keeps_upstream_alignment_and_window_requirements() -> None:
    topology = make_sparse_group_topology(block_size=16, sliding_window=33)
    reachability = HybridReachability(topology.transfer_groups, 64, 16, 256, retention_interval=0)
    block_hashes = tuple(bytes([index]) for index in range(1, 9))
    query_range = TokenRange(0, 128)
    masks = reachability.select_for_lookup(block_hashes, query_range)
    # The 64-token alignment still excludes blocks that cannot serve an aligned hit.
    assert masks[1] == (False, False, True, True, False, False, True, True)
    for window_boundaries, expected_end in (({48, 64, 112, 128}, 128), ({48, 64, 112}, 64)):
        observations = []
        for group, mask in zip(topology.transfer_groups, masks, strict=True):
            rows = lookup_chunk_rows(128, block_hashes, block_size=16, hash_block_size=16, mask=mask)
            available = tuple(
                group.group_id == 1 or int.from_bytes(block_hash, "big") * 16 in window_boundaries
                for block_hash in rows[2]
            )
            observations.append((rows, available))
        assert reachability.resolve_available_end(query_range, block_hashes, observations).end_token == expected_end
