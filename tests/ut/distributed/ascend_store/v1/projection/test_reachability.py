"""Reachable prefixes preserve contiguous hits and partial-tail semantics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    make_topology,
)
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
