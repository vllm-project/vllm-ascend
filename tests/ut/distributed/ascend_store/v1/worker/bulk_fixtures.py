"""Hybrid cache topologies with align-state and sparse original group identities."""

from __future__ import annotations

from dataclasses import replace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    make_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
)


def make_align_state_topology():
    base = make_topology(group_ids=(0, 1), physical_layers=(0,), tp_size=2)
    attention = KVPoolGroupTopology(
        0,
        FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=1, dtype=torch.float32),
        (KVPoolLayerTopology(0, ("layers.0.attention",)),),
        replace(base.groups[0].key_metadata, cache_family="attention"),
    )
    state = KVPoolGroupTopology(
        1,
        MambaSpec(
            block_size=8,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        (KVPoolLayerTopology(1, ("layers.1.state",)),),
        replace(base.groups[1].key_metadata, cache_family="state"),
    )
    return replace(
        base,
        cache_transfer_granularity=8,
        hash_block_size=4,
        groups=(attention, state),
    )


def make_sparse_group_topology():
    base = make_topology(group_ids=(1, 3), physical_layers=(0,))
    attention = KVPoolGroupTopology(
        1,
        FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32),
        (KVPoolLayerTopology(0, ("layers.0.attention",)),),
        replace(base.groups[1].key_metadata, cache_family="attention"),
    )
    sliding = KVPoolGroupTopology(
        3,
        SlidingWindowSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=8,
        ),
        (KVPoolLayerTopology(1, ("layers.1.swa",)),),
        replace(base.groups[3].key_metadata, cache_family="swa"),
    )
    return replace(base, groups=(attention, sliding))
