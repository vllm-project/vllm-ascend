"""Cache-group selection and recovery preserve request-visible token boundaries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec, FullAttentionSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from tests.ut.distributed.ascend_store.v1.helpers import (
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import make_config
from tests.ut.distributed.ascend_store.v1.worker.bulk.fixtures import make_sparse_group_topology
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.coordinator import (
    Coordinator,
    ReachablePrefix,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.protocol.messages import (
    LookupResult,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler.lookup import resolve_load_candidate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.rows import (
    lookup_chunk_ranges,
    store_transfer_records,
)


@pytest.mark.parametrize(
    "start,end,hash_count,block_count,expected_hashes,expected_ids",
    (
        (0, 24, 6, 6, (0, 1, 2, 3, 4, 5), [1, 2, 3, 4, 5, 6]),
        (4, 24, 6, 2, (4, 5), [1, 2]),
        (8, 24, 4, 6, (2, 3), [3, 4]),
        (24, 24, 6, 6, (), []),
    ),
)
def test_store_records_intersect_hashes_and_retained_suffix(
    start, end, hash_count, block_count, expected_hashes, expected_ids
) -> None:
    hashes = tuple(bytes([index]) for index in range(hash_count))
    counts, selected_hashes, ids = store_transfer_records(
        end, hashes, tuple(range(1, block_count + 1)), block_size=4, hash_block_size=4, start_token=start
    )
    assert ids == expected_ids
    assert selected_hashes == [hashes[index] for index in expected_hashes]
    assert counts == [4] * len(expected_ids)


@pytest.mark.parametrize("group_ids", ((0,), (0, 1)), ids=("single", "multiple"))
def test_coordinator_rejects_runtime_observation_shape_mismatches(group_ids) -> None:
    topology = make_topology(group_ids=group_ids)
    coordinator = Coordinator(topology.transfer_groups, 4, 4, 64)
    with pytest.raises(ValueError, match="observation"):
        coordinator.resolve_available_end(0, 4, (b"a",), ())

    rows = (np.asarray((0,)), np.asarray((4,)), (b"a",))
    observations = tuple((rows, ()) for _ in coordinator.group_ids)
    with pytest.raises(ValueError, match="different lengths"):
        coordinator.resolve_available_end(0, 4, (b"a",), observations)


def test_coordinator_rejects_unregistered_manager_during_binding(monkeypatch) -> None:
    monkeypatch.setattr(KVCacheSpecRegistry, "get_manager_class", lambda _spec: None)
    with pytest.raises(ValueError, match="No vLLM KV cache manager is registered"):
        Coordinator(make_topology(group_ids=(0, 1)).transfer_groups, 4, 4, 64)


@pytest.mark.parametrize("local_end,end_token,expected_end", ((0, 16, 8), (8, 16, 8), (12, 16, 16), (12, 14, 12)))
def test_full_attention_combines_local_prefix_with_each_groups_remote_hits(local_end, end_token, expected_end) -> None:
    coordinator = Coordinator(make_topology(group_ids=(3, 7)).transfer_groups, 4, 4, 64)
    hashes = (b"a", b"b", b"c", b"d")
    rows = lookup_chunk_ranges(16, hashes, block_size=4, hash_block_size=4)
    observations = ((rows, (True, True, True, True)), (rows, (True, True, False, True)))
    assert coordinator.resolve_available_end(local_end, end_token, hashes, observations) == ReachablePrefix(
        expected_end
    )


def test_full_attention_honors_custom_registered_manager(monkeypatch) -> None:
    class NoReuseManager(FullAttentionManager):
        @classmethod
        def find_longest_cache_hit(cls, **kwargs):
            return tuple([] for _ in kwargs["kv_cache_group_ids"]), 0

    monkeypatch.setattr(KVCacheSpecRegistry, "get_manager_class", lambda _spec: NoReuseManager)
    coordinator = Coordinator(make_topology().transfer_groups, 4, 4, 64)
    rows = lookup_chunk_ranges(4, (b"a",), block_size=4, hash_block_size=4)
    assert coordinator.resolve_available_end(0, 4, (b"a",), ((rows, (True,)),)) == ReachablePrefix(0)
    with pytest.raises(ValueError, match="cannot cover the confirmed Load endpoint"):
        coordinator.select_for_load((b"a",), 4)


@pytest.mark.parametrize("attention_end", (12, 16))
def test_coordinator_preserves_contiguous_prefixes_and_covering_tail_keys(attention_end) -> None:
    single = Coordinator(make_topology().transfer_groups, 4, 4, 64)
    block_hashes: tuple[bytes, ...] = (b"a", b"b", b"c")
    start_token, end_token = 0, 12
    observations = (((np.asarray((0, 4, 8)), np.asarray((4, 4, 4)), block_hashes), (True, False, True)),)
    assert single.select_for_lookup(end_token) == (None,)
    assert single.resolve_available_end(start_token, end_token, block_hashes, observations) == ReachablePrefix(4)

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
            uses_align_state=True,
        ),
    )
    hybrid = Coordinator(groups, 16, 4, 64)
    block_hashes = (b"a", b"b", b"c", b"d")
    hybrid_observations = tuple(
        (
            (np.asarray((0, 0, 0, 0)), np.asarray((4, 8, 12, 16)), block_hashes),
            tuple(boundary == available_end for boundary in (4, 8, 12, 16)),
        )
        for available_end in (attention_end, 12)
    )
    assert hybrid.resolve_available_end(0, 16, block_hashes, hybrid_observations) == ReachablePrefix(
        12,
        (TailKeyBoundary(0, attention_end), TailKeyBoundary(1, 12)),
    )
    assert hybrid.select_for_load(block_hashes, 12) == (None, (True,))


def test_single_sliding_window_uses_window_and_retention_rules() -> None:
    window = make_sparse_group_topology(block_size=16, sliding_window=33).transfer_groups[1]
    coordinator = Coordinator((window,), 16, 16, 256, retention_interval=0)
    hashes = tuple(bytes([index]) for index in range(1, 9))
    rows = lookup_chunk_ranges(128, hashes, block_size=16, hash_block_size=16)
    tail = (False,) * 6 + (True, True)

    assert coordinator.resolve_available_end(0, 128, hashes, ((rows, tail),)) == ReachablePrefix(128)
    assert coordinator.select_for_load(hashes, 128) == (tail,)
    # Store preserves the replay window at prompt length minus one, aligned down to 112.
    assert coordinator.select_for_store(0, 128, 128) == ((False,) * 5 + (True, True, False),)


def test_sparse_lookup_keeps_upstream_alignment_and_window_requirements() -> None:
    topology = make_sparse_group_topology(block_size=16, sliding_window=33)
    coordinator = Coordinator(topology.transfer_groups, 64, 16, 256, retention_interval=0)
    block_hashes = tuple(bytes([index]) for index in range(1, 9))
    start_token, end_token = 0, 128
    masks = coordinator.select_for_lookup(end_token)
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
            coordinator.resolve_available_end(start_token, end_token, block_hashes, observations).end_token
            == expected_end
        )


def test_checkpoint_reduction_rechecks_the_sliding_window() -> None:
    groups = make_sparse_group_topology(block_size=16, sliding_window=33).transfer_groups
    state = replace(
        groups[0],
        group_id=7,
        kv_cache_spec=MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align"),
    )
    coordinator = Coordinator((*groups, state), 16, 16, 256)
    hashes = tuple(bytes([index]) for index in range(8))
    rows = lookup_chunk_ranges(128, hashes, block_size=16, hash_block_size=16)
    observations = (
        (rows, (True,) * 8),
        (rows, (False, False, True, True, False, False, True, True)),
        (rows, (False, False, False, True, False, True, False, False)),
    )
    # Mamba moves 128 to 96; the window at 96 is absent, so both must settle at 64.
    assert coordinator.resolve_available_end(0, 128, hashes, observations) == ReachablePrefix(64)
    assert coordinator.select_for_load(hashes, 64) == (
        None,
        (False, False, True, True),
        (False, False, False, True),
    )


@pytest.mark.parametrize("eagle,physical_end", ((False, 128), (True, 112)))
def test_sliding_load_uses_scheduler_physical_frontier_without_another_eagle_drop(eagle, physical_end) -> None:
    window = make_sparse_group_topology(block_size=16, sliding_window=33).transfer_groups[1]
    coordinator = Coordinator((window,), 16, 16, 256, use_eagle=eagle)
    candidate = resolve_load_candidate(
        make_config(granularity=16, eagle=eagle),
        LookupResult(128),
        query_start_token=0,
        query_end_token=128,
        local_cached_tokens=0,
        request_token_count=128,
    )
    assert candidate is not None
    assert candidate.load_end_token == physical_end
    hashes = tuple(bytes([index]) for index in range(8))
    assert coordinator.select_for_load(hashes, physical_end) == ((False,) * (physical_end // 16 - 2) + (True, True),)


def test_chunked_local_load_can_skip_every_block_at_a_chunk_boundary() -> None:
    group = replace(
        make_topology().transfer_groups[0],
        kv_cache_spec=ChunkedLocalAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32, attention_chunk_size=64
        ),
    )
    coordinator = Coordinator((group,), 16, 16, 256)
    assert coordinator.select_for_load(tuple(bytes([index]) for index in range(8)), 128) == ((False,) * 8,)


@pytest.mark.parametrize(
    "start,blocks,expected_ids",
    (
        (80, tuple(range(100, 108)), [105, 106]),
        (100, (105, 106, 107), [106]),
        (16, (105, 106, 107), [105, 106]),
        (0, (106, 107), [106]),
        (128, (105, 106, 107), []),
        (144, (105, 106, 107), []),
    ),
)
def test_store_suffix_mask_intersects_requested_and_retained_ranges(start, blocks, expected_ids) -> None:
    window = make_sparse_group_topology(block_size=16, sliding_window=33).transfer_groups[1]
    coordinator = Coordinator((window,), 16, 16, 256, retention_interval=0)
    hashes = tuple(bytes([index]) for index in range(8))
    (mask,) = coordinator.select_for_store(start, 128, 128)
    assert mask is not None and len(mask) == max(8 - start // 16, 0)
    counts, selected_hashes, ids = store_transfer_records(
        128, hashes, blocks, block_size=16, hash_block_size=16, start_token=start, mask=mask
    )
    assert ids == expected_ids
    assert selected_hashes == [hashes[block_id - 100] for block_id in expected_ids]
    assert counts == [16] * len(expected_ids)
