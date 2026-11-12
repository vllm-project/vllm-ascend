# SPDX-License-Identifier: Apache-2.0
"""Request-scoped ownership plans preserve every layer's transfer addresses."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import worker as pull_worker

from ..mooncake_sfa_helpers import make_thread
from .helpers import make_full_spec, make_pp_metadata


def expand_transfers(src, dst, lengths, targets):
    # Compare copied byte ranges independently of descriptor coalescing.
    return [
        (local + offset, remote + offset, target)
        for local, remote, length, target in zip(src, dst, lengths, targets)
        for offset in range(length)
    ]


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("layout", ["contiguous", "strided", "mixed"])
@pytest.mark.parametrize("kind", ["mla", "hnd", "partial_heads"])
@pytest.mark.parametrize("with_targets", [False, True])
def test_shared_plans_preserve_addresses_and_targets_across_layers(rank, layout, kind, with_targets):
    partial_heads = kind == "partial_heads"
    local_heads = 2 if partial_heads else 1
    spec = (
        MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
        if kind == "mla"
        else make_full_spec(num_kv_heads=local_heads)
    )
    names = ["host.0", "host.1", "hbm"]
    local_bases = [[10000, 20000], [100000, 200000], [1000000, 2000000]]
    remote_bases = [[3000000, 4000000], [5000000, 6000000], [7000000, 8000000]]
    remote_lengths = [128, 64]
    local_lengths = [size * local_heads for size in remote_lengths]
    local_strides = [
        size * (2 if layout == "strided" or (layout == "mixed" and index == 1) else 1)
        for index, size in enumerate(local_lengths)
    ]
    remote_strides = [
        size * (2 if layout == "strided" or (layout == "mixed" and index == 1) else 1)
        for index, size in enumerate(remote_lengths)
    ]
    thread = make_thread(
        layer_names=names,
        kv_cache_specs=[spec],
        kv_caches_base_addr=local_bases,
        block_strides=[local_strides] * 3,
        block_lens=[local_lengths] * 3,
        block_shapes=[[(local_heads, 16, 4), (local_heads, 16, 2)]] * 3,
    )
    targets = [[object(), object()] for _ in names]
    thread._transfer_targets = targets
    remote = make_pp_metadata(
        layer_names=names,
        block_strides=[remote_strides] * 3,
        block_lens=[remote_lengths] * 3,
        block_shapes=[[(1, 16, 4), (1, 16, 2)]] * 3,
        tp_base_addrs={peer: remote_bases for peer in range(2 if partial_heads else 1)},
    )
    entries = [("a", [3, 1, 2, 3, 9], [13, 11, 12, 33, 19]), ("b", [8, 6], [28, 26])]
    # Explicit ownership includes an aliased destination and ranks with no B blocks.
    owned_a = [[(1, 11)], [(2, 12)], [(3, 13), (3, 33)], [(9, 19)]][rank]
    owned_b = [[(6, 26)], [(8, 28)], [], []][rank]
    host_bases = {base for layer in local_bases[:2] for base in layer}
    pool = SimpleNamespace(rank=rank, size=4, contains=lambda ptr: ptr in host_bases)
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []
    actual_targets: list[object] = []
    thread._sfa_host_kv = pool
    with (
        patch.object(
            pull_worker, "partition_shared_block_groups", wraps=pull_worker.partition_shared_block_groups
        ) as partition,
        patch.object(
            pull_worker, "group_concurrent_contiguous", wraps=pull_worker.group_concurrent_contiguous
        ) as group,
    ):
        thread._append_spec_transfer_addresses(
            0,
            0,
            1 if partial_heads else 0,
            1,
            2 if partial_heads else 1,
            1,
            {(layer, layer): entries for layer in range(3)},
            remote,
            src,
            dst,
            lengths,
            **({"transfer_targets": actual_targets} if with_targets else {}),
        )
    modes = 2 if layout == "mixed" and not partial_heads else 1
    assert partition.call_count == 2 * modes  # requests × modes, independent of layers/components
    assert group.call_count == (0 if layout == "strided" or partial_heads else 2)
    expected: list[tuple[int, int, object | None]] = []
    for layer in range(3):
        for component in range(2):
            for request, local_ids, remote_ids in entries:
                pairs = (owned_a if request == "a" else owned_b) if layer < 2 else list(zip(local_ids, remote_ids))
                for local_block, remote_block in pairs:
                    local = local_bases[layer][component] + local_block * local_strides[component]
                    if partial_heads:
                        local += remote_lengths[component]
                    peer = remote_bases[layer][component] + remote_block * remote_strides[component]
                    expected.extend(
                        (local + offset, peer + offset, targets[layer][component] if with_targets else None)
                        for offset in range(remote_lengths[component])
                    )
    assert expand_transfers(src, dst, lengths, actual_targets if with_targets else [None] * len(src)) == expected


def test_shared_plan_is_rebuilt_for_changed_request_mapping_and_tp_owner():
    spec = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    thread = make_thread(
        layer_names=["host.0", "host.1"],
        kv_cache_specs=[spec],
        kv_caches_base_addr=[[10000], [20000]],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
    )
    remote = make_pp_metadata(
        layer_names=["host.0", "host.1"],
        block_strides=[[128], [128]],
        block_lens=[[128], [128]],
        tp_base_addrs={0: [[100000], [200000]]},
    )
    pool = SimpleNamespace(rank=0, size=2, contains=lambda ptr: True)
    thread._sfa_host_kv = pool
    with (
        patch.object(
            pull_worker, "partition_shared_block_groups", wraps=pull_worker.partition_shared_block_groups
        ) as partition,
    ):
        for rank, local_ids, remote_ids, owned in [
            (0, [1, 2], [11, 12], (1, 11)),
            (0, [4, 5], [24, 25], (4, 24)),
            (1, [4, 5], [24, 25], (5, 25)),
        ]:
            pool.rank = rank
            src: list[int] = []
            dst: list[int] = []
            lengths: list[int] = []
            entries = [("same-request", local_ids, remote_ids)]
            thread._append_spec_transfer_addresses(
                0, 0, 0, 1, 1, 1, {(0, 0): entries, (1, 1): entries}, remote, src, dst, lengths
            )
            assert src == [10000 + owned[0] * 128, 20000 + owned[0] * 128]
            assert dst == [100000 + owned[1] * 128, 200000 + owned[1] * 128]
            assert lengths == [128, 128]
        assert partition.call_count == 3


def test_shared_plan_keeps_cross_layer_mapping_validation():
    thread = make_thread(layer_names=["host.0", "host.1"], kv_cache_specs=[make_full_spec()])
    with (
        patch.object(pull_worker, "partition_shared_block_groups") as partition,
        pytest.raises(ValueError, match="block IDs differ within spec"),
    ):
        thread._append_spec_transfer_addresses(
            0,
            0,
            0,
            1,
            1,
            1,
            {(0, 0): [("request", [1], [2])], (1, 1): [("request", [1], [3])]},
            make_pp_metadata(),
            [],
            [],
            [],
        )
    partition.assert_not_called()


@pytest.mark.parametrize("rank", [0, 2, 7])
@pytest.mark.parametrize("coalesce", [False, True])
@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("with_targets", [False, True])
def test_host_ownership_uses_raw_pairs_and_bound_target_classification(rank, coalesce, offloaded, with_targets):
    names = ["layer.0", "layer.1", "layer.2"]
    local_bases = [[10000, 20000], [100000, 200000], [1000000, 2000000]]
    remote_bases = [[3000000, 4000000], [5000000, 6000000], [7000000, 8000000]]
    stride = 128 if coalesce else 256
    spec = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)
    thread = make_thread(
        layer_names=names,
        kv_cache_specs=[spec],
        kv_caches_base_addr=local_bases,
        block_strides=[[stride, stride]] * 3,
        block_lens=[[128, 128]] * 3,
    )
    remote = make_pp_metadata(
        layer_names=names,
        block_strides=[[stride, stride]] * 3,
        block_lens=[[128, 128]] * 3,
        tp_base_addrs={0: remote_bases},
    )
    targets = [[SimpleNamespace(offloaded=offloaded) for _ in range(2)] for _ in names]
    thread._transfer_targets = targets
    pool = SimpleNamespace(rank=rank, size=8, contains=lambda ptr: offloaded)
    thread._sfa_host_kv = pool
    entries = [("r", [3, 1, 2, 3, 9], [13, 11, 12, 33, 19])]
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []
    actual_targets: list[object] = []
    with (
        patch.object(pool, "contains", wraps=pool.contains) as contains,
        patch.object(
            pull_worker, "partition_shared_block_groups", wraps=pull_worker.partition_shared_block_groups
        ) as ownership,
        patch.object(
            pull_worker, "group_concurrent_contiguous", wraps=pull_worker.group_concurrent_contiguous
        ) as group,
    ):
        thread._append_spec_transfer_addresses(
            0,
            0,
            0,
            1,
            1,
            1,
            {(layer, layer): entries for layer in range(3)},
            remote,
            src,
            dst,
            lengths,
            **({"transfer_targets": actual_targets} if with_targets else {}),
        )
    assert contains.call_count == (0 if with_targets else 6)
    assert ownership.call_count == int(offloaded)
    assert group.call_count == int(coalesce and not offloaded)
    if offloaded:
        assert ownership.call_args.args[:2] == ([[3, 1, 2, 3, 9]], [[13, 11, 12, 33, 19]])
        pairs = {0: [(1, 11)], 2: [(3, 13), (3, 33)], 7: []}[rank]
    else:
        pairs = [(3, 13), (1, 11), (2, 12), (3, 33), (9, 19)]
    expected: list[tuple[int, int, object | None]] = []
    for layer in range(3):
        for component in range(2):
            for local, peer in pairs:
                target = targets[layer][component] if with_targets else None
                expected.extend(
                    (
                        local_bases[layer][component] + local * stride + offset,
                        remote_bases[layer][component] + peer * stride + offset,
                        target,
                    )
                    for offset in range(128)
                )
    assert expand_transfers(src, dst, lengths, actual_targets if with_targets else [None] * len(src)) == expected
