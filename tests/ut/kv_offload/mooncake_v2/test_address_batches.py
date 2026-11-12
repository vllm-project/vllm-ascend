# SPDX-License-Identifier: Apache-2.0
"""Address ordering, integer arithmetic and descriptor identity across batch sizes."""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import MooncakePullRecvingThread


@pytest.mark.parametrize("groups", [0, 1, 2, 16, 31, 32, 33, 128, 2048])
@pytest.mark.parametrize("carry_targets", [False, True])
def test_address_paths_preserve_order_and_existing_output(groups, carry_targets):
    local = [list(range(3 * i, 3 * i + i % 3 + 1)) for i in range(groups)]
    remote = [list(range(5 * i, 5 * i + len(group))) for i, group in enumerate(local)]
    src, dst, lengths = ([11], [22], [33])
    marker = object()
    targets = [marker] if carry_targets else None
    MooncakePullRecvingThread._append_block_transfer_addresses(
        local, remote, 1000, 2000, 128, 256, 64, src, dst, lengths, 7, 9, targets, marker
    )
    assert src == [11] + [1000 + group[0] * 128 + 7 for group in local]
    assert dst == [22] + [2000 + group[0] * 256 + 9 for group in remote]
    assert lengths == [33] + [64 * len(group) for group in local]
    if carry_targets:
        assert targets is not None
        assert len(targets) == groups + 1 and all(target is marker for target in targets)


@pytest.mark.parametrize("groups", [1, 33])
def test_addresses_keep_python_integer_precision_and_signed_offsets(groups):
    base = (1 << 64) + 10
    local, remote = [[i + 1] for i in range(groups)], [[i + 2] for i in range(groups)]
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []
    MooncakePullRecvingThread._append_block_transfer_addresses(
        local, remote, base, base + 3, 7, 11, 5, src, dst, lengths, -9, -13
    )
    assert src == [base + (i + 1) * 7 - 9 for i in range(groups)]
    assert dst == [base + 3 + (i + 2) * 11 - 13 for i in range(groups)]
    assert lengths == [5] * groups
    assert all(isinstance(value, int) for value in src + dst + lengths)


@pytest.mark.parametrize("counts", [(1, 0), (0, 1), (33, 32), (32, 33)])
def test_mismatched_group_counts_keep_zip_truncation(counts):
    local, remote = [[i] for i in range(counts[0])], [[i] for i in range(counts[1])]
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []
    MooncakePullRecvingThread._append_block_transfer_addresses(local, remote, 10, 20, 3, 5, 7, src, dst, lengths)
    count = min(counts)
    assert src == [10 + 3 * i for i in range(count)]
    assert dst == [20 + 5 * i for i in range(count)]
    assert lengths == [7] * count


@pytest.mark.parametrize("length", [0, -1])
def test_nonpositive_transfer_length_does_not_append(length):
    src: list[int] = []
    dst: list[int] = []
    sizes: list[int] = []
    MooncakePullRecvingThread._append_block_transfer_addresses([[0]], [[0]], 10, 20, 3, 5, length, src, dst, sizes)
    assert not src and not dst and not sizes
