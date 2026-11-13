# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    map_packed_transfer_ids,
    split_if_not_byte_contiguous,
)


@pytest.mark.parametrize("local_pages,remote_pages", [(2, 2), (7, 1), (1, 7)])
def test_packed_transfer_splits_at_actual_row_boundaries(local_pages, remote_pages):
    # The wire owns dense kernel IDs. Independently sized P/D rows insert
    # different holes, which must never be combined into a transfer span.
    local = list(range(14))
    remote = list(range(28, 42))
    src = map_packed_transfer_ids([remote], remote_pages, 64 * 24, 24)
    dst = map_packed_transfer_ids([local], local_pages, 96 * 24, 24)
    src_groups, dst_groups = split_if_not_byte_contiguous(src, dst, 24, 24, 24, force_regroup=True)
    assert [bid for group in src_groups for bid in group] == [
        bid // remote_pages * 64 + bid % remote_pages for bid in remote
    ]
    assert [bid for group in dst_groups for bid in group] == [
        bid // local_pages * 96 + bid % local_pages for bid in local
    ]
    assert list(map(len, src_groups)) == list(map(len, dst_groups))
    for sources, targets in zip(src_groups, dst_groups):
        assert all(b == a + 1 for a, b in zip(sources, sources[1:]))
        assert all(b == a + 1 for a, b in zip(targets, targets[1:]))


def test_legacy_transfer_keeps_original_group_boundaries():
    groups = [[0], [1]]
    assert map_packed_transfer_ids(groups, 1, 0, 24) is groups
    assert split_if_not_byte_contiguous(groups, groups, 24, 24, 24) == (groups, groups)


def test_packed_transfer_rejects_fractional_kernel_stride():
    with pytest.raises(ValueError, match="geometry"):
        map_packed_transfer_ids([[0]], 2, 25, 24)
