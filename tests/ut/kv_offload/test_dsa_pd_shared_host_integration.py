# SPDX-License-Identifier: Apache-2.0
"""Shared Host ownership through the production per-spec address builder."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from .mooncake_sfa_helpers import make_thread
from .mooncake_v2.helpers import make_pp_metadata


@pytest.mark.parametrize(
    "rank,size,shared_host,blocks,geometry,expected",
    [
        pytest.param(3, 8, True, 16, (128, 128, 128), ([6], [26], [256]), id="shared-contiguous"),
        pytest.param(1, 3, True, 8, (256, 512, 64), ([3, 4, 5], [23, 24, 25], [64] * 3), id="partial-blocks"),
        pytest.param(1, 3, True, 8, (128, 256, 128), ([3, 4, 5], [23, 24, 25], [128] * 3), id="remote-strided"),
        pytest.param(1, 3, True, 8, (256, 128, 128), ([3, 4, 5], [23, 24, 25], [128] * 3), id="local-strided"),
        pytest.param(7, 8, True, 2, (128, 128, 128), ([], [], []), id="no-owned-blocks"),
        pytest.param(3, 8, False, 16, (128, 128, 128), ([0], [20], [2048]), id="rank-local-hbm"),
    ],
)
def test_ownership_is_applied_before_address_building(rank, size, shared_host, blocks, geometry, expected):
    local_stride, remote_stride, transfer_len = geometry
    thread = make_thread(
        kv_cache_specs=[MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16)],
        kv_caches_base_addr=[[1024]],
        block_strides=[[local_stride]],
        block_lens=[[transfer_len]],
        _sfa_host_kv=SimpleNamespace(rank=rank, size=size, contains=lambda ptr: shared_host and ptr == 1024),
    )
    remote = make_pp_metadata(
        block_strides=[[remote_stride]],
        block_lens=[[transfer_len]],
        tp_base_addrs={0: [[100000]]},
    )
    src: list[int] = []
    dst: list[int] = []
    lengths: list[int] = []
    thread._append_spec_transfer_addresses(
        0,
        0,
        0,
        1,
        1,
        1,
        {(0, 0): [("request", list(range(blocks)), list(range(20, 20 + blocks)))]},
        remote,
        src,
        dst,
        lengths,
    )
    local_ids, remote_ids, expected_lengths = expected
    assert src == [1024 + block * local_stride for block in local_ids]
    assert dst == [100000 + block * remote_stride for block in remote_ids]
    assert lengths == expected_lengths
