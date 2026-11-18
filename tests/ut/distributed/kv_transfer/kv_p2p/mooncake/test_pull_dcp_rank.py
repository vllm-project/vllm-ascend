# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import vllm.distributed.parallel_state as parallel_state

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_worker


def _sorted_init():
    group_ranks = []
    group_ranks = [sorted(x.tolist()) for x in group_ranks]
    return group_ranks


def _unsorted_init():
    return []


def _reference_group_members(
    remote_rank_start: int,
    remote_dcp_size: int,
    remote_pcp_size: int,
    remote_tp_size: int,
    *,
    sorted_groups: bool,
) -> list[int]:
    members = list(range(remote_rank_start, remote_rank_start + remote_dcp_size))
    if sorted_groups and remote_dcp_size > 1:
        members.sort(key=lambda m: (m % remote_pcp_size) * remote_tp_size + m // remote_pcp_size)
    return members


@pytest.mark.parametrize(
    "remote_tp_size,remote_pcp_size,remote_dcp_size",
    [(2, 2, 4), (4, 2, 4), (4, 2, 8), (8, 1, 4)],
)
def test_pull_dcp_group_members_sorted(monkeypatch, remote_tp_size, remote_pcp_size, remote_dcp_size):
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", _sorted_init)
    thread = object.__new__(pull_worker.MooncakePullRecvingThread)
    thread.tp_rank = 0
    thread.tp_size = remote_tp_size
    thread.pcp_size = remote_pcp_size

    groups = thread._get_attention_remote_tp_rank_groups(
        remote_pcp_size=remote_pcp_size,
        remote_tp_size=remote_tp_size,
        total_num_kv_heads=1,
        local_dcp_size=remote_dcp_size,
        remote_dcp_size=remote_dcp_size,
    )
    expected = [_reference_group_members(0, remote_dcp_size, remote_pcp_size, remote_tp_size, sorted_groups=True)]
    if remote_tp_size * remote_pcp_size > remote_dcp_size:
        for start in range(remote_dcp_size, remote_tp_size * remote_pcp_size, remote_dcp_size):
            expected[0].extend(
                _reference_group_members(start, remote_dcp_size, remote_pcp_size, remote_tp_size, sorted_groups=True)
            )
    assert groups == expected


def test_pull_dcp_group_members_swaps_tp_and_pcp_for_tp2_pcp2_dcp4(monkeypatch):
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", _sorted_init)
    thread = object.__new__(pull_worker.MooncakePullRecvingThread)
    thread.tp_rank = 0
    thread.tp_size = 2
    thread.pcp_size = 2

    groups = thread._get_attention_remote_tp_rank_groups(
        remote_pcp_size=2,
        remote_tp_size=2,
        total_num_kv_heads=1,
        local_dcp_size=4,
        remote_dcp_size=4,
    )
    # Global rank order is (pcp * tp_size + tp): (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3.
    assert groups == [[0, 2, 1, 3]]


def test_pull_dcp_group_members_legacy_unsorted(monkeypatch):
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", _unsorted_init)
    thread = object.__new__(pull_worker.MooncakePullRecvingThread)
    thread.tp_rank = 0
    thread.tp_size = 2
    thread.pcp_size = 2

    groups = thread._get_attention_remote_tp_rank_groups(
        remote_pcp_size=2,
        remote_tp_size=2,
        total_num_kv_heads=1,
        local_dcp_size=4,
        remote_dcp_size=4,
    )
    assert groups == [[0, 1, 2, 3]]
