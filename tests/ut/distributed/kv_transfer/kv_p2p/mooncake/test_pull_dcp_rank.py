# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import vllm.distributed.parallel_state as parallel_state

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import pull_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakePCPTransferMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
)


def _sorted_init():
    group_ranks = []
    group_ranks = [sorted(x.tolist()) for x in group_ranks]
    return group_ranks


def _unsorted_init():
    return []


def _make_remote_pp_metadata(
    remote_tp_size: int,
    remote_pcp_size: int,
    dcp_ranks: dict[tuple[int, int], int] | None,
) -> MooncakePPTransferMetadata:
    return MooncakePPTransferMetadata(
        block_size=4,
        cp_kv_cache_interleave_size=4,
        num_blocks=16,
        layer_names=["model.layers.0.self_attn.attn"],
        layer_block_sizes=[4],
        group_indices=[0],
        block_strides=[[16]],
        block_lens=[[16]],
        block_shapes=[[(1, 4, 2)]],
        block_size_scales=[[1]],
        indexer_replication_sizes=[0],
        dcp_sharded=[True],
        metadata_by_pcp_rank={
            pcp: MooncakePCPTransferMetadata(
                metadata_by_tp_rank={
                    tp: MooncakeTPTransferMetadata(
                        te_rpc_port=9000 + tp,
                        layer_indices=[0],
                        kv_caches_base_addr=[[1000]],
                        local_ip="127.0.0.1",
                        handshake_port=8000 + tp,
                        dcp_rank=None if dcp_ranks is None else dcp_ranks[(tp, pcp)],
                    )
                    for tp in range(remote_tp_size)
                }
            )
            for pcp in range(remote_pcp_size)
        },
    )


def _fresh_thread(tp_size: int, pcp_size: int) -> pull_worker.MooncakePullRecvingThread:
    if hasattr(pull_worker.MooncakePullRecvingThread, "_vllm_sorts_dcp_groups"):
        delattr(pull_worker.MooncakePullRecvingThread, "_vllm_sorts_dcp_groups")
    thread = object.__new__(pull_worker.MooncakePullRecvingThread)
    thread.tp_rank = 0
    thread.tp_size = tp_size
    thread.pcp_size = pcp_size
    return thread


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
def test_pull_dcp_group_members_sorted_fallback(monkeypatch, remote_tp_size, remote_pcp_size, remote_dcp_size):
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", _sorted_init)
    thread = _fresh_thread(remote_tp_size, remote_pcp_size)

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


def test_pull_uses_peer_advertised_dcp_rank_across_mixed_orderings(monkeypatch):
    # Local decoder is sorted, remote prefiller is unsorted: peer dcp_rank must win.
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", _sorted_init)
    thread = _fresh_thread(tp_size=2, pcp_size=2)
    unsorted_peer_ranks = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    remote_pp = _make_remote_pp_metadata(2, 2, unsorted_peer_ranks)

    groups = thread._get_attention_remote_tp_rank_groups(
        remote_pcp_size=2,
        remote_tp_size=2,
        total_num_kv_heads=1,
        local_dcp_size=4,
        remote_dcp_size=4,
        remote_metadata=remote_pp,
    )
    assert groups == [[0, 1, 2, 3]]

    # Local decoder is unsorted, remote prefiller is sorted: peer dcp_rank must win.
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", _unsorted_init)
    thread = _fresh_thread(tp_size=2, pcp_size=2)
    sorted_peer_ranks = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    remote_pp_sorted = _make_remote_pp_metadata(2, 2, sorted_peer_ranks)

    groups = thread._get_attention_remote_tp_rank_groups(
        remote_pcp_size=2,
        remote_tp_size=2,
        total_num_kv_heads=1,
        local_dcp_size=4,
        remote_dcp_size=4,
        remote_metadata=remote_pp_sorted,
    )
    assert groups == [[0, 2, 1, 3]]


def test_pull_rejects_invalid_peer_dcp_ranks():
    thread = _fresh_thread(tp_size=2, pcp_size=2)
    duplicate_ranks = {(0, 0): 0, (0, 1): 0, (1, 0): 2, (1, 1): 3}
    remote_pp = _make_remote_pp_metadata(2, 2, duplicate_ranks)

    with pytest.raises(ValueError, match="Invalid remote DCP in-group ranks"):
        thread._get_attention_remote_tp_rank_groups(
            remote_pcp_size=2,
            remote_tp_size=2,
            total_num_kv_heads=1,
            local_dcp_size=4,
            remote_dcp_size=4,
            remote_metadata=remote_pp,
        )


def test_pull_dcp_group_members_legacy_unsorted_fallback(monkeypatch):
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", _unsorted_init)
    thread = _fresh_thread(tp_size=2, pcp_size=2)
    remote_pp = _make_remote_pp_metadata(2, 2, None)

    groups = thread._get_attention_remote_tp_rank_groups(
        remote_pcp_size=2,
        remote_tp_size=2,
        total_num_kv_heads=1,
        local_dcp_size=4,
        remote_dcp_size=4,
        remote_metadata=remote_pp,
    )
    assert groups == [[0, 1, 2, 3]]
