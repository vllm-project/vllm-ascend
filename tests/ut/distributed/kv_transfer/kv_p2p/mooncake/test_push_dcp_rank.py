# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Regression tests: push derives the DCP in-group rank like upstream.

Upstream parallel_state builds one DCP group per segment of ``dcp_size``
consecutive (TP, PCP) positions, ``lin = tp * pcp_size + pcp_rank``, and
``GroupCoordinator`` derives ``rank_in_group`` from the order of that member
list. Since vllm@25e9f7b3f2 the list is sorted by global rank
(``pcp * tp_size + tp``); the v0.33.0 tag keeps the TP-major order. The push
worker must mirror the installed vLLM, because the destination attention kernel
assigns stripes by ``dcp_group.rank_in_group``.
"""

from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from tests.ut.kv_offload.mooncake_v2 import test_push_sending
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import push_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakePCPTransferMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadataGroups,
)

worker = test_push_sending.worker
LAYER = "model.layers.0.attn"


@pytest.fixture
def sorted_members(monkeypatch):
    """Emulate a vLLM build that sorts DCP group members by global rank."""
    monkeypatch.setattr(push_worker, "_dcp_groups_sorted", lambda: True)


@pytest.fixture
def legacy_members(monkeypatch):
    """Emulate a vLLM build that keeps the TP-major group order."""
    monkeypatch.setattr(push_worker, "_dcp_groups_sorted", lambda: False)


def reference_rank(tp_rank, pcp_rank, tp_size, pcp_size, dcp_size, sort):
    """Independent re-statement of the upstream group and its ordering."""
    lin = tp_rank * pcp_size + pcp_rank
    start = lin - lin % dcp_size
    members = list(range(start, start + dcp_size))
    if sort:
        members.sort(key=lambda member: member % pcp_size * tp_size + member // pcp_size)
    return members.index(lin)


def remote_metadata(tp_size=1, pcp_size=1, dcp_size=1, heads=2):
    return MooncakeTransferMetadataGroups(
        engine_id="d",
        scheduler_host="127.0.0.1",
        scheduler_port=9000,
        pp_size=1,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        tp_size=tp_size,
        use_kv_pp=False,
        metadata_by_pp_rank={
            0: MooncakePPTransferMetadata(
                block_size=4,
                cp_kv_cache_interleave_size=4,
                num_blocks=100,
                layer_names=[LAYER],
                layer_block_sizes=[4],
                group_indices=[0],
                block_strides=[[heads * 16]],
                block_lens=[[heads * 16]],
                block_shapes=[[(heads, 4, 2)]],
                block_size_scales=[[1]],
                indexer_replication_sizes=[0],
                dcp_sharded=[True],
                metadata_by_pcp_rank={
                    pcp: MooncakePCPTransferMetadata(
                        {tp: MooncakeTPTransferMetadata(2, [0], [[10000]], "127.0.0.1", 9001) for tp in range(tp_size)}
                    )
                    for pcp in range(pcp_size)
                },
            )
        },
    )


def derived_rank(tp_rank, pcp_rank, tp_size, pcp_size, dcp_size):
    remote = remote_metadata(tp_size=tp_size, pcp_size=pcp_size, dcp_size=dcp_size)
    return push_worker.MooncakePushConnectorWorker._get_remote_dcp_rank(
        None, remote, {"consumer_rank": [0, pcp_rank, tp_rank]}
    )


def test_sorted_rank_swaps_prefill_and_tensor_parallel_ranks(sorted_members):
    """(TP=2, PCP=2, DCP=4) is the topology where the lin-based shortcut swaps."""
    assert derived_rank(0, 0, 2, 2, 4) == 0
    assert derived_rank(1, 0, 2, 2, 4) == 1
    assert derived_rank(0, 1, 2, 2, 4) == 2
    assert derived_rank(1, 1, 2, 2, 4) == 3


@pytest.mark.parametrize("tp_size,pcp_size,dcp_size", [(2, 2, 2), (2, 2, 4), (4, 2, 4), (4, 2, 8), (8, 1, 4)])
def test_sorted_rank_matches_upstream_group_construction(sorted_members, tp_size, pcp_size, dcp_size):
    for pcp_rank in range(pcp_size):
        for tp_rank in range(tp_size):
            expected = reference_rank(tp_rank, pcp_rank, tp_size, pcp_size, dcp_size, sort=True)
            assert derived_rank(tp_rank, pcp_rank, tp_size, pcp_size, dcp_size) == expected


@pytest.mark.parametrize("tp_size,pcp_size,dcp_size", [(2, 2, 4), (4, 2, 8)])
def test_legacy_order_kept_for_unsorted_upstream(legacy_members, tp_size, pcp_size, dcp_size):
    for pcp_rank in range(pcp_size):
        for tp_rank in range(tp_size):
            lin = tp_rank * pcp_size + pcp_rank
            assert derived_rank(tp_rank, pcp_rank, tp_size, pcp_size, dcp_size) == lin % dcp_size


def test_remote_dcp_rank_zero_when_dcp_disabled():
    assert derived_rank(0, 0, 2, 2, 1) == 0


def _build_layer_request(connector, remote, consumer_rank):
    """Drive the real call site: ``_build_layer_transfer_request``."""
    meta = Mock()
    meta.local_block_ids = [[1]]
    meta.num_prompt_tokens = 8
    state = push_worker.SendReqInfo(local_request_id="req", metadata=meta)
    state.available_layers[LAYER] = (4, Mock())
    params = {
        "consumer_rank": list(consumer_rank),
        "num_computed_tokens": 0,
        "num_external_tokens": 4,
        "block_ids": [[2]],
    }
    destination = push_worker.RemoteReqInfo(params, Future())
    destination.metadata_future.set_result(remote)
    return connector._build_layer_transfer_request("req", state, destination, LAYER)


def test_unsharded_layer_build_skips_rank_lookup(worker, monkeypatch):
    """A layer the destination does not shard never needs the DCP rank, so the
    call site must not evaluate it."""
    remote = remote_metadata(tp_size=2, pcp_size=2, dcp_size=4)
    remote.metadata_by_pp_rank[0].dcp_sharded[0] = False
    monkeypatch.setattr(push_worker.MooncakePushConnectorWorker, "_get_remote_dcp_rank", Mock())
    request, _ = _build_layer_request(worker, remote, (0, 1, 0))
    assert request is not None
    assert request.remote_block_ids == (2,)
    push_worker.MooncakePushConnectorWorker._get_remote_dcp_rank.assert_not_called()


def test_sharded_layer_build_derives_rank(worker, monkeypatch, sorted_members):
    """A layer the destination shards must resolve the derived rank."""
    remote = remote_metadata(tp_size=2, pcp_size=2, dcp_size=4)
    seen = []
    original = push_worker.MooncakePushConnectorWorker._get_remote_dcp_rank

    def spy(self, remote_arg, params):
        seen.append(params["consumer_rank"])
        return original(self, remote_arg, params)

    monkeypatch.setattr(push_worker.MooncakePushConnectorWorker, "_get_remote_dcp_rank", spy)
    request, _ = _build_layer_request(worker, remote, (0, 1, 0))
    assert request is not None
    assert seen == [[0, 1, 0]]
