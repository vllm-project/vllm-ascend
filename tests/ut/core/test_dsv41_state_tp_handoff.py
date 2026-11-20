# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import (
    STATE_PROTOCOL_VERSION,
    CachePlane,
    SourceLease,
    StateSnapshot,
    state_source_rank,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.dsv41 import transfer_addresses


def metadata(tp, capacity, *, fingerprint="same", receiver=False):
    planes = [
        CachePlane("l.long_kv_cache", 0, "history", 0, capacity, 128, 2, 256, 2, "bfloat16", (128, 1)),
        CachePlane("l.swa_cache", 0, "window", 1, capacity, 128, 2, 256, 1, "bfloat16", (128, 1)),
        CachePlane("l.compressor.state_cache", 0, "compressor", 2, capacity, 32, 4, 128, 1, "float32", (32, 1)),
    ]
    layout = {
        "version": STATE_PROTOCOL_VERSION,
        "model_id": "model",
        "tp_size": tp,
        "fingerprint": fingerprint,
        "planes": [asdict(plane) for plane in planes],
    }
    names = [plane.name for plane in planes]
    if receiver:
        return SimpleNamespace(state_layout=layout, layer_names=names, kv_caches_base_addr=[[7000], [8000], [9000]])
    ranks = {
        rank: SimpleNamespace(kv_caches_base_addr=[[1000 + 10000 * rank], [2000 + 10000 * rank], [3000 + 10000 * rank]])
        for rank in range(tp)
    }
    return SimpleNamespace(
        state_layout=layout, layer_names=names, metadata_by_pcp_rank={0: SimpleNamespace(metadata_by_tp_rank=ranks)}
    )


@pytest.mark.parametrize(("d_rank", "p_rank"), [(0, 0), (1, 2)])
def test_tp4_to_tp2_reads_all_planes_from_one_producer_rank(d_rank, p_rank):
    snapshot = StateSnapshot.create("generation", "request", "model", 128, 129, 4)
    source = state_source_rank(4, 2, d_rank)
    assert source == p_rank
    src, dst, sizes = transfer_addresses(
        snapshot, metadata(2, 10, receiver=True), metadata(4, 7), source, ((3,), (2,), (4,)), ((2,), (1,), (3,))
    )
    assert src == [1512 + p_rank * 10000, 2256 + p_rank * 10000, 3384 + p_rank * 10000]
    assert dst == [7768, 8512, 9512]
    assert sizes == [128, 256, 128]


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_equal_tp_keeps_corresponding_rank(tp):
    assert state_source_rank(tp, tp, tp - 1) == tp - 1


@pytest.mark.parametrize(("producer", "consumer", "rank"), [(2, 4, 0), (8, 2, 0), (4, 2, 2), (True, 1, 0)])
def test_unsupported_tp_rejected(producer, consumer, rank):
    with pytest.raises(ValueError):
        state_source_rank(producer, consumer, rank)


def test_topology_or_checkpoint_disagrees_before_read():
    snapshot = StateSnapshot.create("generation", "request", "model", 128, 129, 4)
    ids = ((1,), (1,), (1,))
    with pytest.raises(ValueError, match="snapshot TP"):
        transfer_addresses(snapshot, metadata(2, 10, receiver=True), metadata(2, 10), 0, ids, ids)
    with pytest.raises(ValueError, match="fingerprint"):
        transfer_addresses(snapshot, metadata(2, 10, receiver=True), metadata(4, 10, fingerprint="other"), 0, ids, ids)


def test_producer_lease_records_used_source_ranks_until_aggregate_completion():
    snapshot = StateSnapshot.create("generation", "request", "model", 128, 129, 4)
    lease = SourceLease(snapshot, expires_at=5)
    for rank in range(2):
        assert lease.acquire("generation", "consumer", state_source_rank(4, 2, rank), now=2)
    assert lease.acquired_ranks == {0, 2}
    assert not lease.may_expire(now=6)
    assert not lease.complete("generation", "another")
    assert lease.complete("generation", "consumer")
