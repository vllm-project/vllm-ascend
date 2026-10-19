"""Only complete, same-allocation context shards authorize shared-object publication."""

from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import make_topology
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import make_gva_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.gva.shared_store import (
    GVAWriteOutcome,
    SharedGVAWrites,
)


def outcome(keys=(), address=1000):
    return GVAWriteOutcome(frozenset(keys), {key: (address, 64) for key in keys})


@pytest.mark.parametrize(
    "peer,expected",
    (
        (outcome(("key",)), {"key"}),
        (outcome(), set()),
        (outcome(("other",)), set()),
        (outcome(("key",), address=2000), set()),
    ),
)
def test_completion_requires_every_shard_and_one_allocation(peer, expected):
    local = outcome(("key",))
    channel = Mock(all_gather=Mock(return_value=(local, peer)))
    gate = SharedGVAWrites((0, 1), channel)
    assert gate.complete_keys(local) == expected
    channel.all_gather.assert_called_once_with(local)


def test_replica_nonleaders_and_other_heads_do_not_supply_missing_shards():
    local, empty = outcome(("key",)), outcome()
    # head 1 spans TP ranks 4..7; only ranks 4 and 5 own its two DCP shards.
    outcomes = (local, local, local, local, local, empty, local, local)
    channel = Mock(all_gather=Mock(return_value=outcomes))
    gate = SharedGVAWrites((4, 5), channel)
    assert gate.complete_keys(local) == set()
    channel.all_gather.return_value = (*outcomes[:5], local, empty, empty)
    assert gate.complete_keys(local) == {"key"}


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("put_step,dcp", ((4, 1), (4, 2), (2, 4)))
def test_worker_elects_one_writer_per_context_within_its_head(rank, put_step, dcp):
    topology = make_topology(tp_rank=rank, tp_size=8, put_step=put_step, dcp_size=dcp, dcp_rank=rank % dcp)
    worker, *_ = make_gva_worker(topology=topology)
    try:
        first = rank // put_step * put_step
        expected = tuple(range(first, first + min(put_step, dcp)))
        assert worker._store_writers == expected
        assert worker._layerwise_store_leader is (rank in expected)
    finally:
        worker.close()
