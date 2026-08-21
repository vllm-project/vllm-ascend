# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import MooncakeKVConnectorStats


def test_stats_record_reduce_and_reset() -> None:
    stats = MooncakeKVConnectorStats(data={})
    stats.record_transfer(0.01, 2**20)
    stats.record_transfer(0.03, 3 * 2**20)
    stats.record_failed_transfer()

    reduced = stats.reduce()

    assert reduced["Num successful transfers"] == 2
    assert reduced["Num failed transfers"] == 1
    assert reduced["Avg xfer time (ms)"] == 20.0
    assert reduced["Avg MB per transfer"] == 2.0
    assert reduced["Throughput (MB/s)"] == 100.0

    previous = stats.clone_and_reset()
    assert previous.num_successful_transfers == 2
    assert stats.is_empty()


def test_stats_aggregate_ignores_empty_and_merges_nonempty() -> None:
    stats = MooncakeKVConnectorStats(data={})
    empty = MooncakeKVConnectorStats(data={})
    other = MooncakeKVConnectorStats(data={})
    other.record_transfer(0.1, 1024)
    other.record_failed_transfer()

    assert stats.aggregate(empty) is stats
    stats.aggregate(other)

    assert stats.data["transfer_duration"] == [0.1]
    assert stats.data["bytes_transferred"] == [1024]
    assert stats.data["num_failed_transfers"] == [1]
