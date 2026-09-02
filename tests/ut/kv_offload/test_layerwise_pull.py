from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.read_thread import (
    PullBackend,
    plan_block_reads,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_pull.send_thread import SlotReuseTracker


@pytest.mark.parametrize("container", [list, tuple])
def test_pull_backend_passes_only_read_addresses_to_engine(container):
    engine = MagicMock()
    engine.batch_transfer_sync_read.return_value = 0
    backend = PullBackend(engine, "test", lambda ret: ret != 0)

    local = container([100, 200])
    remote = container([1000, 2000])
    lengths = container([16, 32])
    backend.read("peer", local, remote, lengths)

    engine.batch_transfer_sync_read.assert_called_once_with(
        "peer",
        [100, 200],
        [1000, 2000],
        [16, 32],
    )
    if container is list:
        args = engine.batch_transfer_sync_read.call_args.args
        assert args[1] is local
        assert args[2] is remote
        assert args[3] is lengths


def test_pull_backend_normalizes_engine_failure():
    engine = MagicMock()
    engine.batch_transfer_sync_read.return_value = -1
    backend = PullBackend.mooncake(engine)

    with pytest.raises(RuntimeError, match="Mooncake READ failed"):
        backend.read("peer", [100], [1000], [16])


@pytest.mark.parametrize("use_numpy", [False, True])
def test_block_planner_coalesces_only_when_both_sides_are_contiguous(use_numpy):
    source = [1, 2, 3]
    destination = [4, 5, 7]
    if use_numpy:
        source = np.asarray(source, dtype=np.int64)
        destination = np.asarray(destination, dtype=np.int64)
    remote, local, lengths = plan_block_reads(
        remote_base=1000,
        local_base=2000,
        remote_block_ids=source,
        local_block_ids=destination,
        remote_stride=16,
        local_stride=16,
        length=16,
    )

    np.testing.assert_array_equal(remote, [1016, 1048])
    np.testing.assert_array_equal(local, [2064, 2112])
    np.testing.assert_array_equal(lengths, [32, 16])
    np.testing.assert_array_equal(source, [1, 2, 3])
    np.testing.assert_array_equal(destination, [4, 5, 7])


def test_block_planner_accepts_empty_numpy_arrays():
    remote, local, lengths = plan_block_reads(
        remote_base=1000,
        local_base=2000,
        remote_block_ids=np.empty(0, dtype=np.int64),
        local_block_ids=np.empty(0, dtype=np.int64),
        remote_stride=16,
        local_stride=16,
        length=16,
    )
    assert remote.size == local.size == lengths.size == 0


def test_block_planner_rejects_mismatched_numpy_block_counts():
    with pytest.raises(ValueError, match="block counts differ"):
        plan_block_reads(
            remote_base=1000,
            local_base=2000,
            remote_block_ids=np.asarray([1, 2], dtype=np.int64),
            local_block_ids=np.asarray([4], dtype=np.int64),
            remote_stride=16,
            local_stride=16,
            length=16,
        )


def test_slot_reuse_waits_for_all_endpoints_and_ignores_old_reply():
    tracker = SlotReuseTracker({0: (0,), 4: (0,), 5: (1,)})
    first = (10, "decode-0")
    second = (10, "decode-1")
    tracker.begin(first, [0, 1])
    tracker.begin(second, [0])

    assert not tracker.events[0].is_set()
    # Slot 1 is not shared by another layer, so it never enters the gate.
    assert tracker.events[1].is_set()

    tracker.complete(first)
    assert not tracker.events[0].is_set()
    tracker.complete((9, "decode-1"))
    assert not tracker.events[0].is_set()
    tracker.complete(second)
    assert tracker.events[0].is_set()


def test_slot_reuse_failure_releases_gate_and_keeps_error():
    tracker = SlotReuseTracker({0: (0,), 1: (0,)})
    reader = (1, "decode")
    tracker.begin(reader, [0])

    tracker.complete(reader, "read failed")

    assert tracker.events[0].is_set()
    assert tracker.error(0) == "read failed"
