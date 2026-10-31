# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.kv_read_batcher import KVReadBatcher


@pytest.mark.parametrize("limit", [0, 256, 4096, 262144])
def test_bounded_reads_copy_every_byte(limit):
    source = bytes(range(256)) * 2051
    destination = bytearray(len(source))
    calls = []

    def read(session, sources, destinations, lengths):
        calls.append(lengths)
        for src, dst, size in zip(sources, destinations, lengths):
            destination[dst : dst + size] = source[src : src + size]
        return 0

    engine = Mock(batch_transfer_sync_read=read)
    batcher = KVReadBatcher(limit, 2)
    assert batcher.read(engine, "peer", [0, 511], [0, 511], [511, len(source) - 511]) == 0
    assert destination == source
    if limit:
        assert all(sum(sizes) <= limit and len(sizes) <= 2 for sizes in calls)
    else:
        assert calls == [[511, len(source) - 511]]


def test_descriptor_splitting_preserves_offsets_and_tails():
    rng = random.Random(42)
    for _ in range(500):
        sizes = [rng.randrange(100000) for _ in range(rng.randrange(1, 30))]
        sources = [i * 1000000 for i in range(len(sizes))]
        destinations = [s + 100000000 for s in sources]
        batcher = KVReadBatcher(rng.choice([256, 4096, 262144]), 8)
        offsets = {source: 0 for source in sources}
        for srcs, dsts, lengths in batcher._batches(sources, destinations, sizes):
            assert sum(lengths) <= batcher.max_bytes
            assert len(lengths) <= 8
            for src, dst, size in zip(srcs, dsts, lengths):
                origin = src // 1000000 * 1000000
                assert src - origin == offsets[origin]
                assert (src - origin) % 256 == 0
                assert dst - src == 100000000
                offsets[origin] += size
        assert list(offsets.values()) == sizes


@pytest.mark.parametrize("outcome", [-7, RuntimeError("read failed")])
def test_failed_chunk_stops_request_and_releases_next_reader(outcome):
    batcher = KVReadBatcher(256)
    engine = Mock()
    engine.batch_transfer_sync_read.side_effect = [outcome, 0]
    if isinstance(outcome, Exception):
        with pytest.raises(RuntimeError, match="read failed"):
            batcher.read(engine, "failed", [0], [0], [1024])
    else:
        assert batcher.read(engine, "failed", [0], [0], [1024]) == outcome
    assert engine.batch_transfer_sync_read.call_count == 1
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(batcher.read, engine, "healthy", [0], [0], [1])
        assert result.result(timeout=5) == 0


def test_fifo_admission_between_chunks():
    batcher = KVReadBatcher(256)
    first_read = threading.Event()
    second_queued = threading.Event()
    release = threading.Event()
    calls = []

    # Observe the real wait boundary to order contenders without sleeps.
    original_wait = batcher._condition.wait

    def wait(timeout=None):
        second_queued.set()
        return original_wait(timeout)

    batcher._condition.wait = wait

    def read(session, sources, destinations, sizes):
        calls.append(session)
        if len(calls) == 1:
            first_read.set()
            assert release.wait(5)
        return 0

    engine = Mock(batch_transfer_sync_read=read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(batcher.read, engine, "first", [0], [0], [512])
        try:
            assert first_read.wait(5)
            second = executor.submit(batcher.read, engine, "second", [0], [0], [256])
            assert second_queued.wait(5)
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5) == 0
    assert calls == ["first", "second", "first"]


@pytest.mark.parametrize("size", [-1, 1, 255, True, 256.0])
def test_reject_invalid_byte_limit(size):
    with pytest.raises(ValueError, match="kv_read_batch_bytes"):
        KVReadBatcher(size)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_reject_invalid_descriptor_limit(count):
    with pytest.raises(ValueError, match="kv_read_batch_descriptors"):
        KVReadBatcher(256, count)


def test_empty_and_zero_length_descriptors():
    batcher = KVReadBatcher(256, 2)
    engine = Mock()
    engine.batch_transfer_sync_read.return_value = 0
    assert batcher.read(engine, "peer", [], [], []) == 0
    engine.batch_transfer_sync_read.assert_not_called()
    assert batcher.read(engine, "peer", [1, 2, 3], [4, 5, 6], [0, 0, 0]) == 0
    assert engine.batch_transfer_sync_read.call_count == 2
    for args in [([0], [], [1]), ([0], [0], [-1])]:
        with pytest.raises(ValueError):
            batcher.read(engine, "peer", *args)
