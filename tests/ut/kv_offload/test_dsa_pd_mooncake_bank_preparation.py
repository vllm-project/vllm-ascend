# SPDX-License-Identifier: Apache-2.0
"""Bank ownership, delayed landing, and failure cleanup before adding workers."""

import ctypes
import gc
import threading
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa import transfer_targets as registry_module

from .test_dsa_pd_mooncake_connector_v2 import make_reader

registry = registry_module.MooncakeKVRegistry()

ALIGNMENT = 2 * 1024 * 1024


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    registry.clear()
    monkeypatch.setattr(torch.npu, "synchronize", MagicMock())
    yield
    registry.clear()


class DelayedLanding:
    """Execute byte copies only on event waits, exposing premature slot reuse."""

    def __init__(self, monkeypatch, reader):
        self.reader = reader
        self.original_copy = reader.host_kv.manager.copy_sfa_kv
        self.current = None
        self.pending: dict[int, list[Callable[[], None]]] = {id(stream): [] for stream in reader._landing_streams}
        self.events: list[Any] = []
        self.copy_banks: list[Any] = []
        self.caller = threading.get_ident()
        runtime = self

        class Event:
            def record(self, stream):
                self.stream = stream
                self.tasks = tuple(runtime.pending[id(stream)])
                runtime.events.append(self)

            def synchronize(self):
                assert reader._dsa_pd_mooncake_staging_lock.locked()
                for task in self.tasks:
                    if task in runtime.pending[id(self.stream)]:
                        runtime.pending[id(self.stream)].remove(task)
                        task()
                self.tasks = ()

        @contextmanager
        def stream_context(stream):
            assert self.current is None
            self.current = stream
            try:
                yield
            finally:
                self.current = None

        def submit(*args):
            assert threading.get_ident() == self.caller
            if self.current is None:
                assert reader._get_dsa_pd_mooncake_staging_window() < 2
                self.original_copy(*args)
                return
            bank = next(bank for bank in reader._dsa_pd_mooncake_staging_banks if bank.stream is self.current)
            self.copy_banks.append(bank)
            self.pending[id(bank.stream)].append(lambda: self.original_copy(*args))

        for stream in reader._landing_streams:

            def synchronize(stream=stream):
                for task in tuple(self.pending[id(stream)]):
                    self.pending[id(stream)].remove(task)
                    task()

            stream.synchronize.side_effect = synchronize
        monkeypatch.setattr(torch.npu, "stream", stream_context)
        monkeypatch.setattr(torch.npu, "Event", Event)
        reader.host_kv.manager.copy_sfa_kv = submit


@pytest.mark.parametrize("banks,window", [(1, 1), (1, 4), (2, 4), (4, 4), (4, 5), (8, 4)])
@pytest.mark.parametrize("page_offset", [0, 17, ALIGNMENT - 1])
@pytest.mark.parametrize("metadata", [False, True])
def test_serial_bank_batches_preserve_bytes_with_delayed_events(monkeypatch, banks, window, page_offset, metadata):
    length = (2 * window + 1) * ALIGNMENT + 31
    storage = torch.empty(length + 2 * ALIGNMENT, dtype=torch.uint8)
    offset = (-storage.data_ptr()) % ALIGNMENT + page_offset
    source = storage[offset : offset + length]
    source.random_(0, 255)
    host_storage = torch.full((length + 26,), 251, dtype=torch.uint8)
    host = host_storage[13:-13]
    direct = torch.zeros(8, dtype=torch.uint8)
    registry.register_offloaded_tensor(host)
    reader = make_reader(MagicMock(), pipeline=True, window=window, kv_registry=registry)
    reader._pipeline_banks = banks
    reader._landing_streams = [MagicMock() for _ in range(banks)]
    reader._landing_stream = reader._landing_streams[0]
    runtime = DelayedLanding(monkeypatch, reader)
    caller = threading.get_ident()
    reads = []

    def read(_session, local, remote, lengths):
        assert threading.get_ident() == caller and reader._dsa_pd_mooncake_staging_lock.locked()
        reads.append(tuple(zip(local, remote, lengths)))
        for dst, src, size in reads[-1]:
            ctypes.memmove(dst, src, size)
        return 0

    reader.engine.batch_transfer_sync_read.side_effect = read
    kwargs = {}
    if metadata:
        table = registry.create_transfer_targets([[host], [direct]])
        kwargs["transfer_targets"] = [table[0][0], table[1][0]]
    assert (
        reader.batch_transfer_sync_read(
            "peer",
            [host.data_ptr(), direct.data_ptr()],
            [source.data_ptr(), source.data_ptr()],
            [length, 8],
            "request",
            **kwargs,
        )
        == 0
    )
    assert torch.equal(host, source) and torch.equal(direct, source[:8])
    assert torch.all(host_storage[:13] == 251) and torch.all(host_storage[-13:] == 251)
    assert reads[0] == ((direct.data_ptr(), source.data_ptr(), 8),)
    pieces = [entry for batch in reads[1:] for entry in batch]
    assert [src for _, src, _ in pieces] == [
        source.data_ptr() + sum(p[2] for p in pieces[:i]) for i in range(len(pieces))
    ]
    assert sum(p[2] for p in pieces) == length
    contexts = reader._dsa_pd_mooncake_staging_banks
    expected_banks = min(banks, window) if window >= 2 else 1
    assert len(contexts) == expected_banks
    for index, batch in enumerate(reads[1:]):
        bank = contexts[index % expected_banks]
        for position, (dst, src, size) in enumerate(batch):
            slot_start = (bank.slot_base + position) * ALIGNMENT
            assert dst == reader._dsa_pd_mooncake_staging_tensor.data_ptr() + slot_start + src % ALIGNMENT
            assert dst + size <= reader._dsa_pd_mooncake_staging_tensor.data_ptr() + slot_start + ALIGNMENT
    assert all(not bank.completion_pending for bank in contexts)
    assert all(not bank.inflight_owners for bank in contexts)
    assert all(not pending for pending in runtime.pending.values())
    assert not reader._dsa_pd_mooncake_staging_lock.locked()


@pytest.mark.parametrize("banks", [1, 2, 4, 3, 257])
def test_bank_contexts_reuse_streams_and_events(banks):
    reader = make_reader(MagicMock(), kv_registry=registry)
    reader._landing_streams = [MagicMock() for _ in range(banks)]
    contexts = reader._get_dsa_pd_mooncake_staging_banks(banks, 1, True)
    assert [bank.slot_base for bank in contexts] == list(range(banks))
    assert all(bank.completion_event is None and not bank.completion_pending for bank in contexts)
    assert [bank.stream for bank in contexts] == reader._landing_streams
    assert reader._get_dsa_pd_mooncake_staging_banks(banks, 1, True) is contexts


def test_layout_change_drains_previous_bank_events_before_replacement():
    reader = make_reader(MagicMock(), kv_registry=registry)
    previous = reader._get_dsa_pd_mooncake_staging_banks(2, 2, True)
    events = [MagicMock(), MagicMock()]
    for bank, event in zip(previous, events):
        bank.completion_event = event
        bank.completion_pending = True
    new = reader._get_dsa_pd_mooncake_staging_banks(1, 4, True)
    assert len(new) == 1 and new[0] not in previous
    for event in events:
        event.synchronize.assert_called_once()
    assert all(not bank.completion_pending for bank in previous)


@pytest.mark.parametrize("overlap", [False, True])
def test_plan_rejects_overlaps_and_preserves_disjoint_batch_order(overlap):
    reader = make_reader(MagicMock(), pipeline=True, kv_registry=registry)
    banks = reader._get_dsa_pd_mooncake_staging_banks(2, 1, True)
    host = torch.empty(64, dtype=torch.int8)
    destinations = [host.data_ptr(), host.data_ptr() + (8 if overlap else 16), host.data_ptr() + 32]
    transfers = [(dst, (index + 10) * ALIGNMENT, 16) for index, dst in enumerate(destinations)]
    targets = {dst: (host, dst - host.data_ptr()) for dst in destinations}
    if overlap:
        with pytest.raises(ValueError, match="Overlapping Mooncake Host destination"):
            reader._prepare_dsa_pd_mooncake_bank_plan(
                transfers, targets, reader._dsa_pd_mooncake_staging_tensor, ALIGNMENT, banks
            )
        return
    plan = reader._prepare_dsa_pd_mooncake_bank_plan(
        transfers, targets, reader._dsa_pd_mooncake_staging_tensor, ALIGNMENT, banks
    )
    assert [batch.bank_index for batch in plan.batches] == [0, 1, 0]
    assert [copy.destination for batch in plan.batches for copy in batch.copies] == destinations


def test_all_coalesced_target_bounds_are_checked_before_first_staged_read():
    host = torch.empty(32, dtype=torch.int8)
    reader = make_reader(MagicMock(), pipeline=True, kv_registry=registry)
    banks = reader._get_dsa_pd_mooncake_staging_banks(2, 1, True)
    with pytest.raises(RuntimeError, match="boundary"):
        reader._prepare_dsa_pd_mooncake_bank_plan(
            [(1000, 10 * ALIGNMENT, 8), (2000, 11 * ALIGNMENT, 16)],
            {1000: (host, 0), 2000: (host, 24)},
            reader._dsa_pd_mooncake_staging_tensor,
            ALIGNMENT,
            banks,
        )
    reader.engine.batch_transfer_sync_read.assert_not_called()


def test_failed_event_wait_does_not_skip_other_banks_and_is_retried():
    reader = make_reader(MagicMock(), kv_registry=registry)
    banks = reader._get_dsa_pd_mooncake_staging_banks(2, 2, True)
    events = [MagicMock() for _ in range(2)]
    events[0].synchronize.side_effect = [RuntimeError("wait failed"), None]
    banks[0].stream.synchronize.side_effect = RuntimeError("stream failed")
    for bank, event in zip(banks, events):
        bank.completion_event = event
        bank.completion_pending = True
    with pytest.raises(RuntimeError, match="wait failed"):
        reader._drain_dsa_pd_mooncake_staging_banks(banks)
    for event in events:
        event.synchronize.assert_called_once()
    assert banks[0].completion_pending and banks[0].completion_event is events[0]
    assert not banks[1].completion_pending
    # Retry the failed bank before another request can READ into its staging.
    assert reader._get_dsa_pd_mooncake_staging_banks(2, 2, True) is banks
    assert events[0].synchronize.call_count == 2
    assert not banks[0].completion_pending


def test_failed_cleanup_retains_owners_until_stream_drain_can_finish():
    reader = make_reader(MagicMock(), kv_registry=registry)
    bank = reader._get_dsa_pd_mooncake_staging_banks(1, 2, True)[0]
    target = torch.empty(16, dtype=torch.int8)
    target_ref = weakref.ref(target)
    bank.inflight_owners.append((target,))
    bank.needs_stream_drain = True
    bank.stream.synchronize.side_effect = [RuntimeError("stream failed"), None]
    del target
    with pytest.raises(RuntimeError, match="stream failed"):
        reader._drain_dsa_pd_mooncake_staging_banks((bank,))
    gc.collect()
    assert target_ref() is not None
    reader._drain_dsa_pd_mooncake_staging_banks((bank,))
    gc.collect()
    assert target_ref() is None and not bank.inflight_owners and not bank.needs_stream_drain


def test_failed_event_is_discarded_only_after_successful_stream_drain():
    reader = make_reader(MagicMock(), kv_registry=registry)
    bank = reader._get_dsa_pd_mooncake_staging_banks(1, 2, True)[0]
    event = MagicMock()
    event.synchronize.side_effect = RuntimeError("event failed")
    bank.completion_event = event
    bank.completion_pending = True
    target = torch.empty(16, dtype=torch.int8)
    target_ref = weakref.ref(target)
    bank.inflight_owners.append((target,))
    del target
    with pytest.raises(RuntimeError, match="event failed"):
        reader._drain_dsa_pd_mooncake_staging_banks((bank,))
    bank.stream.synchronize.assert_called_once()
    gc.collect()
    assert bank.completion_event is None and not bank.completion_pending
    assert not bank.inflight_owners and target_ref() is None


def test_staging_growth_cannot_drop_an_allocation_that_failed_to_drain(monkeypatch):
    reader = make_reader(MagicMock(), kv_registry=registry)
    previous = reader._dsa_pd_mooncake_staging_tensor
    bank = reader._get_dsa_pd_mooncake_staging_banks(1, 2, True)[0]
    bank.needs_stream_drain = True
    bank.inflight_owners.append((previous,))
    bank.stream.synchronize.side_effect = RuntimeError("stream failed")
    reader.block_len_per_addr = [[2 * ALIGNMENT]]
    allocate = MagicMock(side_effect=AssertionError("allocated before drain"))
    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(RuntimeError, match="stream failed"):
        reader._ensure_dsa_pd_mooncake_staging_tensor()
    assert reader._dsa_pd_mooncake_staging_tensor is previous
    allocate.assert_not_called()


@pytest.mark.parametrize("failure", ["negative", "read_exception", "copy_exception", "event_record"])
def test_bank_failure_drains_submitted_bytes_before_unlock(monkeypatch, failure):
    source = torch.full((8 * ALIGNMENT,), 19, dtype=torch.uint8)
    host = torch.zeros_like(source)
    registry.register_offloaded_tensor(host)
    reader = make_reader(MagicMock(), pipeline=True, window=4, kv_registry=registry)
    runtime = DelayedLanding(monkeypatch, reader)
    reads = 0

    def read(_session, local, remote, lengths):
        nonlocal reads
        reads += 1
        if reads == 2 and failure in ("negative", "read_exception"):
            if failure == "read_exception":
                raise RuntimeError("read failed")
            return -7
        for dst, src, size in zip(local, remote, lengths):
            ctypes.memmove(dst, src, size)
        return 0

    reader.engine.batch_transfer_sync_read.side_effect = read
    if failure == "copy_exception":
        original = reader.host_kv.manager.copy_sfa_kv

        def copy(*args, **kwargs):
            original(*args, **kwargs)
            if reads == 2:
                raise RuntimeError("copy failed")

        reader.host_kv.manager.copy_sfa_kv = copy
    elif failure == "event_record":
        event_type = torch.npu.Event

        def make_broken_event():
            event = event_type()
            original_record = event.record

            def record(stream):
                if reads == 2:
                    raise RuntimeError("event failed")
                original_record(stream)

            event.record = record
            return event

        monkeypatch.setattr(torch.npu, "Event", make_broken_event)
    args = ("peer", [host.data_ptr()], [source.data_ptr()], [host.numel()], "request")
    if failure == "negative":
        assert reader.batch_transfer_sync_read(*args) == -7
    else:
        with pytest.raises(RuntimeError, match="failed"):
            reader.batch_transfer_sync_read(*args)
    assert reads == 2
    assert all(not pending for pending in runtime.pending.values())
    assert all(not bank.completion_pending for bank in reader._dsa_pd_mooncake_staging_banks)
    assert all(not bank.inflight_owners for bank in reader._dsa_pd_mooncake_staging_banks)
    assert not reader._dsa_pd_mooncake_staging_lock.locked()
    assert torch.count_nonzero(host) > 0
