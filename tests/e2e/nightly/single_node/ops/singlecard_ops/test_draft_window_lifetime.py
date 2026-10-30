# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ownership failures after a real native enqueue, without engine mocking.

These tests prove the owner protocol at a device boundary. They do not prove
that an engine/backend joins all of its secondary-stream readers.
"""

from contextlib import contextmanager
from typing import Any

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (  # noqa: E402
    DraftWindowOwnerPool,
    DraftWindowOwnerState,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU"
)


class NativeCompletion:
    def __init__(self, stream):
        self.event = torch.npu.Event()
        self.event.record(stream)

    def is_complete(self):
        return self.event.query()

    def wait(self):
        self.event.synchronize()


@pytest.mark.parametrize("operation", ["plan", "gather", "pack"])
def test_private_kernel_outputs_survive_exception_after_real_enqueue(monkeypatch, operation):
    from vllm_ascend.ops.triton.v2.spec_decode import draft_window

    pool = DraftWindowOwnerPool(1)
    stream = torch.npu.Stream()
    held_outputs: list[torch.Tensor] = []
    diagnostic_events: list[Any] = []
    kernel_name = {
        "plan": "_plan_draft_window",
        "gather": "_gather_draft_kv_fp32",
        "pack": "_pack_draft_query_indices",
    }[operation]
    original = getattr(draft_window, kernel_name)

    class RaiseAfterEnqueue:
        def __getitem__(self, grid):
            launcher = original[grid]

            def launch(*args):
                outputs = args[7:11] if operation == "plan" else args[3:5] if operation == "gather" else args[1:3]
                assert all(any(value is output for value in owner.retained) for output in outputs)
                held_outputs.extend(outputs)
                launcher(*args)
                # This real event is a diagnostic cleanup token only. The
                # production operation deliberately fails without receiving
                # a token, so its owner must remain quarantined.
                event = torch.npu.Event()
                event.record(stream)
                diagnostic_events.append(event)
                raise RuntimeError("injected host exception after real kernel enqueue")

            return launch

    monkeypatch.setattr(draft_window, kernel_name, RaiseAfterEnqueue())
    with (
        pytest.raises(RuntimeError, match="after real kernel enqueue"),
        torch.npu.stream(stream),
        pool.operation(lambda: NativeCompletion(stream)) as owner,
    ):
        pool.begin_native(owner)
        retain = lambda *objects: pool.retain(owner, *objects)
        if operation == "plan":
            draft_window.plan_draft_window(
                torch.arange(8, dtype=torch.int32, device="npu").reshape(1, 8),
                torch.tensor([0], dtype=torch.int64, device="npu"),
                torch.tensor([17], dtype=torch.int32, device="npu"),
                None,
                block_span=4,
                anchor_span=4,
                window=8,
                capacity=12,
                query_width=3,
                causal=True,
                retain=retain,
            )
        elif operation == "gather":
            draft_window.gather_draft_kv_fp32(
                torch.tensor([[1, 2, 3]], dtype=torch.int64, device="npu"),
                torch.arange(64, dtype=torch.bfloat16, device="npu").reshape(2, 4, 1, 8),
                torch.ones((2, 4, 1, 8), dtype=torch.bfloat16, device="npu"),
                retain=retain,
            )
        else:
            draft_window.pack_draft_query_indices(
                torch.tensor([0, 2, 5], dtype=torch.int32, device="npu"), 5, 3, retain=retain
            )
    assert diagnostic_events and held_outputs
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert all(any(value is output for value in owner.retained) for output in held_outputs)
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
    diagnostic_events[0].synchronize()
    if operation == "plan":
        assert held_outputs[1].cpu().tolist() == [8]
        assert held_outputs[2].cpu().tolist() == [9]
    elif operation == "gather":
        torch.testing.assert_close(held_outputs[0].cpu().flatten(), torch.arange(8, 32, dtype=torch.float32))
        assert held_outputs[1].eq(1).all()
    else:
        assert held_outputs[0].cpu().tolist() == [[0, 0, 1], [2, 3, 4]]
        assert held_outputs[1].cpu().tolist() == [1, 2, 3, 4, 5]
    # Even an external diagnostic wait does not make production's unknown
    # completion contract safe to reuse.
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


@pytest.mark.parametrize("failure", ["host_after_enqueue", "event_create", "event_record", "submit_result"])
def test_native_enqueue_failure_quarantines_backing(failure):
    pool = DraftWindowOwnerPool(1)
    stream = torch.npu.Stream()
    payload = torch.arange(1024, dtype=torch.float32, device="npu")
    stream.wait_stream(torch.npu.current_stream())

    def completion():
        if failure == "event_create":
            raise RuntimeError("injected event creation failure")
        event = torch.npu.Event()
        if failure == "event_record":
            raise RuntimeError("injected event recording failure")
        event.record(stream)
        if failure == "submit_result":
            # Real work and a real recorded event exist, but the owner did
            # not receive a valid completion token.
            return event
        return NativeCompletion(stream)

    with pytest.raises((RuntimeError, TypeError)), torch.npu.stream(stream), pool.operation(completion) as owner:
        pool.retain(owner, payload)
        pool.begin_native(owner)
        observed = payload.square()
        pool.retain(owner, observed)
        if failure == "host_after_enqueue":
            raise RuntimeError("injected host failure after native enqueue")
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert owner.retained[0] is payload
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
    # Only test cleanup waits for the real stream. Production failure does
    # not infer completion or release a quarantined context.
    stream.synchronize()
    torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32).square())
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


def test_native_completion_covers_an_explicit_late_reader_join():
    pool = DraftWindowOwnerPool(1)
    producer, reader = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(producer), pool.operation(lambda: NativeCompletion(producer)) as owner:
        pool.begin_native(owner)
        payload = torch.arange(1024, dtype=torch.float32, device="npu")
        pool.retain(owner, payload)
        reader.wait_stream(producer)
        with torch.npu.stream(reader):
            result = payload * 3
        pool.retain(owner, result)
        # The actual executor must provide an equivalent join for every
        # secondary reader before recording its proposal completion.
        producer.wait_stream(reader)
    replacement = pool.acquire()
    assert replacement.lane == owner.lane
    assert owner.state is DraftWindowOwnerState.RELEASED
    torch.testing.assert_close(result.cpu(), torch.arange(1024, dtype=torch.float32) * 3)


class ObservedNativeCompletion(NativeCompletion):
    """Record native observations without substituting a completion result."""

    def __init__(self, stream):
        super().__init__(stream)
        self.queries = []
        self.waits = 0
        self.failure = None

    def is_complete(self):
        complete = self.event.query()
        self.queries.append(complete)
        if self.failure is not None and complete:
            pytest.skip("reader completed before native pending-completion fault boundary")
        if self.failure == "query":
            raise RuntimeError("injected host error after native pending query")
        return complete

    def wait(self):
        self.waits += 1
        if self.failure == "wait":
            if self.event.query():
                pytest.skip("reader completed before native pending-wait fault boundary")
            raise RuntimeError("injected host error before native completion wait")
        self.event.synchronize()


@contextmanager
def pending_native_reader(pool):
    """Queue bounded device work, then a real reader of the retained payload.

    Matrix multiplication supplies work rather than a host sleep or an event
    whose value was manufactured by a mock. On a device that drains the queue
    before observation this fixture is inconclusive. Keep every allocation
    alive and synchronize only this test's stream during cleanup.
    """
    completion: ObservedNativeCompletion
    matrix_size = 4096
    delay_operations = 128
    stream = torch.npu.Stream()
    try:
        with torch.npu.stream(stream), pool.operation(lambda: completion) as owner:
            pool.begin_native(owner)
            payload = torch.arange(1024, dtype=torch.float32, device="npu")
            observed = torch.empty_like(payload)
            left = torch.full((matrix_size, matrix_size), 1 / matrix_size, dtype=torch.bfloat16, device="npu")
            right = torch.eye(matrix_size, dtype=torch.bfloat16, device="npu")
            scratch = torch.empty_like(left)
            pool.retain(owner, payload, observed, left, right, scratch)
            for _ in range(delay_operations):
                torch.mm(left, right, out=scratch)
                left, scratch = scratch, left
            observed.copy_(payload)
            completion = ObservedNativeCompletion(stream)
        if completion.event.query():
            pytest.skip("bounded native queue completed before an in-flight reader was observed")
        yield owner, completion, payload, observed
    finally:
        stream.synchronize()


def test_native_pool_pressure_waits_before_overwriting_read_backing():
    pool = DraftWindowOwnerPool(1)
    with pending_native_reader(pool) as (owner, completion, payload, observed):
        replacement = pool.acquire()
        if completion.waits == 0:
            pool.abort(replacement)
            pytest.skip("reader completed before pool pressure reached the native wait")
        assert completion.queries[0] is False
        assert completion.queries[-1] is True
        assert completion.waits == 1
        assert owner.state is DraftWindowOwnerState.RELEASED
        assert replacement.lane == owner.lane
        # The replacement may now overwrite the exact backing read by the
        # delayed native copy. That copy must have consumed the old values.
        payload.fill_(-1)
        torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32))
        pool.abort(replacement)


def test_native_reverse_completion_reuses_only_the_completed_lane():
    pool = DraftWindowOwnerPool(2)
    with pending_native_reader(pool) as (slow_owner, slow, payload, observed):
        fast_stream = torch.npu.Stream()
        try:
            with torch.npu.stream(fast_stream), pool.operation(lambda: NativeCompletion(fast_stream)) as fast_owner:
                pool.begin_native(fast_owner)
                fast_payload = torch.ones(1024, device="npu")
                fast_result = fast_payload + 1
                pool.retain(fast_owner, fast_payload, fast_result)
            fast_owner.completion.wait()
            if slow.event.query():
                pytest.skip("device did not exhibit reverse completion of the two native readers")
            replacement = pool.acquire()
            if slow_owner.state is DraftWindowOwnerState.RELEASED:
                pool.abort(replacement)
                pytest.skip("slow reader completed before pool reclamation observed it")
            assert slow.queries[-1] is False
            assert slow.waits == 0
            assert slow_owner.state is DraftWindowOwnerState.SUBMITTED
            assert any(value is payload for value in slow_owner.retained)
            assert replacement.lane == fast_owner.lane
            assert fast_owner.state is DraftWindowOwnerState.RELEASED
            pool.abort(replacement)
            slow.wait()
            torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32))
        finally:
            fast_stream.synchronize()


@pytest.mark.parametrize("failure", ["query", "wait"])
def test_native_pending_completion_fault_quarantines_the_execution(failure):
    pool = DraftWindowOwnerPool(1)
    with pending_native_reader(pool) as (owner, completion, payload, observed):
        completion.failure = failure
        with pytest.raises(RuntimeError, match="injected host error"):
            pool.acquire()
        assert completion.queries[0] is False
        assert completion.waits == (1 if failure == "wait" else 0)
        assert owner.state is DraftWindowOwnerState.QUARANTINED
        assert any(value is payload for value in owner.retained)
        assert owner.completion is completion
        with pytest.raises(RuntimeError, match="quarantined"):
            pool.acquire()
        # A later successful native completion cannot silently unpoison an
        # executor after its bookkeeping failed. No replacement can write.
        completion.event.synchronize()
        torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32))
        with pytest.raises(RuntimeError, match="quarantined"):
            pool.acquire()
