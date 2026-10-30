# SPDX-License-Identifier: Apache-2.0
"""Host ownership tests. Controlled completions are not NPU evidence."""

import gc
import weakref

import pytest

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (
    DraftWindowOwnerPool,
    DraftWindowOwnerState,
)


class Completion:
    def __init__(self, *, complete=False, query_error=False, wait_error=False, false_wait=False):
        self.complete = complete
        self.query_error = query_error
        self.wait_error = wait_error
        self.false_wait = false_wait
        self.queries = 0
        self.waits = 0
        self.on_wait = None

    def is_complete(self):
        self.queries += 1
        if self.query_error:
            raise RuntimeError("event query failed")
        return self.complete

    def wait(self):
        self.waits += 1
        if self.on_wait is not None:
            self.on_wait()
        if self.wait_error:
            raise RuntimeError("event wait failed")
        self.complete = not self.false_wait


class Payload:
    pass


def submit(pool, completion, *payloads):
    owner = pool.acquire()
    pool.retain(owner, *payloads)
    pool.begin_native(owner)
    pool.submit(owner, lambda: completion)
    return owner


def test_pre_native_failure_releases_without_creating_an_event():
    pool = DraftWindowOwnerPool(1)

    def unused_event():
        pytest.fail("no native operation requires no event")

    with pytest.raises(ValueError, match="invalid input"), pool.operation(unused_event) as owner:
        pool.retain(owner, Payload())
        raise ValueError("invalid input")
    assert owner.state is DraftWindowOwnerState.RELEASED
    assert not owner.retained
    assert not pool.poisoned
    assert pool.acquire().lane == owner.lane


def test_no_native_operation_needs_no_event():
    pool = DraftWindowOwnerPool(1)
    with pool.operation(lambda: pytest.fail("unused event")) as owner:
        pool.retain(owner, Payload())
    assert owner.state is DraftWindowOwnerState.RELEASED
    assert not pool.poisoned


def test_partial_native_submission_then_host_error_retains_and_blocks_reuse():
    pool = DraftWindowOwnerPool(2)
    payload = Payload()
    reader_view = weakref.ref(payload)
    with (
        pytest.raises(RuntimeError, match="host failed after enqueue"),
        pool.operation(lambda: pytest.fail("success event must not be recorded")) as owner,
    ):
        pool.retain(owner, payload)
        pool.begin_native(owner)
        del payload
        # This models entry into a possibly submitted region, not an NPU
        # operation. Host failure does not supply a completion boundary.
        raise RuntimeError("host failed after enqueue")
    gc.collect()
    assert reader_view() is not None
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert pool.poisoned
    # Even a free lane must not resume a context whose completion is unknown.
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
    pool.abort(owner)
    assert reader_view() is not None


@pytest.mark.parametrize("failure", ["event construction", "event recording", "submit bookkeeping"])
def test_completion_creation_recording_and_bookkeeping_errors_quarantine(failure):
    pool = DraftWindowOwnerPool(1)
    payload = Payload()

    def failed_completion():
        raise RuntimeError(failure)

    with pytest.raises(RuntimeError, match=failure), pool.operation(failed_completion) as owner:
        pool.retain(owner, payload)
        pool.begin_native(owner)
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert owner.retained == [payload]
    assert pool.poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


def test_invalid_completion_cannot_release_already_submitted_work():
    pool = DraftWindowOwnerPool(1)
    invalid = object()
    with pytest.raises(TypeError, match="completion must implement"), pool.operation(lambda: invalid) as owner:
        pool.retain(owner, Payload())
        pool.begin_native(owner)
    assert owner.completion is invalid
    assert owner.retained
    assert pool.poisoned


def test_completion_protocol_validation_failure_retains_the_token():
    class InvalidCompletion:
        @property
        def is_complete(self):
            raise RuntimeError("bookkeeping property failed")

    pool = DraftWindowOwnerPool(1)
    invalid = InvalidCompletion()
    with pytest.raises(RuntimeError, match="bookkeeping property failed"), pool.operation(lambda: invalid) as owner:
        pool.begin_native(owner)
    assert owner.completion is invalid
    assert pool.poisoned


def test_free_lane_does_not_query_or_wait_for_unrelated_reader():
    pool = DraftWindowOwnerPool(2)
    completion = Completion(query_error=True)
    first = submit(pool, completion, Payload())
    second = pool.acquire()
    assert second.lane != first.lane
    assert completion.queries == completion.waits == 0


def test_pressure_reuses_reverse_completed_lane_without_wait():
    pool = DraftWindowOwnerPool(2)
    slow, fast = Completion(), Completion(complete=True)
    first = submit(pool, slow, Payload())
    second = submit(pool, fast, Payload())
    replacement = pool.acquire()
    assert replacement.lane == second.lane
    assert replacement.sequence > second.sequence
    assert first.retained
    assert not second.retained
    assert slow.waits == fast.waits == 0


def test_pressure_waits_for_oldest_and_retains_until_that_wait_completes():
    pool = DraftWindowOwnerPool(2)
    slow, later = Completion(), Completion()
    first = submit(pool, slow, Payload())
    second = submit(pool, later, Payload())
    before_wait = []
    slow.on_wait = lambda: before_wait.append(bool(first.retained))
    replacement = pool.acquire()
    assert before_wait == [True]
    assert slow.waits == 1
    assert later.waits == 0
    assert replacement.lane == first.lane
    assert second.retained


@pytest.mark.parametrize("failure", ["query", "wait", "false_wait"])
def test_completion_failure_never_releases_the_backing(failure):
    pool = DraftWindowOwnerPool(1)
    completion = Completion(
        query_error=failure == "query", wait_error=failure == "wait", false_wait=failure == "false_wait"
    )
    payload = Payload()
    owner = submit(pool, completion, payload)
    with pytest.raises(RuntimeError):
        pool.acquire()
    assert owner.retained == [payload]
    assert owner.completion is completion
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert pool.poisoned
    completion.query_error = completion.wait_error = False
    completion.complete = True
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


def test_all_eagle_steps_and_nested_derived_state_live_until_proposal_completion():
    pool = DraftWindowOwnerPool(1)
    completion = Completion()
    read_views, derived_views = [], []
    with pool.operation(lambda: completion) as owner:
        for _step in range(4):
            reads, derived = Payload(), Payload()
            read_views.append(weakref.ref(reads))
            derived_views.append(weakref.ref(derived))
            pool.retain(owner, reads, {"builder": {"local_and_index": derived}})
            pool.begin_native(owner)
            del reads, derived
    assert len(owner.retained) == 8
    gc.collect()
    assert all(ref() is not None for ref in read_views + derived_views)
    assert pool.reclaim_completed() == 0
    assert all(ref() is not None for ref in read_views + derived_views)
    completion.complete = True
    assert pool.reclaim_completed() == 1
    gc.collect()
    assert all(ref() is None for ref in read_views + derived_views)


def test_abort_after_record_keeps_real_completion_authoritative():
    pool = DraftWindowOwnerPool(1)
    completion = Completion()
    owner = submit(pool, completion, Payload())
    pool.abort(owner)
    assert owner.retained
    assert owner.state is DraftWindowOwnerState.SUBMITTED
    assert not pool.poisoned
    assert pool.reclaim_completed() == 0
    completion.complete = True
    assert pool.reclaim_completed() == 1
    assert owner.state is DraftWindowOwnerState.RELEASED


def test_late_reader_requires_all_reader_completion_not_primary_only():
    class AllReaders:
        def __init__(self, readers):
            self.readers = readers

        def is_complete(self):
            return all(reader.is_complete() for reader in self.readers)

        def wait(self):
            for reader in self.readers:
                reader.wait()

    primary, communication = Completion(complete=True), Completion()
    pool = DraftWindowOwnerPool(1)
    owner = submit(pool, AllReaders([primary, communication]), Payload())
    assert pool.reclaim_completed() == 0
    assert owner.retained
    communication.complete = True
    assert pool.reclaim_completed() == 1


def test_stale_or_cross_pool_owner_cannot_mutate_another_generation():
    pool = DraftWindowOwnerPool(1)
    other = DraftWindowOwnerPool(1)
    owner = pool.acquire()
    with pytest.raises(RuntimeError, match="stale or belongs"):
        other.begin_native(owner)
    pool.abort(owner)
    replacement = pool.acquire()
    assert replacement.lane == owner.lane
    with pytest.raises(RuntimeError, match="stale or belongs"):
        pool.retain(owner, Payload())
    assert not replacement.retained


def test_unsubmitted_pool_pressure_never_waits_or_discards():
    pool = DraftWindowOwnerPool(1)
    owner = pool.acquire()
    pool.retain(owner, Payload())
    with pytest.raises(RuntimeError, match="unsubmitted proposal"):
        pool.acquire()
    assert owner.retained
    assert owner.state is DraftWindowOwnerState.PREPARING


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        DraftWindowOwnerPool(capacity)
