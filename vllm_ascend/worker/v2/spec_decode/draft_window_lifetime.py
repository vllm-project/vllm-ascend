# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep eager Draft read state alive until its last device reader finishes.

The caller owns tensor allocation and stream ordering. This module only owns
strong references: retaining a view does not prevent another owner from writing
its backing. Each proposal must therefore pass independent read/derived state.
A completion token must cover *all* readers, including communication streams.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol


class DraftWindowCompletion(Protocol):
    def is_complete(self) -> bool: ...

    def wait(self) -> None: ...


class DraftWindowOwnerState(Enum):
    PREPARING = auto()
    ACTIVE = auto()
    SUBMITTED = auto()
    QUARANTINED = auto()
    RELEASED = auto()


@dataclass(eq=False)
class DraftWindowOwner:
    """One proposal, retaining every AR step through a common completion."""

    sequence: int
    lane: int
    state: DraftWindowOwnerState = DraftWindowOwnerState.PREPARING
    retained: list[object] = field(default_factory=list, repr=False)
    completion: DraftWindowCompletion | None = field(default=None, repr=False)


class DraftWindowOwnerPool:
    """Bounded owner pool used from the runner's single host execution thread.

    A free lane is acquired without polling. Only capacity pressure polls and,
    if necessary, waits for the oldest submitted proposal. An error after any
    possible native submission poisons the pool. Its owners remain strongly
    held and cannot be reused; reconstructing this pool is not a recovery
    strategy without first proving device completion outside this owner.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Draft Window owner capacity must be a positive integer")
        self._owners: list[DraftWindowOwner | None] = [None] * capacity
        self._next_sequence = 0
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def acquire(self) -> DraftWindowOwner:
        if self._poisoned:
            raise RuntimeError("Draft Window execution is quarantined: device completion is unknown")
        if None not in self._owners:
            self.reclaim_completed()
        if None not in self._owners:
            submitted = [owner for owner in self._owners if owner is not None]
            if any(owner.state is not DraftWindowOwnerState.SUBMITTED for owner in submitted):
                raise RuntimeError("Draft Window pool is full with an unsubmitted proposal")
            oldest = min(submitted, key=lambda owner: owner.sequence)
            try:
                assert oldest.completion is not None
                oldest.completion.wait()
                if not oldest.completion.is_complete():
                    raise RuntimeError("Draft Window completion wait did not complete the reader")
            except BaseException:
                self._quarantine(oldest)
                raise
            self._release(oldest)
        lane = self._owners.index(None)
        owner = DraftWindowOwner(sequence=self._next_sequence, lane=lane)
        self._owners[lane] = owner
        self._next_sequence += 1
        return owner

    def retain(self, owner: DraftWindowOwner, *objects: object) -> None:
        """Retain read tensors and builder results before handing them to readers."""
        self._require(owner, DraftWindowOwnerState.PREPARING, DraftWindowOwnerState.ACTIVE)
        owner.retained.extend(objects)

    def begin_native(self, owner: DraftWindowOwner) -> None:
        """Enter before the first possibly asynchronous operation, even a copy.

        This is intentionally earlier than model forward: table materialization
        and backend metadata builders can also enqueue work. Repeated calls
        allow a proposal to retain independently allocated state at each step.
        """
        self._require(owner, DraftWindowOwnerState.PREPARING, DraftWindowOwnerState.ACTIVE)
        owner.state = DraftWindowOwnerState.ACTIVE

    def submit(self, owner: DraftWindowOwner, record_completion: Callable[[], DraftWindowCompletion]) -> None:
        """Record the final consumer boundary, retaining ownership on every error.

        Event creation and recording must both happen inside the callback. A
        successful record alone does not release anything; query/wait supplies
        the later completion proof. The callback must join any late readers
        before recording, or return a token covering each reader separately.
        """
        self._require(owner, DraftWindowOwnerState.ACTIVE)
        try:
            completion = record_completion()
            # Retain even an invalid result: it might carry an event already
            # recorded before a bookkeeping/validation failure.
            owner.completion = completion
            if not callable(getattr(completion, "is_complete", None)) or not callable(
                getattr(completion, "wait", None)
            ):
                raise TypeError("Draft Window completion must implement is_complete and wait")
            owner.state = DraftWindowOwnerState.SUBMITTED
        except BaseException:
            self._quarantine(owner)
            raise

    def abort(self, owner: DraftWindowOwner) -> None:
        """Only work known never to have reached native submission is releasable."""
        self._require_active(owner)
        if owner.state is DraftWindowOwnerState.PREPARING:
            self._release(owner)
        elif owner.state is DraftWindowOwnerState.SUBMITTED:
            # The existing completion still owns the reader, even if the host
            # subsequently fails. Pressure/reclaim can retire it normally.
            return
        else:
            self._quarantine(owner)

    def reclaim_completed(self) -> int:
        """Poll submitted readers; never infer completion from host progress."""
        if self._poisoned:
            raise RuntimeError("Draft Window execution is quarantined: device completion is unknown")
        released = 0
        for owner in tuple(self._owners):
            if owner is None or owner.state is not DraftWindowOwnerState.SUBMITTED:
                continue
            try:
                assert owner.completion is not None
                complete = owner.completion.is_complete()
            except BaseException:
                self._quarantine(owner)
                raise
            if complete:
                self._release(owner)
                released += 1
        return released

    @contextmanager
    def operation(self, record_completion: Callable[[], DraftWindowCompletion]) -> Iterator[DraftWindowOwner]:
        """Own one proposal across preparation, all steps and event submission."""
        owner = self.acquire()
        try:
            yield owner
            if owner.state is DraftWindowOwnerState.PREPARING:
                self._release(owner)
            else:
                self.submit(owner, record_completion)
        except BaseException:
            if owner.state is not DraftWindowOwnerState.RELEASED:
                self.abort(owner)
            raise

    def _quarantine(self, owner: DraftWindowOwner) -> None:
        # Do not clear retained objects or completion, even if event recording
        # failed. No token exists that could prove reuse is safe in that case.
        owner.state = DraftWindowOwnerState.QUARANTINED
        self._poisoned = True

    def _release(self, owner: DraftWindowOwner) -> None:
        owner.retained.clear()
        owner.completion = None
        owner.state = DraftWindowOwnerState.RELEASED
        self._owners[owner.lane] = None

    def _require_active(self, owner: DraftWindowOwner) -> None:
        if not 0 <= owner.lane < len(self._owners) or self._owners[owner.lane] is not owner:
            raise RuntimeError("Draft Window owner is stale or belongs to another pool")

    def _require(self, owner: DraftWindowOwner, *states: DraftWindowOwnerState) -> None:
        self._require_active(owner)
        if owner.state not in states:
            raise RuntimeError(f"Draft Window owner cannot be used in state {owner.state.name}")
