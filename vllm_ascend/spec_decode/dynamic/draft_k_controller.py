# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime controller for reducing the physical draft width.

The dynamic confidence policy chooses a logical verify prefix after the draft
model has produced its candidates.  That is not enough to save draft-model
work: a batch can otherwise produce K tokens and then verify only a shorter
prefix.  This controller carries the previous step's logical width back to
the scheduler and caps the next physical K with a one-token safety slack.

The controller is intentionally CPU-only.  It is updated from the existing
``proposal_lengths`` side channel and never reads a device tensor on the
scheduling hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class AdaptiveDraftKController:
    """Hysteretic controller for the physical draft width.

    ``slack`` keeps one extra draft position available for confidence
    exploration.  If the previous logical prefix reaches the current width,
    the controller grows by one position; otherwise it shrinks only when the
    observed prefix plus slack is below the current width.  This prevents
    oscillation while allowing the physical width to track the hardware-aware
    verify policy.
    """

    max_k: int
    min_k: int = 1
    slack: int = 1

    def __post_init__(self) -> None:
        self.max_k = max(int(self.max_k), 0)
        self.min_k = min(max(int(self.min_k), 1), self.max_k) if self.max_k else 0
        self.slack = max(int(self.slack), 0)
        self._current_k: int | None = None

    @property
    def current_k(self) -> int | None:
        """Current physical-width recommendation, or ``None`` before warmup."""

        return self._current_k

    def cap(self, configured_k: int) -> int:
        """Cap a scheduler K without overriding an explicit batch K=0.

        A scheduler-side gate can return zero when the batch is under load. In
        that case the controller preserves its previous recommendation so
        speculation can resume when the gate opens again.
        """

        configured_k = max(min(int(configured_k), self.max_k), 0)
        if configured_k == 0:
            return 0
        if self._current_k is None:
            self._current_k = configured_k
        else:
            self._current_k = min(self._current_k, configured_k)
        return min(self._current_k, configured_k)

    def update(self, lengths: Iterable[int]) -> None:
        """Update the recommendation from the previous logical prefixes."""

        if self.max_k <= 0:
            return
        observed = max((max(int(length), 0) for length in lengths), default=0)
        observed = min(observed, self.max_k)
        if self._current_k is None:
            self._current_k = max(self.min_k, min(self.max_k, observed + self.slack))
            return

        # If the previous policy used the full physical width, allow one-step
        # growth.  Otherwise only shrink after the observed prefix is safely
        # below the current width.
        if observed >= self._current_k:
            self._current_k = min(self.max_k, self._current_k + 1)
            return

        target = max(self.min_k, min(self.max_k, observed + self.slack))
        if target < self._current_k:
            self._current_k = target
