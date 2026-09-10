# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only physical-width control and optional batch-level proposal gating.

The upstream adaptive-verification manager owns the logical per-request
verification prefix. This controller is deliberately independent: it observes
the drafts actually accepted by the target and caps the next physical draft K
with a safety slack. It never rewrites confidence probabilities or the
upstream verification budget.

The controller is intentionally CPU-only. It consumes the model runner's
already-copied output and never reads a device tensor on the scheduling hot
path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


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
    percentile: float = 0.5
    hybrid_enabled: bool = False
    hybrid_min_batch_size: int = 8
    hybrid_acceptance_threshold: float = 0.6
    hybrid_low_steps: int = 4
    hybrid_high_steps: int = 2
    hybrid_probe_interval: int = 32

    def __post_init__(self) -> None:
        self.max_k = max(int(self.max_k), 0)
        self.min_k = min(max(int(self.min_k), 1), self.max_k) if self.max_k else 0
        self.slack = max(int(self.slack), 0)
        self.percentile = min(max(float(self.percentile), 0.0), 1.0)
        self.hybrid_min_batch_size = max(int(self.hybrid_min_batch_size), 1)
        self.hybrid_acceptance_threshold = min(max(float(self.hybrid_acceptance_threshold), 0.0), 1.0)
        self.hybrid_low_steps = max(int(self.hybrid_low_steps), 1)
        self.hybrid_high_steps = max(int(self.hybrid_high_steps), 1)
        self.hybrid_probe_interval = max(int(self.hybrid_probe_interval), 0)
        self._current_k: int | None = None
        self._low_acceptance_steps = 0
        self._high_acceptance_steps = 0
        self.observation_count = 0
        self.last_scheduled_widths: list[int] = []
        self.last_accepted_lengths: list[int] = []
        self.last_reason = "warmup"

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
        """Update the recommendation from observed usable prefix lengths."""

        if self.max_k <= 0:
            return
        observed_lengths = sorted(max(int(length), 0) for length in lengths)
        if not observed_lengths:
            return
        percentile_idx = math.ceil(self.percentile * (len(observed_lengths) - 1))
        observed = observed_lengths[percentile_idx]
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

    def observe(
        self,
        scheduled_widths: Sequence[int],
        sampled_token_ids: Sequence[Sequence[int]],
    ) -> None:
        """Observe actual acceptance without adding a device synchronization.

        A speculative result contains the accepted draft prefix followed by
        one target/bonus token, so ``len(sampled) - 1`` is the number of draft
        tokens that were useful. Clamp it by the width that the scheduler sent
        because prefill-only requests have no speculative capacity.
        """

        if len(scheduled_widths) != len(sampled_token_ids):
            return
        pairs = [
            (max(int(width), 0), tokens)
            for width, tokens in zip(scheduled_widths, sampled_token_ids, strict=True)
            if int(width) > 0
        ]
        if not pairs:
            return

        widths = [width for width, _ in pairs]
        accepted = [min(width, max(len(tokens) - 1, 0)) for width, tokens in pairs]
        self.last_scheduled_widths = widths
        self.last_accepted_lengths = accepted
        self.observation_count += 1
        if self.hybrid_enabled:
            self._update_hybrid(widths, accepted)
            return
        self.update(accepted)

    def _update_hybrid(self, scheduled_widths: Sequence[int], accepted_lengths: Sequence[int]) -> None:
        """Apply batch/acceptance gating to the next physical draft width."""

        batch_size = len(scheduled_widths)
        if batch_size < self.hybrid_min_batch_size:
            self._current_k = self.max_k
            self._low_acceptance_steps = 0
            self._high_acceptance_steps = 0
            self.last_reason = "small_batch_full_k"
            return

        if self.hybrid_probe_interval and self.observation_count % self.hybrid_probe_interval == 0:
            self._current_k = self.max_k
            self.last_reason = "periodic_full_k_probe"
            return

        scheduled = sum(scheduled_widths)
        acceptance = sum(accepted_lengths) / scheduled if scheduled else 1.0
        if acceptance >= self.hybrid_acceptance_threshold:
            self._high_acceptance_steps += 1
            self._low_acceptance_steps = 0
            if self._high_acceptance_steps >= self.hybrid_high_steps:
                self._current_k = self.max_k
                self.last_reason = "high_acceptance_full_k"
            else:
                self.last_reason = "high_acceptance_hysteresis"
            return

        self._low_acceptance_steps += 1
        self._high_acceptance_steps = 0
        if self._low_acceptance_steps < self.hybrid_low_steps:
            self.last_reason = "low_acceptance_hysteresis"
            return

        self.update(accepted_lengths)
        self.last_reason = "low_acceptance_dynamic_k"


@dataclass
class ProposalGate:
    """Hysteresis gate that switches between throughput and latency profiles.

    A low-load streak enters the latency profile (speculation enabled), while a
    high-load streak or queued work immediately returns to the throughput
    profile (``K=0``).  Hysteresis prevents oscillation around the boundary.
    """

    max_num_seqs: int
    enter_ratio: float = 0.5
    exit_ratio: float = 0.8
    max_avg_scheduled_tokens: float = 32.0
    enter_steps: int = 2
    exit_steps: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        self.max_num_seqs = max(int(self.max_num_seqs), 1)
        self.enter_ratio = min(max(float(self.enter_ratio), 0.0), 1.0)
        self.exit_ratio = min(max(float(self.exit_ratio), self.enter_ratio), 1.0)
        self.max_avg_scheduled_tokens = max(float(self.max_avg_scheduled_tokens), 0.0)
        self.enter_steps = max(int(self.enter_steps), 1)
        self.exit_steps = max(int(self.exit_steps), 1)
        self._latency_profile = False
        self._low_load_streak = 0
        self._high_load_streak = 0

    @property
    def latency_profile(self) -> bool:
        """Whether the next decode step may launch speculative decoding."""

        return self.enabled and self._latency_profile

    def observe(
        self,
        *,
        num_running: int,
        num_waiting: int,
        total_num_scheduled_tokens: int,
        num_scheduled_requests: int,
        prefill_scheduled: bool = False,
    ) -> bool:
        """Update the gate and return the current profile.

        Prefill and queued work are treated as throughput pressure.  Decode
        batches enter the latency profile only after ``enter_steps`` consecutive
        low-load observations and leave it after ``exit_steps`` high-load
        observations.
        """

        if not self.enabled:
            self._latency_profile = False
            return False

        running_ratio = max(float(num_running), 0.0) / self.max_num_seqs
        avg_tokens = float(total_num_scheduled_tokens) / num_scheduled_requests if num_scheduled_requests > 0 else 0.0
        low_load = (
            num_waiting == 0
            and not prefill_scheduled
            and running_ratio <= self.enter_ratio
            and avg_tokens <= self.max_avg_scheduled_tokens
        )
        high_load = (
            num_waiting > 0
            or prefill_scheduled
            or running_ratio >= self.exit_ratio
            or avg_tokens > self.max_avg_scheduled_tokens
        )

        if high_load:
            self._low_load_streak = 0
            self._high_load_streak += 1
            if self._high_load_streak >= self.exit_steps:
                self._latency_profile = False
        elif low_load:
            self._high_load_streak = 0
            self._low_load_streak += 1
            if self._low_load_streak >= self.enter_steps:
                self._latency_profile = True
        else:
            self._low_load_streak = 0
            self._high_load_streak = 0

        return self.latency_profile

    def select_k(self, configured_k: int, **load: int | bool) -> int:
        """Return the batch K after observing current scheduler load."""

        profile = self.observe(**load)
        configured_k = max(int(configured_k), 0)
        return configured_k if profile else 0
