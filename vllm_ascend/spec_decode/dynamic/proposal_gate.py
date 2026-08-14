# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Batch-level proposal gating for hardware-aware speculative decoding.

The confidence scheduler chooses *how many* tokens each request should verify.
This module decides whether the batch should launch a drafter at all.  It is
deliberately CPU-only: all inputs are scheduler counters and no device tensor
is touched on the scheduling path.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        self.max_avg_scheduled_tokens = max(
            float(self.max_avg_scheduled_tokens), 0.0
        )
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
        avg_tokens = (
            float(total_num_scheduled_tokens) / num_scheduled_requests
            if num_scheduled_requests > 0
            else 0.0
        )
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
