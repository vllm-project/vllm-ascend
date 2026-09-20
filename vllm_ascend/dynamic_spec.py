# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler-side control for Ascend variable physical draft width.

The worker owns confidence processing and NPU cost modelling. It publishes a
compact recommendation; the scheduler applies it to a matching batch bucket.
The acceptance policy remains a safe fallback if AV profiling is unavailable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any


_ACCEPTANCE_EMA_ALPHA = 0.2
_BUCKET_SWITCH_STEPS = 2
_MIN_K_DWELL_STEPS = 8
PHYSICAL_K_MIN_TUNED_BATCH_SIZE = 9
_ACCEPTANCE_THRESHOLD = 0.6
_DOWNSHIFT_STEPS = 3
_UPSHIFT_STEPS = 2
_PROBE_INTERVAL = 32


def resolve_physical_k(dynamic_config: dict[str, Any]) -> dict[str, Any] | None:
    physical = dynamic_config.get("physical_k")
    if physical is None:
        return None
    if not isinstance(physical, dict):
        raise ValueError("physical_k must be an object")
    if dynamic_config.get("method") not in ("dspark", "dflash"):
        raise ValueError("physical_k requires dspark or dflash")
    unknown = set(physical) - {"min_k", "auto_tune"}
    if unknown:
        raise ValueError(f"Unknown physical_k fields: {sorted(unknown)}")
    min_k = physical.get("min_k", 3)
    auto_tune = physical.get("auto_tune", True)
    if type(min_k) is not int or min_k < 1:
        raise ValueError(f"physical_k.min_k has invalid value {min_k!r}")
    if not isinstance(auto_tune, bool):
        raise ValueError(f"physical_k.auto_tune has invalid value {auto_tune!r}")
    return {"min_k": min_k, "auto_tune": auto_tune}


def v2_physical_k_enabled(dynamic_config: dict[str, Any]) -> bool:
    return isinstance(dynamic_config.get("physical_k"), dict)


@dataclass
class _BucketState:
    stable_k: int
    profile_k: int
    cost_floor_k: int
    empirical_k: int
    survival: list[float]
    survival_seen: list[bool]
    observations: int = 0
    down_steps: int = 0
    up_steps: int = 0
    dwell_steps: int = 0
    last_probe_observation: int = -1


@dataclass
class AdaptiveDraftKController:
    """Combine AV cost recommendations with observed acceptance by batch.

    RL rollouts commonly begin with a large decode batch and shrink as
    requests encounter EOS.  Each power-of-two batch bucket therefore owns
    independent acceptance and K state.  Bucket changes and K changes are
    debounced so a short-lived rollout tail does not make graph width flap.
    """

    max_k: int
    min_k: int = 1
    auto_tune: bool = True

    def __post_init__(self) -> None:
        self.max_k = max(int(self.max_k), 0)
        self.min_k = min(max(int(self.min_k), 1), self.max_k) if self.max_k else 0
        self._current_k: int | None = None
        self._bucket_states: dict[int, _BucketState] = {}
        self._active_bucket: int | None = None
        self._pending_bucket: int | None = None
        self._pending_bucket_steps = 0

    @property
    def current_k(self) -> int | None:
        return self._current_k

    @staticmethod
    def _batch_bucket(batch_size: int) -> int:
        return 1 << (max(int(batch_size), 1) - 1).bit_length()

    def _state(self, bucket: int) -> _BucketState:
        state = self._bucket_states.get(bucket)
        if state is None:
            state = _BucketState(
                stable_k=self.max_k,
                profile_k=self.max_k,
                cost_floor_k=self.min_k,
                empirical_k=self.max_k,
                survival=[1.0] * self.max_k,
                survival_seen=[False] * self.max_k,
            )
            self._bucket_states[bucket] = state
        return state

    def _settled_bucket(self, batch_size: int) -> int:
        """Require two consecutive steps before following a shrinking batch."""

        bucket = self._batch_bucket(batch_size)
        if self._active_bucket is None:
            self._active_bucket = bucket
        elif bucket == self._active_bucket:
            self._pending_bucket = None
            self._pending_bucket_steps = 0
        elif bucket == self._pending_bucket:
            self._pending_bucket_steps += 1
            if self._pending_bucket_steps >= _BUCKET_SWITCH_STEPS:
                self._active_bucket = bucket
                self._pending_bucket = None
                self._pending_bucket_steps = 0
        else:
            self._pending_bucket = bucket
            self._pending_bucket_steps = 1
        return self._active_bucket

    def _desired_k(self, state: _BucketState) -> int:
        desired = self.max_k
        if self.auto_tune:
            desired = min(desired, max(state.profile_k, state.cost_floor_k))
        empirical_k = state.empirical_k
        if self.auto_tune:
            empirical_k = max(empirical_k, state.cost_floor_k)
        desired = min(desired, empirical_k)
        return max(self.min_k, min(desired, self.max_k))

    def _advance_state(self, state: _BucketState) -> None:
        desired = self._desired_k(state)
        if state.dwell_steps:
            state.dwell_steps -= 1
        if desired < state.stable_k:
            state.down_steps += 1
            state.up_steps = 0
            if state.down_steps >= _DOWNSHIFT_STEPS and state.dwell_steps == 0:
                state.stable_k = desired
                state.down_steps = 0
                state.dwell_steps = _MIN_K_DWELL_STEPS
        elif desired > state.stable_k:
            state.up_steps += 1
            state.down_steps = 0
            if state.up_steps >= _UPSHIFT_STEPS and state.dwell_steps == 0:
                state.stable_k = desired
                state.up_steps = 0
                state.dwell_steps = _MIN_K_DWELL_STEPS
        else:
            state.down_steps = 0
            state.up_steps = 0

    def recommend(
        self,
        batch_size: int,
        physical_k: int,
        cost_floor_k: int | None = None,
    ) -> None:
        physical_k = max(self.min_k, min(int(physical_k), self.max_k))
        state = self._state(self._batch_bucket(batch_size))
        state.profile_k = physical_k
        if cost_floor_k is not None:
            state.cost_floor_k = max(
                self.min_k,
                min(int(cost_floor_k), self.max_k),
            )

    def cap(self, configured_k: int, batch_size: int | None = None) -> int:
        configured_k = max(min(int(configured_k), self.max_k), 0)
        if configured_k == 0:
            return 0
        if batch_size is not None and batch_size < PHYSICAL_K_MIN_TUNED_BATCH_SIZE:
            self._current_k = configured_k
            return configured_k
        if batch_size:
            bucket = self._settled_bucket(batch_size)
            state = self._state(bucket)
            if (
                _PROBE_INTERVAL
                and state.observations
                and state.observations % _PROBE_INTERVAL == 0
                and state.last_probe_observation != state.observations
            ):
                state.last_probe_observation = state.observations
                self._current_k = configured_k
                return configured_k
            self._current_k = min(state.stable_k, configured_k)
            return self._current_k
        self._current_k = configured_k if self._current_k is None else self._current_k
        return min(self._current_k, configured_k)

    def observe(
        self,
        scheduled_widths: Sequence[int],
        sampled_token_ids: Sequence[Sequence[int]],
    ) -> None:
        if len(scheduled_widths) != len(sampled_token_ids):
            return
        pairs = [(int(width), tokens) for width, tokens in zip(scheduled_widths, sampled_token_ids) if width > 0]
        if not pairs:
            return
        widths = [width for width, _ in pairs]
        accepted = [min(width, max(len(tokens) - 1, 0)) for width, tokens in pairs]
        if len(widths) < PHYSICAL_K_MIN_TUNED_BATCH_SIZE:
            self._current_k = self.max_k
            return
        bucket = self._batch_bucket(len(widths))
        state = self._state(bucket)
        state.observations += 1
        alpha = _ACCEPTANCE_EMA_ALPHA
        for position in range(1, self.max_k + 1):
            eligible = [index for index, width in enumerate(widths) if width >= position]
            if not eligible:
                continue
            observed = sum(accepted[index] >= position for index in eligible) / len(eligible)
            index = position - 1
            if state.survival_seen[index]:
                state.survival[index] = (1.0 - alpha) * state.survival[index] + alpha * observed
            else:
                state.survival[index] = observed
                state.survival_seen[index] = True

        # Acceptance survival is monotone by definition. Enforce that after
        # EMA updates where later positions may have fewer eligible samples.
        for index in range(1, self.max_k):
            state.survival[index] = min(state.survival[index], state.survival[index - 1])
        useful_positions = sum(
            probability >= _ACCEPTANCE_THRESHOLD
            for probability, seen in zip(state.survival, state.survival_seen)
            if seen
        )
        ordered = sorted(accepted)
        quantile_position = math.ceil(0.5 * (len(ordered) - 1))
        quantile_k = ordered[quantile_position]
        state.empirical_k = max(
            self.min_k,
            min(
                self.max_k,
                useful_positions,
                quantile_k,
            ),
        )
        self._advance_state(state)
        if self._active_bucket == bucket:
            self._current_k = state.stable_k

    def observe_proposals(self, lengths: Sequence[int]) -> None:
        """Fallback for outputs that expose proposal lengths but no tokens."""

        normalized = [max(0, min(int(length), self.max_k)) for length in lengths]
        if not normalized:
            return
        widths = [self.max_k] * len(normalized)
        sampled = [[0] * (length + 1) for length in normalized]
        self.observe(widths, sampled)


def _create_controller(vllm_config: Any) -> AdaptiveDraftKController | None:
    if not getattr(vllm_config, "use_v2_model_runner", False):
        return None
    dynamic = (getattr(vllm_config, "additional_config", None) or {}).get("dynamic_spec_config", {})
    if not isinstance(dynamic, dict):
        return None
    params = resolve_physical_k(dynamic)
    if params is None:
        return None
    max_k = int(getattr(getattr(vllm_config, "speculative_config", None), "num_speculative_tokens", 0))
    return AdaptiveDraftKController(
        max_k=max_k,
        min_k=params["min_k"],
        auto_tune=params["auto_tune"],
    )


def _update_controller(controller, scheduler_output, model_runner_output) -> None:
    recommendation = getattr(model_runner_output, "physical_k_recommendation", None)
    if recommendation is not None:
        batch_size, physical_k, cost_floor_k = recommendation
        controller.recommend(batch_size, physical_k, cost_floor_k)
    sampled = getattr(model_runner_output, "sampled_token_ids", None)
    req_ids = getattr(model_runner_output, "req_ids", ())
    scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    if sampled is not None and len(req_ids) == len(sampled):
        controller.observe(
            [len(scheduled.get(req_id, ())) for req_id in req_ids],
            sampled,
        )
    elif (lengths := getattr(model_runner_output, "proposal_lengths", None)) is not None:
        controller.observe_proposals(lengths)


def install_scheduler_policy() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__
    if not getattr(original_init, "_vllm_ascend_physical_k_patched", False):
        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            config = args[0] if args else kwargs.get("vllm_config", getattr(self, "vllm_config", None))
            self._ascend_physical_k_controller = _create_controller(config)

        patched_init._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler.__init__ = patched_init

    original_after = Scheduler._update_after_schedule
    if not getattr(original_after, "_vllm_ascend_physical_k_patched", False):
        @wraps(original_after)
        def patched_after(self, scheduler_output):
            controller = getattr(self, "_ascend_physical_k_controller", None)
            if controller is not None:
                batch_size = len(scheduler_output.scheduled_spec_decode_tokens)
                scheduler_output.num_spec_tokens_to_schedule = controller.cap(
                    scheduler_output.num_spec_tokens_to_schedule, batch_size
                )
            return original_after(self, scheduler_output)

        patched_after._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler._update_after_schedule = patched_after

    original_output = Scheduler.update_from_output
    if not getattr(original_output, "_vllm_ascend_physical_k_patched", False):
        @wraps(original_output)
        def patched_output(self, scheduler_output, model_runner_output):
            outputs = original_output(self, scheduler_output, model_runner_output)
            if (controller := getattr(self, "_ascend_physical_k_controller", None)) is not None:
                _update_controller(controller, scheduler_output, model_runner_output)
            return outputs

        patched_output._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler.update_from_output = patched_output
