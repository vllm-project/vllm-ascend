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


_PHYSICAL_DEFAULTS = {"enabled": True, "min_k": 3, "slack": 0, "percentile": 0.5}
_HYBRID_DEFAULTS = {
    "enabled": True,
    "min_batch_size": 8,
    "acceptance_threshold": 0.6,
    "low_steps": 3,
    "high_steps": 2,
    "probe_interval": 32,
}

_ACCEPTANCE_EMA_ALPHA = 0.2
_BUCKET_SWITCH_STEPS = 2
_MIN_K_DWELL_STEPS = 8


def _validate_number(name: str, value: Any, default: Any) -> None:
    if isinstance(default, bool):
        valid = isinstance(value, bool)
    elif isinstance(default, int):
        minimum = 0 if name in ("slack", "probe_interval") else 1
        valid = type(value) is int and value >= minimum
    else:
        valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    if not valid:
        raise ValueError(f"physical_k.{name} has invalid value {value!r}")


def resolve_physical_k(dynamic_config: dict[str, Any]) -> dict[str, Any] | None:
    physical = dynamic_config.get("physical_k")
    if physical is None:
        return None
    if not isinstance(physical, dict):
        raise ValueError("physical_k must be an object")
    if dynamic_config.get("policy") != "hardware_aware" or dynamic_config.get("method") not in ("dspark", "dflash"):
        raise ValueError("physical_k requires hardware_aware policy and dspark/dflash method")
    unknown = set(physical) - set(_PHYSICAL_DEFAULTS) - {"capture_k", "hybrid", "auto_tune"}
    if unknown:
        raise ValueError(f"Unknown physical_k fields: {sorted(unknown)}")

    result: dict[str, Any] = {}
    for name, default in _PHYSICAL_DEFAULTS.items():
        value = physical.get(name, default)
        _validate_number(name, value, default)
        result[name] = value
    capture_k = physical.get("capture_k")
    if capture_k is not None:
        if (
            not isinstance(capture_k, (list, tuple))
            or not capture_k
            or any(type(value) is not int or value < 1 for value in capture_k)
        ):
            raise ValueError("physical_k.capture_k must be a non-empty list of positive integers")
        result["capture_k"] = tuple(sorted(set(capture_k)))

    hybrid = physical.get("hybrid", {})
    if not isinstance(hybrid, dict):
        raise ValueError("physical_k.hybrid must be an object")
    unknown = set(hybrid) - set(_HYBRID_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown physical_k.hybrid fields: {sorted(unknown)}")
    for name, default in _HYBRID_DEFAULTS.items():
        value = hybrid.get(name, default)
        _validate_number(name, value, default)
        result[f"hybrid_{name}"] = value

    auto_tune = physical.get("auto_tune", {})
    if not isinstance(auto_tune, dict):
        raise ValueError("physical_k.auto_tune must be an object")
    unknown = set(auto_tune) - {"enabled"}
    if unknown:
        raise ValueError(
            "physical_k.auto_tune only supports 'enabled'; AV profiling owns "
            f"K selection, unknown fields: {sorted(unknown)}"
        )
    result["auto_tune_enabled"] = auto_tune.get("enabled", True)
    _validate_number("auto_tune.enabled", result["auto_tune_enabled"], True)
    return result


def v2_physical_k_enabled(dynamic_config: dict[str, Any]) -> bool:
    physical = dynamic_config.get("physical_k")
    return isinstance(physical, dict) and physical.get("enabled", True) is True


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
    slack: int = 0
    percentile: float = 0.5
    hybrid_enabled: bool = True
    hybrid_min_batch_size: int = 8
    hybrid_acceptance_threshold: float = 0.6
    hybrid_low_steps: int = 3
    hybrid_high_steps: int = 2
    hybrid_probe_interval: int = 32
    auto_tune_enabled: bool = True

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
        self._bucket_states: dict[int, _BucketState] = {}
        self._active_bucket: int | None = None
        self._pending_bucket: int | None = None
        self._pending_bucket_steps = 0
        self.observation_count = 0
        self.last_scheduled_widths: list[int] = []
        self.last_accepted_lengths: list[int] = []
        self.last_reason = "awaiting_profile_recommendation"

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
                self.last_reason = "batch_bucket_settled"
        else:
            self._pending_bucket = bucket
            self._pending_bucket_steps = 1
            self.last_reason = "batch_bucket_hysteresis"
        return self._active_bucket

    def _desired_k(self, state: _BucketState) -> int:
        desired = self.max_k
        if self.auto_tune_enabled:
            desired = min(desired, max(state.profile_k, state.cost_floor_k))
        if self.hybrid_enabled or not self.auto_tune_enabled:
            empirical_k = state.empirical_k
            if self.auto_tune_enabled:
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
            if state.down_steps >= self.hybrid_low_steps and state.dwell_steps == 0:
                state.stable_k = desired
                state.down_steps = 0
                state.dwell_steps = _MIN_K_DWELL_STEPS
                self.last_reason = "combined_downshift"
            else:
                self.last_reason = "downshift_hysteresis"
        elif desired > state.stable_k:
            state.up_steps += 1
            state.down_steps = 0
            if state.up_steps >= self.hybrid_high_steps and state.dwell_steps == 0:
                state.stable_k = desired
                state.up_steps = 0
                state.dwell_steps = _MIN_K_DWELL_STEPS
                self.last_reason = "combined_upshift"
            else:
                self.last_reason = "upshift_hysteresis"
        else:
            state.down_steps = 0
            state.up_steps = 0
            self.last_reason = "combined_stable"

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
        self.last_reason = "av_profile_recommendation"

    def cap(self, configured_k: int, batch_size: int | None = None) -> int:
        configured_k = max(min(int(configured_k), self.max_k), 0)
        if configured_k == 0:
            return 0
        if self.hybrid_enabled and batch_size is not None and batch_size < self.hybrid_min_batch_size:
            self._current_k = configured_k
            self.last_reason = "small_batch_fixed_k"
            return configured_k
        if batch_size:
            bucket = self._settled_bucket(batch_size)
            state = self._state(bucket)
            if (
                self.hybrid_probe_interval
                and state.observations
                and state.observations % self.hybrid_probe_interval == 0
                and state.last_probe_observation != state.observations
            ):
                state.last_probe_observation = state.observations
                self._current_k = configured_k
                self.last_reason = "periodic_full_k_probe"
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
        self.last_scheduled_widths = widths
        self.last_accepted_lengths = accepted
        if self.hybrid_enabled and len(widths) < self.hybrid_min_batch_size:
            self._current_k = self.max_k
            self.last_reason = "small_batch_fixed_k"
            return
        self.observation_count += 1
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
            probability >= self.hybrid_acceptance_threshold
            for probability, seen in zip(state.survival, state.survival_seen)
            if seen
        )
        ordered = sorted(accepted)
        quantile_position = math.ceil(self.percentile * (len(ordered) - 1))
        quantile_k = ordered[quantile_position]
        state.empirical_k = max(
            self.min_k,
            min(
                self.max_k,
                useful_positions + self.slack,
                quantile_k + self.slack,
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
    if params is None or not params["enabled"]:
        return None
    max_k = int(getattr(getattr(vllm_config, "speculative_config", None), "num_speculative_tokens", 0))
    return AdaptiveDraftKController(
        max_k=max_k,
        min_k=params["min_k"],
        slack=params["slack"],
        percentile=params["percentile"],
        hybrid_enabled=params["hybrid_enabled"],
        hybrid_min_batch_size=params["hybrid_min_batch_size"],
        hybrid_acceptance_threshold=params["hybrid_acceptance_threshold"],
        hybrid_low_steps=params["hybrid_low_steps"],
        hybrid_high_steps=params["hybrid_high_steps"],
        hybrid_probe_interval=params["hybrid_probe_interval"],
        auto_tune_enabled=params["auto_tune_enabled"],
    )


def _update_controller(controller, scheduler_output, model_runner_output) -> None:
    recommendation = getattr(model_runner_output, "physical_k_recommendation", None)
    if recommendation is not None:
        batch_size, physical_k, _score, *extra = recommendation
        controller.recommend(
            batch_size,
            physical_k,
            extra[0] if extra else None,
        )
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
