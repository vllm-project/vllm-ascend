# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side policy for Ascend variable physical draft width.

Confidence estimation and logical verification budgets are owned by vLLM.
This module only feeds their result, plus observed acceptance, into the next
Ascend draft width.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

_PHYSICAL_DEFAULTS = {
    "enabled": True,
    "min_k": 1,
    "slack": 0,
    "percentile": 0.5,
}
_HYBRID_DEFAULTS = {
    "enabled": True,
    "min_batch_size": 8,
    "acceptance_threshold": 0.6,
    "low_steps": 4,
    "high_steps": 2,
    "probe_interval": 32,
}
_AUTO_TUNE_DEFAULTS = {
    "enabled": False,
    "warmup_steps": 64,
    "explore_ratio": 0.05,
    "update_interval": 32,
    "min_gain": 0.02,
    "ema_decay": 0.9,
}


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
    """Validate and normalize the compact ``physical_k`` interface."""

    physical = dynamic_config.get("physical_k")
    if physical is None:
        return None
    if not isinstance(physical, dict):
        raise ValueError("physical_k must be an object")
    if dynamic_config.get("policy") != "hardware_aware" or dynamic_config.get("method") not in ("dspark", "dflash"):
        raise ValueError("physical_k requires hardware_aware policy and dspark/dflash method")

    unknown = set(physical) - set(_PHYSICAL_DEFAULTS) - {"capture_k", "hybrid"}
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
    unknown = set(auto_tune) - set(_AUTO_TUNE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown physical_k.auto_tune fields: {sorted(unknown)}")
    for name, default in _AUTO_TUNE_DEFAULTS.items():
        value = auto_tune.get(name, default)
        _validate_number(name, value, default)
        result[f"auto_tune_{name}"] = value
    return result


def v2_physical_k_enabled(dynamic_config: dict[str, Any]) -> bool:
    physical = dynamic_config.get("physical_k")
    return isinstance(physical, dict) and physical.get("enabled", True) is True


@dataclass
class _CostEstimate:
    """EMA of effective output tokens per millisecond for one K candidate."""

    ema_score: float = 0.0
    samples: int = 0

    def observe(self, score: float, decay: float) -> None:
        self.ema_score = score if not self.samples else decay * self.ema_score + (1 - decay) * score
        self.samples += 1


@dataclass
class AdaptiveDraftKController:
    """Choose the next physical K from acceptance or an online cost model."""

    max_k: int
    min_k: int = 1
    slack: int = 0
    percentile: float = 0.5
    hybrid_enabled: bool = True
    hybrid_min_batch_size: int = 8
    hybrid_acceptance_threshold: float = 0.6
    hybrid_low_steps: int = 4
    hybrid_high_steps: int = 2
    hybrid_probe_interval: int = 32
    capture_k: tuple[int, ...] = ()
    graph_mode: str = "unknown"
    auto_tune_enabled: bool = False
    auto_tune_warmup_steps: int = 64
    auto_tune_explore_ratio: float = 0.05
    auto_tune_update_interval: int = 32
    auto_tune_min_gain: float = 0.02
    auto_tune_ema_decay: float = 0.9

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
        self.auto_tune_warmup_steps = max(int(self.auto_tune_warmup_steps), 1)
        self.auto_tune_explore_ratio = min(max(float(self.auto_tune_explore_ratio), 0.0), 1.0)
        self.auto_tune_update_interval = max(int(self.auto_tune_update_interval), 1)
        self.auto_tune_min_gain = min(max(float(self.auto_tune_min_gain), 0.0), 1.0)
        self.auto_tune_ema_decay = min(max(float(self.auto_tune_ema_decay), 0.0), 1.0)
        candidates = {
            int(k)
            for k in self.capture_k
            if self.min_k <= int(k) <= self.max_k
        }
        if self.max_k >= self.min_k:
            candidates.add(self.max_k)
        self._candidate_k = tuple(sorted(candidates))
        self._cost_model: dict[tuple[str, int, int], _CostEstimate] = {}
        self._last_auto_decision = 0
        self._current_k: int | None = None
        self._low_acceptance_steps = 0
        self._high_acceptance_steps = 0
        self.observation_count = 0
        self.last_scheduled_widths: list[int] = []
        self.last_accepted_lengths: list[int] = []
        self.last_reason = "warmup"

    @property
    def current_k(self) -> int | None:
        return self._current_k

    def cap(self, configured_k: int) -> int:
        configured_k = max(min(int(configured_k), self.max_k), 0)
        if configured_k == 0:
            return 0
        if self._current_k is None:
            self._current_k = configured_k
        else:
            self._current_k = min(self._current_k, configured_k)
        return self._current_k

    @staticmethod
    def _batch_bucket(batch_size: int) -> int:
        batch_size = max(int(batch_size), 1)
        return 1 << (batch_size - 1).bit_length()

    def _cost_key(self, batch_size: int, k: int) -> tuple[str, int, int]:
        return self.graph_mode, self._batch_bucket(batch_size), int(k)

    def _select_least_sampled(self, batch_size: int, exclude: int | None = None) -> int | None:
        choices = [k for k in self._candidate_k if k != exclude]
        if not choices:
            return None
        return min(
            choices,
            key=lambda k: (
                self._cost_model.get(self._cost_key(batch_size, k), _CostEstimate()).samples,
                k,
            ),
        )

    def _observe_cost(
        self,
        widths: Sequence[int],
        accepted: Sequence[int],
        elapsed_ms: float,
    ) -> None:
        if elapsed_ms <= 0 or not self._candidate_k:
            return
        batch_size = len(widths)
        observed_k = max(widths)
        if observed_k not in self._candidate_k:
            return
        # Every request produces one target token, in addition to accepted
        # draft tokens.  This makes candidates with different K comparable.
        effective_tokens = sum(accepted) + batch_size
        score = effective_tokens / elapsed_ms
        key = self._cost_key(batch_size, observed_k)
        estimate = self._cost_model.setdefault(key, _CostEstimate())
        estimate.observe(score, self.auto_tune_ema_decay)

    def _choose_auto_k(self, batch_size: int) -> None:
        if not self._candidate_k:
            return
        # During warmup, deliberately cover every configured candidate.  This
        # is bounded exploration and does not require extra graph capture.
        if self.observation_count <= self.auto_tune_warmup_steps:
            selected = self._select_least_sampled(batch_size)
            if selected is not None:
                self._current_k = selected
                self.last_reason = "auto_warmup_explore"
            return

        if self.observation_count - self._last_auto_decision < self.auto_tune_update_interval:
            return
        self._last_auto_decision = self.observation_count

        if self.auto_tune_explore_ratio:
            period = max(round(1 / self.auto_tune_explore_ratio), 1)
            if self.observation_count % period == 0:
                selected = self._select_least_sampled(batch_size, self._current_k)
                if selected is not None:
                    self._current_k = selected
                    self.last_reason = "auto_periodic_explore"
                    return

        estimates = {
            k: self._cost_model.get(self._cost_key(batch_size, k))
            for k in self._candidate_k
        }
        usable = {k: value for k, value in estimates.items() if value is not None and value.samples > 0}
        if not usable:
            selected = self._select_least_sampled(batch_size)
            if selected is not None:
                self._current_k = selected
                self.last_reason = "auto_new_batch_bucket_explore"
            return
        best_k = max(usable, key=lambda k: usable[k].ema_score)
        current = usable.get(self._current_k)
        best = usable[best_k]
        if current is None or best.ema_score >= current.ema_score * (1 + self.auto_tune_min_gain):
            self._current_k = best_k
            self.last_reason = "auto_cost_model_best_k"
        else:
            self.last_reason = "auto_cost_model_keep_k"

    def update(self, lengths: Iterable[int]) -> None:
        if self.max_k <= 0:
            return
        observed_lengths = sorted(max(int(length), 0) for length in lengths)
        if not observed_lengths:
            return
        index = math.ceil(self.percentile * (len(observed_lengths) - 1))
        observed = min(observed_lengths[index], self.max_k)
        if self._current_k is None:
            self._current_k = max(self.min_k, min(self.max_k, observed + self.slack))
        elif observed >= self._current_k:
            self._current_k = min(self.max_k, self._current_k + 1)
        else:
            self._current_k = min(self._current_k, max(self.min_k, observed + self.slack))

    def observe(
        self,
        scheduled_widths: Sequence[int],
        sampled_token_ids: Sequence[Sequence[int]],
        elapsed_ms: float | None = None,
    ) -> None:
        if len(scheduled_widths) != len(sampled_token_ids):
            return
        pairs = [
            (max(int(width), 0), tokens)
            for width, tokens in zip(scheduled_widths, sampled_token_ids)
            if int(width) > 0
        ]
        if not pairs:
            return

        widths = [width for width, _ in pairs]
        accepted = [min(width, max(len(tokens) - 1, 0)) for width, tokens in pairs]
        self.last_scheduled_widths = widths
        self.last_accepted_lengths = accepted
        self.observation_count += 1
        if self.auto_tune_enabled and elapsed_ms is not None:
            self._observe_cost(widths, accepted, float(elapsed_ms))
            self._choose_auto_k(len(widths))
            return
        if not self.hybrid_enabled:
            self.update(accepted)
            return
        self._update_hybrid(widths, accepted)

    def _update_hybrid(self, widths: Sequence[int], accepted: Sequence[int]) -> None:
        if len(widths) < self.hybrid_min_batch_size:
            self._current_k = self.max_k
            self._low_acceptance_steps = 0
            self._high_acceptance_steps = 0
            self.last_reason = "small_batch_full_k"
            return
        if self.hybrid_probe_interval and self.observation_count % self.hybrid_probe_interval == 0:
            self._current_k = self.max_k
            self.last_reason = "periodic_full_k_probe"
            return

        total_width = sum(widths)
        acceptance = sum(accepted) / total_width if total_width else 1.0
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
        self.update(accepted)
        self.last_reason = "low_acceptance_dynamic_k"


def _create_controller(vllm_config: Any) -> AdaptiveDraftKController | None:
    if not getattr(vllm_config, "use_v2_model_runner", False):
        return None
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    dynamic_config = additional_config.get("dynamic_spec_config", {})
    if not isinstance(dynamic_config, dict):
        return None
    params = resolve_physical_k(dynamic_config)
    if params is None or not params["enabled"]:
        return None
    speculative_config = getattr(vllm_config, "speculative_config", None)
    max_k = int(getattr(speculative_config, "num_speculative_tokens", 0))
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
        capture_k=tuple(params.get("capture_k", ())),
        graph_mode=_graph_mode(vllm_config),
        auto_tune_enabled=params["auto_tune_enabled"],
        auto_tune_warmup_steps=params["auto_tune_warmup_steps"],
        auto_tune_explore_ratio=params["auto_tune_explore_ratio"],
        auto_tune_update_interval=params["auto_tune_update_interval"],
        auto_tune_min_gain=params["auto_tune_min_gain"],
        auto_tune_ema_decay=params["auto_tune_ema_decay"],
    )


def _graph_mode(vllm_config: Any) -> str:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation_config, "cudagraph_mode", "unknown")
    return getattr(mode, "name", str(mode))


def _update_controller(controller, scheduler_output, model_runner_output, elapsed_ms: float | None = None) -> None:
    sampled = getattr(model_runner_output, "sampled_token_ids", None)
    req_ids = getattr(model_runner_output, "req_ids", ())
    scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    if sampled is not None and len(req_ids) == len(sampled):
        controller.observe(
            [len(scheduled.get(req_id, ())) for req_id in req_ids],
            sampled,
            elapsed_ms=elapsed_ms,
        )
        return
    lengths = getattr(model_runner_output, "proposal_lengths", None)
    if lengths is not None:
        controller.update(lengths)


def install_scheduler_policy() -> None:
    """Install the minimal scheduler hooks needed for next-step physical K."""

    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__
    if not getattr(original_init, "_vllm_ascend_physical_k_patched", False):

        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            vllm_config = args[0] if args else kwargs.get("vllm_config")
            self._ascend_physical_k_controller = _create_controller(vllm_config)

        patched_init._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler.__init__ = patched_init

    original_update_after_schedule = Scheduler._update_after_schedule
    if not getattr(original_update_after_schedule, "_vllm_ascend_physical_k_patched", False):

        @wraps(original_update_after_schedule)
        def patched_update_after_schedule(self, scheduler_output):
            controller = getattr(self, "_ascend_physical_k_controller", None)
            if controller is not None and controller.auto_tune_enabled:
                self._ascend_physical_k_profile_start_ns = time.perf_counter_ns()
            if controller is not None:
                scheduler_output.num_spec_tokens_to_schedule = controller.cap(
                    scheduler_output.num_spec_tokens_to_schedule
                )
            return original_update_after_schedule(self, scheduler_output)

        patched_update_after_schedule._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler._update_after_schedule = patched_update_after_schedule

    original_update_from_output = Scheduler.update_from_output
    if not getattr(original_update_from_output, "_vllm_ascend_physical_k_patched", False):

        @wraps(original_update_from_output)
        def patched_update_from_output(self, scheduler_output, model_runner_output):
            outputs = original_update_from_output(self, scheduler_output, model_runner_output)
            controller = getattr(self, "_ascend_physical_k_controller", None)
            if controller is not None:
                start_ns = getattr(self, "_ascend_physical_k_profile_start_ns", None)
                elapsed_ms = (
                    (time.perf_counter_ns() - start_ns) / 1_000_000
                    if start_ns is not None
                    else None
                )
                _update_controller(controller, scheduler_output, model_runner_output, elapsed_ms)
                if start_ns is not None:
                    del self._ascend_physical_k_profile_start_ns
            return outputs

        patched_update_from_output._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler.update_from_output = patched_update_from_output
