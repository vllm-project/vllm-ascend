# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-side configuration, policy and scheduler integration for physical K.

This module depends only on the standard library at import time: configuration
and platform initialization must not import model proposers or Torch. vLLM
scheduler classes are imported only when installing the existing integration.
Confidence estimation and logical verification budgets remain upstream-owned.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

# Reuse the upstream logger identity without importing vLLM during config init.
logger = logging.getLogger("vllm.logger")


# Configuration normalization.
_PHYSICAL_FIELDS = (
    ("enabled", "adaptive_draft_k", True),
    ("min_k", "adaptive_draft_k_min", 1),
    ("slack", "adaptive_draft_k_slack", 0),
    ("percentile", "adaptive_draft_k_percentile", 0.5),
)
_HYBRID_FIELDS = (
    ("enabled", "hybrid_policy_enabled", True),
    ("min_batch_size", "hybrid_min_batch_size", 8),
    ("acceptance_threshold", "hybrid_acceptance_threshold", 0.6),
    ("low_steps", "hybrid_low_steps", 4),
    ("high_steps", "hybrid_high_steps", 2),
    ("probe_interval", "hybrid_probe_interval", 32),
)


def _validate_fields(values: dict, fields: tuple, path: str) -> dict[str, Any]:
    result = {}
    for name, legacy_name, default in fields:
        value = values.get(name, default)
        valid = False
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            minimum = 0 if name in ("slack", "probe_interval") else 1
            valid = type(value) is int and value >= minimum
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
        if not valid:
            raise ValueError(f"{path}.{name} has invalid value {value!r}")
        result[legacy_name] = value
    return result


def resolve_method_params(dynamic_config: dict[str, Any]) -> dict[str, Any]:
    """Return scheduler/worker parameters; never mutate the caller's config.

    ``physical_k`` is opt-in. Its defaults enable physical K, V2 variable-width
    support and hybrid together, with zero slack. Absent/None preserves every
    legacy default, including slack=1 and disabled hybrid/V2 switches.
    """
    legacy = dynamic_config.get("method_params", {})
    physical = dynamic_config.get("physical_k")
    if physical is None:
        return dict(legacy) if isinstance(legacy, dict) else {}
    if not isinstance(legacy, dict) or not isinstance(physical, dict):
        raise ValueError("physical_k and method_params must be objects")
    if dynamic_config.get("policy") != "hardware_aware" or dynamic_config.get("method") not in ("dspark", "dflash"):
        raise ValueError("physical_k requires hardware_aware policy and dspark/dflash method")
    unknown = set(physical) - {name for name, _, _ in _PHYSICAL_FIELDS} - {"capture_k", "hybrid"}
    if unknown:
        raise ValueError(f"Unknown physical_k fields: {sorted(unknown)}")
    hybrid = physical.get("hybrid", {})
    if not isinstance(hybrid, dict):
        raise ValueError("physical_k.hybrid must be an object")
    unknown = set(hybrid) - {name for name, _, _ in _HYBRID_FIELDS}
    if unknown:
        raise ValueError(f"Unknown physical_k.hybrid fields: {sorted(unknown)}")
    translated = _validate_fields(physical, _PHYSICAL_FIELDS, "physical_k")
    translated.update(_validate_fields(hybrid, _HYBRID_FIELDS, "physical_k.hybrid"))
    translated["v2_varlen_physical_k"] = translated["adaptive_draft_k"]
    capture = physical.get("capture_k")
    if capture is not None:
        if not isinstance(capture, (list, tuple)) or not capture or any(type(k) is not int or k < 1 for k in capture):
            raise ValueError("physical_k.capture_k must be a non-empty list of positive integers")
        translated["v2_varlen_capture_k"] = list(capture)
    legacy_keys = set(translated) | {"v2_varlen_capture_k", "adaptive_draft_k_v2"}
    conflict = legacy_keys.intersection(legacy)
    if conflict:
        raise ValueError(f"physical_k cannot be combined with legacy method_params fields: {sorted(conflict)}")
    return {**legacy, **translated}


def v2_physical_k_enabled(dynamic_config: dict[str, Any]) -> bool:
    """Constant-size hot-path lookup; full validation happens at config init."""
    physical = dynamic_config.get("physical_k")
    if physical is not None:
        return physical.get("enabled", True)
    params = dynamic_config.get("method_params", {})
    if not isinstance(params, dict):
        return False
    return bool(params.get("v2_varlen_physical_k", params.get("adaptive_draft_k_v2", False)))


# CPU-only policy state machines.


def validate_v1_dynamic_policy(method: str, dynamic_config: Any) -> None:
    """Keep the policy guard formerly passed by the two V1 constructors.

    DSpark also runs the DFlash parent constructor, which may initialize its
    scheduler when the configured dynamic method is DFlash. An unrelated or
    disabled dynamic method must not start validating an unused policy.
    """
    constructs_scheduler = method in ("dspark", "dflash") and (
        dynamic_config.method == method or (method == "dspark" and dynamic_config.method == "dflash")
    )
    if constructs_scheduler and dynamic_config.policy != "confidence_budget":
        raise ValueError(
            "The legacy V1 scheduler supports only confidence_budget. "
            "Use V2 DSpark with enable_adaptive_verification=true "
            "for hardware_aware physical K."
        )


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


# Scheduler installation and feedback; preserve async placeholder ordering.


def _supports_adaptive_physical_k(vllm_config) -> bool:
    """Return whether the active worker can consume a variable draft width.

    V2 is enabled by physical_k or the legacy varlen graph switch. The worker then
    captures width-specific descriptors and updates the draft metadata in the
    same runtime scope as the scheduler-selected physical K.  Without that
    switch, retain the historical fixed-width safety behavior.
    """

    if bool(getattr(vllm_config, "use_v2_model_runner", False)):
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        dynamic_config = additional_config.get("dynamic_spec_config", {})
        return v2_physical_k_enabled(dynamic_config if isinstance(dynamic_config, dict) else {})
    # Some older vLLM configs do not expose the field yet; the launch
    # environment is still authoritative for the Ascend V2 worker.
    return os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "1"


def install_output_fields() -> None:
    from vllm.v1 import outputs as outputs_mod

    def _patch_optional_field(cls, field_name: str) -> None:
        """Backport an optional dataclass field without touching vLLM files.

        The Ascend worker can run against a vLLM checkout that predates the
        speculative-decoding side-channel fields.  Adding the field at runtime
        keeps the editable vllm-ascend package self-contained and preserves the
        upstream class layout/serialization contract.
        """

        fields = getattr(cls, "__dataclass_fields__", {})
        if field_name in fields:
            return

        setattr(cls, field_name, None)
        original_init = cls.__init__
        marker = f"_vllm_ascend_optional_{field_name}_patched"
        if getattr(original_init, marker, False):
            return

        @wraps(original_init)
        def _patched_init(self, *args, **kwargs):
            value = kwargs.pop(field_name, None)
            original_init(self, *args, **kwargs)
            setattr(self, field_name, value)

        setattr(_patched_init, marker, True)
        cls.__init__ = _patched_init

    # ``spec_token_ids`` is required by the PP/MTP compatibility path, while
    # ``proposal_lengths`` carries the logical dynamic-draft width.  Both are
    # optional on newer upstream vLLM and are installed here only when absent.
    _patch_optional_field(outputs_mod.ModelRunnerOutput, "spec_token_ids")
    _patch_optional_field(outputs_mod.ModelRunnerOutput, "proposal_lengths")
    _patch_optional_field(outputs_mod.DraftTokenIds, "proposal_lengths")

    empty_output = outputs_mod.EMPTY_MODEL_RUNNER_OUTPUT
    if not hasattr(empty_output, "spec_token_ids"):
        empty_output.spec_token_ids = None


def install_scheduler_policy() -> None:
    """Backport the Ascend proposal gate to older vLLM schedulers.

    The hardware-aware scheduler lives in vllm-ascend's copied scheduler
    classes (BalanceScheduler/RecomputeScheduler).  They call the helper that
    was added to upstream vLLM in 6ec76df8.  Install the same helper and the
    minimal constructor state on the upstream base class when that commit is
    not present, keeping all compatibility code in this repository.
    """

    from vllm.v1.core.sched.scheduler import Scheduler

    original_gate_helper = getattr(Scheduler, "_apply_ascend_proposal_gate", None)
    if not getattr(original_gate_helper, "_vllm_ascend_adaptive_k_patched", False):

        def _apply_ascend_proposal_gate(
            self,
            configured_k: int,
            *,
            total_num_scheduled_tokens: int,
            num_scheduled_requests: int,
            prefill_scheduled: bool = False,
        ) -> int:
            if original_gate_helper is not None:
                selected_k = original_gate_helper(
                    self,
                    configured_k,
                    total_num_scheduled_tokens=total_num_scheduled_tokens,
                    num_scheduled_requests=num_scheduled_requests,
                    prefill_scheduled=prefill_scheduled,
                )
            else:
                gate = getattr(self, "_ascend_proposal_gate", None)
                if gate is None:
                    selected_k = configured_k
                else:
                    streaming_waiting = getattr(self, "num_waiting_for_streaming_input", 0)
                    if not isinstance(streaming_waiting, int):
                        streaming_waiting = len(streaming_waiting)
                    selected_k = gate.select_k(
                        configured_k,
                        num_running=len(getattr(self, "running", ())) + streaming_waiting,
                        num_waiting=len(getattr(self, "waiting", ())) + len(getattr(self, "skipped_waiting", ())),
                        total_num_scheduled_tokens=total_num_scheduled_tokens,
                        num_scheduled_requests=num_scheduled_requests,
                        prefill_scheduled=prefill_scheduled,
                    )

            controller = getattr(self, "_ascend_physical_k_controller", None)
            return controller.cap(selected_k) if controller is not None else selected_k

        _apply_ascend_proposal_gate._vllm_ascend_adaptive_k_patched = True  # type: ignore[attr-defined]
        Scheduler._apply_ascend_proposal_gate = _apply_ascend_proposal_gate

    original_init = Scheduler.__init__
    if getattr(original_init, "_vllm_ascend_dynamic_gate_patched", False):
        return

    @wraps(original_init)
    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._latest_proposal_lengths = {}
        self._ascend_proposal_gate = None
        self._ascend_physical_k_controller = None

        vllm_config = args[0] if args else kwargs.get("vllm_config")
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        dynamic_spec_config = additional_config.get("dynamic_spec_config", {})
        if not isinstance(dynamic_spec_config, dict):
            dynamic_spec_config = {}

        method_params = resolve_method_params(dynamic_spec_config)

        # The confidence scheduler chooses a logical verify prefix after the
        # draft has run.  Opt-in adaptive K feeds that result back to the next
        # scheduler step so unused draft positions are not computed again.
        if (
            dynamic_spec_config.get("policy") == "hardware_aware"
            and dynamic_spec_config.get("method") in ("dspark", "dflash")
            and bool(method_params.get("adaptive_draft_k", False))
        ):
            if not _supports_adaptive_physical_k(vllm_config):
                logger.info(
                    "Adaptive physical draft K is disabled for the V2 model "
                    "runner; keeping the fixed DSpark query width for FULL "
                    "graph attention metadata compatibility."
                )
            else:
                try:
                    speculative_config = getattr(vllm_config, "speculative_config", None)
                    max_k = int(getattr(speculative_config, "num_speculative_tokens", 0))
                    self._ascend_physical_k_controller = AdaptiveDraftKController(
                        max_k=max_k,
                        min_k=int(method_params.get("adaptive_draft_k_min", 1)),
                        slack=int(method_params.get("adaptive_draft_k_slack", 1)),
                        percentile=float(method_params.get("adaptive_draft_k_percentile", 0.5)),
                        hybrid_enabled=bool(method_params.get("hybrid_policy_enabled", False)),
                        hybrid_min_batch_size=int(method_params.get("hybrid_min_batch_size", 8)),
                        hybrid_acceptance_threshold=float(method_params.get("hybrid_acceptance_threshold", 0.6)),
                        hybrid_low_steps=int(method_params.get("hybrid_low_steps", 4)),
                        hybrid_high_steps=int(method_params.get("hybrid_high_steps", 2)),
                        hybrid_probe_interval=int(method_params.get("hybrid_probe_interval", 32)),
                    )
                except (ImportError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Failed to initialize adaptive physical draft K controller: %s",
                        exc,
                    )

        if dynamic_spec_config.get("proposal_gate_enabled", False):
            try:
                gate_params = dynamic_spec_config.get("proposal_gate_params", {})
                if not isinstance(gate_params, dict):
                    raise TypeError("proposal_gate_params must be a dict")
                accepted = {
                    key: value
                    for key, value in gate_params.items()
                    if key
                    in {
                        "enter_ratio",
                        "exit_ratio",
                        "max_avg_scheduled_tokens",
                        "enter_steps",
                        "exit_steps",
                    }
                }
                self._ascend_proposal_gate = ProposalGate(
                    max_num_seqs=getattr(self, "max_num_running_reqs", 1),
                    **accepted,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Ascend proposal gate compatibility patch: %s",
                    exc,
                )

    _patched_init._vllm_ascend_dynamic_gate_patched = True  # type: ignore[attr-defined]
    Scheduler.__init__ = _patched_init

    # The copied Ascend scheduler variants call
    # ``_apply_ascend_proposal_gate`` directly.  The upstream base Scheduler
    # does not, yet it is the scheduler selected by the default V2 launch.
    # Select the next width BEFORE AsyncScheduler creates its placeholders.
    # Capping after schedule() returns changes the worker's draft width but
    # leaves the next request.spec_token_ids at max K. The base hook is called
    # by AsyncScheduler before it consumes num_spec_tokens_to_schedule.
    original_update = Scheduler._update_after_schedule
    if not getattr(original_update, "_vllm_ascend_physical_k_patched", False):

        @wraps(original_update)
        def _patched_update_after_schedule(self, scheduler_output):
            controller = getattr(self, "_ascend_physical_k_controller", None)
            if controller is None:
                return original_update(self, scheduler_output)

            configured_k = int(getattr(scheduler_output, "num_spec_tokens_to_schedule", 0))
            selected_k = controller.cap(configured_k)
            scheduler_output.num_spec_tokens_to_schedule = selected_k
            log_count = getattr(self, "_ascend_physical_k_schedule_log_count", 0)
            if log_count < 16:
                logger.warning(
                    "V2 physical-K dispatch #%d: configured_k=%d selected_k=%d",
                    log_count + 1,
                    configured_k,
                    selected_k,
                )
                self._ascend_physical_k_schedule_log_count = log_count + 1
            return original_update(self, scheduler_output)

        _patched_update_after_schedule._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler._update_after_schedule = _patched_update_after_schedule


def _observe_physical_k(
    self,
    scheduler_output,
    model_runner_output,
) -> None:
    controller = getattr(self, "_ascend_physical_k_controller", None)
    if controller is None:
        return
    scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    req_ids = getattr(model_runner_output, "req_ids", ())
    sampled = getattr(model_runner_output, "sampled_token_ids", None)
    if sampled is None or len(req_ids) != len(sampled):
        return
    widths = [len(scheduled.get(req_id, ())) for req_id in req_ids]
    before = controller.current_k
    controller.observe(widths, sampled)
    log_count = getattr(self, "_ascend_physical_k_observe_log_count", 0)
    if controller.observation_count and log_count < 16:
        logger.warning(
            "V2 physical-K feedback #%d: scheduled=%s accepted=%s previous_k=%s next_k=%s",
            log_count + 1,
            controller.last_scheduled_widths,
            controller.last_accepted_lengths,
            before,
            controller.current_k,
        )
        logger.warning(
            "V2 physical-K policy reason #%d: %s",
            log_count + 1,
            controller.last_reason,
        )
        self._ascend_physical_k_observe_log_count = log_count + 1


def _update_proposal_lengths(self, scheduler_output, model_runner_output) -> None:
    lengths = getattr(model_runner_output, "proposal_lengths", None)
    if lengths is None:
        _observe_physical_k(self, scheduler_output, model_runner_output)
        return
    log_count = getattr(self, "_ascend_proposal_lengths_log_count", 0)
    if log_count < 8:
        logger.warning(
            "V2 hardware-aware K consumed by scheduler #%d: reqs=%d lengths=%s",
            log_count + 1,
            len(getattr(model_runner_output, "req_ids", ())),
            lengths,
        )
        self._ascend_proposal_lengths_log_count = log_count + 1
    req_ids = getattr(model_runner_output, "req_ids", ())
    if len(req_ids) != len(lengths):
        logger.warning(
            "Ignoring malformed proposal_lengths: %d request ids vs %d lengths",
            len(req_ids),
            len(lengths),
        )
        return
    latest = getattr(self, "_latest_proposal_lengths", None)
    if latest is None:
        latest = self._latest_proposal_lengths = {}
    for req_id, length in zip(req_ids, lengths):
        length = max(int(length), 0)
        latest[req_id] = length
        request = getattr(self, "requests", {}).get(req_id)
        if request is None or request.is_finished():
            continue
        if request.is_prefill_chunk:
            request.spec_token_ids = []
        else:
            request.spec_token_ids = [-1] * length

    controller = getattr(self, "_ascend_physical_k_controller", None)
    if controller is not None:
        controller.update(lengths)


def update_dynamic_feedback(scheduler, scheduler_output, model_runner_output, *, native_proposal_lengths: bool) -> None:
    """Consume completed output without changing current-step accounting."""
    if native_proposal_lengths:
        controller = getattr(scheduler, "_ascend_physical_k_controller", None)
        lengths = getattr(model_runner_output, "proposal_lengths", None)
        if controller is not None and lengths is not None:
            controller.update(lengths)
        elif controller is not None:
            _observe_physical_k(scheduler, scheduler_output, model_runner_output)
    else:
        _update_proposal_lengths(scheduler, scheduler_output, model_runner_output)


def trim_proposal_tokens(req_ids, draft_token_ids, lengths_by_req):
    """Align CPU proposal lengths by request ID, preserving fallback widths."""
    lengths_by_req = lengths_by_req or {}
    lengths = [lengths_by_req.get(req_id, len(tokens)) for req_id, tokens in zip(req_ids, draft_token_ids)]
    return [tokens[: max(0, min(int(k), len(tokens)))] for tokens, k in zip(draft_token_ids, lengths)], lengths
