# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler integration and CPU proposal bookkeeping for dynamic physical K.

This module must remain importable during platform patch initialization. Policy
classes are loaded only when constructing a scheduler, not while importing the
platform (the spec_decode package loads model proposers).
PP/MTP still owns its output ordering; update_dynamic_feedback is called after
upstream accounting and before PP draft writeback.
"""

from __future__ import annotations

import os
from functools import wraps

from vllm.logger import logger

from vllm_ascend.dynamic_spec_config import resolve_method_params, v2_physical_k_enabled


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
                    from vllm_ascend.spec_decode.dynamic.policy import (
                        AdaptiveDraftKController,
                    )

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
                from vllm_ascend.spec_decode.dynamic.policy import ProposalGate

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
