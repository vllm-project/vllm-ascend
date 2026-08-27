# SPDX-License-Identifier: Apache-2.0
"""Async D2H selected-probs queue + controller cache update."""
from __future__ import annotations


from .controller import _dcut_enable_drafter_probs
from .globals import logger
from .utils import (
    _dcut_selected_probs_from_graph,
    _dcut_selected_probs_from_reused_logits,
)


def _dcut_bypass_prob_capture_for_prefill(self) -> None:
    """Disable D-Cut decision work while the target runs real prefill.

    Prefill-containing batches are never truncated and run through the
    native eager model path. Keeping ``needs_draft_probs`` enabled in that
    target path would retain state that belongs to the previous proposal.
    Drop that stale decision here. A mixed batch may re-enable probability
    capture later, immediately before its draft proposal, because those newly
    proposed tokens are consumed by the next decode-only verifier step. Pure
    prefill keeps probability capture disabled until decode resumes.
    """
    drafter = getattr(self, "drafter", None)
    if drafter is not None:
        if hasattr(drafter, "needs_draft_probs"):
            drafter.needs_draft_probs = False
        drafter._dcut_last_draft_ran_python = False
        drafter._dcut_last_logits_for_probs = None
        drafter._last_selected_probs = None

    self._adaptive_probs_pending = False
    self._adaptive_probs_expired = False
    self._adaptive_probs_source = "prefill_bypass"
    self._adaptive_probs_last_consumed_source = "prefill_bypass"
    self._adaptive_num_reqs = 0
    self._adaptive_req_ids = []
    self._adaptive_active = set()
    controller = getattr(self, "_verify_adaptive_controller", None)
    if controller is not None:
        controller.clear_adaptive_decision()


def _dcut_prepare_prob_capture(self, scheduler_output) -> None:
    """Reset per-step execution state before the drafter runs or replays."""
    drafter = getattr(self, "drafter", None)
    if drafter is not None:
        drafter._dcut_last_draft_ran_python = False
        drafter._dcut_current_graph_descriptor = None
        drafter._dcut_last_graph_prob_source = "none"
        # Graph replay updates the fixed-address tensor retained per bucket;
        # it does not reassign this Python attribute. Clear it so a replay can
        # never consume the final bucket captured during startup by accident.
        drafter._last_selected_probs = None
        # Force graph replay to select retained logits by the current graph
        # descriptor instead of reusing the final startup-capture bucket.
        drafter._dcut_last_logits_for_probs = None

    # The decision was consumed by the truncation immediately before this call.
    # Make every decision single-use so a failed probability capture cannot
    # silently apply the previous cap to another verifier step.
    controller = getattr(self, "_verify_adaptive_controller", None)
    if controller is not None:
        controller.clear_adaptive_decision()
    if not getattr(self, "_adaptive_probs_pending", False):
        self._adaptive_probs_source = "capture_pending"


def _dcut_probability_req_ids(self, num_reqs: int) -> list[str]:
    """Return the request IDs aligned with the captured draft-prob rows.

    ``_copy_draft_token_ids_to_cpu`` snapshots ``_draft_token_req_ids`` at
    proposal time. Use that same snapshot instead of re-classifying rows from
    ``num_computed_tokens_cpu``: a request that just finished prefill already
    has valid next-step draft tokens, while its computed-token bookkeeping can
    still look like prefill until the following runner iteration.
    """
    draft_req_ids = getattr(self, "_draft_token_req_ids", None)
    if draft_req_ids is None:
        draft_req_ids = self.input_batch.req_ids
    return list(draft_req_ids[:num_reqs])


def _dcut_queue_probs(self, zeros_only: bool) -> None:
    """Queue this step's selected_probs D2H (non-blocking) for next-step use.

    Device-agnostic apart from the async D2H copy + event record, which work
    the same on NPU (torch_npu supports non_blocking copies + npu.Event).
    """
    if (
        zeros_only
        or self._adaptive_probs_pending
        or self._adaptive_probs_pinned is None
        or self._adaptive_probs_event is None
    ):
        return
    self._adaptive_probs_source = "missing"
    _dcut_enable_drafter_probs(self)
    drafter = getattr(self, "drafter", None)
    if drafter is None or not hasattr(drafter, "take_last_selected_probs"):
        self._adaptive_probs_source = "no_selected_probs_hook"
        cnt = getattr(self, "_dcut_missing_probs_steps", 0) + 1
        self._dcut_missing_probs_steps = cnt
        if cnt <= 3 or cnt % 200 == 0:
            logger.warning(
                "D-Cut: drafter has no selected-probs hook; decision stats "
                "will not update (count=%s).",
                cnt,
            )
        return
    draft_token_ids = getattr(self, "_draft_token_ids", None)
    ran_python = getattr(drafter, "_dcut_last_draft_ran_python", False)
    if ran_python:
        probs = drafter.take_last_selected_probs()
        prob_source = "eager_python"
    else:
        probs = _dcut_selected_probs_from_graph(
            drafter,
            draft_token_ids,
        )
        prob_source = getattr(drafter, "_dcut_last_graph_prob_source", "missing")
        if (
            probs is not None
            and not getattr(
                self,
                "_dcut_logged_graph_selected_probs",
                False,
            )
        ):
            logger.info(
                "D-Cut: using selected probabilities produced by the "
                "replayed draft graph."
            )
            self._dcut_logged_graph_selected_probs = True
    if probs is None:
        if (
            not ran_python
            and not getattr(
                self,
                "_dcut_logged_graph_probs_fallback",
                False,
            )
        ):
            logger.warning(
                "D-Cut: replayed draft graph has no matching selected-prob "
                "buffer; trying the guarded logits compatibility path."
            )
            self._dcut_logged_graph_probs_fallback = True
        try:
            probs = _dcut_selected_probs_from_reused_logits(
                drafter,
                draft_token_ids,
            )
            if probs is not None:
                prob_source = "guarded_reused_logits"
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "D-Cut: deriving selected probs from reused logits failed: %s",
                e,
            )
            probs = None
    if probs is None:
        self._adaptive_probs_source = f"missing_{prob_source}"
        cnt = getattr(self, "_dcut_missing_probs_steps", 0) + 1
        self._dcut_missing_probs_steps = cnt
        if cnt <= 3 or cnt % 200 == 0:
            graph_ready = bool(
                getattr(drafter, "_dcut_graph_selected_probs_ready", False)
            )
            draft_shape = (
                tuple(getattr(draft_token_ids, "shape", ()))
                if draft_token_ids is not None
                else None
            )
            descriptor = getattr(
                drafter, "_dcut_current_graph_descriptor", None
            )
            logger.warning(
                "D-Cut: drafter did not expose selected draft probs; decision "
                "stats will not update (count=%s source=%s graph_ready=%s "
                "draft_shape=%s descriptor=%s).",
                cnt,
                prob_source,
                graph_ready,
                draft_shape,
                descriptor,
            )
        return
    num_reqs = self.input_batch.num_reqs
    num_spec = self.num_spec_tokens
    if probs.dim() == 1:
        needed = num_reqs * num_spec
        if probs.numel() < needed:
            self._adaptive_probs_source = "prob_length_mismatch"
            logger.warning(
                "D-Cut: selected draft probs too short: got=%s need=%s",
                probs.numel(),
                needed,
            )
            return
        probs = probs[:needed].view(num_reqs, num_spec)
    else:
        probs = probs[:num_reqs]
        if probs.shape[-1] != num_spec:
            self._adaptive_probs_source = "prob_shape_mismatch"
            logger.warning(
                "D-Cut: selected draft probs shape mismatch: shape=%s num_spec=%s",
                tuple(probs.shape),
                num_spec,
            )
            return
    prob_req_ids = _dcut_probability_req_ids(self, num_reqs)
    if len(prob_req_ids) != num_reqs:
        self._adaptive_probs_source = "request_id_mismatch"
        logger.warning(
            "D-Cut: draft probability request IDs are misaligned: "
            "got=%s expected=%s",
            len(prob_req_ids),
            num_reqs,
        )
        return
    self._adaptive_probs_pending = True
    self._adaptive_probs_expired = False
    self._adaptive_probs_source = prob_source
    self._adaptive_probs_generation = getattr(self, "_adaptive_probs_generation", 0) + 1
    self._adaptive_num_reqs = num_reqs
    self._adaptive_req_ids = prob_req_ids
    # Every probability row came from this proposal output. In particular,
    # include a request that completed prefill in this iteration; the scheduler
    # can place its freshly proposed tokens in the very next spec batch.
    self._adaptive_active = set(prob_req_ids)
    # Non-blocking D2H on the default stream (the drafter runs there too). The
    # next execute_model consumes it immediately before truncation, allowing
    # this copy to overlap the remainder of the current scheduler step.
    self._adaptive_probs_pinned[:num_reqs].copy_(
        probs.contiguous(),
        non_blocking=True,
    )
    self._adaptive_probs_event.record()


def _maybe_process_adaptive_probs(
    self,
    stage: str = "pre_truncate",
) -> None:
    """Consume queued probs before truncating the next verifier batch."""
    if not self._adaptive_probs_pending:
        return
    controller = self._verify_adaptive_controller
    assert self._adaptive_probs_event is not None
    if not self._adaptive_probs_event.query():
        if getattr(self, "_dcut_skip_unready_probs", False):
            self._adaptive_probs_expired = True
            if controller is not None:
                controller.clear_adaptive_decision()
            return
        # In the default pre_truncate path the copy has had the rest of the
        # previous iteration to complete. Synchronize only if it is still late,
        # so this step uses fresh probabilities and the next D2H queue is free.
        self._adaptive_probs_event.synchronize()
    self._adaptive_probs_pending = False
    if getattr(self, "_adaptive_probs_expired", False):
        self._adaptive_probs_expired = False
        self._adaptive_probs_source = "expired"
        if controller is not None:
            controller.clear_adaptive_decision()
        return
    self._adaptive_probs_last_consumed_source = getattr(
        self, "_adaptive_probs_source", "unknown"
    )
    self._adaptive_probs_last_consumed_generation = (
        getattr(self, "_adaptive_probs_generation", 0)
    )

    num_reqs = self._adaptive_num_reqs
    active = self._adaptive_active
    if active:
        current_req_ids = set(
            self.input_batch.req_ids[
                : getattr(self.input_batch, "num_reqs", 0)
            ]
        )
        active = active & current_req_ids
    if getattr(self, "_dcut_debug_stats_enabled", False):
        self._adaptive_probs_last_consumed_mean_by_position = []
        active_indices = [
            index
            for index, req_id in enumerate(
                self._adaptive_req_ids[:num_reqs]
            )
            if req_id in active
        ]
        if active_indices:
            assert self._adaptive_probs_pinned is not None
            mean_by_position = self._adaptive_probs_pinned[
                active_indices
            ].float().mean(dim=0)
            self._adaptive_probs_last_consumed_mean_by_position = [
                round(float(value), 6)
                for value in mean_by_position.tolist()
            ]
    if active and controller is not None:
        assert self._adaptive_probs_pinned is not None
        controller.process_draft_output(
            selected_probs=self._adaptive_probs_pinned[:num_reqs],
            req_ids=self._adaptive_req_ids,
            active_draft_req_ids=active,
            batch_size=num_reqs,
        )
    elif controller is not None:
        controller.clear_adaptive_decision()


def profile_adaptive_cost(self) -> None:
    """Profile verifier ITL after warmup (called from NPUWorker)."""
    if getattr(self, "_verify_adaptive_controller", None) is not None:
        self._verify_adaptive_controller.profile_cost_table(self)
