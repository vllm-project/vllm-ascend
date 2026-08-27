# SPDX-License-Identifier: Apache-2.0
"""Patch NPUWorker: warmup hook + cost profiling trigger."""
from __future__ import annotations

from .globals import logger
from .patch_full_graph import _dcut_validate_gdn_full_graph_weights


def _patch_worker() -> None:
    import vllm_ascend.worker.worker as m

    W = m.NPUWorker
    if getattr(W, "_dcut_patched", False):
        return

    _orig = W.compile_or_warm_up_model

    def compile_or_warm_up_model(self, *a, **k):
        runner = getattr(self, "model_runner", None)
        if runner is not None and hasattr(runner, "_dcut_enable_drafter_probs"):
            try:
                runner._dcut_enable_drafter_probs()
            except Exception as e:
                logger.warning("D-Cut: enabling draft probs before warmup failed: %s", e)
        if runner is not None:
            dropped = _dcut_drop_pre_warmup_draft_graphs(runner)
            if dropped:
                logger.warning(
                    "D-Cut: dropped %d draft graph(s) captured before "
                    "probability collection; recapturing during warmup.",
                    dropped,
                )
        if runner is not None:
            _dcut_validate_gdn_full_graph_weights(runner)
        # Mark that we're in the REAL warmup (not profile_cudagraph_memory).
        # _build_attention_metadata patch checks this to force use_spec_decode=True
        # only during real warmup, not during profile_cudagraph_memory (which uses
        # a minimal KV cache that can't support spec-decode conv1d).
        if runner is not None:
            runner._dcut_in_real_warmup = True
        try:
            ret = _orig(self, *a, **k)
            if runner is not None and hasattr(runner, "profile_adaptive_cost"):
                try:
                    runner.profile_adaptive_cost()
                except Exception as e:
                    # Empty cost table => controller no-ops => graceful fall back to
                    # vanilla DFlash (full-length verify).
                    import traceback
                    logger.error("D-Cut: cost profiling failed; falling back: %s", e)
                    logger.error("D-Cut: full traceback: %s", traceback.format_exc())
                    ctrl = getattr(runner, "_verify_adaptive_controller", None)
                    if ctrl is not None:
                        ctrl._cost_table.clear()
                        ctrl._sorted_bs.clear()
                        ctrl._sorted_sql_per_bs.clear()
        finally:
            if runner is not None:
                runner._dcut_in_real_warmup = False
        return ret

    W.compile_or_warm_up_model = compile_or_warm_up_model
    W._dcut_patched = True
    logger.info("D-Cut: patched NPUWorker.")


def _dcut_drop_pre_warmup_draft_graphs(runner) -> int:
    """Drop draft graphs captured before probability collection was enabled."""
    drafter = getattr(runner, "drafter", None)
    if drafter is None or not getattr(drafter, "needs_draft_probs", False):
        return 0

    runnable = getattr(drafter, "_runnable", None)
    runnable_state = getattr(runnable, "__dict__", None)
    if not isinstance(runnable_state, dict):
        return 0
    entries = runnable_state.get("concrete_aclgraph_entries")
    if not entries:
        return 0

    by_descriptor = getattr(
        drafter,
        "_dcut_graph_selected_probs_by_descriptor",
        None,
    ) or {}
    stale_descriptors = [
        descriptor
        for descriptor, entry in entries.items()
        if getattr(entry, "aclgraph", None) is not None
        and by_descriptor.get(descriptor) is None
    ]
    for descriptor in stale_descriptors:
        entries.pop(descriptor, None)
    return len(stale_descriptors)
