# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend PIECEWISE profiling adapters for the upstream verification manager.

Confidence estimation and prefix allocation remain upstream-owned. The runner
keeps explicit compact/reallocate call sites because their order relative to
input preparation and attention metadata construction is part of the contract.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from types import MethodType

import numpy as np
from vllm.logger import logger


def configure_piecewise_manager(
    manager,
):
    """Make one upstream manager consume Ascend PIECEWISE timings.

    Upstream seeds its step-cost tables from FULL-decode-graph dummy runs
    (``full_cudagraph=True`` samples) and profile sizes derived from the
    captured full-graph token counts. Under Plan A the target decode runs
    through PIECEWISE graphs, so there are no full graphs to price: profile
    a representative grid of piecewise batch sizes instead and price the
    drafter curve from every sample (not only ``full_cudagraph`` ones).

    Keep the object returned by the upstream factory. It owns the
    confidence buffers, copy stream, events, and validation state used by
    the rest of #47808; constructing and discarding it would allocate all
    of those resources twice during startup.
    """

    def batches_to_profile(self, capture_sizes):
        del capture_sizes
        # No FULL graphs: leave ``_cudagraph_limit`` at 0 so the cost
        # tables stay smooth (nothing pads to a captured size).
        self._cudagraph_limit = 0
        max_num_tokens = self.req_states.max_num_batched_tokens
        base_size = max(1, self.num_speculative_steps + 1)
        grid = [base_size]
        while grid[-1] < max_num_tokens:
            grid.append(min(grid[-1] * 2, max_num_tokens))
        from vllm import envs

        context_len = envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN
        for num_tokens in grid:
            for _ in range(3):
                yield {
                    "num_tokens": num_tokens,
                    "context_len": context_len,
                }

    def set_initial_cost_curves(self, samples):
        from collections import defaultdict

        def median_curve(points):
            grouped: dict[int, list[float]] = defaultdict(list)
            for key, value in points:
                grouped[key].append(value)
            return [(k, float(np.median(v))) for k, v in sorted(grouped.items())]

        draft_curve = median_curve((s.num_reqs, s.drafter_ms) for s in samples)
        verify_curve = median_curve((s.num_target_tokens, s.forward_ms) for s in samples)
        self.set_cost_curves(draft_curve, verify_curve)
        logger.debug("ASCEND_AV_COST_CURVES draft=%s verify=%s", draft_curve, verify_curve)

    manager.batches_to_profile = MethodType(  # type: ignore[method-assign]
        batches_to_profile, manager
    )
    manager.set_initial_cost_curves = MethodType(  # type: ignore[method-assign]
        set_initial_cost_curves, manager
    )
    enable_budget_debug(manager, logger)
    return manager


@contextmanager
def adaptive_verification_gate_wrapper(runner_module):
    """Relax the upstream ``AttentionCGSupport.ALWAYS`` requirement on Ascend.

    Upstream adaptive verification captures varlen FULL decode graphs, so its
    factory refuses to create the manager unless every attention builder
    reports ``AttentionCGSupport.ALWAYS`` (``adaptive_verification.py``).
    Ascend attention backends only report ``UNIFORM_BATCH`` today, which would
    make ``enable_adaptive_verification=true`` fail at startup. Under Plan A
    the decode runs through PIECEWISE graphs (see ``graph_manager_wrapper``),
    so the ALWAYS hard gate is relaxed here while every other upstream
    validation (device/CPU query-len mismatch support, etc.) still runs.
    """
    original_factory = getattr(runner_module, "maybe_create_adaptive_verification_manager", None)
    if original_factory is None:
        yield
        return

    # Keep the allocator algorithm from vLLM PR #47808, but do not use its
    # torch.compile wrapper on Ascend.  The compiled NPU graph corrupts the
    # in-place ``capacities`` result for dynamic request counts (for example a
    # budget of 2 has produced [1, 0, 10]); the identical eager function has
    # exact budget conservation across the same NPU shape matrix.  Runtime
    # ``reallocate_drafts`` resolves this module global on every call, so the
    # Ascend plugin can replace only the execution wrapper without forking the
    # confidence or prefix-allocation logic.
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive_mod

    if adaptive_mod._assign_draft_token_budget_compiled is not adaptive_mod._assign_draft_token_budget:
        adaptive_mod._assign_draft_token_budget_compiled = adaptive_mod._assign_draft_token_budget
        logger.warning(
            "Adaptive verification on Ascend uses the upstream eager prefix "
            "allocator because its torch.compile wrapper corrupts dynamic "
            "capacity outputs on NPU."
        )

    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )

    def relaxed_factory(
        *,
        enable_adaptive_verification: bool,
        attn_groups,
        attn_cg_support,
        req_states,
        query_start_loc,
        num_bonus_tokens,
        max_total_logits,
        **factory_kwargs,
    ):
        # Keep upstream validation inputs (config, target layer names and
        # additional attention support) intact; only relax the graph gate.
        if not enable_adaptive_verification:
            return original_factory(
                enable_adaptive_verification=enable_adaptive_verification,
                attn_groups=attn_groups,
                attn_cg_support=attn_cg_support,
                req_states=req_states,
                query_start_loc=query_start_loc,
                num_bonus_tokens=num_bonus_tokens,
                max_total_logits=max_total_logits,
                **factory_kwargs,
            )
        try:
            manager = original_factory(
                enable_adaptive_verification=enable_adaptive_verification,
                attn_groups=attn_groups,
                attn_cg_support=attn_cg_support,
                req_states=req_states,
                query_start_loc=query_start_loc,
                num_bonus_tokens=num_bonus_tokens,
                max_total_logits=max_total_logits,
                **factory_kwargs,
            )
        except ValueError as exc:
            # Only the ALWAYS requirement is relaxed on Ascend; any other
            # validation failure must keep failing loudly.
            if "AttentionCGSupport.ALWAYS" not in str(exc):
                raise
            logger.warning(
                "Relaxing the adaptive-verification AttentionCGSupport.ALWAYS "
                "gate for Ascend; decode runs through PIECEWISE graphs: %s",
                exc,
            )
            manager = None

        # Preserve the upstream manager whenever validation succeeds. If only
        # the ALWAYS gate rejected Ascend, instantiate the same upstream class
        # once, then configure either instance for PIECEWISE profiling.
        if manager is None:
            manager = AdaptiveVerificationManager(
                req_states,
                query_start_loc,
                num_bonus_tokens,
                max_total_logits=max_total_logits,
            )
        logger.info("Using the upstream adaptive-verification manager with Ascend PIECEWISE cost profiling.")
        return configure_piecewise_manager(manager)

    try:
        runner_module.maybe_create_adaptive_verification_manager = relaxed_factory
        yield
    finally:
        runner_module.maybe_create_adaptive_verification_manager = original_factory


def enable_budget_debug(manager, logger) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    original = manager.get_num_tokens

    @wraps(original)
    def traced(*args, **kwargs):
        start = perf_counter()
        result = original(*args, **kwargs)
        state = manager._batch_budget
        if state is not None:
            capacities, non_drafts, budget = state
            logger.debug(
                "ASCEND_AV_BUDGET batch=%d available=%d budget=%d min_k=%d max_k=%d non_drafts=%d cpu_ms=%.3f",
                len(capacities),
                sum(capacities.values()),
                budget,
                min(capacities.values(), default=0),
                max(capacities.values(), default=0),
                sum(non_drafts.values()),
                (perf_counter() - start) * 1000,
            )
        return result

    manager.get_num_tokens = traced
