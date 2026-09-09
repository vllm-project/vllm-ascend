# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in CPU-only diagnostics; INFO leaves upstream callables untouched."""

import logging
from functools import wraps
from time import perf_counter


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


def enable_draft_graph_debug(manager, logger) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    original = manager.dispatch
    logger.debug("ASCEND_DRAFT_CAPTURES descriptors=%s", manager._capture_descs)

    @wraps(original)
    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        # Only descriptor metadata is logged, never device tensor contents.
        logger.debug(
            "ASCEND_DRAFT_DISPATCH mode=%s tokens=%s batch=%s width=%s",
            getattr(result, "cg_mode", None),
            getattr(result, "num_tokens", None),
            getattr(result, "num_reqs", None),
            getattr(result, "uniform_token_count", None),
        )
        return result

    manager.dispatch = traced
