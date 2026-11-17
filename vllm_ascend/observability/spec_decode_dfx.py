# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative-decode DFX probes for the v2 rejection sampler.

Observation-only module: it owns no business logic and holds no state.
The rejection sampler carries exactly two call sites (input-side NaN
census, output-side accept0 census); everything diagnostic lives here so
the instrumentation can be reviewed, gated, and removed wholesale.

All probes are gated on ``logger.isEnabledFor(DEBUG)`` -- on the default
INFO path each call site pays one integer compare; under
``VLLM_LOGGING_LEVEL=DEBUG`` probes pay device reductions + D2H syncs
(diagnostic mode only). Every probe is wrapped in never-raise guards:
diagnostics must not take down the sampler.
"""

import logging

import torch
from vllm.logger import logger as _vllm_logger

# Reuse the shared 'vllm' namespace logger: vllm's handler tree only
# covers that namespace, so init_logger(__name__) inside vllm_ascend
# would create an unhandled logger whose output is silently dropped
# (enforced by the repository logger-convention check).
_logger = _vllm_logger

_NAN_ROW_FMT_LIMIT = 8


def _fmt_nan_rows(bad_rows, nan_per_row, rows_per_req: int, unit: str) -> str:
    """Render (row_id, nan_count) pairs as req/step coordinates.

    Both host transfers are batched: indexing ``nan_per_row`` per element
    inside the loop would pay one D2H sync per iteration (up to
    ``_NAN_ROW_FMT_LIMIT``); slicing first and copying the whole batch
    collapses it to two syncs (one per list).
    """
    limit_rows = bad_rows[:_NAN_ROW_FMT_LIMIT]
    bad_rows_list = limit_rows.tolist()
    bad_counts_list = nan_per_row[limit_rows].tolist()
    parts = []
    for i, count in zip(bad_rows_list, bad_counts_list):
        parts.append(f"{unit}={i // rows_per_req} step={i % rows_per_req} nan={int(count)}")
    return "[" + ", ".join(parts) + "]"


def log_nan_census(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    num_speculative_steps: int,
) -> None:
    """[dfx/nan] Per-row NaN census of the verification inputs.

    Row-level reductions (logsumexp, probability mass) turn a single NaN
    element into a whole-row verdict failure -- the acceptance test
    ``target_log_prob > tl.log(u) + draft_log_prob`` is False whenever
    either side is NaN -- and the resample zero-mass fallback then emits a
    degenerate but legal token (id 0). A single census record therefore
    identifies exactly which request/step was poisoned.

    DEBUG-gated: INFO pays one integer compare per step; DEBUG pays one
    device reduction + D2H sync per input tensor (diagnostic mode only).
    """
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    try:
        rows_per_req = num_speculative_steps + 1
        nan_per_row = torch.isnan(target_logits).sum(dim=-1)
        bad_rows = nan_per_row.nonzero(as_tuple=True)[0]
        if bad_rows.numel() > 0:
            _logger.debug(
                "[dfx/nan] target_logits: rows_with_nan=%d/%d total_nan=%d %s",
                bad_rows.numel(),
                target_logits.shape[0],
                int(nan_per_row.sum()),
                _fmt_nan_rows(bad_rows, nan_per_row, rows_per_req, "req"),
            )
        if draft_logits is not None and draft_logits.numel() > 1:
            d_nan = torch.isnan(draft_logits).sum(dim=-1).view(-1)
            d_bad = d_nan.nonzero(as_tuple=True)[0]
            if d_bad.numel() > 0:
                _logger.debug(
                    "[dfx/nan] draft_logits: rows_with_nan=%d total_nan=%d %s",
                    d_bad.numel(),
                    int(d_nan.sum()),
                    _fmt_nan_rows(d_bad, d_nan, num_speculative_steps, "slot"),
                )
    except Exception as e:  # diagnostics must never take down the sampler
        _logger.debug("[dfx/nan] target/draft census failed: %s", e)


def log_accept0_census(num_sampled: torch.Tensor, num_reqs: int) -> None:
    """[dfx/nan] Post-resample accepted-count census.

    ``num_sampled`` stores ``resample_idx + 1`` unconditionally for every
    request (finalize kernel), so ``num_sampled == 1`` marks a request whose
    drafts were ALL rejected this step. Correlate with the input NaN census
    of the same step: ``rows_with_nan`` rising together with ``accept0``
    confirms the NaN -> forced-reject -> garbage-token causality in
    production; NaN present but accept0 normal means the row-level NaN did
    NOT reach the verdict (reduction NaN semantics evidence).
    """
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    try:
        zero_accept = (num_sampled == 1).nonzero(as_tuple=True)[0]
        if zero_accept.numel() > 0:
            _logger.debug(
                "[dfx/nan] post-resample: accept0_reqs=%d/%d req_ids_first8=%s",
                zero_accept.numel(),
                num_reqs,
                zero_accept[:_NAN_ROW_FMT_LIMIT].tolist(),
            )
    except Exception as e:  # diagnostics must never take down the sampler
        _logger.debug("[dfx/nan] accept0 census failed: %s", e)
