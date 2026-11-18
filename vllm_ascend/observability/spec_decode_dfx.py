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

import bisect
import logging

import torch
from vllm.logger import logger as _vllm_logger

# Reuse the shared 'vllm' namespace logger: vllm's handler tree only
# covers that namespace; a module-local logger inside vllm_ascend would
# have no handler and silently drop every record (the repository
# logger-convention check enforces this).
_logger = _vllm_logger

_NAN_ROW_FMT_LIMIT = 8


def _fmt_rows(rows_counts: list[tuple[int, int, int]], unit: str) -> str:
    """Render (owner_id, step, nan_count) triples.

    Both host transfers are batched upstream: indexing ``nan_per_row`` per
    element inside the loop would pay one D2H sync per iteration (up to
    ``_NAN_ROW_FMT_LIMIT``); slicing first and copying the whole batch
    collapses it to two syncs (one per list).
    """
    parts = [f"{unit}={r} step={s} nan={c}" for r, s, c in rows_counts]
    return "[" + ", ".join(parts) + "]"


def _uniform_coords(bad_rows_list: list[int], nan_counts: list[int], rows_per_req: int) -> list[tuple[int, int, int]]:
    """Row coordinates for a uniform batch: every request occupies exactly
    ``rows_per_req`` rows."""
    return [(i // rows_per_req, i % rows_per_req, int(c)) for i, c in zip(bad_rows_list, nan_counts)]


def _packed_coords(
    bad_rows_list: list[int],
    nan_counts: list[int],
    boundaries: list[int],
) -> list[tuple[int, int, int]]:
    """Row coordinates from inclusive-cumsum row boundaries.

    ``target_logits`` is packed with ``cu_num_logits``: a request occupies
    ``end - start`` rows, which is NOT always ``num_speculative_steps + 1``
    (mixed prefill/decode batches, requests with fewer scheduled drafts).
    Row ``i`` belongs to request ``bisect_right(boundaries, i) - 1`` and its
    step offset is ``i - boundaries[r]``.
    """
    coords = []
    for i, c in zip(bad_rows_list, nan_counts):
        req = bisect.bisect_right(boundaries, i) - 1
        coords.append((req, i - boundaries[req], int(c)))
    return coords


def log_nan_census(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    num_speculative_steps: int,
    cu_num_logits: torch.Tensor | None = None,
) -> None:
    """[dfx/nan] Per-row NaN census of the verification inputs.

    Row-level reductions (logsumexp, probability mass) turn a single NaN
    element into a whole-row verdict failure -- the acceptance test
    ``target_log_prob > tl.log(u) + draft_log_prob`` is False whenever
    either side is NaN -- and the resample zero-mass fallback then emits a
    degenerate but legal token (id 0). A single census record therefore
    identifies exactly which request/step was poisoned.

    ``cu_num_logits`` (inclusive cumsum over rows per request) should be
    passed whenever available: without it the coordinates assume a uniform
    ``num_speculative_steps + 1`` layout, which misassigns rows in mixed
    prefill/decode or shorter-draft batches.

    DEBUG-gated: INFO pays one integer compare per step; DEBUG pays one
    device reduction + D2H sync per input tensor (diagnostic mode only).
    """
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    try:
        nan_per_row = torch.isnan(target_logits).sum(dim=-1)
        bad_rows = nan_per_row.nonzero(as_tuple=True)[0]
        if bad_rows.numel() > 0:
            limit_rows = bad_rows[:_NAN_ROW_FMT_LIMIT]
            bad_rows_list = limit_rows.tolist()
            bad_counts_list = nan_per_row[limit_rows].tolist()
            if cu_num_logits is not None:
                boundaries = cu_num_logits.tolist() if torch.is_tensor(cu_num_logits) else list(cu_num_logits)
                coords = _packed_coords(bad_rows_list, bad_counts_list, boundaries)
            else:
                coords = _uniform_coords(bad_rows_list, bad_counts_list, num_speculative_steps + 1)
            _logger.debug(
                "[dfx/nan] target_logits: rows_with_nan=%d/%d total_nan=%d %s",
                bad_rows.numel(),
                target_logits.shape[0],
                int(nan_per_row.sum()),
                _fmt_rows(coords, "req"),
            )
        if draft_logits is not None and draft_logits.numel() > 1:
            d_nan = torch.isnan(draft_logits).sum(dim=-1).view(-1)
            d_bad = d_nan.nonzero(as_tuple=True)[0]
            if d_bad.numel() > 0:
                limit_rows = d_bad[:_NAN_ROW_FMT_LIMIT]
                d_counts = d_nan[limit_rows].tolist()
                coords = _uniform_coords(limit_rows.tolist(), d_counts, num_speculative_steps)
                _logger.debug(
                    "[dfx/nan] draft_logits: rows_with_nan=%d total_nan=%d %s",
                    d_bad.numel(),
                    int(d_nan.sum()),
                    _fmt_rows(coords, "slot"),
                )
    except Exception as e:  # diagnostics must never take down the sampler
        _logger.debug("[dfx/nan] target/draft census failed: %s", e)


def log_accept0_census(
    num_sampled: torch.Tensor,
    num_reqs: int,
    cu_num_logits: torch.Tensor | None = None,
) -> None:
    """[dfx/nan] Post-resample accepted-count census.

    ``num_sampled`` stores ``resample_idx + 1`` unconditionally for every
    request (finalize kernel), so ``num_sampled == 1`` marks a request whose
    drafts were ALL rejected this step -- but only for requests that
    actually had drafts: a request with no scheduled drafts (a single row
    under packed ``cu_num_logits``) also stores ``num_sampled == 1`` with
    nothing to reject. When boundaries are given, both the numerator and
    the denominator are restricted to requests with ``cu[i+1] - cu[i] > 1``.

    Correlate with the input NaN census of the same step: ``rows_with_nan``
    rising together with ``accept0`` confirms the NaN -> forced-reject ->
    garbage-token causality in production; NaN present but accept0 normal
    means the row-level NaN did NOT reach the verdict (reduction NaN
    semantics evidence).
    """
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    try:
        ns_list = num_sampled.tolist()
        zero_accept_all = [i for i, n in enumerate(ns_list) if n == 1]
        if cu_num_logits is not None:
            bounds = cu_num_logits.tolist() if torch.is_tensor(cu_num_logits) else list(cu_num_logits)
            draft_reqs = [i for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 1]
            zero_accept = [i for i in zero_accept_all if i in draft_reqs]
            denom = len(draft_reqs)
            suffix = "(had_drafts)"
        else:
            zero_accept = zero_accept_all
            denom = num_reqs
            suffix = ""
        if zero_accept:
            _logger.debug(
                "[dfx/nan] post-resample: accept0_reqs=%d/%d%s req_ids_first8=%s",
                len(zero_accept),
                denom,
                suffix,
                zero_accept[:_NAN_ROW_FMT_LIMIT],
            )
    except Exception as e:  # diagnostics must never take down the sampler
        _logger.debug("[dfx/nan] accept0 census failed: %s", e)
