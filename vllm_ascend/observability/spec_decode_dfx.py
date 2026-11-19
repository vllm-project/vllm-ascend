# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative-decode DFX probes for the v2 rejection sampler.

Observation-only module: it owns no business logic and holds no state.
The rejection sampler carries exactly two call sites (input-side NaN
census as an issue/consume pair, output-side accept0 census); everything
diagnostic lives here so the instrumentation can be reviewed, gated, and
removed wholesale.

The NaN census is split in two phases to survive async scheduling:
``issue_nan_census`` runs at the rejection_sample entry, BEFORE any
verify kernel is enqueued, so its per-row NaN reductions enter the
stream ahead of every reader (and ahead of any next-step kernel that
reuses/overwrites the logits buffers under overlapped scheduling).
Stream order therefore guarantees the reductions observe THIS step's
logits regardless of how far ahead the scheduler runs. The reductions
produce fresh small tensors held by the returned pending object -- the
allocator cannot recycle them while referenced -- and ``consume_nan_census``
reads only those small tensors at the exit, never the (possibly already
recycled) logits buffers.

All probes are gated on ``logger.isEnabledFor(DEBUG)`` -- on the default
INFO path each call site pays one integer compare; under
``VLLM_LOGGING_LEVEL=DEBUG`` probes pay device reductions + D2H syncs
(diagnostic mode only). Every probe is wrapped in never-raise guards:
diagnostics must not take down the sampler.
"""

import bisect
import logging
from dataclasses import dataclass

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


@dataclass
class _NanCensusPending:
    """Stream-order snapshot of the input-side NaN census.

    Holds ONLY fresh small tensors produced by the entry reductions plus
    host scalars. The logits buffers themselves are deliberately not
    referenced: under async scheduling they may be recycled and rewritten
    by the next step before the exit runs, so consuming them at the exit
    would observe a foreign step's data (reviewer finding). The pending
    tensors are safe to defer -- they are our own allocations and stay
    valid as long as referenced.
    """

    # [num_target_rows] NaN counts per packed row (device, int64)
    target_nan_per_row: torch.Tensor
    # [num_target_rows] NaN counts per packed row, restricted to the ACTIVE
    # (slot, step) pairs the verify kernels read -- bonus rows (each
    # request's last row, which reads no draft) are zeroed. Or None when
    # no valid draft logits exist this step.
    draft_nan_per_row: torch.Tensor | None
    # [num_target_rows] draft-buffer slot of each packed row (device,
    # expanded_idx_mapping), or None when the active-pair restriction is
    # unavailable (legacy full-scan path).
    draft_slot_per_row: torch.Tensor | None
    num_speculative_steps: int
    num_target_rows: int
    # Reference only; boundaries are read (tolist) at consume time, which
    # sits after the verdict's sync -- same pattern as accept0 census.
    # cu_num_logits is per-step host-built metadata, not the recycled
    # logits buffer, so deferring the read is safe.
    cu_num_logits: torch.Tensor | None


def issue_nan_census(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    num_speculative_steps: int,
    cu_num_logits: torch.Tensor | None = None,
    draft_slot_per_row: torch.Tensor | None = None,
    draft_step_per_row: torch.Tensor | None = None,
    vocab_size: int | None = None,
    temperature: torch.Tensor | None = None,
) -> _NanCensusPending | None:
    """[dfx/nan] Phase 1 -- enqueue the per-row NaN reductions.

    Row-level reductions (logsumexp, probability mass) turn a single NaN
    element into a whole-row verdict failure -- the acceptance test
    ``target_log_prob > tl.log(u) + draft_log_prob`` is False whenever
    either side is NaN -- and the resample zero-mass fallback then emits a
    degenerate but legal token (id 0). A single census record therefore
    identifies exactly which request/step was poisoned.

    The census input set MUST equal the verifier's read set; three
    restrictions align it (a NaN the verifier never loads cannot be a
    verification input and must not be reported):

    * **Padded vocabulary** -- both logits tensors may be wider than the
      effective vocab; the verifier masks columns to ``vocab_size``
      (``block_offsets < vocab_size``). Pass ``vocab_size`` (= the
      min(target, draft) width computed by the caller) to slice both
      reductions to the effective columns; NaNs in padded columns are
      ignored.
    * **Greedy requests** -- for ``temperature == 0`` requests the
      verifier computes target argmax only and skips draft reads
      entirely. Pass ``temperature`` (``[max_num_reqs]``) to zero the
      draft counts of greedy rows (the target census keeps them: greedy
      still reads target logits for argmax).
    * **Active (slot, step) pairs** -- ``draft_logits`` is a PERSISTENT
      ``[max_num_reqs, K, V]`` buffer: only the slots selected by
      ``draft_slot_per_row`` (expanded_idx_mapping) at the steps
      scheduled by ``draft_step_per_row`` (expanded_local_pos) carry
      this step's drafts. The draft census GATHERS exactly those rows
      and zeroes bonus rows (each request's last row, which reads no
      draft -- mirroring the verifier's ``_is_bonus_logit``
      early-return). The gather also cuts the read traffic from
      ``max_num_reqs * K * V`` to ``num_rows * V``.

    MUST be called before the first verify kernel is enqueued: the
    reductions then enter the stream ahead of every logits reader, so
    stream order pins them to this step's data even when async scheduling
    recycles the logits buffers before the exit runs. Pure device work --
    no host sync, no host-data dependency, INFO-gated to zero cost.

    Returns None when the DEBUG gate is closed; returns a pending snapshot
    otherwise (always non-None when the caller reaches the exit).
    """
    if not _logger.isEnabledFor(logging.DEBUG):
        return None
    try:
        # Padded-vocabulary slice: a view, so the reduce reads only the
        # effective columns (zero-copy, stream-order safe).
        if vocab_size is not None and vocab_size < target_logits.shape[1]:
            target_input = target_logits[:, :vocab_size]
        else:
            target_input = target_logits
        target_nan_per_row = torch.isnan(target_input).sum(dim=-1)
        draft_nan_per_row = None
        slot_per_row = None
        if draft_logits is not None and draft_logits.numel() > 1:
            # The caller may pass a dummy 1-element tensor when there are
            # no draft logits (numel() == 1) -- skip it.
            if draft_slot_per_row is not None and draft_step_per_row is not None:
                # Active-pair gather: the packed row i reads draft row
                # [slot_i, step_i]. Bonus rows carry step == K (one past
                # the last draft step); the verifier never dereferences
                # them, so clamp the gather into bounds and zero their
                # counts -- the clamp reads a real neighbor row, the mask
                # discards the result.
                step = draft_step_per_row.clamp(max=draft_logits.shape[1] - 1)
                active = draft_logits[draft_slot_per_row, step]
                if vocab_size is not None and vocab_size < active.shape[1]:
                    active = active[:, :vocab_size]
                draft_nan_per_row = torch.isnan(active).sum(dim=-1)
                next_step = torch.cat((draft_step_per_row[1:], draft_step_per_row.new_zeros(1)))
                draft_nan_per_row = draft_nan_per_row * (next_step != 0)
                if temperature is not None:
                    # Greedy requests (temperature == 0) never read draft
                    # logits in the verifier; zero their draft counts.
                    temp_per_row = temperature[draft_slot_per_row]
                    draft_nan_per_row = draft_nan_per_row * (temp_per_row != 0)
                slot_per_row = draft_slot_per_row
            else:
                # Legacy path: no row mapping provided. The buffer layout
                # is padded and fixed, so a full scan with uniform slot
                # coordinates stays correct-shaped; the greedy mask cannot
                # be applied without per-row slots (production callers
                # always pass the mappings above).
                draft_input = draft_logits.view(-1, draft_logits.shape[-1])
                if vocab_size is not None and vocab_size < draft_input.shape[1]:
                    draft_input = draft_input[:, :vocab_size]
                draft_nan_per_row = torch.isnan(draft_input).sum(dim=-1).view(-1)
        return _NanCensusPending(
            target_nan_per_row=target_nan_per_row,
            draft_nan_per_row=draft_nan_per_row,
            draft_slot_per_row=slot_per_row,
            num_speculative_steps=num_speculative_steps,
            num_target_rows=target_logits.shape[0],
            cu_num_logits=cu_num_logits,
        )
    except Exception as e:  # diagnostics must never take down the sampler
        _logger.debug("[dfx/nan] target/draft census issue failed: %s", e)
        return None


def consume_nan_census(pending: _NanCensusPending | None) -> None:
    """[dfx/nan] Phase 2 -- read the entry snapshot and emit the records.

    Call AFTER the accept0 verdict of the same step. The verdict's own
    sync has by then drained the stream, so the entry reductions are
    complete and the small-tensor reads below pay only microsecond-scale
    D2H -- all after the verdict (observer-effect free). This function
    never touches the logits buffers: under async scheduling those may
    already hold the next step's data.

    ``cu_num_logits`` (inclusive cumsum over rows per request) should be
    present whenever available: without it the coordinates assume a
    uniform ``num_speculative_steps + 1`` layout, which misassigns rows in
    mixed prefill/decode or shorter-draft batches. With the active-pair
    restriction, draft rows are aligned with the packed target rows and
    carry req/step/slot coordinates; the legacy full-scan path keeps
    uniform slot coordinates.
    """
    if pending is None or not _logger.isEnabledFor(logging.DEBUG):
        return
    try:
        boundaries = None
        if pending.cu_num_logits is not None:
            boundaries = (
                pending.cu_num_logits.tolist()
                if torch.is_tensor(pending.cu_num_logits)
                else list(pending.cu_num_logits)
            )
        nan_per_row = pending.target_nan_per_row
        bad_rows = nan_per_row.nonzero(as_tuple=True)[0]
        if bad_rows.numel() > 0:
            limit_rows = bad_rows[:_NAN_ROW_FMT_LIMIT]
            bad_rows_list = limit_rows.tolist()
            bad_counts_list = nan_per_row[limit_rows].tolist()
            if boundaries is not None:
                coords = _packed_coords(bad_rows_list, bad_counts_list, boundaries)
            else:
                coords = _uniform_coords(bad_rows_list, bad_counts_list, pending.num_speculative_steps + 1)
            _logger.debug(
                "[dfx/nan] target_logits: rows_with_nan=%d/%d total_nan=%d %s",
                bad_rows.numel(),
                pending.num_target_rows,
                int(nan_per_row.sum()),
                _fmt_rows(coords, "req"),
            )
        if pending.draft_nan_per_row is not None:
            d_nan = pending.draft_nan_per_row
            d_bad = d_nan.nonzero(as_tuple=True)[0]
            if d_bad.numel() > 0:
                limit_rows = d_bad[:_NAN_ROW_FMT_LIMIT]
                rows_list = limit_rows.tolist()
                d_counts = d_nan[limit_rows].tolist()
                if pending.draft_slot_per_row is not None and boundaries is not None:
                    # Active-pair layout: draft rows are aligned with the
                    # packed target rows. Coordinates mirror the target
                    # census (req/step via boundaries) plus the persistent
                    # buffer slot, so a poisoned row can be traced to both
                    # the request and its buffer cell.
                    slot_list = pending.draft_slot_per_row[limit_rows].tolist()
                    coords = []
                    for i, c, slot in zip(rows_list, d_counts, slot_list):
                        req = bisect.bisect_right(boundaries, i) - 1
                        coords.append((req, i - boundaries[req], slot, c))
                    parts = [f"req={r} step={s} slot={o} nan={c}" for r, s, o, c in coords]
                    coord_str = "[" + ", ".join(parts) + "]"
                else:
                    # Legacy full-scan path: rows are per-slot, uniform K
                    # steps per slot.
                    coord_str = _fmt_rows(_uniform_coords(rows_list, d_counts, pending.num_speculative_steps), "slot")
                _logger.debug(
                    "[dfx/nan] draft_logits: rows_with_nan=%d total_nan=%d %s",
                    d_bad.numel(),
                    int(d_nan.sum()),
                    coord_str,
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
