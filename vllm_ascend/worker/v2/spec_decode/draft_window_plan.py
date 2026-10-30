# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Scalar Draft Window intervals and projections, independent of device state.

All lengths and starts are absolute logical token counts. A table-column span
must be supplied by its actual owner/provider; scheduler block sizes, physical
cache blocks and compressed storage units are not interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


def _nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _positive_integer(name: str, value: int) -> None:
    _nonnegative_integer(name, value)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class DraftWindowLengthBounds:
    """An exact end and a distinct scheduling upper bound for the same row.

    Construct only when the CPU end is exact for the current preparation phase.
    An upper bound alone cannot construct this value or plan a read interval.
    """

    exact_end: int
    upper_end: int

    def __post_init__(self) -> None:
        _nonnegative_integer("exact_end", self.exact_end)
        _nonnegative_integer("upper_end", self.upper_end)
        if self.upper_end < self.exact_end:
            raise ValueError("upper_end must cover exact_end")

    def window_lengths(self, start: int) -> tuple[int, int]:
        _nonnegative_integer("start", start)
        if start > self.exact_end:
            raise ValueError("window start exceeds exact end")
        return self.exact_end - start, self.upper_end - start


@dataclass(frozen=True)
class DraftWindowPlan:
    full_exact_end: int
    full_horizon_end: int
    window_start: int
    window_capacity_tokens: int

    @property
    def window_visible_length(self) -> int:
        return self.full_exact_end - self.window_start

    def at_exact_end(self, exact_end: int) -> DraftWindowPlan:
        """Advance an Eagle step without moving its frozen proposal anchor.

        Rejection may decrease the exact end. The proposal's configured
        rejection bound must guarantee that it cannot move before the anchor.
        No truncation of queries or zero-length clamping is performed here.
        """
        _nonnegative_integer("exact_end", exact_end)
        if exact_end < self.window_start:
            raise ValueError("exact end after rejection precedes the frozen window start")
        if exact_end > self.full_horizon_end:
            raise ValueError("exact end exceeds the frozen proposal horizon")
        return replace(self, full_exact_end=exact_end)


def compute_draft_window_plan(
    *,
    full_exact_end: int,
    window_size: int,
    block_span_tokens: int,
    future_offset: int = 0,
) -> DraftWindowPlan:
    """Compute H=E+future, S=floor(max(H-W,0)/B)*B, V=E-S.

    ``block_span_tokens`` is the historical anchor's alignment, not a selected
    group's guessed physical block size. Projection onto other groups must
    preserve this S. Capacity is an allocation bound, never proof of residency.
    """
    _nonnegative_integer("full_exact_end", full_exact_end)
    _nonnegative_integer("future_offset", future_offset)
    _positive_integer("window_size", window_size)
    _positive_integer("block_span_tokens", block_span_tokens)
    horizon = full_exact_end + future_offset
    start = max(horizon - window_size, 0) // block_span_tokens * block_span_tokens
    if start > full_exact_end:
        raise ValueError("window start exceeds exact end; window is smaller than the proposal horizon requires")
    return DraftWindowPlan(full_exact_end, horizon, start, window_size + block_span_tokens - 1)


def freeze_eagle_window_plan(
    *,
    proposal_exact_end: int,
    static_k: int,
    max_rejected_tokens: int,
    window_size: int,
    block_span_tokens: int,
) -> DraftWindowPlan:
    """Freeze an anchor that remains valid through bounded rejection.

    A startup contract W >= K + R ensures S <= max(E-R,0) for every E,
    independent of block alignment. R must come from the actual first-pass
    rejection contract, not from a CPU upper bound. A violating configuration
    is rejected before serving, rather than failing on a later legal rejection.
    """
    _positive_integer("static_k", static_k)
    _nonnegative_integer("max_rejected_tokens", max_rejected_tokens)
    _positive_integer("window_size", window_size)
    if window_size < static_k + max_rejected_tokens:
        raise ValueError("Eagle3 draft_window_size must cover static K plus the maximum rejected tokens")
    return compute_draft_window_plan(
        full_exact_end=proposal_exact_end,
        window_size=window_size,
        block_span_tokens=block_span_tokens,
        future_offset=static_k,
    )


@dataclass(frozen=True)
class DirectPrefixProjection:
    source_start_column: int
    first_token_offset: int
    required_columns: int

    @property
    def source_end_column(self) -> int:
        return self.source_start_column + self.required_columns


def project_direct_prefix(
    *,
    window_start: int,
    full_exact_end: int,
    block_span_tokens: int,
    valid_columns: int,
    resident_start: int,
    resident_end: int,
    supports_first_token_offset: bool = False,
) -> DirectPrefixProjection:
    """Project a fixed logical window onto one direct-paged group.

    The source owner must separately certify the contiguous resident interval.
    Valid columns describe initialized table entries, not resident tokens. A
    native/recycling/striped/compressed layout needs its own provider rather
    than using this direct-prefix projection without proof.

    Different groups receive the same logical S. An unrepresentable offset is
    rejected; rounding S again or using an LCM would change window semantics.
    """
    for name, value in (
        ("window_start", window_start),
        ("full_exact_end", full_exact_end),
        ("valid_columns", valid_columns),
        ("resident_start", resident_start),
        ("resident_end", resident_end),
    ):
        _nonnegative_integer(name, value)
    _positive_integer("block_span_tokens", block_span_tokens)
    if full_exact_end < window_start:
        raise ValueError("window start exceeds exact end")
    if resident_start > resident_end:
        raise ValueError("resident interval is reversed")
    if window_start < resident_start or full_exact_end > resident_end:
        raise ValueError("requested read interval is not fully resident")
    column, offset = divmod(window_start, block_span_tokens)
    if offset and not supports_first_token_offset:
        raise NotImplementedError("group cannot represent the logical window start without a first-token offset")
    visible = full_exact_end - window_start
    columns = (offset + visible + block_span_tokens - 1) // block_span_tokens if visible else 0
    if columns and column + columns > valid_columns:
        raise ValueError("requested read interval exceeds valid source columns")
    return DirectPrefixProjection(column, offset, columns)
