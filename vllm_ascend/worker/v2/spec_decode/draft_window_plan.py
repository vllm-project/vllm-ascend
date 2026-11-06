# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Block-aligned Draft read geometry in absolute logical tokens.

Policy and consumer initialization validate W and the actual table-column
span. Callers supply completed integer exact ends, never scheduling uppers.
"""


def window_capacity_tokens(window_size: int, block_span_tokens: int) -> int:
    """Allocation bound for every start aligned to the consumer's block span."""
    return window_size + block_span_tokens - 1


def compute_draft_window_start(
    *,
    full_exact_end: int,
    window_size: int,
    block_span_tokens: int,
    future_offset: int = 0,
) -> int:
    """Return S=floor(max(E+future-W,0)/B)*B in absolute tokens."""
    start = max(full_exact_end + future_offset - window_size, 0) // block_span_tokens * block_span_tokens
    if start > full_exact_end:
        raise ValueError("window start exceeds exact end; window is smaller than the proposal horizon requires")
    return start
