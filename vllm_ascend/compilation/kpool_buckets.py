# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only length bounds for graph-updatable GLM KPool selection."""

from dataclasses import dataclass

MIN_KPOOL_SCORE_BUCKET = 2048


def kpool_score_bucket(length: int | None, capacity: int, selected: int) -> int:
    """Round an authoritative pool-count bound up, retaining the full fallback.

    Missing bounds must never truncate history. The final bucket need not be a
    power of two: block-table padding can extend beyond the model's limit.
    """
    if capacity < selected or selected <= 0:
        raise ValueError("KPool capacity must cover the positive selection width.")
    if length is None:
        return capacity
    if length < 0 or length > capacity:
        raise ValueError("KPool length bound must be within cache capacity.")
    required = max(length, selected, MIN_KPOOL_SCORE_BUCKET)
    return min(capacity, 1 << (required - 1).bit_length())


@dataclass(frozen=True)
class KPoolBucketProvider:
    layer_name: str
    capacity: int
    selected: int
    token_offset: int = 0

    def resolve(self, context) -> dict[str, int]:
        metadata = context[self.layer_name]
        return {"max_pool_seq_len": kpool_score_bucket(metadata.max_pool_seq_len, self.capacity, self.selected)}
