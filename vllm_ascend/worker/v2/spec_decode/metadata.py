# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-metadata compatibility helpers for V2 speculative decoding."""

from __future__ import annotations

from typing import Any


def align_padded_query_lengths(
    attn_metadata: dict[str, Any],
    num_tokens_padded: int,
) -> None:
    """Make TND query lengths cover the padded FULL-graph query tensor.

    Upstream DSpark/DFlash builds a uniform query-start array from the number
    of real requests. Ascend FULL graphs replay a padded tensor with
    ``num_reqs_padded * num_query_per_req`` query rows. When the graph batch
    has an extra dummy request, the upstream array ends at the real token
    count while the query tensor has the padded rows. FIA rejects this as
    ``queryT != actualSequenceLengthQ[-1]``. The dummy rows are already
    masked/padded by the graph manager; only the cumulative Q length needs to
    include them.

    Non-padded batches are unchanged. The helper intentionally updates the
    Ascend metadata object instead of the upstream vLLM checkout.
    """

    for metadata in attn_metadata.values():
        actual = getattr(metadata, "actual_seq_lengths_q", None)
        if not actual:
            continue
        if int(actual[-1]) >= num_tokens_padded:
            continue
        padded = list(actual)
        padded[-1] = int(num_tokens_padded)
        metadata.actual_seq_lengths_q = padded
