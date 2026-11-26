# SPDX-License-Identifier: Apache-2.0
"""Host metadata for mask-free, single-query MLA FIA tasks."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import NamedTuple

import torch


class CompactFIAKV(NamedTuple):
    """Reader output whose valid rows already form a stable prefix."""

    nope: torch.Tensor
    rope: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, eq=False)
class FIALengthProvider:
    # Each captured call owns a provider, including repeated calls in one layer.
    layer_name: str
    row_start: int
    rows: int
    capacity: int
    window: int | None
    ratio: int = 4
    budget: int = 511
    graph_queries: int | None = None

    def resolve(self, context):
        meta = context[self.layer_name] if isinstance(context, dict) else context
        lengths = []
        for row in range(self.row_start, self.row_start + self.rows):
            if self.graph_queries is not None:
                request, offset = divmod(row, self.graph_queries)
                count = self.graph_queries
            else:
                request = bisect_right(meta.query_start_loc, row) - 1
                offset = row - meta.query_start_loc[request]
                count = meta.query_start_loc[request + 1] - meta.query_start_loc[request]
            if request >= len(meta.seq_lens):
                # Padded graph requests use a zero KV row and are zeroed at output.
                lengths.append(1)
                continue
            visible = meta.seq_lens[request] - count + offset
            if self.window is not None:
                length = min(visible + 1, self.window)
            elif visible // self.ratio > self.budget:
                length = self.budget * self.ratio + visible % self.ratio + 1
            else:
                length = visible + 1
            lengths.append(max(1, min(length, self.capacity)))
        return {"actual_seq_kvlen": lengths}


def compact_fia_rows(selected, nope, rope):
    """Stable-pack valid KV rows before supplying lengths instead of a mask.

    Sliding windows have leading padding; sparse selections can have internal
    holes. Dropping the mask without this permutation changes attention.
    All operations stay on device and are capturable.
    """
    import torch

    order = torch.argsort((selected < 0).to(torch.int32), dim=-1, stable=True)
    valid = selected.gather(1, order) >= 0

    def pack(value):
        index = order[:, :, None, None].expand(-1, -1, value.shape[2], value.shape[3])
        return torch.gather(value, 1, index).masked_fill(~valid[:, :, None, None], 0).contiguous()

    return pack(nope), pack(rope)
