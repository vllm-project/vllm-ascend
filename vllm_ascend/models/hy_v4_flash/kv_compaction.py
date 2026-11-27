# SPDX-License-Identifier: Apache-2.0
"""Stable device-side KV compaction for FP32 attention."""

from typing import NamedTuple

import torch


class CompactKV(NamedTuple):
    """Reader output whose valid rows already form a stable prefix."""

    nope: torch.Tensor
    rope: torch.Tensor
    mask: torch.Tensor


def compact_kv_rows(selected, nope, rope):
    """Stable-pack valid KV rows and zero padding for attention readers.

    Sliding windows have leading padding; sparse selections can have internal
    holes. Dropping the mask without this permutation changes attention.
    All operations stay on device and are capturable.
    """
    order = torch.argsort((selected < 0).to(torch.int32), dim=-1, stable=True)
    valid = selected.gather(1, order) >= 0

    def pack(value):
        index = order[:, :, None, None].expand(-1, -1, value.shape[2], value.shape[3])
        return torch.gather(value, 1, index).masked_fill(~valid[:, :, None, None], 0).contiguous()

    return pack(nope), pack(rope)
