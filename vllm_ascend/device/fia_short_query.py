# SPDX-License-Identifier: Apache-2.0
"""Opt-in workaround for the reproduced A3 BF16 paged FIA short-query defect."""

import torch
import torch.nn.functional as F

_FIA_SHORT_QUERY_LIMIT = 16
_FIA_BAD_QUERY_MIN = 9
FIA_PADDED_QUERY_LENGTH = 32


def pad_short_fia_queries(query: torch.Tensor, cumulative_lengths: list[int]):
    """Prepend dummy queries while preserving every real query's causal position.

    For a request with q real queries and K cached tokens, query j attends up
    to K-q+j. Prepending p queries changes that to K-(q+p)+(j+p), identically.
    The extra output rows are discarded; no K/V or recurrent state is changed.
    CPU lengths are already supplied by the caller; no device read is needed.
    """
    if not isinstance(cumulative_lengths, (list, tuple)) or not cumulative_lengths:
        return None
    lengths = []
    previous = 0
    for end in cumulative_lengths:
        lengths.append(end - previous)
        previous = end
    if previous != query.shape[0] or any(n < 0 for n in lengths):
        return None
    if max(lengths) > _FIA_SHORT_QUERY_LIMIT or not any(
        _FIA_BAD_QUERY_MIN <= n < _FIA_SHORT_QUERY_LIMIT for n in lengths
    ):
        return None
    pieces, padded_lengths, slices = [], [], []
    source_start = padded_start = 0
    for n in lengths:
        padded_n = FIA_PADDED_QUERY_LENGTH if n else 0
        pad = padded_n - n
        pieces.append(F.pad(query[source_start : source_start + n], (0, 0, 0, 0, pad, 0)))
        slices.append(slice(padded_start + pad, padded_start + padded_n))
        source_start += n
        padded_start += padded_n
        padded_lengths.append(padded_start)
    return torch.cat(pieces, dim=0), padded_lengths, slices


def unpad_short_fia_output(output: torch.Tensor, slices: list[slice]) -> torch.Tensor:
    return torch.cat([output[part] for part in slices], dim=0)
