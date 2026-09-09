# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-storage, batched writes for upstream DSpark's active-K results."""

import torch


def _request_count(key: slice, maximum: int) -> int:
    if not isinstance(key, slice) or key.start not in (None, 0) or key.step not in (None, 1):
        raise TypeError("unsupported request buffer index")
    count = maximum if key.stop is None else key.stop
    if not isinstance(count, int) or not 0 <= count <= maximum:
        raise IndexError("request count outside buffer")
    return count


class IndexedDraftTokenBuffer:
    """One device write per column, independent of the request count.

    Construct before graph capture against the full contiguous backing tensor.
    Indices and flat storage views are persistent across capture and replay.
    """

    def __init__(self, buffer: torch.Tensor):
        if buffer.ndim != 2 or not buffer.is_contiguous():
            raise ValueError("draft token backing buffer must be contiguous and 2-D")
        self._buffer = buffer
        self._flat = buffer.view(-1)
        rows = torch.arange(buffer.shape[0], device=buffer.device, dtype=torch.int64) * buffer.shape[1]
        self._columns = tuple(rows + col for col in range(buffer.shape[1]))

    def __setitem__(self, key, value) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("unsupported draft token buffer index")
        reqs, col = key
        count = _request_count(reqs, self._buffer.shape[0])
        if not isinstance(col, int) or not 0 <= col < self._buffer.shape[1]:
            raise IndexError("draft column outside buffer")
        self._flat.index_copy_(0, self._columns[col][:count], value.reshape(count).to(self._buffer.dtype))


class IndexedConfidenceBuffer:
    """Write an active prefix without flattening a noncontiguous narrow view."""

    def __init__(self, buffer: torch.Tensor, active_k: int):
        if buffer.ndim != 2 or not buffer.is_contiguous():
            raise ValueError("confidence backing buffer must be contiguous and 2-D")
        if not 1 <= active_k <= buffer.shape[1]:
            raise ValueError("active K outside confidence buffer")
        self._buffer = buffer
        self._active_k = active_k
        self._flat = buffer.view(-1)
        rows = torch.arange(buffer.shape[0], device=buffer.device, dtype=torch.int64) * buffer.shape[1]
        cols = torch.arange(active_k, device=buffer.device, dtype=torch.int64)
        self._indices = (rows[:, None] + cols).flatten()

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self):
        return (self._buffer.shape[0], self._active_k)

    def __setitem__(self, key, value) -> None:
        count = _request_count(key, self._buffer.shape[0]) * self._active_k
        self._flat.index_copy_(0, self._indices[:count], value.reshape(count).to(self._buffer.dtype))
