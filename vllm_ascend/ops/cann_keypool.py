# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State gather for CANN KeyPool."""

import torch


def _cache_rows(cache):
    width = cache.shape[-1]
    if cache.stride(0) % width:
        raise ValueError("KeyPool cache pages must align to complete rows.")
    rows = (cache.shape[0] - 1) * (cache.stride(0) // width) + cache.shape[1]
    return cache.as_strided((rows, width), (width, 1))


def prepare_keypool_state(tail, starts, tail_metadata, pool):
    n, width = starts.numel(), tail.shape[-1]
    inputs = tail_metadata.cann_inputs
    if inputs.page_rows * width != tail.stride(0):
        raise ValueError("KeyPool tail metadata must preserve the physical cache page stride.")
    state = tail.new_empty((1 + 2 * n, pool, width))
    torch.index_select(
        _cache_rows(tail), 0, inputs.history_indices.flatten(), out=state[1 : n + 1].view(n * pool, width)
    )
    return state


def gather_retained_hidden(hidden, indices):
    return torch.index_select(hidden, 0, indices.flatten())
