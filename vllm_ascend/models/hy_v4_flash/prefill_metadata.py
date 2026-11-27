# SPDX-License-Identifier: Apache-2.0
"""Share immutable, shape-derived tensors across layers of a Prefill step."""

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available


def reuse_prefill_metadata(device, key, create):
    if device.type != "npu" or not is_forward_context_available():
        return create()
    context = get_forward_context()
    if context.cudagraph_runtime_mode is not CUDAGraphMode.NONE:
        return create()
    stream = torch.npu.current_stream(device).npu_stream
    cache = context.additional_kwargs.setdefault("hy4_prefill_metadata", {})
    key = (device, stream, *key)
    if key not in cache:
        cache[key] = create()
    return cache[key]


def logical_positions(first, end, device):
    return reuse_prefill_metadata(
        device, ("logical_positions", first, end), lambda: torch.arange(first, end, device=device)
    )


def _make_window_candidates(first, queries, width, device):
    starts = torch.arange(first, first + queries, dtype=torch.int32, device=device)
    offsets = torch.arange(width, dtype=torch.int32, device=device)
    return starts[:, None] + offsets


def window_indices(first, queries, width, device):
    return reuse_prefill_metadata(
        device,
        ("c8_window_indices", first, queries, width),
        lambda: _make_window_candidates(first, queries, width, device),
    )


def _make_sparse_lengths(num_pages, queries, length, device):
    table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    query_ends = torch.full((1,), queries, dtype=torch.int32, device=device)
    kv_lengths = torch.full((1,), length, dtype=torch.int32, device=device)
    return table, query_ends, kv_lengths


def sparse_lengths(num_pages, queries, length, device):
    return reuse_prefill_metadata(
        device,
        ("c8_sparse_lengths", num_pages, queries, length),
        lambda: _make_sparse_lengths(num_pages, queries, length, device),
    )
