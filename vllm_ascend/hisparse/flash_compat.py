# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bound Flash attention views on the stock Core HiSparse binding path."""

from types import MethodType

import torch
from vllm.v1.hisparse.runtime import HiSparseCacheHandle, HiSparseRuntime


def _bound_slab(raw_tensor: torch.Tensor, runtime: HiSparseRuntime, placement: dict) -> torch.Tensor:
    pages = placement["num_blocks"]
    if pages:
        page_bytes = placement["block_size"] * runtime.row_width * runtime.kv_dtype.itemsize
        extent = (pages - 1) * placement["block_stride"] + page_bytes
    else:
        extent = 0
    return raw_tensor[: placement["byte_offset"] + extent]


def _bind_resident(self, raw_tensor, **placement):
    bounded = _bound_slab(raw_tensor, self.runtime, placement)
    HiSparseCacheHandle.bind_cache(self, bounded, **placement)


def _bind_hot(self, raw_tensor, **placement):
    bounded = _bound_slab(raw_tensor, self, placement)
    HiSparseRuntime.bind_hot_cache(self, bounded, **placement)
    # DMA addresses use the capacity of the original shared arena.
    self.hot_backing = raw_tensor


def install_flash_layer_view_bounds(handle: HiSparseCacheHandle) -> None:
    """Apply the same bounds to initial binding and device cache rebind."""
    handle.bind_cache = MethodType(_bind_resident, handle)
    handle.runtime.bind_hot_cache = MethodType(_bind_hot, handle.runtime)
