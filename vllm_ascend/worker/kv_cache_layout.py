# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.v1.attention.backends.utils import get_kv_cache_layout

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

HND_KV_CACHE_LAYOUT = "HND"


def check_hnd_kv_cache_layout_supported(cache_layout: str | None = None) -> None:
    if cache_layout is None:
        cache_layout = get_kv_cache_layout()
    if cache_layout == HND_KV_CACHE_LAYOUT and get_ascend_device_type() != AscendDeviceType.A5:
        raise RuntimeError("HND KV cache layout is only supported on Ascend A5.")


def get_kv_cache_stride_order(
    attn_backend: Any,
    kv_cache_shape: tuple[int, ...],
) -> tuple[int, ...]:
    check_hnd_kv_cache_layout_supported()
    try:
        stride_order = attn_backend.get_kv_cache_stride_order()
        assert len(stride_order) == len(kv_cache_shape)
    except (AttributeError, NotImplementedError):
        stride_order = tuple(range(len(kv_cache_shape)))
    return stride_order


def view_kv_cache_with_stride_order(
    raw_tensor: torch.Tensor,
    dtype: torch.dtype,
    logical_shape: tuple[int, ...],
    stride_order: tuple[int, ...],
) -> torch.Tensor:
    storage_shape = tuple(logical_shape[i] for i in stride_order)
    inv_order = [stride_order.index(i) for i in range(len(stride_order))]
    return raw_tensor.view(dtype).view(storage_shape).permute(*inv_order)


def view_split_kv_cache_with_stride_order(
    raw_tensor: torch.Tensor,
    dtype: torch.dtype,
    full_kv_cache_shape: tuple[int, ...],
    side_logical_shape: tuple[int, ...],
    attn_backend: Any,
) -> torch.Tensor:
    full_stride_order = get_kv_cache_stride_order(attn_backend, full_kv_cache_shape)
    if len(full_stride_order) == len(side_logical_shape) + 1 and 0 in full_stride_order:
        side_stride_order = tuple(dim - 1 for dim in full_stride_order if dim != 0)
    else:
        side_stride_order = tuple(range(len(side_logical_shape)))
    return view_kv_cache_with_stride_order(
        raw_tensor,
        dtype,
        side_logical_shape,
        side_stride_order,
    )
