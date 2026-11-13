# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from math import prod

import torch
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class CacheComponent:
    """A backend tensor view relative to the first byte of a layer placement."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    offset_bytes: int
    strides_bytes: tuple[int, ...]

    @classmethod
    def paged(cls, count, shape, dtype, offset_bytes, page_stride_bytes):
        inner = tuple(prod(shape[i + 1 :]) * dtype.itemsize for i in range(len(shape)))
        return cls((count, *shape), dtype, offset_bytes, (page_stride_bytes, *inner))

    def end_bytes(self) -> int:
        return (
            self.offset_bytes
            + sum((length - 1) * stride for length, stride in zip(self.shape, self.strides_bytes, strict=True))
            + self.dtype.itemsize
        )


def describe_mamba_cache_components(spec, blocks, kernel_size, page_stride):
    """Describe the per-page state layout shared by Mamba and GDN."""
    if not isinstance(spec, MambaSpec) or kernel_size != spec.block_size:
        return None
    offset = 0
    components = []
    for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True):
        components.append(CacheComponent.paged(blocks, shape, dtype, offset, page_stride))
        offset += prod(shape) * dtype.itemsize
    return tuple(components) if offset <= spec.page_size_bytes else None


def bind_component(manager_view: torch.Tensor, component: CacheComponent, span_bytes: int) -> torch.Tensor:
    """Construct a zero-copy view within the layer's physical byte span."""
    start = manager_view.storage_offset() * manager_view.element_size()
    offset = start + component.offset_bytes
    itemsize = component.dtype.itemsize
    if offset % itemsize or any(stride % itemsize for stride in component.strides_bytes):
        raise ValueError("KV cache component is not dtype aligned")
    if component.offset_bytes < 0 or component.end_bytes() > span_bytes:
        raise ValueError("KV cache component exceeds its layer placement")
    storage = manager_view.untyped_storage()
    if start + span_bytes > storage.nbytes():
        raise ValueError("KV cache layer placement exceeds its backing")
    return torch.empty(0, dtype=component.dtype, device=manager_view.device).set_(
        storage, offset // itemsize, component.shape, tuple(s // itemsize for s in component.strides_bytes)
    )
