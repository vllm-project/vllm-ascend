# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from dataclasses import dataclass
from typing import Any

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass(frozen=True)
class KVCachePage:
    """A strided byte view and the inputs needed to interpret its layout."""

    pages: torch.Tensor
    backend: type[AttentionBackend]
    spec: KVCacheSpec
    layout: Any
    kernel_block_size: int

    def component_views(self) -> list[torch.Tensor]:
        num_blocks = self.pages.shape[0]
        shapes = self.backend.get_runtime_shapes(
            self.spec, num_blocks, self.layout, kernel_block_size=self.kernel_block_size
        )
        dtypes = (self.spec.dtype,)
        if getattr(self.spec, "scale_dim", 0):
            dtypes += (self.spec.scale_dtype,)

        # Backends describe [B,H,N,C] in bytes. Kernels consume NHD with
        # the last dimension expressed in dtype elements.
        components = []
        page_rows = self.pages[:, 0, 0, :]
        offset = 0
        for (blocks, heads, tokens, content_bytes), dtype in zip(shapes, dtypes, strict=True):
            itemsize = get_dtype_size(dtype)
            if blocks != num_blocks or content_bytes % itemsize:
                raise ValueError("Backend shape is incompatible with the manager page view.")
            size = heads * tokens * content_bytes
            component = page_rows.narrow(1, offset, size).view(dtype)
            components.append(component.unflatten(1, (tokens, heads, content_bytes // itemsize)))
            offset += size
        return components


class KVCachePageBinding:
    """Bind pages for layers that consume a list of NHD cache components."""

    def bind_kv_cache(self, kv_cache: KVCachePage) -> None:
        self.kv_cache = kv_cache.component_views()
