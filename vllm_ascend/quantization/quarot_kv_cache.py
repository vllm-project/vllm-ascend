#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""QuaRot KV-cache helpers for the lean H-only attention contract."""

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, spec_manager_map
from typing_extensions import Self

_LEAN_ATTENTION_CONTRACT = "lean_h_only"
_QUAROT_NATIVE_KV_CACHE_ENV = "VLLM_ASCEND_QUAROT_USE_NATIVE_KV_CACHE"
_DEFAULT_KV_SCALE_DTYPE = torch.float16
_DEFAULT_KV_GROUP_SIZE = 1


@dataclass(frozen=True, kw_only=True)
class QuaRotInt4KVCacheSpec(FullAttentionSpec):
    """vLLM-style QuaRot KV4 cache spec.

    The cache payload stays split as key/value tensors. Quantization metadata
    stays split as k_scale/v_scale tensors. This follows the upstream vLLM
    pattern used by quantized KV-cache backends.
    """

    group_size: int
    scale_dtype: torch.dtype = _DEFAULT_KV_SCALE_DTYPE
    codec: str = "int4_symmetric"

    @property
    def packed_head_size(self) -> int:
        return cdiv(self.head_size, 2)

    @property
    def num_scale_groups(self) -> int:
        return cdiv(self.head_size, self.group_size)

    @property
    def key_cache_page_size_bytes(self) -> int:
        return self.block_size * self.num_kv_heads * self.packed_head_size

    @property
    def value_cache_page_size_bytes(self) -> int:
        return self.key_cache_page_size_bytes

    @property
    def key_scale_page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * self.num_scale_groups
            * get_dtype_size(self.scale_dtype)
        )

    @property
    def value_scale_page_size_bytes(self) -> int:
        return self.key_scale_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        return (
            self.key_cache_page_size_bytes
            + self.value_cache_page_size_bytes
            + self.key_scale_page_size_bytes
            + self.value_scale_page_size_bytes
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, QuaRotInt4KVCacheSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be QuaRotInt4KVCacheSpec."
        )
        sliding_window = set(spec.sliding_window for spec in specs if spec.sliding_window is not None)
        attention_chunk_size = set(
            spec.attention_chunk_size for spec in specs if spec.attention_chunk_size is not None
        )
        group_sizes = set(spec.group_size for spec in specs)
        scale_dtypes = set(spec.scale_dtype for spec in specs)
        codecs = set(spec.codec for spec in specs)
        assert len(group_sizes) == 1, "QuaRot KV4 layers in one cache group must share group_size."
        assert len(scale_dtypes) == 1, "QuaRot KV4 layers in one cache group must share scale_dtype."
        assert len(codecs) == 1, "QuaRot KV4 layers in one cache group must share codec."

        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            group_size=group_sizes.pop(),
            scale_dtype=scale_dtypes.pop(),
            codec=codecs.pop(),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All QuaRot KV4 layers in the same cache group must share the same attention spec."
                )
            assert spec.group_size == merged_spec.group_size
            assert spec.scale_dtype == merged_spec.scale_dtype
            assert spec.codec == merged_spec.codec
        return merged_spec

    def raw_split_sizes(self, total_bytes: int) -> tuple[int, int, int, int]:
        if total_bytes % self.page_size_bytes != 0:
            raise ValueError(
                f"KV-cache byte size {total_bytes} is not divisible by page_size_bytes={self.page_size_bytes}."
            )
        num_pages = total_bytes // self.page_size_bytes
        return (
            num_pages * self.key_cache_page_size_bytes,
            num_pages * self.value_cache_page_size_bytes,
            num_pages * self.key_scale_page_size_bytes,
            num_pages * self.value_scale_page_size_bytes,
        )

    def key_cache_shape(self, num_blocks: int) -> tuple[int, int, int, int]:
        return (num_blocks, self.block_size, self.num_kv_heads, self.packed_head_size)

    def value_cache_shape(self, num_blocks: int) -> tuple[int, int, int, int]:
        return self.key_cache_shape(num_blocks)

    def key_scale_shape(self, num_blocks: int) -> tuple[int, int, int, int]:
        return (num_blocks, self.block_size, self.num_kv_heads, self.num_scale_groups)

    def value_scale_shape(self, num_blocks: int) -> tuple[int, int, int, int]:
        return self.key_scale_shape(num_blocks)


def use_native_quarot_kv_cache() -> bool:
    return os.getenv(_QUAROT_NATIVE_KV_CACHE_ENV, "1").strip().lower() not in {"0", "false", "no"}


def uses_quarot_kv4_cache(layer: torch.nn.Module) -> bool:
    config = getattr(layer, "quarot_config", None) or {}
    return config.get("attention_contract") == _LEAN_ATTENTION_CONTRACT and not use_native_quarot_kv_cache()


spec_manager_map.setdefault(QuaRotInt4KVCacheSpec, FullAttentionManager)


def maybe_get_quarot_kv4_cache_spec(
    layer: torch.nn.Module,
    spec: FullAttentionSpec | None,
) -> FullAttentionSpec | None:
    if spec is None or not isinstance(spec, FullAttentionSpec):
        return spec
    if not uses_quarot_kv4_cache(layer):
        return spec

    config = getattr(layer, "quarot_config", None) or {}
    group_size = int(config.get("kv_group_size", _DEFAULT_KV_GROUP_SIZE))
    if group_size <= 0:
        raise ValueError(f"QuaRot kv_group_size must be positive, got {group_size!r}.")

    return QuaRotInt4KVCacheSpec(
        block_size=spec.block_size,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        head_size_v=spec.head_size_v,
        dtype=torch.uint8,
        page_size_padded=spec.page_size_padded,
        sliding_window=spec.sliding_window,
        attention_chunk_size=spec.attention_chunk_size,
        group_size=group_size,
    )


def iter_kv_cache_tensors(kv_cache: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(kv_cache, torch.Tensor):
        return [kv_cache]
    return [tensor for tensor in kv_cache]


def iter_kv_cache_records(
    kv_caches: torch.Tensor | Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]],
) -> list[list[torch.Tensor]]:
    if isinstance(kv_caches, torch.Tensor):
        return [[kv_caches]]
    caches = list(kv_caches)
    if not caches:
        return []
    if isinstance(caches[0], torch.Tensor):
        return [[tensor for tensor in caches]]
    return [[tensor for tensor in cache] for cache in caches]


def kv_cache_tensor_count(kv_cache: torch.Tensor | Sequence[torch.Tensor]) -> int:
    return len(iter_kv_cache_tensors(kv_cache))


def kv_cache_block_lengths(kv_cache: torch.Tensor | Sequence[torch.Tensor]) -> list[int]:
    lengths: list[int] = []
    for cache in iter_kv_cache_tensors(kv_cache):
        block_shape = cache.shape[1:]
        lengths.append(cache.element_size() * math.prod(block_shape))
    return lengths


def quantize_int4_symmetric_groupwise_packed(
    x: torch.Tensor,
    group_size: int,
    *,
    scale_dtype: torch.dtype = _DEFAULT_KV_SCALE_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size!r}.")
    if x.ndim < 1:
        raise ValueError("Expected tensor rank >= 1 for group-wise int4 quantization.")

    original_shape = x.shape
    d = original_shape[-1]
    if d % 2 != 0:
        raise ValueError(f"Packed int4 cache requires even last dimension, got {d}.")

    rows = x.to(torch.float16).reshape(-1, d)
    num_groups = cdiv(d, group_size)
    packed = torch.empty((rows.shape[0], d // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((rows.shape[0], num_groups), dtype=scale_dtype, device=x.device)

    quant = torch.empty((rows.shape[0], d), dtype=torch.int8, device=x.device)
    for g in range(num_groups):
        start = g * group_size
        end = min(d, start + group_size)
        chunk = rows[:, start:end]
        row_max = chunk.amax(dim=-1, keepdim=True)
        row_min = chunk.amin(dim=-1, keepdim=True)
        divisor = torch.clamp(torch.maximum(row_max, -row_min), min=1e-6)
        scale = torch.clamp(divisor / 7.0, min=1e-6)
        quant[:, start:end] = torch.round((chunk / divisor) * 7.0).clamp(-8, 7).to(torch.int8)
        scales[:, g] = scale.squeeze(-1).to(scale_dtype)

    low = (quant[:, 0::2] & 0x0F).to(torch.uint8)
    high = ((quant[:, 1::2] & 0x0F).to(torch.uint8)) << 4
    packed.copy_(low | high)
    return packed.reshape(*original_shape[:-1], d // 2), scales.reshape(*original_shape[:-1], num_groups)


def dequantize_int4_symmetric_groupwise_packed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    out_dtype: torch.dtype,
    head_size: int | None = None,
) -> torch.Tensor:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size!r}.")
    if packed.ndim < 1:
        raise ValueError("Expected tensor rank >= 1 for packed int4 dequantization.")

    packed_last_dim = packed.shape[-1]
    d = head_size if head_size is not None else packed_last_dim * 2
    if d != packed_last_dim * 2:
        raise ValueError(f"Packed int4 cache expects head_size={packed_last_dim * 2}, got {d}.")

    packed_rows = packed.reshape(-1, packed_last_dim)
    scale_rows = scales.reshape(-1, scales.shape[-1]).to(torch.float32)
    num_groups = cdiv(d, group_size)
    if scale_rows.shape[-1] != num_groups:
        raise ValueError(
            f"Scale shape mismatch for packed int4 dequantization: got {tuple(scales.shape)}, expected last dim {num_groups}."
        )

    low = (packed_rows & 0x0F).to(torch.int8)
    high = ((packed_rows >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    quant = torch.empty((packed_rows.shape[0], d), dtype=torch.int8, device=packed.device)
    quant[:, 0::2] = low
    quant[:, 1::2] = high

    out = torch.empty((packed_rows.shape[0], d), dtype=torch.float32, device=packed.device)
    for g in range(num_groups):
        start = g * group_size
        end = min(d, start + group_size)
        out[:, start:end] = quant[:, start:end].to(torch.float32) * scale_rows[:, g].unsqueeze(-1)
    return out.reshape(*packed.shape[:-1], d).to(out_dtype)
