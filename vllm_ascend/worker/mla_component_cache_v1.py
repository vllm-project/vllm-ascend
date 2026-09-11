# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The vllm-ascend contributors
"""V1 MLA component-major cache views.

A manager block may contain multiple kernel blocks.  Each kernel slot is laid
out as ``[all nope | all rope | optional hybrid padding]``.  The returned MLA
cache therefore keeps the familiar ``(nope, rope)`` contract while both tensors
are first-axis strided over the same physical slot.
"""

from collections.abc import Sequence

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend import envs


def use_mla_component_cache(vllm_config: VllmConfig) -> bool:
    """Return whether V1 should preserve exact MLA specs for this runner."""
    return not getattr(vllm_config, "use_v2_model_runner", False) and bool(envs.VLLM_ASCEND_ENABLE_MLA_COMPONENT_CACHE)


def _typed_empty_like_storage(raw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bind an int8 raw region to a one-dimensional typed tensor view."""
    # raw int8 storage -> 目标dtype的一维typed view。
    raw_byte_offset = raw.storage_offset()
    dtype_size = torch.empty((), dtype=dtype).element_size()
    if raw_byte_offset % dtype_size != 0 or raw.numel() % dtype_size != 0:
        raise ValueError(
            f"Raw MLA cache range [{raw_byte_offset}, {raw_byte_offset + raw.numel()}) is not aligned to {dtype}"
        )

    typed = torch.empty(0, dtype=dtype, device=raw.device)
    typed.set_(
        raw.untyped_storage(),
        raw_byte_offset // dtype_size,
        (raw.numel() // dtype_size,),
        (1,),
    )
    return typed


def _component_views(
    typed_raw: torch.Tensor,
    *,
    num_blocks: int,
    kernel_blocks_per_manager: int,
    kernel_block_size: int,
    num_kv_heads: int,
    nope_dim: int,
    rope_dim: int,
    slot_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct first-axis-strided nope/rope views over component slots."""
    element_size = typed_raw.element_size()
    slot_elements = slot_bytes // element_size
    nope = torch.as_strided(
        typed_raw,
        (num_blocks * kernel_blocks_per_manager, kernel_block_size, num_kv_heads, nope_dim),
        (slot_elements, num_kv_heads * nope_dim, nope_dim, 1),
        typed_raw.storage_offset(),
    )
    rope = torch.as_strided(
        typed_raw,
        (num_blocks * kernel_blocks_per_manager, kernel_block_size, num_kv_heads, rope_dim),
        (slot_elements, num_kv_heads * rope_dim, rope_dim, 1),
        typed_raw.storage_offset() + kernel_block_size * num_kv_heads * nope_dim,
    )
    return nope, rope


def build_mla_component_cache(
    raw: torch.Tensor,
    *,
    layer: MLAAttention,
    spec: MLAAttentionSpec,
    kernel_block_size: int | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one standardized MLA raw layer region into component views."""
    # component cache进入常规allocate/reshape链路：这里只做必要几何推导。
    if isinstance(kernel_block_size, Sequence):
        kernel_block_size = kernel_block_size[0]
    manager_block_size = spec.block_size
    if manager_block_size % kernel_block_size != 0:
        raise ValueError(
            f"MLA manager block size {manager_block_size} is not divisible by kernel block size {kernel_block_size}"
        )

    ratio = manager_block_size // kernel_block_size
    physical_page_bytes = spec.page_size_bytes
    element_size = torch.empty((), dtype=spec.dtype).element_size()
    if physical_page_bytes % ratio != 0 or physical_page_bytes % element_size != 0:
        raise ValueError(
            f"MLA physical page {physical_page_bytes} cannot be split into {ratio} dtype-aligned kernel slots"
        )

    slot_bytes = physical_page_bytes // ratio
    component_bytes = kernel_block_size * spec.num_kv_heads * layer.head_size * element_size
    if component_bytes > slot_bytes:
        raise ValueError(f"MLA kernel slot {slot_bytes} is smaller than nope+rope {component_bytes}")

    num_blocks = raw.numel() // physical_page_bytes
    typed_raw = _typed_empty_like_storage(raw, spec.dtype)
    return _component_views(
        typed_raw,
        num_blocks=num_blocks,
        kernel_blocks_per_manager=ratio,
        kernel_block_size=kernel_block_size,
        num_kv_heads=spec.num_kv_heads,
        nope_dim=layer.kv_lora_rank,
        rope_dim=layer.qk_rope_head_dim,
        slot_bytes=slot_bytes,
    )


def is_mla_component_pair(kv_cache: object) -> bool:
    """Recognize a component-major MLA tuple without model-name checks."""
    if not isinstance(kv_cache, tuple) or len(kv_cache) != 2:
        return False
    nope, rope = kv_cache
    if not isinstance(nope, torch.Tensor) or not isinstance(rope, torch.Tensor):
        return False
    if nope.ndim != 4 or rope.ndim != 4 or nope.dtype != rope.dtype:
        return False
    if nope.device != rope.device or nope.shape[:3] != rope.shape[:3]:
        return False
    if nope.shape[3] <= 0 or rope.shape[3] < 0:
        return False

    element_size = nope.element_size()
    slot_bytes = nope.stride(0) * element_size
    if slot_bytes <= 0 or slot_bytes != rope.stride(0) * element_size:
        return False

    nope_strides = (
        slot_bytes // element_size,
        nope.shape[2] * nope.shape[3],
        nope.shape[3],
        1,
    )
    rope_dim = rope.shape[3]
    rope_inner_stride = rope.shape[2] * rope_dim if rope_dim else 0
    rope_strides = (
        slot_bytes // element_size,
        rope_inner_stride,
        rope_dim,
        1,
    )
    if nope.stride() != nope_strides or rope.stride() != rope_strides:
        return False

    nope_storage = nope.untyped_storage()
    rope_storage = rope.untyped_storage()
    nope_slot_bytes = nope.shape[1] * nope.shape[2] * nope.shape[3] * element_size
    rope_slot_bytes = rope.shape[1] * rope.shape[2] * rope_dim * element_size
    return (
        nope_storage.data_ptr() == rope_storage.data_ptr()
        and nope_storage.nbytes() == rope_storage.nbytes()
        and rope.data_ptr() == nope.data_ptr() + nope_slot_bytes
        and nope_slot_bytes + rope_slot_bytes <= slot_bytes
    )
