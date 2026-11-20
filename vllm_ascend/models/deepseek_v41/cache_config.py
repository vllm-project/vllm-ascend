# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend V4.1 storage specs for upstream cache planning."""

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    declared_kwarg,
)
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    get_current_hardware_profile,
)

STATE_RING_ROWS = 32
MAX_MAIN_BLOCK_SIZE = 128
SWA_BLOCK_SIZE = 32

# Fixed dimensions accepted by the packaged mixed-quant attention ABI.
# The actual head sizes come from the model config; these bounds reject
# unsupported layouts before allocating caches, rather than override config.
MLA_HEAD_DIM = 512
INDEX_HEAD_DIM = 128
INDEX_FOLD_ROWS = 8


def uses_a5_packed_cache() -> bool:
    """Select the V4.1 cache ABI from the detected hardware profile."""
    return get_current_hardware_profile().supports(HardwareCapability.DSV41_PACKED_CACHE)


def make_swa_cache_spec(*, block_size, window_size, head_size, dtype, cache_dtype, bounded_replay=False):
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # FP8 values followed by one BF16 scale per 32 values.
        head_size += (head_size // 32) * torch.bfloat16.itemsize
        dtype = torch.uint8
        cache_dtype = "a5_mxfp8_bf16_scale"
    return AscendSlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        sliding_window=window_size,
        cache_dtype_str=cache_dtype,
        model_version="deepseek_v41",
        alignment=None,
        **declared_kwarg(AscendSlidingWindowMLASpec, "bounded_replay", bounded_replay),
    )


def _check_main_block_size(block_size):
    if block_size > MAX_MAIN_BLOCK_SIZE:
        raise ValueError("DeepSeek V4.1 uses one kernel page per manager block; set --block-size to 128 or smaller.")


def make_mla_cache_spec(*, block_size, head_size, compress_ratio):
    _check_main_block_size(block_size)
    if uses_a5_packed_cache():
        if head_size != MLA_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {MLA_HEAD_DIM}, got {head_size}")
        # Two FP4 values per byte, followed by one BF16 scale per 16 values.
        head_size = head_size // 2 + (head_size // 16) * torch.bfloat16.itemsize
        dtype = torch.uint8
        scale_dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
        scale_dtype = torch.int8
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
        scale_dtype=scale_dtype,
    )


def make_index_cache_spec(*, block_size, head_size, compress_ratio):
    _check_main_block_size(block_size)
    if uses_a5_packed_cache():
        if head_size != INDEX_HEAD_DIM:
            raise ValueError(f"A5 DeepSeek V4.1 requires head size {INDEX_HEAD_DIM}, got {head_size}")
        scale_dim = head_size // 32
        head_size //= 2
        dtype = torch.uint8
        scale_dtype = torch.uint8
    else:
        dtype = torch.int8
        scale_dim = 1
        scale_dtype = torch.float16
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=dtype,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
        state_content_bytes=head_size * get_dtype_size(dtype) + scale_dim * get_dtype_size(scale_dtype),
        scale_dim=scale_dim,
        scale_dtype=scale_dtype,
    )


def make_folded_index_cache_spec(*, block_size):
    """QSLI candidate-source view: eight K/scale rows per 544-byte row."""
    _check_main_block_size(block_size)
    if block_size % INDEX_FOLD_ROWS:
        raise ValueError("A5 folded index page must contain full 8-token groups")
    return AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=INDEX_FOLD_ROWS * (INDEX_HEAD_DIM // 2 + INDEX_HEAD_DIM // 32),
        dtype=torch.uint8,
        tokens_per_state=INDEX_FOLD_ROWS,
        model_version="deepseek_v41",
        # Both supported vLLM lanes (#57169) omit MLA `storage_block_size`.
        **{},
    )


def is_deepseek_v41_cache(specs_or_groups):
    if isinstance(specs_or_groups, dict):
        specs = list(specs_or_groups.values())
    else:
        specs = []
        for item in specs_or_groups:
            spec = getattr(item, "kv_cache_spec", item)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                specs.extend(spec.kv_cache_specs.values())
            else:
                specs.append(spec)
    return any(getattr(spec, "model_version", None) == "deepseek_v41" for spec in specs)
