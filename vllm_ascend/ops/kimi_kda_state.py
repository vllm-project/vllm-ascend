# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Cache-state shape helpers for Kimi KDA."""

from vllm.distributed import divide
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first


def kimi_kda_state_shape(
    tp_world_size: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    num_spec: int = 0,
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """Return KDA convolution and recurrent cache shapes.

    vLLM 0.23's generic KDA helper accepts ``num_spec`` but does not add it
    to the short-convolution state length. Speculative decode needs those
    extra slots, as the existing Qwen GDN shape calculation already does.
    """
    if conv_kernel_size < 1:
        raise ValueError("conv_kernel_size must be positive")
    if num_spec < 0:
        raise ValueError("num_spec must be non-negative")

    local_heads = divide(num_heads, tp_world_size)
    local_conv_dim = divide(3 * num_heads * head_dim, tp_world_size)
    state_len = conv_kernel_size - 1 + num_spec
    conv_state_shape = (local_conv_dim, state_len) if is_conv_state_dim_first() else (state_len, local_conv_dim)
    recurrent_state_shape = (local_heads, head_dim, head_dim)
    return conv_state_shape, recurrent_state_shape


__all__ = ["kimi_kda_state_shape"]
