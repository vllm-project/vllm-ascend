# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/tests/kernels/test_rotary_embedding.py
# Copyright 2023 The vLLM team.

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from typing import Optional

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform

import vllm_ascend.platform  # noqa: F401

# Only Neox style true scenario is supported for now
IS_NEOX_STYLE = [True]
DTYPES = [torch.half, torch.bfloat16]
HEAD_SIZES = [64, 96, 128, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [17]  # Arbitrary values for testing
BATCH_SIZES = [5]  # Arbitrary values for testing
SEQ_LENS = [11, 4096]  # Arbitrary values for testing
SEEDS = [0]
DEVICES = [f"npu:{0}"]
# Set tolerance to 1 for quant ops
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


# test with leading dimension and merge seqlen and batch_size as num_tokens
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_rotary_embedding_quant_with_leading_dim(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size

    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype=dtype)
    num_tokens = batch_size * seq_len
    positions = torch.randint(0, max_position, (batch_size, seq_len))
    qkv_tensor = torch.randn(num_tokens,
                             num_heads * head_size * 3,
                             dtype=dtype)
    query, key, _ = qkv_tensor.split(
        [num_heads * head_size, num_heads * head_size, num_heads * head_size],
        dim=-1,
    )

    # because the custom kernel is in-place.
    rope.cos_sin_cache = rope.cos_sin_cache.float()
    ref_query, ref_key = rope.forward_native(positions, query.float(),
                                             key.float())
    torch.ops._C.rotary_embedding(
        positions,
        query,
        key,
        rope.head_size,
        rope.cos_sin_cache.to(dtype),
        rope.is_neox_style,
    )

    # Compare the results.
    torch.testing.assert_close(query,
                               ref_query,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    torch.testing.assert_close(key,
                               ref_key,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
