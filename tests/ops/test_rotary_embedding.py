# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/tests/kernels/test_rotary_embedding.py
# Copyright 2023 The vLLM team.

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from typing import Optional, Tuple, Union

import pytest
import torch
import torch.nn as nn
import torch_npu  # noqa: F401

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


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


# adapted from https://github.com/vllm-project/vllm/vllm/model_executor/layers/rotary_embedding.py
class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


# test with leading dimension and merge seqlen and batch_size as num_tokens
@pytest.mark.skip(
    reason="vllm ascend's ci will disable custom kernel build, disable this "
    "unittest until build custom kernels open by default")
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

    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                           is_neox_style, dtype)
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
