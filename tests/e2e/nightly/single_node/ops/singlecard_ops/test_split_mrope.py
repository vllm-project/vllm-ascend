# Copyright 2023 The vLLM team.

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

import gc
from typing import Union

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Arbitrary values for testing
NUM_TOKENS = [4096]
NUM_Q_HEADS = [28]
NUM_KV_HEADS = [4]
HEAD_SIZE = [128]
ROTARY_DIM = [128]
MROPE_SECTION = [[16, 24, 24]]
IS_NEOX_STYLE = [True, False]
DTYPE = [torch.bfloat16, torch.half]
DEVICES = [f"npu:{4}"]

# Set tolerance to 1 for quant ops
DEFAULT_ATOL = 1e-1
DEFAULT_RTOL = 1e-1


def mrope_golden(positions, query, key, cos_sin_cache, head_size,
                 mrope_section, is_neox_style):
    num_tokens = positions.shape[-1]
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)

    if positions.ndim == 2:
        cos = torch.cat(
            [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
            dim=-1)
        sin = torch.cat(
            [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
            dim=-1)

    rotary_dim = cos_sin.shape[-1]

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb_torch(query_rot, cos, sin,
                                       bool(is_neox_style))
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb_torch(key_rot, cos, sin, bool(is_neox_style))
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style,
) -> torch.Tensor:
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


def _compute_inv_freq(base: Union[int, float], rotary_dim) -> torch.Tensor:
    inv_freq = 1.0 / (base**(
        torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    return inv_freq


def _compute_cos_sin_cache(base, max_position_embeddings,
                           rotary_dim) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = _compute_inv_freq(base, rotary_dim)
    t = torch.arange(max_position_embeddings, dtype=torch.float)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache


# test with leading dimension and merge seqlen and batch_size as num_tokens
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads", NUM_Q_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZE)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIM)
@pytest.mark.parametrize("mrope_section", MROPE_SECTION)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_mrope(num_tokens: int, num_q_heads: int, num_kv_heads: int,
                     head_size: int, rotary_dim: int, mrope_section: list[int],
                     is_neox_style: bool, dtype: torch.dtype,
                     device: str) -> None:

    torch.set_default_device(device)

    in_positions = torch.randint(0, num_tokens, [3, num_tokens],
                                 dtype=torch.int64, device="npu")
    in_qkv = torch.randn(num_tokens,
                         (num_q_heads + 2 * num_kv_heads) * head_size,
                         dtype=dtype, device="npu")
    in_cos_sin_cache = _compute_cos_sin_cache(num_tokens, num_tokens,
                                              rotary_dim).to(device="npu")

    # Execute golden cases.
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    in_q, in_k, golden_v = in_qkv.split([q_size, kv_size, kv_size], dim=-1)

    golden_q, golden_k = mrope_golden(in_positions, in_q, in_k,
                                      in_cos_sin_cache,
                                      head_size,
                                      mrope_section,
                                      is_neox_style)

    # Execute real cases.
    if is_neox_style:
        rotary_mode = 'half'
    else:
        rotary_mode = 'interleave'

    q, k, v = torch.ops._C_ascend.npu_split_mrope(
        in_positions,
        in_qkv,
        in_cos_sin_cache.to(dtype=dtype),
        q_size,
        kv_size,
        num_q_heads,
        num_kv_heads,
        head_size,
        mrope_section=mrope_section,
        rotary_mode=rotary_mode)

    # Compare the results.
    torch.testing.assert_close(q.view(golden_q.size()),
                               golden_q,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    torch.testing.assert_close(k.view(golden_k.size()),
                               golden_k,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)
    torch.testing.assert_close(v.view(golden_v.size()),
                               golden_v,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
