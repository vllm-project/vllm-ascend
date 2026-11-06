# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache import minimax_qknorm_rope_cache


def _norm_rope_reference(projection, weight, cos, sin, eps):
    x = projection.unflatten(-1, (-1, weight.numel())).float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * (1 + weight.float())
    x = x.to(projection.dtype).float()
    half = cos.shape[-1]
    first, second = x[..., :half], x[..., half : 2 * half]
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin, x[..., 2 * half :]),
        dim=-1,
    )


@pytest.mark.parametrize("tokens", [1, 193])
@pytest.mark.parametrize("fp8_cache", [False, True])
@torch.inference_mode()
def test_fused_sparse_prepare_with_mixed_head_dimensions(tokens, fp8_cache):
    if fp8_cache and not any(tag in torch.npu.get_device_name(0).lower() for tag in ("950", "910_95")):
        pytest.skip("FP8 cache conversion requires A5")

    main_dim, index_dim, rope_dim = 256, 128, 64
    q_heads, kv_heads, index_heads = 8, 2, 2
    widths = (q_heads * main_dim, kv_heads * main_dim, kv_heads * main_dim, index_heads * index_dim, index_dim)
    eps, main_block, index_block = 1e-6, 16, 8
    generator = torch.Generator().manual_seed(41)
    packed = torch.randn((tokens, sum(widths)), generator=generator, dtype=torch.bfloat16)
    host_inputs = packed.split(widths, dim=-1)
    # Split views share a padded projection row with nonunit column stride.
    backing = torch.empty((tokens, 2 * packed.shape[1] + 32), device="npu", dtype=packed.dtype)
    projections = backing[:, : 2 * packed.shape[1] : 2]
    projections.copy_(packed)
    weights = tuple(
        torch.randn(dim, generator=generator, dtype=packed.dtype) * 0.1
        for dim in (main_dim, main_dim, index_dim, index_dim)
    )
    angles = torch.randn((tokens, rope_dim // 2), generator=generator)
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1).to(packed.dtype)
    positions = torch.arange(tokens - 1, -1, -1)
    actual = max(1, tokens - 3)
    slots = torch.arange(1, tokens + 1)
    index_slots = slots.flip(0)
    slots[1::7] = -1
    index_slots[2::11] = -1
    dtype = torch.float8_e4m3fn if fp8_cache else packed.dtype
    main_shape = (tokens // main_block + 1, main_block, kv_heads, main_dim)
    index_shape = (tokens // index_block + 1, index_block, index_dim)
    key_cache = torch.zeros(main_shape, device="npu", dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    index_cache = torch.zeros(index_shape, device="npu", dtype=dtype)

    q, index_q = minimax_qknorm_rope_cache(
        *projections.split(widths, dim=-1),
        rope.npu(),
        positions.npu(),
        *(weight.npu() for weight in weights),
        key_cache,
        value_cache,
        index_cache,
        slots.npu(),
        index_slots.npu(),
        actual,
        eps,
    )

    cos, sin = rope[positions].float().unsqueeze(1).chunk(2, dim=-1)
    want_q, want_k, want_index_q, want_index_k = (
        _norm_rope_reference(host_inputs[i], weight, cos, sin, eps)
        for i, weight in zip((0, 1, 3, 4), weights, strict=True)
    )
    assert q.shape == (tokens, q_heads * main_dim) and q.dtype == packed.dtype
    assert index_q.shape == (tokens, index_heads * index_dim) and index_q.dtype == dtype
    torch.testing.assert_close(q.cpu().float(), want_q.flatten(1).to(q.dtype).float(), atol=0.032, rtol=0.01)
    torch.testing.assert_close(
        index_q.cpu().float(), want_index_q.flatten(1).to(dtype).float(), atol=0.25 if fp8_cache else 0.032, rtol=0.01
    )

    for cache, rows, mapping in (
        (key_cache, want_k.to(packed.dtype).flatten(1), slots),
        (value_cache, host_inputs[2], slots),
        (index_cache, want_index_k.flatten(1), index_slots),
    ):
        observed = cache.cpu().float().reshape(cache.shape[0] * cache.shape[1], -1)
        expected = torch.zeros_like(observed)
        valid = mapping[:actual] >= 0
        expected[mapping[:actual][valid]] = rows[:actual][valid].to(dtype).float()
        tolerance = 0 if cache is value_cache else (0.25 if fp8_cache else 0.032)
        torch.testing.assert_close(observed, expected, atol=tolerance, rtol=0.01 if tolerance else 0)
        untouched = torch.ones(observed.shape[0], dtype=torch.bool)
        untouched[mapping[:actual][valid]] = False
        assert torch.count_nonzero(observed[untouched]) == 0
