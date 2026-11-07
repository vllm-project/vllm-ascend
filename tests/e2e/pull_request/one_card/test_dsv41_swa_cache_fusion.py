# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.dsv41_swa_cache import dsv41_rmsnorm_rope_cache

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = HEAD_DIM - NOPE_DIM
EPS = 1e-6


def _reference(projected, weight, cos, sin):
    value = projected.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS) * weight.float()
    value = value.to(torch.bfloat16).float()
    rotary = value[:, NOPE_DIM:].clone()
    paired = rotary.reshape(rotary.shape[0], -1, 2).flip(-1).reshape_as(rotary)
    signs = torch.ones(ROPE_DIM)
    signs[0::2] = -1
    value[:, NOPE_DIM:] = rotary * cos[:, 0, 0].float() + paired * sin[:, 0, 0].float() * signs
    return value.to(torch.bfloat16)


def _require_fusion_hardware():
    if not get_current_hardware_profile().supports(HardwareCapability.DSV41_SWA_CACHE_FUSION):
        pytest.skip("The DSV4.1 Triton fusion targets A2/A3 standard SWA caches")


@pytest.mark.parametrize("tokens", [1, 8, 192, 1024, 8192])
@torch.inference_mode()
def test_dsv41_rmsnorm_rope_cache_matches_interleaved_baseline(tokens):
    _require_fusion_hardware()
    torch.manual_seed(41)
    block_size = 32
    blocks = (tokens * 2 + block_size - 1) // block_size
    projected_cpu = torch.randn(tokens, HEAD_DIM, dtype=torch.bfloat16)
    weight_cpu = torch.randn(HEAD_DIM, dtype=torch.bfloat16)
    angles = torch.randn(tokens, 1, 1, ROPE_DIM)
    cos_cpu, sin_cpu = angles.cos(), angles.sin()
    expected = _reference(projected_cpu, weight_cpu, cos_cpu, sin_cpu)

    # Padded projection/RoPE rows and cache pages verify that all physical
    # strides, rather than shape-derived contiguous strides, reach the kernel.
    projected = torch.empty(tokens, HEAD_DIM + 16, dtype=torch.bfloat16, device="npu")[:, :HEAD_DIM]
    projected.copy_(projected_cpu)
    cos = torch.empty(tokens, 1, 1, ROPE_DIM + 8, device="npu")[:, :, :, :ROPE_DIM]
    sin = torch.empty_like(cos)
    cos.copy_(cos_cpu)
    sin.copy_(sin_cpu)
    weight = weight_cpu.npu()
    backing = torch.full((blocks, block_size + 3, 1, HEAD_DIM), -7.0, dtype=torch.bfloat16, device="npu")
    cache = backing[:, :block_size]
    flat_slots = torch.randperm(blocks * block_size)[:tokens]
    coordinates = torch.stack((flat_slots // block_size, flat_slots % block_size), -1).to(torch.int32)
    coordinates[::7] = -1

    dsv41_rmsnorm_rope_cache(projected, weight, cos, sin, cache, coordinates.npu(), EPS, NOPE_DIM)
    torch.npu.synchronize()

    actual = cache.cpu()
    expected_cache = torch.full(cache.shape, -7.0, dtype=torch.bfloat16)
    for token, (block, offset) in enumerate(coordinates.tolist()):
        if block >= 0:
            expected_cache[block, offset, 0] = expected[token]
    torch.testing.assert_close(actual.float(), expected_cache.float(), atol=0.035, rtol=0.01)
    # Rows outside the sliced physical page must remain untouched as well.
    assert torch.equal(backing[:, block_size:].cpu(), torch.full_like(backing[:, block_size:].cpu(), -7.0))


@torch.inference_mode()
def test_dsv41_rmsnorm_rope_cache_graph_replay_uses_updated_inputs_and_slots():
    _require_fusion_hardware()
    torch.manual_seed(73)
    tokens, block_size = 8, 16
    projected = torch.randn(tokens, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="npu")
    angles = torch.randn(tokens, 1, 1, ROPE_DIM, device="npu")
    cos, sin = angles.cos(), angles.sin()
    cache = torch.zeros(2, block_size + 1, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")[:, :block_size]
    slots = torch.full((tokens, 2), -1, dtype=torch.int32, device="npu")

    def run():
        dsv41_rmsnorm_rope_cache(projected, weight, cos, sin, cache, slots, EPS, NOPE_DIM)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()

    projected.copy_(torch.randn_like(projected))
    angles.copy_(torch.randn_like(angles))
    cos.copy_(angles.cos())
    sin.copy_(angles.sin())
    slots.copy_(torch.tensor([[0, 3], [1, 2], [-1, -1], [0, 8], [1, 5], [-1, -1], [0, 1], [1, 9]]))
    cache.zero_()
    graph.replay()
    torch.npu.synchronize()

    expected_rows = _reference(projected.cpu(), weight.cpu(), cos.cpu(), sin.cpu())
    expected_cache = torch.zeros_like(cache.cpu())
    for token, (block, offset) in enumerate(slots.cpu().tolist()):
        if block >= 0:
            expected_cache[block, offset, 0] = expected_rows[token]
    torch.testing.assert_close(cache.cpu().float(), expected_cache.float(), atol=0.035, rtol=0.01)
