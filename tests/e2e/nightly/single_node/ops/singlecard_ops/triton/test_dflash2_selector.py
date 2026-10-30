# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""DFlash2 selector reference, FP32 tail precision, and ACL replay checks."""

import math

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample import gumbel as gumbel_module
from vllm.v1.worker.gpu.sample.gumbel import _log1p_neg_stable

from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import _selector_walk_kernel_ascend


def _uniform_reference(seed, step, position, token):
    """Independent scalar Murmur3 reference; preserve both seed halves."""
    mask = (1 << 32) - 1

    def rotate(value, shift):
        return ((value << shift) | (value >> (32 - shift))) & mask

    def mix(value, key):
        key = rotate((key * 0xCC9E2D51) & mask, 15)
        value ^= (key * 0x1B873593) & mask
        return (rotate(value, 13) * 5 + 0xE6546B64) & mask

    if hasattr(gumbel_module, "spec_step_seed"):
        seed ^= step << 32
    position = position - 1 + (1 << 30)
    h = 0
    for key in (seed & mask, (seed >> 32) & mask, position & mask, token & mask):
        h = mix(h, key)
    h ^= 16
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & mask
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & mask
    h ^= h >> 16
    # Match upstream's two FP32 terms, rather than rounding the sum in FP64.
    return float(
        np.float32(np.float32(h >> 16) * np.float32(2**-16) + np.float32((h & 0xFFFF) + 0.5) * np.float32(2**-32))
    )


def _walk_reference(scores, candidates, positions, mapping, temperatures, seeds, probabilistic):
    rows, steps, top_k = candidates.shape
    tokens = torch.full((rows, steps), -123, dtype=torch.int64)
    realized = torch.full((rows, steps, top_k), -777.0)
    for row in range(rows):
        req = int(mapping[row, 0])
        if req < 0:
            continue
        previous = 0
        for step in range(steps):
            values = scores[row, step, previous].float()
            realized[row, step] = values
            temperature = float(temperatures[req]) if probabilistic else 0.0
            if temperature:
                noise = []
                for token in candidates[row, step].tolist():
                    uniform = _uniform_reference(int(seeds[req]), step, int(positions[row, step]), token)
                    # FP32 uniforms may round to one; upstream clamps this tail.
                    log_complement = math.log1p(-uniform) if uniform < 0.25 else math.log(max(1 - uniform, 2**-24))
                    noise.append(-math.log(-log_complement))
                values = values / temperature + torch.tensor(noise)
            previous = int(values.argmax())
            tokens[row, step] = candidates[row, step, previous]
    return tokens, realized


@pytest.mark.parametrize("top_k", [3, 16])
@pytest.mark.parametrize("num_steps", [1, 4])
@pytest.mark.parametrize("probabilistic", [False, True])
@torch.inference_mode()
def test_selector_eager_and_acl_replay_match_reference(top_k, num_steps, probabilistic):
    torch.npu.set_device(0)
    rows = 4
    generator = torch.Generator().manual_seed(2026)
    scores = torch.randn(rows, num_steps, top_k, top_k, generator=generator).npu()
    candidates = torch.arange(rows * num_steps * top_k).reshape(rows, num_steps, top_k).long().npu()
    positions = torch.arange(rows * num_steps).reshape(rows, num_steps).long().npu()
    mapping = torch.tensor([2, -1, 0, 1]).repeat_interleave(num_steps).reshape(rows, num_steps).npu()
    temperatures = torch.tensor([0.7, 0.0, 1.3]).npu()
    seeds = torch.tensor([13, 13 + (1 << 32), -(1 << 55) + 9], dtype=torch.int64).npu()
    tokens = torch.full((rows, num_steps), -123, dtype=torch.int64, device="npu")
    realized = torch.full((rows, num_steps, top_k), -777.0, device="npu")

    def launch():
        _selector_walk_kernel_ascend[(rows,)](
            scores,
            candidates,
            positions,
            mapping,
            temperatures,
            seeds,
            tokens,
            realized,
            num_steps=num_steps,
            top_k=top_k,
            BLOCK_K=triton.next_power_of_2(top_k),
            SAMPLE_PROBABILISTIC=probabilistic,
            USE_FP64=False,
            num_warps=1,
        )

    # Warm both data-dependent branches before capturing, with a padding row.
    for _ in range(3):
        launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()

    for revision in range(3):
        scores.copy_(torch.randn(scores.shape, generator=generator))
        candidates.add_(top_k)
        positions.add_(17)
        mapping.copy_(mapping.roll(1, dims=0))
        seeds.add_(1 << 33)
        temperatures.copy_(torch.tensor([0.0, 0.8 + revision, 1.3]))
        expected = _walk_reference(
            scores.cpu(),
            candidates.cpu(),
            positions.cpu(),
            mapping.cpu(),
            temperatures.cpu(),
            seeds.cpu(),
            probabilistic,
        )
        for execute in (launch, graph.replay):
            tokens.fill_(-123)
            realized.fill_(-777)
            execute()
            torch.npu.synchronize()
            torch.testing.assert_close(tokens.cpu(), expected[0], rtol=0, atol=0)
            torch.testing.assert_close(realized.cpu(), expected[1], rtol=0, atol=0)


@triton.jit
def _gumbel_tail_kernel(uniform_ptr, noise_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    uniform = tl.load(uniform_ptr + offsets, offsets < size, other=0.5)
    noise = -tl.log(-_log1p_neg_stable(uniform))
    tl.store(noise_ptr + offsets, noise, offsets < size)


def test_gumbel_fp32_tail_is_finite_and_accurate():
    torch.npu.set_device(0)
    values = torch.tensor([2**-33, 2**-30, 2**-26, 0.01, 0.249, 0.25, 0.5, 0.9, 1.0])
    uniform = values.npu()
    noise = torch.empty_like(uniform)
    _gumbel_tail_kernel[(1,)](uniform, noise, size=values.numel(), BLOCK=16)
    expected = torch.tensor(
        [-math.log(-(math.log1p(-float(u)) if u < 0.25 else math.log(max(1 - float(u), 2**-24)))) for u in values]
    )
    assert torch.isfinite(noise).all()
    torch.testing.assert_close(noise.cpu(), expected, rtol=2e-6, atol=2e-6)
