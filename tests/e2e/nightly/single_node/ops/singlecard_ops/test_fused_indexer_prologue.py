# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

INPUT_DIM = 512
HEAD_DIM = 128
ROPE_DIM = 64
EPSILON = 1e-6
BLOCK_SIZE = 16


def _unfused_reference(latent, weight, norm_weight, cos, sin):
    projected = F.linear(latent, weight)
    normalized, _ = torch_npu.npu_rms_norm(projected, norm_weight, EPSILON)
    key = normalized.view(-1, 1, HEAD_DIM)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        key.unsqueeze(1),
        cos,
        sin,
        rotary_mode="interleave",
        partial_slice=[HEAD_DIM - ROPE_DIM, HEAD_DIM],
        negate_sin=False,
    )
    return torch_npu.npu_dynamic_quant(key.squeeze(1), dst_type=torch.int8)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="Requires Ascend A2/A3",
)
@pytest.mark.parametrize("num_tokens", [1, 17, 65])
def test_fused_indexer_prologue_matches_unfused_path(num_tokens):
    torch.manual_seed(7)
    device = "npu"
    latent = torch.randn((num_tokens, INPUT_DIM), dtype=torch.bfloat16, device=device)
    weight = torch.randn((HEAD_DIM, INPUT_DIM), dtype=torch.bfloat16, device=device) * 0.02
    norm_weight = torch.randn((HEAD_DIM,), dtype=torch.bfloat16, device=device) * 0.05 + 1.0
    angles = torch.randn((num_tokens, 1, 1, ROPE_DIM), dtype=torch.float32, device=device)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    slots = torch.stack(
        (
            torch.arange(num_tokens, device=device, dtype=torch.int32) // BLOCK_SIZE,
            torch.arange(num_tokens, device=device, dtype=torch.int32) % BLOCK_SIZE,
        ),
        dim=1,
    )
    if num_tokens > 1:
        slots[-1] = -1

    # Views deliberately retain gaps in their physical storage. This matches
    # the cache-pool layouts for which scatter_cache_sk previously handled the
    # final publication step.
    k_backing = torch.full((num_blocks, BLOCK_SIZE, 2, HEAD_DIM), -5, dtype=torch.int8, device=device)
    scale_backing = torch.full((num_blocks, BLOCK_SIZE, 2, 1), -1.0, dtype=torch.float16, device=device)
    k_cache = k_backing[:, :, ::2, :]
    scale_cache = scale_backing[:, :, ::2, :]

    reference_q, reference_scale = _unfused_reference(latent, weight, norm_weight, cos, sin)
    torch.ops._C_ascend.fused_indexer_prologue(
        latent,
        weight,
        norm_weight,
        cos,
        sin,
        slots,
        k_cache,
        scale_cache,
        EPSILON,
    )

    actual_k = k_cache.cpu()
    actual_scale = scale_cache.cpu()
    expected_k = torch.full_like(actual_k, -5)
    expected_scale = torch.full_like(actual_scale, -1.0)
    written_key_mask = torch.zeros_like(expected_k, dtype=torch.bool)
    reference_q = reference_q.cpu()
    reference_scale = reference_scale.cpu().to(torch.float16)
    slots_cpu = slots.cpu()
    for token, (block_id, block_offset) in enumerate(slots_cpu.tolist()):
        if block_id >= 0:
            expected_k[block_id, block_offset, 0] = reference_q[token]
            expected_scale[block_id, block_offset, 0, 0] = reference_scale[token]
            written_key_mask[block_id, block_offset, 0] = True

    valid_scales = expected_scale >= 0
    torch.testing.assert_close(actual_scale[valid_scales], expected_scale[valid_scales], rtol=5e-3, atol=1e-4)
    assert torch.equal(actual_scale[~valid_scales], expected_scale[~valid_scales])

    # Matmul accumulation order may move a quantized value by one code while
    # retaining the same BF16 result envelope.
    max_code_error = (
        (actual_k[written_key_mask].to(torch.int16) - expected_k[written_key_mask].to(torch.int16)).abs().max()
    )
    assert max_code_error <= 1
    assert torch.equal(actual_k[~written_key_mask], expected_k[~written_key_mask])

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
