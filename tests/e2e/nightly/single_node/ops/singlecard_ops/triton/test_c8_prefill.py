# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.c8_prefill import try_fused_c8_prefill

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="C8 cached prefill regression coverage on Ascend A2/A3",
)


def _to_nz(cache):
    pages, tokens, heads, head_size = cache.shape
    return cache.view(pages, tokens, heads, head_size // 32, 32).permute(0, 2, 3, 1, 4).contiguous().npu()


def _strided_scales(scale):
    backing = torch.empty((*scale.shape[:-1], 2 * scale.shape[-1]), dtype=scale.dtype, device="npu")
    view = backing[..., ::2]
    view.copy_(scale)
    return view


def _assert_bitwise(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    bits = torch.int32 if expected.dtype == torch.float32 else torch.int16
    assert torch.equal(actual.cpu().contiguous().view(bits), expected.contiguous().view(bits))


@pytest.mark.parametrize(
    ("target_dtype", "k_scale_dtype", "v_scale_dtype", "block_size", "heads", "table_dtype"),
    [
        pytest.param(
            torch.bfloat16, torch.bfloat16, torch.float32, 32, 1, torch.int32, id="bf16-bf16-fp32-single-head"
        ),
        pytest.param(torch.bfloat16, torch.float16, torch.float16, 64, 4, torch.int64, id="bf16-fp16-fp16"),
        pytest.param(torch.bfloat16, torch.float32, torch.bfloat16, 96, 2, torch.int32, id="bf16-fp32-bf16"),
        pytest.param(torch.float16, torch.float16, torch.float32, 128, 8, torch.int32, id="fp16-fp16-fp32"),
        pytest.param(
            torch.float16, torch.bfloat16, torch.bfloat16, 128, 1, torch.int64, id="fp16-bf16-bf16-single-head"
        ),
        pytest.param(torch.float16, torch.float32, torch.float16, 256, 3, torch.int32, id="fp16-fp32-fp16"),
    ],
)
@torch.inference_mode()
def test_fused_c8_prefill_matches_dense_reference(
    target_dtype, k_scale_dtype, v_scale_dtype, block_size, heads, table_dtype
):
    generator = torch.Generator().manual_seed(19003)
    head_size = 128
    shape = (4, block_size, heads, head_size)
    key = torch.randint(-128, 128, shape, dtype=torch.int8, generator=generator)
    value = torch.randint(-128, 128, shape, dtype=torch.int8, generator=generator)
    scales = torch.linspace(0.005, 0.24, heads * head_size).view(1, heads, head_size)
    k_scale = scales.to(k_scale_dtype)
    v_scale = scales.flip(-1).to(v_scale_dtype)
    # The unused columns contain invalid physical page IDs. The backing
    # buffer adds an extra column so the table's row stride exceeds its width.
    table = torch.tensor([[3, 1, -1, -1], [-1, -1, -1, -1], [3, -1, -1, -1]], dtype=table_dtype, device="npu")[:, :3]
    lengths = [block_size + 1, 0, block_size - 3]
    actual = try_fused_c8_prefill(
        _to_nz(key),
        _to_nz(value),
        table,
        lengths,
        block_size,
        heads,
        head_size,
        target_dtype,
        _strided_scales(k_scale),
        _strided_scales(v_scale),
    )

    assert actual is not None
    for packed, cache, scale in zip(actual, (key, value), (k_scale, v_scale)):
        live = torch.cat((cache[3], cache[1, :1], cache[3, : lengths[2]]))
        _assert_bitwise(packed, live.to(target_dtype) * scale)


@torch.inference_mode()
def test_fused_c8_prefill_large_launch():
    # 18,000 live pages times four channel tiles gives a grid of 72,000
    # with one output token per page to keep the output allocation small.
    rows, block_size, heads, head_size = 18_000, 32, 8, 128
    page = (
        torch.arange(block_size * heads * head_size)
        .remainder(256)
        .sub(128)
        .to(torch.int8)
        .view(1, block_size, heads, head_size)
    )
    value = page.flip(-1)
    scale = torch.full((1, heads, head_size), 0.125, dtype=torch.bfloat16)
    device_scale = scale.npu()
    actual = try_fused_c8_prefill(
        _to_nz(page),
        _to_nz(value),
        torch.zeros((rows, 1), dtype=torch.int32, device="npu"),
        [1] * rows,
        block_size,
        heads,
        head_size,
        torch.bfloat16,
        device_scale,
        device_scale,
    )

    assert actual is not None
    for packed, cache in zip(actual, (page, value)):
        expected = (cache[0, :1].to(torch.bfloat16) * scale).expand(rows, heads, head_size)
        _assert_bitwise(packed, expected)


@torch.inference_mode()
def test_fused_c8_prefill_declines_stream_capture():
    block_size, head_size = 32, 64
    key = torch.zeros((1, block_size, 1, head_size), dtype=torch.int8, device="npu")
    value = torch.ones_like(key)
    table = torch.zeros((1, 1), dtype=torch.int32, device="npu")
    scale = torch.ones((1, 1, head_size), dtype=torch.bfloat16, device="npu")
    # Record a real operation so the test also checks that capture and replay
    # work after the wrapper declines to launch.
    counter = torch.zeros(1, dtype=torch.int32, device="npu")
    counter.add_(1)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = try_fused_c8_prefill(key, value, table, [1], block_size, 1, head_size, torch.bfloat16, scale, scale)
        counter.add_(1)
    assert actual is None
    before = counter.item()
    graph.replay()
    torch.npu.synchronize()
    assert counter.item() == before + 1
