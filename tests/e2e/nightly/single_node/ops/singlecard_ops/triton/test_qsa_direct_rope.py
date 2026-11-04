# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.linearnorm.qsa_rmsnorm_mrope import triton_qsa_rmsnorm_mrope
from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope import triton_split_qkv_rmsnorm_mrope

pytestmark = [
    pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
]


@pytest.mark.parametrize("rows", [0, 1, 32, 129])
@pytest.mark.parametrize("axes", [1, 3])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_direct_rope_matches_gathered_cache(rows: int, axes: int, weight_dtype: torch.dtype) -> None:
    torch.manual_seed(17)
    cache = torch.randn(129, 64, dtype=torch.bfloat16, device="npu")
    # Model-runner buffers can have much larger axis strides than active rows.
    storage = torch.randint(-4, 140, (3, 2 * rows + 17), device="npu", dtype=torch.int64)
    positions = storage[:, : 2 * rows : 2] if axes == 3 else storage[0, : 2 * rows : 2]
    gathered = cache[positions.clamp(0, 128)]
    qkv = torch.randn(rows, 6 * 256, dtype=torch.bfloat16, device="npu")
    qw, kw = [torch.randn(256, dtype=weight_dtype, device="npu") for _ in range(2)]
    expected = triton_split_qkv_rmsnorm_mrope(
        qkv,
        1 + qw,
        1 + kw,
        gathered if axes == 3 else gathered.repeat(3, 1, 1),
        num_q_heads=2,
        num_kv_heads=1,
        head_size=256,
        eps=1e-6,
        mrope_section=[11, 11, 10],
        is_interleaved=True,
        rope_dim=64,
        has_gate=True,
    )
    actual = triton_split_qkv_rmsnorm_mrope(
        qkv,
        qw,
        kw,
        cache,
        positions=positions,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=256,
        eps=1e-6,
        mrope_section=[11, 11, 10],
        is_interleaved=True,
        rope_dim=64,
        has_gate=True,
    )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    packed = torch.randn(rows, 5, 128, dtype=torch.bfloat16, device="npu")
    query = packed[:, :4]  # Preserve the projection's non-contiguous token stride.
    weight = torch.randn(128, dtype=torch.bfloat16, device="npu")
    expected_query = triton_qsa_rmsnorm_mrope(query, weight, gathered, 1e-6, [11, 11, 10], 64)
    actual_query = triton_qsa_rmsnorm_mrope(query, weight, cache, 1e-6, [11, 11, 10], 64, positions)
    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)


@torch.inference_mode()
def test_direct_rope_replay_uses_updated_positions() -> None:
    rows = 32
    query = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device="npu")
    qkv = torch.randn(rows, 1536, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(128, dtype=torch.bfloat16, device="npu")
    main_weight = torch.randn(256, dtype=torch.bfloat16, device="npu")
    cache = torch.randn(129, 64, dtype=torch.bfloat16, device="npu")
    positions = torch.zeros(3, rows + 7, dtype=torch.int64, device="npu")[:, :rows]

    def run():
        indexer = torch.ops.vllm.triton_qsa_rmsnorm_mrope(query, weight, cache, 1e-6, [11, 11, 10], 64, positions)
        main = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv, main_weight, main_weight, cache, 2, 1, 256, 1e-6, [11, 11, 10], True, 64, None, None, True, positions
        )
        return (indexer, *main)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for offset in [-3, 37, 135]:
        positions.copy_(torch.arange(3 * rows, device="npu").reshape(3, rows) + offset)
        query.add_(0.125)
        qkv.mul_(0.875)
        graph.replay()
        expected = run()
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
