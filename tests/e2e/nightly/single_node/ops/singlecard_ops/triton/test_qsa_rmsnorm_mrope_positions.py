# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.linearnorm.qsa_rmsnorm_mrope import triton_qsa_rmsnorm_mrope

pytestmark = [
    pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
]


@pytest.mark.parametrize("rows", [0, 1, 32, 129])
@pytest.mark.parametrize("axes", [1, 3])
@torch.inference_mode()
def test_direct_rope_matches_gathered_cache(rows: int, axes: int) -> None:
    torch.manual_seed(17)
    cache = torch.randn(129, 64, dtype=torch.bfloat16, device="npu")
    # Model-runner buffers can have much larger axis strides than active rows.
    storage = torch.randint(-4, 140, (3, 2 * rows + 17), device="npu", dtype=torch.int64)
    positions = storage[:, : 2 * rows : 2] if axes == 3 else storage[0, : 2 * rows : 2]
    gathered = cache[positions.clamp(0, 128)]
    packed = torch.randn(rows, 5, 128, dtype=torch.bfloat16, device="npu")
    query = packed[:, :4]  # Preserve the projection's non-contiguous token stride.
    weight = torch.randn(128, dtype=torch.bfloat16, device="npu")
    expected = triton_qsa_rmsnorm_mrope(query, weight, gathered, 1e-6, [11, 11, 10], 64)
    actual = triton_qsa_rmsnorm_mrope(query, weight, cache, 1e-6, [11, 11, 10], 64, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_direct_rope_replay_uses_updated_positions() -> None:
    rows = 32
    query = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(128, dtype=torch.bfloat16, device="npu")
    cache = torch.randn(129, 64, dtype=torch.bfloat16, device="npu")
    positions = torch.zeros(3, rows + 7, dtype=torch.int64, device="npu")[:, :rows]

    def run():
        return torch.ops.vllm.triton_qsa_rmsnorm_mrope(query, weight, cache, 1e-6, [11, 11, 10], 64, positions)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for offset in [-3, 37, 135]:
        positions.copy_(torch.arange(3 * rows, device="npu").reshape(3, rows) + offset)
        query.add_(0.125)
        graph.replay()
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)
