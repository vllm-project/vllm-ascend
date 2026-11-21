import pytest
import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.linearnorm.qsa_rmsnorm_mrope import (
    triton_qsa_rmsnorm_mrope,
)

pytestmark = [
    pytest.mark.skipif(
        get_ascend_device_type() not in (AscendDeviceType.A3, AscendDeviceType.A5),
        reason="Requires Ascend A3/A5",
    ),
]


def _reference(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    values = tensor.float()
    reciprocal_std = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (values * reciprocal_std * (weight.float() + 1.0)).to(torch.bfloat16)
    rotary_offsets = torch.arange(32)
    h_mask = (rotary_offsets % 3 == 1) & (rotary_offsets < 33)
    w_mask = (rotary_offsets % 3 == 2) & (rotary_offsets < 30)
    axis = torch.where(h_mask, 1, torch.where(w_mask, 2, 0))
    token = torch.arange(tensor.shape[0])[:, None]
    offset = rotary_offsets[None, :]
    cos = cos_sin[axis[None, :], token, offset].float()
    sin = cos_sin[axis[None, :], token, offset + 32].float()
    first = normalized[..., :32]
    second = normalized[..., 32:64]
    output = normalized.clone()
    output[..., :32] = (
        (first.float() * cos[:, None, :]).to(torch.bfloat16) - (second.float() * sin[:, None, :]).to(torch.bfloat16)
    ).to(torch.bfloat16)
    output[..., 32:64] = (
        (second.float() * cos[:, None, :]).to(torch.bfloat16) + (first.float() * sin[:, None, :]).to(torch.bfloat16)
    ).to(torch.bfloat16)
    return output


@pytest.mark.parametrize("num_heads", [4, 1], ids=["prefill", "decode"])
@torch.inference_mode()
def test_qsa_rmsnorm_mrope(num_heads: int) -> None:
    torch.manual_seed(17)
    num_tokens, head_dim, rope_dim = 4096, 128, 64
    tensor_cpu = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16)
    weight_cpu = torch.randn(head_dim, dtype=torch.bfloat16) * 0.05
    cos_sin_cpu = torch.randn(3, num_tokens, rope_dim, dtype=torch.bfloat16)
    expected = _reference(tensor_cpu, weight_cpu, cos_sin_cpu, 1e-6)

    actual = triton_qsa_rmsnorm_mrope(
        tensor_cpu.npu(),
        weight_cpu.npu(),
        cos_sin_cpu.npu(),
        1e-6,
        [11, 11, 10],
        rope_dim,
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("rows", [0, 1, 32, 129])
@pytest.mark.parametrize("axes", [1, 3])
@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@torch.inference_mode()
def test_qsa_rmsnorm_mrope_with_positions(rows: int, axes: int) -> None:
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


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires Ascend A3")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available")
@torch.inference_mode()
def test_qsa_rmsnorm_mrope_positions_replay() -> None:
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
