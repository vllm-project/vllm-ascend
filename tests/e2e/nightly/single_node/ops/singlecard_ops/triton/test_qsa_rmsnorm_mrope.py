import pytest
import torch
import torch_npu

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
