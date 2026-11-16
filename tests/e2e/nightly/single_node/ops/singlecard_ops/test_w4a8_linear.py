# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicLinearMethod
from vllm_ascend.quantization.prepared_linear_input import PreparedLinearInput


@pytest.mark.parametrize("shape", [(1, 5120, 24), (7, 1536, 1280), (2, 4352, 5120)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_w4a8_linear_quant_matmul(shape, dtype):
    """Compare the loaded INT4 path with an unpacked CPU reference."""
    torch.manual_seed(42)
    m, k, n = shape
    weight = torch.randint(-8, 8, (n, k), dtype=torch.int8)
    scale = torch.rand(n) * 0.01
    transposed = weight.t().contiguous()
    packed = ((transposed[:, 0::2] & 15) | ((transposed[:, 1::2] & 15) << 4)).to(torch.int8)
    layer = torch.nn.Module()
    for name, tensor in {
        "weight": packed.t().contiguous(),
        "weight_scale": scale[:, None],
        "weight_offset": torch.zeros(n, 1),
        "scale_bias": 8 * (weight.float() * scale[:, None]).sum(dim=1, keepdim=True),
    }.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor.npu(), requires_grad=False))

    method = AscendW4A8DynamicLinearMethod.__new__(AscendW4A8DynamicLinearMethod)
    method.process_weights_after_loading(layer)
    assert layer.weight.shape == (k, n // 8)
    assert layer.weight.dtype == torch.int32

    x = torch.randn(m, k, dtype=dtype).npu()
    quantized_x, token_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    reference = ((quantized_x.cpu().float() @ weight.float().t()) * token_scale.cpu()[:, None] * scale[None, :]).to(
        dtype
    )
    output = method.apply(layer, x)
    torch.testing.assert_close(output.cpu(), reference, rtol=0.02, atol=0.02)

    # Prepared inputs may be reused across projections and graph executions.
    before = quantized_x.clone()
    prepared = PreparedLinearInput(quantized_x, token_scale, dtype)
    for _ in range(2):
        torch.testing.assert_close(method.apply(layer, prepared), output, rtol=0, atol=0)
        assert torch.equal(quantized_x, before)

    bias = torch.randn(n, dtype=dtype).npu()
    torch.testing.assert_close(method.apply(layer, x, bias), output + bias, rtol=0, atol=0)
    torch.testing.assert_close(method.apply(layer, x.reshape(1, m, k)).reshape(m, n), output, rtol=0, atol=0)
    empty = method.apply(layer, x[:0])
    assert empty.shape == (0, n)
    assert empty.dtype == dtype


def test_w4a8_linear_quant_matmul_fake_shape():
    """Per-channel scales must work during graph shape propagation."""
    with FakeTensorMode():
        x = torch.empty(7, 1536, dtype=torch.int8, device="npu")
        weight = torch.empty(1536, 160, dtype=torch.int32, device="npu")
        scale = torch.empty(1280, dtype=torch.int64, device="npu")
        output = torch_npu.npu_quant_matmul(
            x,
            weight,
            scale,
            offset=torch.empty(1280, device="npu"),
            pertoken_scale=torch.empty(7, device="npu"),
            group_sizes=[0, 0, 0],
            output_dtype=torch.bfloat16,
        )
        assert output.shape == (7, 1280)
        assert output.dtype == torch.bfloat16
