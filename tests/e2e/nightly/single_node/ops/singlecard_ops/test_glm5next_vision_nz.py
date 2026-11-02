# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu
from torch import nn

from vllm_ascend.device.device_config import is_950
from vllm_ascend.models.glm5next.multimodal import _vision_projection
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

pytestmark = pytest.mark.e2e_coverage(
    arch="", feature="", parallel="", deploy="", hardware="A5", quantization="", graph_mode="eager"
)


@pytest.fixture(scope="module", autouse=True)
def require_a5():
    if not torch.npu.is_available() or not is_950():
        pytest.skip("GLM vision NZ regression requires Ascend 950")


class VisionProjection(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = bias

    def forward(self, x):
        # RowParallelLinear computes the local GEMM without bias, then adds
        # bias after the TP reduction. Do not fuse it into this reference.
        output = torch.ops.vllm.unquantized_gemm(x, self.weight, None)
        return output if self.bias is None else output + self.bias, None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("use_nz", [False, True])
@pytest.mark.parametrize("shape", [(32,), (1, 32), (7, 32), (2, 3, 32), (2, 2, 3, 32), (0, 32), (2, 0, 32)])
@torch.inference_mode()
def test_glm5next_vision_nz_projection(dtype, has_bias, noncontiguous, use_nz, shape):
    generator = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(shape, generator=generator, dtype=dtype).npu()
    if noncontiguous:
        x = torch.stack((x, x), dim=-1)[..., 0]
    weight = torch.randn(64, 32, generator=generator, dtype=dtype).npu()
    bias = torch.randn(64, generator=generator, dtype=dtype).npu() if has_bias else None
    expected = torch.nn.functional.linear(x, weight)
    if bias is not None:
        expected = expected + bias
    stored_weight = torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ) if use_nz else weight
    layer = VisionProjection(stored_weight, bias)
    actual, _ = _vision_projection(layer, x, use_nz)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    assert actual.shape == (*x.shape[:-1], weight.shape[0])
    if use_nz:
        assert torch_npu.get_npu_format(layer.weight) == ACL_FORMAT_FRACTAL_NZ
    else:
        assert torch.equal(actual, expected)
