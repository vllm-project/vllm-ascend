# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.quantization.nz import cast_mxfp4_swiglu_oai_weight, mxfp8_nz
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env()

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5")


@pytest.mark.parametrize("fp4", [True, False], ids=["mxfp4_alg2", "mxfp8_alg1"])
@pytest.mark.parametrize("group_list_type", [0, 1], ids=["prefix", "count"])
@torch.inference_mode()
def test_gmm_swiglu_oai_mx_quant(fp4, group_list_type):
    torch.npu.set_device(0)
    assert enable_custom_op()
    e, k, n = 3, 128, 128
    counts = torch.tensor([2, 0, 3] if fp4 else [768, 0, 771], dtype=torch.int64)
    m = int(counts.sum())
    # Use an isolated K coordinate and different gate/up values per expert and
    # output block so an incorrect NZ orientation cannot give the right result.
    gate = torch.tensor([[1.0, 4.0], [1.0, 1.0], [4.0, 1.0]])
    up = torch.tensor([[2.0, 6.0], [1.0, 1.0], [-6.0, 2.0]])
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    columns = torch.cat((gate, up), dim=1).repeat_interleave(32, dim=1)
    x = torch.zeros((m + 1, k), dtype=torch.uint8)
    w = torch.zeros((e, k, n), dtype=torch.uint8)
    if fp4:
        x[:, 33] = 2
        code = torch.searchsorted(magnitudes, columns.abs()).to(torch.uint8)
        w[:, 33] = code | ((columns < 0).to(torch.uint8) << 3)
        x = (x[:, 0::2] | (x[:, 1::2] << 4)).npu()
        w = w.transpose(1, 2)
        w = cast_mxfp4_swiglu_oai_weight((w[:, :, 0::2] | (w[:, :, 1::2] << 4)).contiguous().npu())
        ws = torch.full((e, n, k // 64, 2), 127, dtype=torch.uint8, device="npu")
        ws[0, :, 0, 1] = 126
        ws[2, :, 0, 1] = 128
    else:
        x[:, 33] = 1
        w = w.float()
        w[:, 33] = columns
        x = x.float().to(torch.float8_e4m3fn).npu()
        w = mxfp8_nz().cast(w.to(torch.float8_e4m3fn).npu())
        ws = torch.full((e, k // 64, n, 2), 127, dtype=torch.uint8, device="npu")
        ws[0, 0, :, 1] = 126
        ws[2, 0, :, 1] = 128
    xs = torch.full((m + 1, k // 64, 2), 127, dtype=torch.uint8, device="npu")
    xs[:, 0, 1] = 128
    groups = (counts.cumsum(0) if group_list_type == 0 else counts).npu()
    output, scale = torch.ops._C_ascend.gmm_swiglu_oai_mx_quant(
        x, [w], [ws], xs, groups, group_list_type=group_list_type
    )

    # BF16 is the rounding boundary before MX encoding in the original PR.
    dequant = torch.tensor([[1.0], [2.0], [4.0]])
    gate = (gate * dequant).clamp(max=7.0)
    up = (up * dequant).clamp(-7.0, 7.0)
    act = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).to(torch.bfloat16).float()
    exponent = torch.ceil(torch.log2(act.abs() / (6 if fp4 else 448)))
    want_scale = ((exponent + 127).to(torch.uint8).repeat_interleave(counts, dim=0)).view(m, 1, 2)
    scaled = act * torch.exp2(-exponent)
    if fp4:
        # RNE picks the even code on ties.
        codes = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        nearest = (scaled.abs()[..., None] - magnitudes[codes]).abs().argmin(dim=-1)
        code = codes[nearest].to(torch.uint8)
        code |= (scaled < 0).to(torch.uint8) << 3
        want_output = (code * 17).repeat_interleave(16, dim=1)
        assert output.dtype == torch.float4_e2m1fn_x2
        assert output.shape == (m + 1, n // 4)
    else:
        want_output = scaled.repeat_interleave(32, dim=1).to(torch.float8_e4m3fn).view(torch.uint8)
        assert output.dtype == torch.float8_e4m3fn
        assert output.shape == (m + 1, n // 2)
    want_output = want_output.repeat_interleave(counts, dim=0)
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.shape == (m + 1, n // 128, 2)
    torch.testing.assert_close(output.view(torch.uint8).cpu()[:m], want_output, rtol=0, atol=0)
    torch.testing.assert_close(scale.view(torch.uint8).cpu()[:m], want_scale, rtol=0, atol=0)
    assert torch.count_nonzero(scale.view(torch.uint8).cpu()[m:]) == 0


def test_gmm_swiglu_oai_mx_quant_meta():
    assert enable_custom_op()
    op = torch.ops._C_ascend.gmm_swiglu_oai_mx_quant
    e, k, n = 2, 128, 256
    w = torch.empty((e, k, n), dtype=torch.float8_e4m3fn, device="meta")
    ws = torch.empty((e, k // 64, n, 2), dtype=torch.int8, device="meta")
    groups = torch.empty(e, dtype=torch.int64, device="meta")

    def run(m, **kwargs):
        x = torch.empty((m, k), dtype=w.dtype, device="meta")
        xs = torch.empty((m, k // 64, 2), dtype=torch.uint8, device="meta")
        return op(x, [w], [ws], xs, groups, **kwargs)

    with pytest.raises(RuntimeError, match="513 tokens per expert"):
        run(512 * e + 1)
    with pytest.raises(RuntimeError, match="dst_type_max"):
        run(513 * e, dst_type_max=6)
    y, scale = run(513 * e, group_list_type=None)
    assert y.shape == (513 * e, n // 2) and y.dtype == torch.float8_e4m3fn
    assert scale.shape == (513 * e, n // 128, 2) and scale.dtype == torch.float8_e8m0fnu
