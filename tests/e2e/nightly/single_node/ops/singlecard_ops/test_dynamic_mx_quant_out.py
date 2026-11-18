# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="MXFP8 requires A5")
@pytest.mark.parametrize("tokens,hidden", [(0, 5120), (6, 5120), (129, 5120), (384, 15360), (4096, 5120)])
@torch.inference_mode()
def test_dynamic_mx_quant_out(tokens, hidden):
    """Caller-owned packed scale storage, changing inputs and graph replay."""
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="npu")
    storage = torch.full((tokens + 1, hidden), 23, dtype=torch.uint8, device="npu")
    scale_storage = torch.full((tokens + 1, hidden // 64), 23, dtype=torch.int16, device="npu")
    out = storage[:tokens].view(torch.float8_e4m3fn)
    scale = scale_storage[:tokens].view(torch.uint8).view(tokens, hidden // 64, 2)
    op = torch.ops._C_ascend.dynamic_mx_quant_out
    op(x, out, scale)
    if tokens == 0:
        assert torch.all(storage == 23)
        assert torch.all(scale_storage == 23)
        return
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        op(x, out, scale)
    for magnitude in (0, 1, 1024):
        x.normal_().mul_(magnitude)
        expected, expected_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(scale.view(torch.uint8), expected_scale.view(torch.uint8))
        assert torch.all(storage[tokens:] == 23)
        assert torch.all(scale_storage[tokens:] == 23)
    if tokens == 6:

        def quant_into_buffers(value, result, result_scale):
            op(value, result, result_scale)
            return result, result_scale

        compiled = torch.compile(
            quant_into_buffers, backend="npugraph_ex", fullgraph=True, dynamic=False, options={"clone_input": False}
        )
        for _ in range(3):
            x.normal_()
            expected, expected_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            result, result_scale = compiled(x, out, scale)
            assert result.data_ptr() == out.data_ptr()
            assert result_scale.data_ptr() == scale.data_ptr()
            assert torch.equal(result.view(torch.uint8), expected.view(torch.uint8))
            assert torch.equal(result_scale, expected_scale.view(torch.uint8))
