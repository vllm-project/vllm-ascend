# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.ops  # noqa: F401 -- initialize registered ops before model helpers
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41.engram import npu as engram

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires A5 UVA")


@pytest.fixture(scope="module", params=["bf16", "int8", "mxfp8"])
def host_table(request):
    torch.npu.set_device(0)
    kind = request.param
    dtype = {"bf16": torch.bfloat16, "int8": torch.int8, "mxfp8": torch.float8_e4m3fn}[kind]
    codes = engram.HostUvaBuffer((17, 128), dtype, torch.device("npu:0"))
    scales = None
    try:
        values = (torch.arange(17 * 128).reshape(17, 128) % 13 - 6).float()
        codes.tensor.copy_(values.to(dtype))
        reference = codes.tensor.float()
        if kind != "bf16":
            scales = engram.HostUvaBuffer(
                (17, 4), torch.uint8 if kind == "mxfp8" else torch.float32, torch.device("npu:0")
            )
            exponents = torch.arange(68).reshape(17, 4) % 5 - 2
            factors = 2.0 ** exponents.float()
            scales.tensor.copy_(exponents + 127 if kind == "mxfp8" else factors)
            reference = reference * factors.repeat_interleave(32, dim=1)
        yield codes, scales, reference
    finally:
        if scales is not None:
            scales.close()
        codes.close()


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens", [0, 1, 7, 129])
@pytest.mark.parametrize("limited", [False, True])
@torch.inference_mode()
def test_uva_grid_stride_preserves_rows_and_padding(host_table, monkeypatch, id_dtype, tokens, limited):
    codes, scales, reference = host_table
    monkeypatch.setattr(engram, "get_ascend_config", lambda: SimpleNamespace(engram_prefill_uva_aiv_limit=limited))
    # Heads are contiguous, while row stride exceeds the number of heads.
    storage = (torch.arange(tokens * 10).reshape(tokens, 10) % 21 + 98).to(id_dtype).npu()
    ids = storage[:, :5]
    selected = ids.cpu()[:, 1:4].long()
    owned = (selected >= 100) & (selected < 117)
    expected = torch.full((tokens, 4, 128), -320, dtype=torch.bfloat16)
    expected[:, :3] = reference[(selected - 100).clamp(0, 16)].bfloat16()
    expected[:, :3][~owned] = 0
    output = torch.full((tokens * 4, 128), -320, dtype=torch.bfloat16, device="npu")

    def run():
        return engram.gather_dequantize_host_uva(
            codes,
            scales,
            ids,
            head_start=1,
            local_heads=3,
            pad_heads=4,
            output=output,
            vocab_start=100,
            vocab_end=117,
        )

    actual = run()
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual.cpu().view(tokens, 4, 128), expected, rtol=0, atol=0)
    if tokens == 129:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        output.fill_(123)
        graph.replay()
        torch.npu.synchronize()
        expected[:, 3] = 123  # Neither execution nor replay may overwrite padded heads.
        torch.testing.assert_close(output.cpu().view(tokens, 4, 128), expected, rtol=0, atol=0)
