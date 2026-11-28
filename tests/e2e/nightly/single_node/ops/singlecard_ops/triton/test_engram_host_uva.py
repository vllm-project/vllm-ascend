# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import ExitStack

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41.engram.npu import SCALE_GROUP, HostUvaBuffer, gather_dequantize_host_uva


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
@pytest.mark.parametrize("tokens", [0, 1, 257])
def test_a2_host_uva_capped_grid(dtype, tokens):
    if get_ascend_device_type() != AscendDeviceType.A2:
        pytest.skip("Requires Atlas A2 Host-UVA")
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    owner = torch.nn.Module()
    table_rows, width = 17, 128
    with ExitStack() as stack:
        codes = HostUvaBuffer((table_rows, width), dtype, device, "codes", owner)
        stack.callback(codes.close)
        values = (torch.arange(table_rows * width).reshape(table_rows, width) % 23 - 11).to(dtype)
        assert codes.tensor is not None
        codes.tensor.copy_(values)
        expected_rows = values.float()
        scales = None
        if dtype == torch.int8:
            groups = width // SCALE_GROUP
            scales = HostUvaBuffer((table_rows, groups), torch.float32, device, "scales", owner)
            stack.callback(scales.close)
            factors = (torch.arange(table_rows * groups).reshape(table_rows, groups) % 5 + 1) / 8
            assert scales.tensor is not None
            scales.tensor.copy_(factors)
            expected_rows = expected_rows * factors.repeat_interleave(SCALE_GROUP, dim=1)

        # Exercise nonzero head and vocabulary offsets, invalid IDs and untouched padding.
        host_ids = torch.arange(tokens * 4).reshape(tokens, 4) % (table_rows + 2) + 9
        ids = host_ids.to(device)[:, :3]
        local = host_ids[:, 1:3] - 10
        valid = (local >= 0) & (local < table_rows)
        expected = torch.full((tokens, 3, width), -123, dtype=torch.bfloat16)
        expected[:, :2] = torch.where(valid[..., None], expected_rows[local.clamp(0, table_rows - 1)], 0)
        for percentage in (None, 1, 75):
            output = torch.full_like(expected, -123, device=device).view(-1, width)
            actual = gather_dequantize_host_uva(
                codes,
                scales,
                ids,
                head_start=1,
                local_heads=2,
                pad_heads=3,
                output=output,
                vocab_start=10,
                vocab_end=10 + table_rows,
                core_percentage=percentage,
            )
            assert actual is output
            torch.testing.assert_close(actual.view_as(expected).cpu(), expected, rtol=0, atol=0)
