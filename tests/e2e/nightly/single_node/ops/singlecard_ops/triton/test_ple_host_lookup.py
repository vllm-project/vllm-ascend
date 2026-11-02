# SPDX-License-Identifier: Apache-2.0
"""Real NPU regression for the PLE host-UVA lookup Triton kernel."""

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer
from vllm_ascend.ops.triton.qwen4_exp.ple_lookup import lookup

pytestmark = [
    pytest.mark.skipif(
        get_ascend_device_type() == AscendDeviceType.A5,
        reason="A5 host-UVA lookup not yet validated",
    ),
]


@pytest.mark.parametrize("graph", [False, True])
def test_host_lookup(graph):
    torch.npu.set_device(0)
    table = HostUvaBuffer((513, 160), torch.bfloat16, "npu:0")
    try:
        values = torch.arange(513 * 160).reshape(513, 160).remainder(127).bfloat16()
        table.tensor.copy_(values)
        # Exercise pointer boundaries without allocating a multi-GiB table.
        base = table.ptrs[0].item()
        pointers = torch.tensor([base + i * 160 * 2 for i in range(0, 513, 128)], device="npu", dtype=torch.int64)
        ids = torch.tensor([[9, 10, 137, 138, 521, 522, 523, -1] * 2], device="npu")
        output = torch.empty((1, 16, 160), dtype=torch.bfloat16, device="npu")

        def run():
            lookup(pointers, ids, output, 10, 523, 128)

        for _ in range(3):
            run()
        torch.npu.synchronize()
        capture = torch.npu.NPUGraph() if graph else None
        if graph:
            with torch.npu.graph(capture):
                run()
        for shift in (0, 1, -4, 17):
            host_ids = torch.tensor([[9, 10, 137, 138, 521, 522, 523, -1] * 2]) + shift
            ids.copy_(host_ids)
            output.fill_(float("nan"))
            capture.replay() if graph else run()
            owned = (host_ids >= 10) & (host_ids < 523)
            expected = values[(host_ids - 10).clamp(0, 512)]
            expected = torch.where(owned.unsqueeze(-1), expected, 0)
            torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        if capture:
            capture.reset()
    finally:
        table.close()
        table.close()
