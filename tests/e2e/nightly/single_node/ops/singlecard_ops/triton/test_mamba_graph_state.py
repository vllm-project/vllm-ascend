# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.mamba.graph_state import GDNGraphStateUpdater

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Mamba graph-state tests require an Ascend NPU",
)

# Cases cover 1-D/2-D layouts, partial batches, padding, clear tables and no-op rows.
_CASES = [
    dict(groups=1, actual_rows=2, graph_rows=4, width=3, ndim=2, pad=-1, clear_width=None),
    dict(groups=2, actual_rows=0, graph_rows=3, width=2, ndim=2, pad=-1, clear_width=None),
    dict(groups=3, actual_rows=3, graph_rows=3, width=1, ndim=1, pad=-1, clear_width=None),
    dict(groups=2, actual_rows=2, graph_rows=5, width=4, ndim=2, pad=77, clear_width=2),
    dict(groups=1, actual_rows=1, graph_rows=1, width=2, ndim=2, pad=0, clear_width=1),
    dict(groups=4, actual_rows=4, graph_rows=8, width=1, ndim=1, pad=-3, clear_width=None),
    dict(groups=1, actual_rows=0, graph_rows=0, width=2, ndim=2, pad=-1, clear_width=None),
]


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("case", _CASES)
def test_graph_state_updater_matches_torch_copy(case):
    device = torch.device("npu")
    groups = case["groups"]
    actual_rows = case["actual_rows"]
    graph_rows = case["graph_rows"]
    width = case["width"]
    ndim = case["ndim"]
    pad = case["pad"]
    clear_width = case["clear_width"]

    updates = []
    sources = []
    destinations = []
    clears = []
    for group in range(groups):
        if ndim == 2:
            source = (
                torch.arange(actual_rows * width, dtype=torch.int32, device=device).view(actual_rows, width)
                + group * 100
            )
        else:
            source = torch.arange(actual_rows, dtype=torch.int32, device=device) + group * 100
        destination = torch.full(
            (graph_rows, width) if ndim == 2 else (graph_rows,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        clear = (
            torch.full((graph_rows, clear_width), -1, dtype=torch.int32, device=device)
            if clear_width is not None
            else None
        )
        updates.append((source, destination, actual_rows, graph_rows, pad, clear))
        sources.append(source)
        destinations.append(destination)
        clears.append(clear)

    GDNGraphStateUpdater().apply(updates)

    for source, destination, clear in zip(sources, destinations, clears):
        expected = torch.full_like(destination, pad)
        expected[:actual_rows] = source
        assert torch.equal(destination, expected)
        if clear is not None:
            assert torch.equal(clear, torch.full_like(clear, PAD_SLOT_ID))
