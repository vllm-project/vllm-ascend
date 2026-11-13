# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise token-history addressing beyond the signed 32-bit element range."""

import pytest
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.sample.thinking_budget import (
    _load_effective_token_ascend,
    _update_committed_marker_cache_kernel_ascend,
)

DEVICE = torch.device("npu")
REQUEST_SLOT = 2
HISTORY_WIDTH = 8
LARGE_ROW_STRIDE = 2**30
START = 90
END = 91


@triton.jit
def _load_history_token(history, stride, requests, output, inputs):
    slot = tl.load(requests)
    token = _load_effective_token_ascend(history, stride, inputs, 0, slot, 5, 1)
    tl.store(output, token)


@pytest.mark.parametrize("stride", [HISTORY_WIDTH, LARGE_ROW_STRIDE])
@pytest.mark.parametrize("operation", ["load", "cold_scan", "incremental_scan"])
def test_thinking_budget_history_addressing(stride, operation):
    # Use a valid strided tensor whose third row crosses 2**31 int32 elements.
    # Only eight tokens are populated; allocating the backing storage is needed
    # to distinguish address overflow from an invalid input pointer.
    elements = REQUEST_SLOT * stride + HISTORY_WIDTH
    free_bytes, _ = torch.npu.mem_get_info()
    required_bytes = elements * torch.tensor([], dtype=torch.int32).element_size()
    if free_bytes < required_bytes + 1024**3:
        pytest.skip("History-address regression needs 8 GiB of free NPU memory plus headroom")
    storage = torch.empty(elements, dtype=torch.int32, device=DEVICE)
    history = storage.as_strided((REQUEST_SLOT + 1, HISTORY_WIDTH), (stride, 1))

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=DEVICE)

    history[REQUEST_SLOT].copy_(tensor([1, START, 10, 11, 12, END, 0, 0]))
    assert history[REQUEST_SLOT, :6].cpu().tolist() == [1, START, 10, 11, 12, END]
    requests = tensor([REQUEST_SLOT])
    if operation == "load":
        output = tensor([0])
        _load_history_token[(1,)](history, stride, requests, output, tensor([12]))
        assert output.cpu().tolist() == [START]
        return

    incremental = operation == "incremental_scan"
    starts = tensor([17, 18, 1 if incremental else -1])
    ends = tensor([19, 20, -1])
    scans = tensor([21, 22, 5 if incremental else 0])
    _update_committed_marker_cache_kernel_ascend[(1,)](
        requests,
        tensor([-1, -1, 3]),
        history,
        stride,
        tensor([0, 0, 6 if incremental else 5]),
        starts,
        ends,
        scans,
        tensor([START]),
        tensor([END]),
        START_LEN=1,
        NATURAL_END_LEN=1,
        MAX_LEN=1,
        BLOCK=256,
    )
    assert starts.cpu().tolist() == [17, 18, 1]
    assert ends.cpu().tolist() == [19, 20, 5 if incremental else -1]
    assert scans.cpu().tolist() == [21, 22, 6 if incremental else 5]
