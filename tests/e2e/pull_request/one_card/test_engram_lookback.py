# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU backend
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("num_reqs", [1, 4])
@pytest.mark.parametrize("mapping_dtype", [torch.int32, torch.int64])
def test_gather_lookback_masked_history(depth, num_reqs, mapping_dtype):
    """New slot-zero requests must never form an address before history."""
    max_num_reqs = 4
    seq_stride = 16384
    tokens_cpu = (
        torch.arange(max_num_reqs, dtype=torch.int32)[:, None] * (seq_stride + 1)
        + torch.arange(seq_stride, dtype=torch.int32)[None, :]
    )
    # Allocate history first, so a fresh allocator can place row zero at its
    # segment boundary. A masked negative position must still use a safe address.
    tokens = tokens_cpu.to("npu")
    mapping_cpu = torch.tensor([0] if num_reqs == 1 else [3, 2, 1, 0], dtype=mapping_dtype)
    mapping = mapping_cpu.to("npu")
    computed = torch.empty(max_num_reqs, dtype=torch.int32, device="npu")
    window = torch.empty((max_num_reqs, depth), dtype=torch.int32, device="npu")
    # Include the observed mixed prefill/decode batch, short histories, a full
    # lookback, and the last valid history column. Row zero is cold in the first
    # two cases, both alone and at the end of a reordered batch.
    computed_cases = [(0, 0, 0, 17)]
    for cursor in dict.fromkeys((0, 1, depth - 1, depth, 18, seq_stride)):
        computed_cases.append((cursor,) * max_num_reqs)
    for computed_values in computed_cases:
        computed_cpu = torch.tensor(computed_values, dtype=torch.int32)
        computed.copy_(computed_cpu)
        window.fill_(-777)
        expected = torch.full((max_num_reqs, depth), -1, dtype=torch.int32)
        for batch_idx, req_idx in enumerate(mapping_cpu.tolist()):
            cursor = int(computed_cpu[req_idx])
            available = min(cursor, depth)
            if available:
                expected[batch_idx, :available] = tokens_cpu[req_idx, cursor - available : cursor].flip(0)
        _gather_lookback_kernel[(max_num_reqs,)](
            window,
            mapping,
            computed,
            tokens,
            tokens.stride(0),
            num_reqs,
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        torch.npu.synchronize()
        torch.testing.assert_close(window.cpu(), expected, rtol=0, atol=0)
