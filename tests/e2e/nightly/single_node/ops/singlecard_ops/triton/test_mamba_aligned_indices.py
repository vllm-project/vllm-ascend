# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.mamba.aligned_indices import compute_aligned_state_indices

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Mamba aligned-index tests require an Ascend NPU",
)


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize(
    "num_groups,max_reqs,slots,lengths,seq_dtype",
    [(1, 4, 1, [], torch.int32), (3, 9, 3, [0, 16, 17, 129], torch.int64), (24, 4, 17, [1, 16, 17, 241], torch.int32)],
    ids=["empty", "partial-unaligned", "full-wide"],
)
def test_aligned_state_indices_match_torch_gather(num_groups, max_reqs, slots, lengths, seq_dtype):
    num_reqs, columns, block_size = len(lengths), 32, 16
    storage = torch.arange(num_groups * max_reqs * columns * 2, dtype=torch.int32).view(num_groups, max_reqs, -1)
    # Exercise exact INT32 IDs, including bit patterns that would change in an FP32 conversion.
    storage[0, 0, :8] = torch.tensor([0, 1, -1, 2**24 + 1, 2**31 - 1, -(2**31), 0x7F800001, -0x7FFFFF])
    tables = storage.to("npu")[:, :, :columns]
    output = torch.full((num_groups, max_reqs, slots), -99, dtype=torch.int32, device="npu")
    ctx = SimpleNamespace(
        is_initialized=True,
        block_table_ptrs=torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu"),
        block_table_stride_req=tables.stride(1),
        block_size=block_size,
        aligned_state_indices=output,
    )
    seq_lens = torch.tensor(lengths, dtype=seq_dtype)
    actual = compute_aligned_state_indices(ctx, seq_lens.to("npu"), num_reqs)

    first_slot = ((seq_lens.long() - 1) // block_size).clamp_min(0)
    indices = (first_slot[:, None] + torch.arange(slots)).expand(num_groups, -1, -1)
    expected = storage[:, :num_reqs, :columns].gather(2, indices)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.all(output[:, num_reqs:].cpu() == -99)
