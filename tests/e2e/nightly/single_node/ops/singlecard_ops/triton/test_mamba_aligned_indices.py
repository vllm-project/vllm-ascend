# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.mamba.aligned_indices import compute_aligned_state_indices

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Mamba aligned-index tests require an Ascend NPU",
)

# Real callers size the table from the aligned Mamba block table and the
# per-request slot window; these cases cover the window and flat dispatches.
_CASES = [
    dict(num_groups=1, max_reqs=1, num_reqs=1, state_slots=1, table_cols=16, cache_block_size=16, seq_lens=[1]),
    dict(num_groups=1, max_reqs=2, num_reqs=2, state_slots=4, table_cols=16, cache_block_size=16, seq_lens=[33, 63]),
    dict(
        num_groups=2,
        max_reqs=8,
        num_reqs=3,
        state_slots=4,
        table_cols=20,
        cache_block_size=16,
        seq_lens=[1, 17, 65],
        table_stride=24,
        seq_dtype=torch.int64,
    ),
    dict(
        num_groups=1,
        max_reqs=3,
        num_reqs=2,
        state_slots=1,
        table_cols=8,
        cache_block_size=16,
        seq_lens=[1, 40],
        table_stride=12,
    ),
    dict(num_groups=1, max_reqs=4, num_reqs=0, state_slots=2, table_cols=8, cache_block_size=16, seq_lens=[]),
    dict(
        num_groups=24,
        max_reqs=4,
        num_reqs=4,
        state_slots=4,
        table_cols=16,
        cache_block_size=16,
        seq_lens=[1, 16, 31, 49],
    ),
]


def _golden(tables, seq_lens, num_reqs, state_slots, cache_block_size, shape):
    expected = torch.full(shape, -99, dtype=torch.int32, device=tables.device)
    for group in range(tables.shape[0]):
        for row in range(num_reqs):
            first_slot = max((int(seq_lens[row]) - 1) // cache_block_size, 0)
            for slot in range(state_slots):
                source_col = first_slot + min(slot, state_slots - 1)
                expected[group, row, slot] = tables[group, row, source_col]
    return expected


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("case", _CASES)
def test_aligned_state_indices_match_torch_gather(case):
    device = torch.device("npu")
    num_groups = case["num_groups"]
    max_reqs = case["max_reqs"]
    num_reqs = case["num_reqs"]
    state_slots = case["state_slots"]
    table_cols = case["table_cols"]
    table_stride = case.get("table_stride", table_cols)
    seq_dtype = case.get("seq_dtype", torch.int32)

    storage = torch.arange(
        num_groups * max_reqs * table_stride,
        dtype=torch.int32,
        device=device,
    ).view(num_groups, max_reqs, table_stride)
    tables = storage[:, :, :table_cols]
    table_ptrs = torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device=device)
    seq_lens = torch.tensor(case["seq_lens"], dtype=seq_dtype, device=device)
    output = torch.full((num_groups, max_reqs, state_slots), -99, dtype=torch.int32, device=device)

    actual = compute_aligned_state_indices(
        table_ptrs,
        seq_lens,
        output,
        num_reqs,
        storage.stride(1),
        table_cols,
        case["cache_block_size"],
    )
    expected = _golden(tables, case["seq_lens"], num_reqs, state_slots, case["cache_block_size"], output.shape)

    assert torch.equal(actual, expected[:, :num_reqs])
    assert torch.equal(output[:, num_reqs:], torch.full_like(output[:, num_reqs:], -99))
