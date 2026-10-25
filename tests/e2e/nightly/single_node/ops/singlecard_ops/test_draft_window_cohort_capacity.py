# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real GQA consumers preserve row identity with bounded Full-row padding."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import (  # noqa: E402
    DenseDraftRead,
    dense_draft_attention,
    prepare_dense_draft_read,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("decode_exact", [1151, 1537])
def test_full_row_capacity_cohorts_have_independent_native_operands(decode_exact):
    exact = torch.tensor([decode_exact, 8191, 1152], dtype=torch.int32)
    upper = torch.tensor([decode_exact + 1, 8192, 1152], dtype=torch.int32)
    full = torch.tensor([False, True, True])
    rows = torch.tensor([2, 0, 1], dtype=torch.int64)
    table_cpu = torch.arange(3 * 64, dtype=torch.int32).reshape(3, 64)
    table = table_cpu.to("npu")
    held = []
    read = prepare_dense_draft_read(
        DirectPrefixSource(table, tuple(rows.tolist()), tuple(((upper + 127) // 128).tolist()), 128),
        exact_device=exact.to("npu"),
        row_indices=rows.to("npu"),
        upper_cpu=upper,
        window_size=1024,
        anchor_span=128,
        query_width=8,
        causal=False,
        full_rows_cpu=full,
        retain=lambda *items: held.extend(items),
    )
    keys = torch.zeros((3 * 64, 128, 4, 192), dtype=torch.bfloat16, device="npu")
    values_cpu = torch.arange(1, 4, dtype=torch.bfloat16).repeat_interleave(64)[:, None, None, None]
    values_cpu = values_cpu.expand(3 * 64, 128, 4, 192).contiguous()
    values = values_cpu.to("npu")
    query = torch.ones((3 * 8, 8, 192), dtype=torch.bfloat16, device="npu")
    output = dense_draft_attention(read, query, keys, values, 1.0)
    torch.npu.synchronize()
    expected = torch.tensor([3, 1, 2], dtype=torch.bfloat16).repeat_interleave(8)[:, None, None]
    torch.testing.assert_close(output.cpu(), expected.expand(24, 8, 192), atol=2**-7, rtol=2**-6)
    starts = torch.where(full, 0, (exact - 1024).clamp(min=0) // 128 * 128)
    assert torch.equal(read.starts.cpu(), starts)
    assert torch.equal(read.seq_lens.cpu(), exact - starts)
    assert torch.equal(table.cpu(), table_cpu)
    assert not keys.count_nonzero().item()
    assert torch.equal(values.cpu(), values_cpu)
    parts = (read,) if isinstance(read, DenseDraftRead) else tuple(part.read for part in read.parts)
    private_buffers = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for part in parts
        for pair in part._attention_workspaces.values()
        for tensor in pair
    }
    bounds = torch.where(full, upper, torch.minimum(upper, torch.tensor(1024 + 128 - 1)))
    expected_tokens = int(((bounds + 31) // 32 * 32).sum())
    assert sum(private_buffers.values()) == expected_tokens * 2 * 4 * 192 * 4
    assert len(private_buffers) == 6, "distinct Full/read cohorts share native scratch"
