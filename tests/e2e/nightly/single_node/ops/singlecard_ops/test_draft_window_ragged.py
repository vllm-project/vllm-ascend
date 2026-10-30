# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real device exact planning for packed queries and mixed Full/window rows."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import (  # noqa: E402
    DenseDraftReadBatch,
    dense_draft_attention,
    prepare_dense_draft_read,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
TOLERANCES = {torch.bfloat16: (2**-7, 2**-6), torch.float16: (2**-10, 2**-9)}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim,kv_heads", [(192, 4), (128, 1)])
@pytest.mark.parametrize("case", ["parallel", "eagle", "interleaved_eagle"])
def test_native_ragged_exact_read_with_full_continuation(dtype, dim, kv_heads, case):
    eagle = case != "parallel"
    block, window, heads = 128, 1035 if eagle else 1024, 4
    ends: tuple[int, ...] = (1795, 2177) if eagle else (1151, 1408)
    widths: tuple[int, ...] = (4, 257) if eagle else (3, 5)
    starts: tuple[int, ...] = (640, 0) if eagle else (0, 0)
    frozen: tuple[int, ...] | None = (640, 1024) if eagle else None
    native_ends: tuple[int, ...] = (1796, ends[1]) if eagle else ends
    if case == "interleaved_eagle":
        ends, widths, starts = (1151, 2177, 1795, 1535), (4, 33, 4, 33), (0, 0, 640, 0)
        native_ends, frozen = (1152, 2177, 1796, 1535), (0, 1024, 640, 512)
    generator = torch.Generator().manual_seed(4337 + dim)
    canonical = (torch.randperm(128, generator=generator) + 1).reshape(4, 32).to(torch.int32)
    rows = (3, 1, 0, 2)[: len(ends)]
    keys_cpu = (torch.randn(129, block, kv_heads, dim, generator=generator) * 0.2).to(dtype)
    values_cpu = (torch.randn(129, block, kv_heads, dim, generator=generator) * 0.2).to(dtype)
    for row, end in zip(rows, ends):
        for position in range(end, 32 * block):
            physical = int(canonical[row, position // block])
            keys_cpu[physical, position % block] = torch.nan
            values_cpu[physical, position % block] = torch.nan
    query_cpu = (torch.randn(sum(widths), heads, dim, generator=generator) * 0.2).to(dtype)
    # Match combined K/V cache storage with distinct offsets and page stride.
    backing = torch.full((129, 2, 2, block, kv_heads, dim), torch.nan, dtype=dtype, device="npu")
    keys, values = backing[:, 1, 0], backing[:, 1, 1]
    keys.copy_(keys_cpu)
    values.copy_(values_cpu)
    table_backing = torch.full((4, 64), -1, dtype=torch.int32, device="npu")
    table = table_backing[:, ::2]
    table.copy_(canonical)
    table_before = table_backing.clone()
    source = DirectPrefixSource(table, rows, (32,) * len(rows), block)
    offsets = torch.tensor([0, *torch.tensor(widths).cumsum(0).tolist()], dtype=torch.int32)
    retained = []
    read = prepare_dense_draft_read(
        source,
        exact_device=torch.tensor(ends, dtype=torch.int32, device="npu"),
        query_ends=torch.tensor(native_ends, dtype=torch.int32, device="npu"),
        row_indices=torch.tensor(rows, dtype=torch.int64, device="npu"),
        upper_cpu=torch.tensor(ends, dtype=torch.int32) + 1,
        window_size=window,
        anchor_span=block,
        query_width=max(widths),
        causal=eagle,
        frozen_starts=torch.tensor(frozen, dtype=torch.int32, device="npu") if eagle else None,
        full_rows_cpu=torch.tensor([False, True] * (len(rows) // 2)),
        query_start_loc_cpu=offsets,
        query_start_loc_device=offsets.npu(),
        retain=lambda *objects: retained.extend(objects),
    )
    result = dense_draft_attention(read, query_cpu.npu(), keys, values, 1 / math.sqrt(dim))
    torch.npu.synchronize()
    assert read.starts.cpu().tolist() == list(starts)
    assert read.seq_lens.cpu().tolist() == [end - start for end, start in zip(ends, starts)]
    references, query_offset = [], 0
    for index, (row, start, end, width) in enumerate(zip(rows, starts, ends, widths)):
        expected_slots = [int(canonical[row, p // block]) * block + p % block for p in range(start, end)]
        assert isinstance(read, DenseDraftReadBatch)
        part = next(part for part in read.parts if index in part.request_indices)
        local_row = part.request_indices.index(index)
        private = part.read
        local_queries = part.query_indices.reshape(len(part.request_indices), width)[local_row].cpu().tolist()
        assert local_queries == list(range(query_offset, query_offset + width))
        actual_slots = private.slots[local_row].cpu().tolist()
        assert actual_slots[: end - start] == expected_slots
        assert actual_slots[end - start :] == [expected_slots[-1]] * (private.capacity - end + start)
        for local in range(width):
            visible_end = min(end, native_ends[index] - width + local + 1) if eagle else end
            # Independent absolute-coordinate reference: no tested mask or
            # slot tensor participates in the reference calculation.
            k = keys_cpu[canonical[row].long()].flatten(0, 1)[start:visible_end].double()
            v = values_cpu[canonical[row].long()].flatten(0, 1)[start:visible_end].double()
            k, v = k.repeat_interleave(heads // kv_heads, 1), v.repeat_interleave(heads // kv_heads, 1)
            q = query_cpu[query_offset + local].double()
            scores = torch.einsum("hd,khd->hk", q, k) / math.sqrt(dim)
            references.append(torch.einsum("hk,khd->hd", scores.softmax(-1), v))
        query_offset += width
    expected = torch.stack(references).float()
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.cpu().float(), expected, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
    torch.testing.assert_close(table_backing, table_before, atol=0, rtol=0)
    torch.testing.assert_close(keys.cpu(), keys_cpu, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(values.cpu(), values_cpu, atol=0, rtol=0, equal_nan=True)
