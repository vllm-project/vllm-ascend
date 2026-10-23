# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Device exact GQA read-set and uninitialized-tail regression; no engine claim."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import (  # noqa: E402
    dense_draft_attention,
    prepare_dense_draft_read,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
TOLERANCES = {torch.bfloat16: (2**-7, 2**-6), torch.float16: (2**-10, 2**-9)}
# Prespecified dtype tolerances vs an independent FP32 oracle.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("page_layers", [0, 3])
@pytest.mark.parametrize("geometry", [(192, 4, 4), (128, 32, 8)])
@pytest.mark.parametrize(("query_width", "future"), [(1, 0), (1, 2), (8, 0), (9, 0)])
@pytest.mark.parametrize("causal", [False, True])
def test_dense_device_exact_reader(geometry, query_width, future, causal, page_layers, dtype):
    D, NQ, NK = geometry
    B, W = 128, 1024
    ends = (1151, 1152 if query_width == 8 else 1281)
    generator = torch.Generator().manual_seed(914 + D + query_width)
    canonical = (torch.randperm(48, generator=generator) + 1).reshape(4, 12).to(torch.int32)
    canonical[3, 0] = canonical[1, 0]  # Actual shared prefix page, then reorder.
    rows = (3, 1)
    keys_cpu = (torch.randn(49, B, NK, D, generator=generator) * 0.2).to(dtype)
    values_cpu = (torch.randn(49, B, NK, D, generator=generator) * 0.2).to(dtype)
    for row, end in zip(rows, ends):
        for logical in range(end, 12 * B):
            physical = int(canonical[row, logical // B])
            keys_cpu[physical, logical % B] = torch.nan
            values_cpu[physical, logical % B] = torch.nan
    query_cpu = (torch.randn(2 * query_width, NQ, D, generator=generator) * 0.2).to(dtype)
    reference, slots_reference = [], []
    for index, (row, end) in enumerate(zip(rows, ends)):
        start = max(end + future - W, 0) // B * B
        k = keys_cpu[canonical[row].long()].flatten(0, 1)[start:end].float().repeat_interleave(NQ // NK, dim=1)
        v = values_cpu[canonical[row].long()].flatten(0, 1)[start:end].float().repeat_interleave(NQ // NK, dim=1)
        q = query_cpu[index * query_width : (index + 1) * query_width].float()
        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(D)
        if causal:
            scores.masked_fill_(
                torch.arange(start, end)[None, :] > torch.arange(end - query_width, end)[:, None], -torch.inf
            )
        reference.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        slots_reference.append(
            [int(canonical[row, position // B]) * B + position % B for position in range(start, end)]
        )
    keys, values, query = keys_cpu.npu(), values_cpu.npu(), query_cpu.npu()
    if page_layers:
        # Match MRV2 combined, page-strided K/V with a nonzero layer offset.
        backing = torch.full((49, page_layers, 2, B, NK, D), torch.nan, dtype=dtype, device="npu")
        keys, values = backing[:, 2, 0], backing[:, 2, 1]
        keys.copy_(keys_cpu)
        values.copy_(values_cpu)
        assert not keys.is_contiguous() and values.storage_offset() > 0
        backing_before = backing.clone()
    source_table = canonical.npu()
    if page_layers:
        # The planner must use real table strides, without a Full-table copy.
        table_backing = torch.full((4, 24), -1, dtype=torch.int32, device="npu")
        source_table = table_backing[:, ::2]
        source_table.copy_(canonical.npu())
        assert source_table.stride(1) == 2
    source = DirectPrefixSource(source_table, rows, (12, 12), B)
    upper_cpu = torch.tensor(ends, dtype=torch.int32) + 1
    upper_device, rejected = upper_cpu.npu(), torch.ones(2, dtype=torch.int32, device="npu")
    device_rows = torch.tensor(rows, dtype=torch.int64, device="npu")
    # E0=end-1, K=3 freezes H=end+2, independently of later current E.
    frozen_starts = (
        torch.div(torch.clamp(upper_device - rejected - 1 + 3 - W, min=0), B, rounding_mode="floor") * B
        if future
        else None
    )
    retained = []

    def execute():
        read = prepare_dense_draft_read(
            source,
            exact_device=upper_device - rejected,
            row_indices=device_rows,
            upper_cpu=upper_cpu,
            window_size=W,
            anchor_span=B,
            query_width=query_width,
            causal=causal,
            retain=lambda *objects: retained.extend(objects),
            frozen_starts=frozen_starts,
        )
        return read, dense_draft_attention(read, query, keys, values, 1 / math.sqrt(D))

    read, output = execute()
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu().float(), torch.cat(reference), atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1]
    )
    for index, expected in enumerate(slots_reference):
        actual = read.slots[index].cpu().tolist()
        assert actual[: len(expected)] == expected
        assert actual[len(expected) :] == [expected[-1]] * (read.capacity - len(expected))
        assert set(actual) == set(expected)
    for device, original in ((keys, keys_cpu), (values, values_cpu), (query, query_cpu), (source_table, canonical)):
        torch.testing.assert_close(device.cpu(), original, atol=0, rtol=0, equal_nan=True)

    if page_layers:
        torch.testing.assert_close(backing, backing_before, atol=0, rtol=0, equal_nan=True)


def test_bfloat16_probability_cancellation():
    # Two nearly equal probabilities multiply large opposite values. Rounding
    # both probabilities to BF16 first loses the small output; keep FP32 PV.
    block, dim, query_width = 128, 128, 8
    query = torch.zeros((query_width, 1, dim), dtype=torch.bfloat16, device="npu")
    query[:, :, 0] = 1
    keys = torch.zeros((2, block, 1, dim), dtype=torch.bfloat16, device="npu")
    keys[:, :, :, 0] = -9984
    keys[1, 0, :, 0] = 0.0439453125
    keys[1, 1, :, 0] = 0
    values = torch.zeros_like(keys)
    values[1, 0] = 128
    values[1, 1] = -128.5
    source = DirectPrefixSource(torch.tensor([[1]], dtype=torch.int32, device="npu"), (0,), (1,), block)
    retained = []
    read = prepare_dense_draft_read(
        source,
        exact_device=torch.tensor([8], dtype=torch.int32, device="npu"),
        row_indices=torch.tensor([0], device="npu"),
        upper_cpu=torch.tensor([8], device="cpu"),
        window_size=128,
        anchor_span=128,
        query_width=query_width,
        causal=False,
        retain=lambda *objects: retained.extend(objects),
    )
    q, k, v = query.cpu().float(), keys[1, :8].cpu().float(), values[1, :8].cpu().float()
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(dim)
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
    actual = dense_draft_attention(read, query, keys, values, 1 / math.sqrt(dim))
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2**-7, rtol=2**-6)


def test_small_causal_window_empty_rows_are_zero():
    block, query_width, dim = 128, 8, 128
    query = torch.ones((query_width, 1, dim), dtype=torch.bfloat16, device="npu")
    keys = torch.full((3, block, 1, dim), torch.nan, dtype=torch.bfloat16, device="npu")
    values = torch.full_like(keys, torch.nan)
    keys[2, 0] = 1
    values[2, 0] = 7
    source = DirectPrefixSource(torch.tensor([[1, 2]], dtype=torch.int32, device="npu"), (0,), (2,), block)
    retained = []
    read = prepare_dense_draft_read(
        source,
        exact_device=torch.tensor([129], dtype=torch.int32, device="npu"),
        row_indices=torch.tensor([0], device="npu"),
        upper_cpu=torch.tensor([130], device="cpu"),
        window_size=1,
        anchor_span=block,
        query_width=query_width,
        causal=True,
        retain=lambda *objects: retained.extend(objects),
    )
    # Independent per-query read set: positions121..127 have no key after S128;
    # position128 has one key whose value is7. W remains legal and unchanged.
    expected = torch.zeros((query_width, 1, dim), dtype=torch.bfloat16, device="cpu")
    expected[-1] = 7
    actual = dense_draft_attention(read, query, keys, values, 1 / math.sqrt(dim))
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
