# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exact Draft reads against an independent CPU reference; no engine claim."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window.dense import (  # noqa: E402
    dense_draft_attention,
    gather_dense_draft_kv,
    prepare_dense_draft_read,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
# Preserve the existing dtype tolerances; the reference does not use read slots/masks.
TOLERANCES = {torch.bfloat16: (2**-7, 2**-6), torch.float16: (2**-10, 2**-9)}


@torch.inference_mode()
def test_wide_kv_gather_preserves_fp32_strides_and_nan_tail():
    # GLM DSpark TP4 has 16 local KV heads of width 192: a 4096-channel tile.
    block, heads, dim = 128, 16, 192
    generator = torch.Generator().manual_seed(4317)
    key_cpu = torch.full((4, block, heads, dim), torch.nan, dtype=torch.bfloat16)
    value_cpu = torch.full_like(key_cpu, torch.nan)
    for cache in (key_cpu, value_cpu):
        cache[1, 0] = torch.randn(heads, dim, generator=generator)
        cache[2, :2] = torch.randn(2, heads, dim, generator=generator)
    key_backing = torch.full((4, 2, block, heads, dim), torch.nan, dtype=torch.bfloat16, device="npu")
    value_backing = torch.full((4, 3, block, heads, dim), torch.nan, dtype=torch.bfloat16, device="npu")
    # ND storage must exercise the fused gather rather than the format-aware fallback.
    key_backing = torch_npu.npu_format_cast(key_backing, 2)
    value_backing = torch_npu.npu_format_cast(value_backing, 2)
    keys, values = key_backing[:, 1], value_backing[:, 2]
    keys.copy_(key_cpu)
    values.copy_(value_cpu)
    key_before, value_before = key_backing.cpu(), value_backing.cpu()
    table_cpu = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    table = table_cpu.npu()
    read = prepare_dense_draft_read(
        table,
        (2, 2),
        block,
        exact_cpu=torch.tensor([129, 130], dtype=torch.int32),
        query_ends_cpu=torch.tensor([129, 130], dtype=torch.int32),
        upper_cpu=torch.tensor([129, 130], dtype=torch.int32),
        frozen_starts=torch.tensor([128, 128], dtype=torch.int32),
        full_rows_cpu=torch.zeros(2, dtype=torch.bool),
        query_width=1,
        causal=True,
        retain=lambda *objects: None,
    )
    assert keys.stride(0) != values.stride(0) and keys.storage_offset() and values.storage_offset()
    assert torch_npu.get_npu_format(keys) in (0, 2) and torch_npu.get_npu_format(values) in (0, 2)
    for cache, actual in zip((key_cpu, value_cpu), gather_dense_draft_kv(read, keys, values)):
        expected = torch.stack(
            (cache[1, 0].expand(32, -1, -1), torch.cat((cache[2, :2], cache[2, 1].expand(30, -1, -1))))
        ).float()
        assert actual.dtype == torch.float32 and actual.isfinite().all()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(table.cpu(), table_cpu, rtol=0, atol=0)
    torch.testing.assert_close(key_backing.cpu(), key_before, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(value_backing.cpu(), value_before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@torch.inference_mode()
def test_ragged_exact_read_preserves_page_strides_and_uninitialized_tail(dtype, causal):
    block, dim, kv_heads, q_heads = 128, 128, 2, 4
    rows, ends, widths = (1, 0), (1151, 1281), (3, 4)
    generator = torch.Generator().manual_seed(4317)
    canonical = torch.randperm(24, generator=generator).reshape(2, 12).to(torch.int32)
    key_cpu = (torch.randn(24, block, kv_heads, dim, generator=generator) * 0.2).to(dtype)
    value_cpu = (torch.randn(24, block, kv_heads, dim, generator=generator) * 0.2).to(dtype)
    for row, end in zip(rows, ends):
        for position in range(end, 12 * block):
            page = int(canonical[row, position // block])
            key_cpu[page, position % block] = value_cpu[page, position % block] = torch.nan
    query_cpu = (torch.randn(sum(widths), q_heads, dim, generator=generator) * 0.2).to(dtype)
    key_backing = torch.full((24, 3, 2, block, kv_heads, dim), torch.nan, dtype=dtype, device="npu")
    value_backing = torch.full((24, 5, 2, block, kv_heads, dim), torch.nan, dtype=dtype, device="npu")
    keys, values = key_backing[:, 1, 0], value_backing[:, 2, 1]
    keys.copy_(key_cpu)
    values.copy_(value_cpu)
    key_before, value_before = key_backing.cpu(), value_backing.cpu()
    assert keys.stride(0) != values.stride(0) and keys.storage_offset() and values.storage_offset()
    full_table = canonical.npu()
    compact_backing = torch.full((2, 24), -1, dtype=torch.int32, device="npu")
    compact = compact_backing[:, ::2]
    compact.copy_(full_table[list(rows)])
    starts = torch.tensor([max(end - 1024, 0) // block * block for end in ends], dtype=torch.int32)
    exact = torch.tensor(ends, dtype=torch.int32)
    boundaries = torch.tensor([0, 3, 7], dtype=torch.int32)
    retained = []
    read = prepare_dense_draft_read(
        compact,
        (12, 12),
        block,
        exact_cpu=exact,
        query_ends_cpu=exact + 1,
        upper_cpu=exact + 2,
        frozen_starts=starts,
        full_rows_cpu=torch.zeros(2, dtype=torch.bool),
        query_width=max(widths),
        query_start_loc_cpu=boundaries,
        query_start_loc_device=boundaries.npu(),
        causal=causal,
        retain=lambda *items: retained.extend(items),
    )
    references = []
    query_begin = 0
    for row, end, width, start in zip(rows, ends, widths, starts.tolist()):
        key = key_cpu[canonical[row].long()].flatten(0, 1)[start:end].double().repeat_interleave(2, dim=1)
        value = value_cpu[canonical[row].long()].flatten(0, 1)[start:end].double().repeat_interleave(2, dim=1)
        q = query_cpu[query_begin : query_begin + width].double()
        scores = torch.einsum("qhd,khd->hqk", q, key) / math.sqrt(dim)
        if causal:
            positions = torch.arange(end + 1 - width, end + 1)
            scores.masked_fill_(torch.arange(start, end)[None, :] > positions[:, None], -torch.inf)
        references.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value))
        query_begin += width
    output = dense_draft_attention(read, query_cpu.npu(), keys, values, 1 / math.sqrt(dim))
    actual = output.cpu().double()
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, torch.cat(references), atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
    # Neither read preparation nor attention may alter Full source/write operands.
    torch.testing.assert_close(full_table.cpu(), canonical, atol=0, rtol=0)
    torch.testing.assert_close(compact.cpu(), canonical[list(rows)], atol=0, rtol=0)
    torch.testing.assert_close(key_backing.cpu(), key_before, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(value_backing.cpu(), value_before, atol=0, rtol=0, equal_nan=True)


@torch.inference_mode()
def test_bfloat16_probability_cancellation_keeps_fp32_attention():
    block, dim, width = 128, 128, 8
    query = torch.zeros((width, 1, dim), dtype=torch.bfloat16)
    query[:, :, 0] = 1
    keys = torch.zeros((2, block, 1, dim), dtype=torch.bfloat16)
    keys[:, :, :, 0] = -9984
    keys[1, 0, :, 0], keys[1, 1, :, 0] = 0.0439453125, 0
    values = torch.zeros_like(keys)
    values[1, 0], values[1, 1] = 128, -128.5
    retained = []
    read = prepare_dense_draft_read(
        torch.tensor([[1]], dtype=torch.int32, device="npu"),
        (1,),
        block,
        exact_cpu=torch.tensor([8], dtype=torch.int32),
        query_ends_cpu=torch.tensor([8], dtype=torch.int32),
        upper_cpu=torch.tensor([8], dtype=torch.int32),
        frozen_starts=torch.tensor([0], dtype=torch.int32),
        full_rows_cpu=torch.zeros(1, dtype=torch.bool),
        query_width=width,
        causal=False,
        retain=lambda *items: retained.extend(items),
    )
    # Rounding the two leading probabilities to BF16 before PV loses this residual.
    q, k, v = query.double(), keys[1, :8].double(), values[1, :8].double()
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(dim)
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
    actual = dense_draft_attention(read, query.npu(), keys.npu(), values.npu(), 1 / math.sqrt(dim))
    torch.testing.assert_close(actual.cpu().double(), expected, atol=2**-7, rtol=2**-6)
