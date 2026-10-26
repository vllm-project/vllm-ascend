# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real native causal-SWA band intersected with an independent extra Window."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_read import (  # noqa: E402
    NativeSWASource,
    materialize_native_swa_intersection,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU"
)


@pytest.mark.parametrize("ends", [(1151, 1152), (4303, 4352)])
@pytest.mark.parametrize("window", [1000, 1024, 8192])
def test_native_swa_intersection_uses_current_exact_and_canonical_pages(ends, window):
    block, width, heads, kv_heads, dim, native_window = 128, 8, 8, 1, 128, 4096
    generator = torch.Generator().manual_seed(261026)
    columns = math.ceil(max(ends) / block)
    pages = 2 * columns + 1
    canonical = (torch.randperm(2 * columns, generator=generator) + 1).reshape(2, columns).int()
    rows = (1, 0)
    for row, end in zip(rows, ends):
        # Evicted columns precede the earliest key of this request's native
        # band. Nulling them must not create a resident-Full-prefix claim.
        old_columns = max(end - width - native_window, 0) // block
        canonical[row, :old_columns] = 0
    key = (torch.randn(pages, block, kv_heads, dim, generator=generator) * 0.2).bfloat16()
    value = (torch.randn(key.shape, generator=generator) * 0.2).bfloat16()
    query = (torch.randn(2 * width, heads, dim, generator=generator) * 0.2).bfloat16()
    key[0], value[0] = 123, -123
    key_device, value_device = key.npu(), value.npu()
    table = canonical.npu()
    exact = torch.tensor(ends, dtype=torch.int32)
    starts = torch.tensor([max(end - window, 0) // block * block for end in ends], dtype=torch.int32)
    retained = []
    read = materialize_native_swa_intersection(
        NativeSWASource(table, rows, (columns, columns), block, native_window),
        exact_cpu=exact,
        exact_device=exact.npu(),
        upper_cpu=exact + 1,
        starts_cpu=starts,
        padded_rows=2,
        query_width=width,
        row_indices=torch.tensor(rows, dtype=torch.int64).npu(),
        retain=lambda *items: retained.extend(items),
    )
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu()
    actual, _ = torch_npu.npu_fused_infer_attention_score(
        query=query.npu(),
        key=key_device.view(pages, block, -1),
        value=value_device.view(pages, block, -1),
        block_table=read.block_table,
        block_size=block,
        input_layout="TND",
        actual_seq_lengths=[width, 2 * width],
        actual_seq_lengths_kv=read.seq_lens_cpu.tolist(),
        num_key_value_heads=kv_heads,
        num_heads=heads,
        scale=dim**-0.5,
        sparse_mode=4,
        pre_tokens=native_window,
        next_tokens=0,
        atten_mask=mask,
    )
    expected = []
    for compact_row, (persistent_row, end, start) in enumerate(zip(rows, ends, starts.tolist())):
        first = max(end - width - native_window, start, 0)
        logical = torch.arange(first, end)
        page_ids = canonical[persistent_row, logical // block].long()
        assert bool((page_ids > 0).all())
        k = key[page_ids, logical % block].float().repeat_interleave(heads // kv_heads, dim=1)
        v = value[page_ids, logical % block].float().repeat_interleave(heads // kv_heads, dim=1)
        q = query[compact_row * width : (compact_row + 1) * width].float()
        positions = torch.arange(end - width, end)
        scores = torch.einsum("qhd,khd->hqk", q, k) * dim**-0.5
        disallowed = (logical[None] > positions[:, None]) | (logical[None] < positions[:, None] - native_window)
        scores.masked_fill_(disallowed[None], -torch.inf)
        expected.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        stop_column = (end + block - 1) // block
        wanted = canonical[persistent_row, start // block : stop_column]
        torch.testing.assert_close(read.block_table[compact_row, : len(wanted)].cpu(), wanted, atol=0, rtol=0)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.cpu().float(), torch.cat(expected), atol=2**-7, rtol=2**-6)
    torch.testing.assert_close(table.cpu(), canonical, atol=0, rtol=0)
    torch.testing.assert_close(read.seq_lens.cpu(), exact - starts, atol=0, rtol=0)
    torch.testing.assert_close(key_device.cpu(), key, atol=0, rtol=0)
    torch.testing.assert_close(value_device.cpu(), value, atol=0, rtol=0)
