# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DCP8 rank-local QSFA boundary and changed-input graph replay on A3/A5.

The physical page is 768 tokens and the operator page is 128 tokens. This
exercises the 138-column view at the 135807 + 4 global-token boundary without
requiring a full model or issuing an illegal device access on the old version.
"""

import importlib
import math

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    get_sfa_dcp_max_local_block_table_cols,
)


@pytest.mark.parametrize("batch", [2, 30])
@torch.inference_mode()
def test_local_block_table_refresh_between_graph_replays(batch):
    backing = torch.zeros(batch, 144, dtype=torch.int32, device="npu")
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.max_local_block_table_cols = 138
    builder.dcp_local_seq_lens_buf = torch.empty(batch, dtype=torch.int32, device="npu")
    builder._local_block_table_buffers = {}
    table = builder._get_dcp_local_block_table(backing, batch)
    pages = torch.arange(batch * 2, dtype=torch.float32, device="npu")

    def read_pages():
        return torch.index_select(pages, 0, table[:, 0].to(torch.int64))

    for _ in range(3):
        read_pages()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = read_pages()
    for offset in (batch, 0):
        expected = torch.arange(batch, dtype=torch.int32) + offset
        backing[:, 0].copy_(expected.npu())
        table = builder._get_dcp_local_block_table(backing, batch)
        graph.replay()
        assert torch.equal(result.cpu(), expected.float())
        torch.testing.assert_close(result, read_pages(), rtol=0, atol=0)
    graph.reset()


@pytest.mark.parametrize("batch", [1, 30])
@pytest.mark.parametrize("local_len", [17535, 17536, 17537, 17539, 17664])
@torch.inference_mode()
def test_qsfa_dcp_tail_eager_and_graph(batch, local_len):
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    cols = get_sfa_dcp_max_local_block_table_cols(140000, 768, 8, 6)
    assert cols == 138
    page_size, heads, nope, rope, queries = 128, 96, 512, 64, 4
    dtype = torch.float8_e4m3fn if "950" in torch.npu.get_device_name() else torch.int8
    # Each request has its own pages. The final column has a different value
    # so dropping or misaddressing the tail changes the numerical result.
    base = (torch.arange(batch) % 4 + 1).float()
    values = base[:, None].expand(batch, cols).clone()
    values[:, 0] += 16
    values[:, -1] += 4
    # Match the cache's storage rounding in the reference (e.g. FP8 rounds 19 to 20).
    values = values.to(dtype).float()
    key_nope = values[:, :, None, None, None].expand(batch, cols, page_size, 1, nope).contiguous().to(dtype)
    key_rope = torch.zeros(batch, cols, page_size, 1, rope, dtype=torch.bfloat16)
    scales = torch.ones(batch, cols, page_size, 1, 4, dtype=torch.float32)
    packed = torch.cat((key_nope.view(torch.uint8), key_rope.view(torch.uint8), scales.view(torch.uint8)), -1)
    cache = packed.reshape(batch * cols, page_size, 1, 656).view(dtype).npu()
    query = torch.zeros(batch * queries, heads, nope + rope, dtype=torch.bfloat16, device="npu")
    backing = torch.zeros(batch, cols + 6, dtype=torch.int32, device="npu")
    backing[:, :cols].copy_(torch.arange(batch * cols, dtype=torch.int32, device="npu").reshape(batch, cols))
    builder = AscendSFADCPMetadataBuilder.__new__(AscendSFADCPMetadataBuilder)
    builder.max_local_block_table_cols = cols
    builder.dcp_local_seq_lens_buf = torch.empty(batch, dtype=torch.int32, device="npu")
    builder._local_block_table_buffers = {}
    table = builder._get_dcp_local_block_table(backing, batch)
    indices = torch.full((batch * queries, 1, 2048), -1, dtype=torch.int32, device="npu")
    q_lens = torch.arange(queries, (batch + 1) * queries, queries, dtype=torch.int32, device="npu")
    kv_lens = torch.full((batch,), local_len, dtype=torch.int32, device="npu")

    def run():
        return torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
            query,
            cache,
            cache,
            indices,
            (nope + rope) ** -0.5,
            key_quant_mode=2,
            value_quant_mode=2,
            block_table=table,
            actual_seq_lengths_query=q_lens,
            actual_seq_lengths_kv=kv_lens,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=0,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            rope_head_dim=rope,
            return_softmax_lse=True,
        )

    def select(count):
        selected = torch.arange(local_len - count, local_len, dtype=torch.int32)
        indices.fill_(-1)
        indices[:, 0, :count].copy_(selected.to("npu"))
        return selected

    def check(result, selected):
        output, maximum, total = [tensor.cpu().float() for tensor in result]
        page_ids = backing.cpu()[:, :cols][:, selected.long() // page_size]
        reference = values.flatten()[page_ids.long()].mean(-1)
        reference = reference.repeat_interleave(queries)[:, None, None].expand_as(output)
        torch.testing.assert_close(output, reference, atol=0.04, rtol=0.01)
        lse = maximum + total.log()
        torch.testing.assert_close(lse, torch.full_like(lse, math.log(selected.numel())), atol=1e-5, rtol=1e-5)

    selected = select(2048)
    for _ in range(3):
        result = run()
    check(result, selected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = run()
    # Rebuild metadata outside capture after changing the physical KV pages.
    last_page = (local_len - 1) // page_size
    backing[:, [0, last_page]] = backing[:, [last_page, 0]]
    table = builder._get_dcp_local_block_table(backing, batch)
    selected = select(32)
    graph.replay()
    torch.npu.synchronize()
    check(result, selected)
    eager = run()
    torch.testing.assert_close(result[0], eager[0])
    graph.reset()
