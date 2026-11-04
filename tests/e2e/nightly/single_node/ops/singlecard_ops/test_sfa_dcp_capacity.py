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

from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    get_sfa_dcp_local_block_table,
    get_sfa_dcp_max_local_block_table_cols,
)


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
    values[:, -1] += 4
    key_nope = values[:, :, None, None, None].expand(batch, cols, page_size, 1, nope).contiguous().to(dtype)
    key_rope = torch.zeros(batch, cols, page_size, 1, rope, dtype=torch.bfloat16)
    scales = torch.ones(batch, cols, page_size, 1, 4, dtype=torch.float32)
    packed = torch.cat((key_nope.view(torch.uint8), key_rope.view(torch.uint8), scales.view(torch.uint8)), -1)
    cache = packed.reshape(batch * cols, page_size, 1, 656).view(dtype).npu()
    query = torch.zeros(batch * queries, heads, nope + rope, dtype=torch.bfloat16, device="npu")
    table = get_sfa_dcp_local_block_table(
        torch.arange(batch * cols, dtype=torch.int32, device="npu").reshape(batch, cols), batch, cols
    )
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
        reference = values[:, selected.long() // page_size].mean(-1)
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
    # Replay must consume newly selected tail tokens at the captured pointers.
    selected = select(32)
    graph.replay()
    torch.npu.synchronize()
    check(result, selected)
