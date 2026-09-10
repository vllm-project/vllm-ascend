# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 NoPE SMLA: absent sinks, exact sparse lengths and graph replay."""

import pytest
import torch
import torch_npu  # noqa: F401
from tests.ut.ops.helpers.c_ascend_loader import ensure_c_ascend_loaded

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile


@pytest.mark.parametrize("selected_count", [1, 128, 2048])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_no_sink_and_exact_sparse_length(selected_count, use_graph):
    if get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED:
        pytest.skip("SparseFlashMla original KV requires A5")
    ensure_c_ascend_loaded(required_op="npu_sparse_flash_mla_metadata")
    ensure_c_ascend_loaded(required_op="npu_sparse_flash_mla")
    heads, dim, block, width = 4, 512, 128, 2051
    pages = (selected_count + block - 1) // block
    query = torch.zeros(1, heads, dim, dtype=torch.bfloat16, device="npu")
    cache = torch.ones(pages, block, 1, dim, dtype=torch.bfloat16, device="npu")
    indices = torch.full((1, 1, width), -1, dtype=torch.int32, device="npu")
    indices[0, 0, :selected_count] = torch.arange(selected_count, dtype=torch.int32, device="npu")
    lengths = torch.tensor([[selected_count]], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([selected_count], dtype=torch.int32, device="npu")
    table = torch.arange(pages, dtype=torch.int32, device="npu")[None]
    metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
        num_heads_q=heads,
        num_heads_kv=1,
        head_dim=dim,
        cu_seqlens_q=starts,
        seqused_ori_kv=seq_lens,
        ori_topk_length=lengths,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_ori_kv=selected_count,
        max_seqlen_cmp_kv=0,
        ori_topk=width,
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=3,
        cmp_mask_mode=3,
        ori_win_left=0,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device="npu",
    )

    def run():
        return torch.ops._C_ascend.npu_sparse_flash_mla(
            query,
            ori_kv=cache,
            ori_sparse_indices=indices,
            ori_block_table=table,
            cu_seqlens_q=starts,
            seqused_ori_kv=seq_lens,
            ori_topk_length=lengths,
            sinks=None,
            metadata=metadata,
            softmax_scale=dim**-0.5,
            cmp_ratio=1,
            ori_mask_mode=3,
            cmp_mask_mode=3,
            ori_win_left=0,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=False,
        )[0]

    output = run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph() if use_graph else None
    try:
        if graph is not None:
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                output = run()
        for value in (1.0, 2.0):
            cache.fill_(value)
            if graph is not None:
                graph.replay()
            else:
                output = run()
            # Zero query makes every selected key equiprobable. Any finite
            # sink adds a denominator term and fails the one-key case.
            torch.testing.assert_close(output.cpu(), torch.full_like(query.cpu(), value), rtol=0.01, atol=0.01)
    finally:
        if graph is not None:
            graph.reset()
