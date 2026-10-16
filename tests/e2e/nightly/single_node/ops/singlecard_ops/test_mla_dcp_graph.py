# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP MLA graph updates must retain hybrid-cache block strides."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.attention.context_parallel import mla_cp
from vllm_ascend.attention.context_parallel.common_cp import CPKVScope
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl


@pytest.mark.parametrize("num_heads", [24, 48, 96])
@pytest.mark.parametrize("query_len", [1, 8])
def test_mla_dcp_graph_strided_history(num_heads: int, query_len: int):
    torch.manual_seed(42)
    device = "npu"
    num_reqs, num_blocks, block_size, nope_dim, rope_dim = 4, 96, 128, 512, 64
    num_tokens = num_reqs * query_len
    query_ends = [query_len * (i + 1) for i in range(num_reqs)]
    q = torch.randn(num_tokens, num_heads, nope_dim, device=device, dtype=torch.bfloat16)
    q_rope = torch.randn(num_tokens, num_heads, rope_dim, device=device, dtype=torch.bfloat16)
    cache = torch.randn(num_blocks, block_size * (nope_dim + rope_dim), device=device, dtype=torch.bfloat16)
    # Each physical page contains separate K and RoPE segments. K is dense
    # within each block, while the next block begins after both segments.
    k = cache[:, : block_size * nope_dim].view(num_blocks, 1, block_size, nope_dim)
    k_rope = cache[:, block_size * nope_dim :].view(num_blocks, 1, block_size, rope_dim)
    assert not k.is_contiguous()
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(num_reqs, -1)
    lengths = [66, 386, 769, 0]
    scale = (nope_dim + rope_dim) ** -0.5
    # The contiguous eager V1 call is independent of the implementation's V2
    # graph path, including its argument translation and task-update binding.
    expected, expected_lse = torch_npu.npu_fused_infer_attention_score(
        q,
        k.contiguous(),
        k.contiguous(),
        query_rope=q_rope,
        key_rope=k_rope.contiguous(),
        num_heads=num_heads,
        num_key_value_heads=1,
        input_layout="TND",
        scale=scale,
        actual_seq_lengths=query_ends,
        actual_seq_lengths_kv=lengths,
        block_table=block_table,
        block_size=block_size,
        softmax_lse_flag=True,
    )
    torch.npu.synchronize()
    workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
        q,
        k,
        k,
        query_rope=q_rope,
        key_rope=k_rope,
        num_query_heads=num_heads,
        num_key_value_heads=1,
        input_layout="TND",
        softmax_scale=scale,
        actual_seq_qlen=query_ends,
        actual_seq_kvlen=lengths,
        block_table=block_table,
        block_size=block_size,
        return_softmax_lse=True,
    )
    params = SimpleNamespace(
        workspaces={num_tokens: workspace},
        events={num_tokens: []},
        handles={num_tokens: []},
        attn_params={num_tokens: []},
    )
    impl = object.__new__(AscendMlaDCPImpl)
    impl.num_kv_heads, impl.scale, impl.layer_name = 1, scale, "mla"
    context = SimpleNamespace(capturing=True, is_draft_model=False, is_draft_model_prefill=False)
    graph = torch.npu.NPUGraph()
    capture_stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    with patch.object(mla_cp, "get_graph_params", return_value=params), patch.object(mla_cp, "_EXTRA_CTX", context):
        with torch.npu.graph(graph, stream=capture_stream):
            output, lse = impl._run_dcp_mtp_split_attention_op(
                q,
                q_rope,
                k,
                k_rope,
                attn_mask=None,
                sparse_mode=0,
                block_table=block_table,
                block_size=block_size,
                actual_seq_lengths=query_ends,
                actual_seq_lengths_kv=[0] * num_reqs,
                attention_kind=CPKVScope.HISTORY,
            )
        decode = SimpleNamespace(actual_seq_lengths_q=query_ends, cp_history_seq_len=lengths, block_table=block_table)
        forward_context = SimpleNamespace(attn_metadata={"mla": SimpleNamespace(decode=decode)})
        for _ in range(2):
            update_stream.wait_stream(torch.npu.current_stream())
            graph.replay()
            impl.update_graph_params(update_stream, forward_context, num_tokens)
            torch.npu.synchronize()
            # The fourth request is graph padding with no history.
            live = 3 * query_len
            assert torch.isfinite(output[:live]).all()
            torch.testing.assert_close(output[:live], expected[:live], atol=0.02, rtol=0.02)
            torch.testing.assert_close(lse[:live], expected_lse[:live], atol=0.02, rtol=0.02)
